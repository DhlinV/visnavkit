from functools import partial
from pathlib import Path

import nvidia.dali as dali
import nvidia.dali.fn as fn
import torch
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from navigators.data.dali_augmentations import (
    apply_color_jitter,
    apply_crop,
    apply_horizontal_flip,
    apply_jitter,
    apply_warp_perspective,
)
from navigators.data.file_list import resolve_path, video_files_from_file_list
from navigators.data.pose_targets import dali_pose_target_loader
from navigators.utils.common import build_idxs

NOMINAL_FPS = 20


def _video_reader_and_frames(
    file_list,
    sequence_length,
    shard_id,
    num_shards,
    frame_step,
    seq_step,
    steps_between_samples,
    crop_xy,
    crop_wh,
    reader_seed,
    shuffle,
    reader_kwargs,
    frame_wh,
    use_augs,
    p_hflip,
    use_warp_perspective,
    p_warp_perspective,
    warp_angle_range,
    warp_focal,
    use_jitter,
    p_jitter,
    jitter_n_degree,
    color_jitter_brightness,
    color_jitter_contrast,
    color_jitter_saturation,
    color_jitter_hue,
):
    """
    DALI subgraph: decode video, augment, crop, pair consecutive frames, transpose to NCHW.

    Returns ``(videos, label, start_frame_num, do_hflip)`` where ``videos`` is paired-frame
    tensors for the rest of the training pipeline or vision-only consumers.
    """
    # `sequence_length` is the final number of paired frames we want to output.
    # DALI decodes this sequence with stride=frame_step, so adjacent decoded frames
    # are `frame_step` source-video frames apart. We then build pairs from decoded
    # indices (i, i + 1), where i = k * seq_step.
    # max decoded index used is (sequence_length - 1) * seq_step + 1.
    reader_sequence_length = (sequence_length - 1) * seq_step + 2

    videos, label, start_frame_num = fn.readers.video(  # pyright: ignore[reportGeneralTypeIssues]
        name="video_reader",
        device="gpu",
        file_list=file_list,
        file_list_frame_num=True,
        enable_frame_num=True,
        file_list_include_preceding_frame=True,
        random_shuffle=shuffle,
        seed=reader_seed,
        shard_id=shard_id,
        num_shards=num_shards,
        stick_to_shard=True,
        sequence_length=reader_sequence_length,
        stride=frame_step,
        step=steps_between_samples,
        **reader_kwargs,
    )

    videos, warp_homography = apply_warp_perspective(
        videos=videos,
        frame_wh=frame_wh,
        enabled=use_augs and use_warp_perspective,
        probability=p_warp_perspective,
        angle_range=warp_angle_range,
        focal=warp_focal,
    )
    videos = apply_crop(videos, crop_xy, crop_wh)
    videos, do_hflip = apply_horizontal_flip(videos, enabled=use_augs, probability=p_hflip)
    videos = apply_jitter(
        videos,
        reader_sequence_length=reader_sequence_length,
        enabled=use_augs and use_jitter,
        probability=p_jitter,
        n_degree=jitter_n_degree,
    )
    videos = apply_color_jitter(
        videos,
        brightness=color_jitter_brightness if use_augs else 0.0,
        contrast=color_jitter_contrast if use_augs else 0.0,
        saturation=color_jitter_saturation if use_augs else 0.0,
        hue=color_jitter_hue if use_augs else 0.0,
    )

    paired_frames = []
    for k in range(sequence_length):
        i = k * seq_step
        f0 = fn.slice(videos, axes=[0], start=[i], shape=[1])
        f1 = fn.slice(videos, axes=[0], start=[i + 1], shape=[1])
        paired_frames.append(fn.cat(f0, f1, axis=3))
    videos = fn.cat(*paired_frames, axis=0)

    # NHWC -> NCHW
    videos = fn.transpose(videos, perm=[0, 3, 1, 2])
    return videos, label, start_frame_num, do_hflip, warp_homography


@dali.pipeline_def(enable_conditionals=True)
def dali_pipeline(
    file_list,
    video_files,
    sequence_length,
    shard_id,
    num_shards,
    frame_step,
    seq_step,
    steps_between_samples,
    crop_xy,
    crop_wh,
    t_anchors,
    num_pts,
    use_full_pose,
    reader_seed,
    shuffle,
    reader_kwargs={},
    frame_wh=None,
    # ---- augmentations ----
    use_augs=True,
    p_hflip=0.5,
    use_warp_perspective=True,
    p_warp_perspective=0.5,
    warp_angle_range=(-3.0, 3.0),
    warp_focal=None,
    use_jitter=True,
    p_jitter=0.5,
    jitter_n_degree=2,
    color_jitter_brightness=0.0,
    color_jitter_contrast=0.0,
    color_jitter_saturation=0.0,
    color_jitter_hue=0.0,
    # -----------------------
):
    videos, label, start_frame_num, do_hflip, warp_homography = _video_reader_and_frames(
        file_list,
        sequence_length,
        shard_id,
        num_shards,
        frame_step,
        seq_step,
        steps_between_samples,
        crop_xy,
        crop_wh,
        reader_seed,
        shuffle,
        reader_kwargs,
        frame_wh,
        use_augs,
        p_hflip,
        use_warp_perspective,
        p_warp_perspective,
        warp_angle_range,
        warp_focal,
        use_jitter,
        p_jitter,
        jitter_n_degree,
        color_jitter_brightness,
        color_jitter_contrast,
        color_jitter_saturation,
        color_jitter_hue,
    )

    # load pose data
    pose_fn = partial(
        dali_pose_target_loader,
        video_files,
        sequence_length=sequence_length,
        seq_step=seq_step,
        frame_step=frame_step,
        t_anchors=t_anchors,
        num_pts=num_pts,
        use_full_pose=use_full_pose,
    )
    frame_times_s, poses, speeds = fn.python_function(
        label.cpu(),
        start_frame_num.cpu(),
        do_hflip.cpu(),
        function=pose_fn,
        device="cpu",
        num_outputs=3,
        batch_processing=False,
    )

    return videos, frame_times_s, poses, speeds, label, start_frame_num, do_hflip


class DaliDataset:
    """
    Video decode + pose targets. Batches include ``reader_label``, ``reader_start_frame``,
    ``reader_hflip`` from the video reader (clip id / debug); training usually ignores them.
    """

    def __init__(
        self,
        file_list,
        data_root,
        batch_size=1,
        seq_len=10,
        num_threads=2,
        device_id=0,
        n_streams=1,
        rank=0,
        world_size=1,
        base_seed=123456,
        frame_step=1,
        seq_step=1,
        steps_between_samples=20,
        plan_len_seconds=3,
        plan_len_points=10,
        offset_t_anchors=False,
        use_full_pose=False,
        crop_xy=(0, 0),
        crop_wh=(1920, 1080),
        downscale_factor=4,
        shuffle=False,
        reader_kwargs=None,
        frame_wh=(480, 270),
        # ---- augmentations ----
        use_augs=True,
        p_hflip=0.5,
        use_warp_perspective=True,
        p_warp_perspective=0.5,
        warp_angle_range=(-3.0, 3.0),
        warp_focal=None,
        use_jitter=True,
        p_jitter=0.5,
        jitter_n_degree=2,
        color_jitter_brightness=0.0,
        color_jitter_contrast=0.0,
        color_jitter_saturation=0.0,
        color_jitter_hue=0.0,
        # -----------------------
        last_batch_policy=LastBatchPolicy.DROP,
    ):
        if reader_kwargs is None:
            reader_kwargs = {}
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1 (final paired length), got {seq_len}")
        if seq_step < 1:
            raise ValueError(f"seq_step must be >= 1, got {seq_step}")
        if n_streams < 1:
            raise ValueError(f"n_streams must be >= 1, got {n_streams}")
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank must be in [0, world_size), got rank={rank}, world_size={world_size}")
        for name, value in {
            "color_jitter_brightness": color_jitter_brightness,
            "color_jitter_contrast": color_jitter_contrast,
            "color_jitter_saturation": color_jitter_saturation,
        }.items():
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if not (0.0 <= color_jitter_hue <= 0.5):
            raise ValueError(f"color_jitter_hue must be in [0, 0.5], got {color_jitter_hue}")
        if downscale_factor < 1:
            raise ValueError(f"downscale_factor must be >= 1, got {downscale_factor}")
        if not (0.0 <= p_hflip <= 1.0):
            raise ValueError(f"p_hflip must be in [0, 1], got {p_hflip}")
        if not (0.0 <= p_warp_perspective <= 1.0):
            raise ValueError(f"p_warp_perspective must be in [0, 1], got {p_warp_perspective}")
        if not (0.0 <= p_jitter <= 1.0):
            raise ValueError(f"p_jitter must be in [0, 1], got {p_jitter}")
        assert (batch_size % n_streams) == 0, "batch_size must be divisble by n_streams"

        # offset: drop the degenerate t=0 anchor; all anchors strictly in the future
        if offset_t_anchors:
            self.t_anchors = build_idxs(plan_len_seconds, plan_len_points + 1)[1:]
        else:
            self.t_anchors = build_idxs(plan_len_seconds, plan_len_points)
        self.num_pts = plan_len_seconds * NOMINAL_FPS

        self.data_root = Path(data_root)
        self.file_list = resolve_path(file_list, self.data_root)
        self.video_files = video_files_from_file_list(self.file_list, data_root=self.data_root)

        frame_wh = tuple(frame_wh)
        crop_xy = [crop_xy[0] // downscale_factor, crop_xy[1] // downscale_factor]
        crop_wh = [crop_wh[0] // downscale_factor, crop_wh[1] // downscale_factor]

        global_num_shards = world_size * n_streams
        self.pipes = []
        for local_shard_id in range(n_streams):
            global_shard_id = rank * n_streams + local_shard_id
            pipe_seed = base_seed + global_shard_id
            self.pipes.append(
                dali_pipeline(
                    file_list=str(self.file_list),
                    video_files=self.video_files,
                    sequence_length=seq_len,
                    shard_id=global_shard_id,
                    num_shards=global_num_shards,
                    batch_size=batch_size // n_streams,
                    num_threads=num_threads,
                    device_id=device_id,
                    frame_step=frame_step,
                    seq_step=seq_step,
                    steps_between_samples=steps_between_samples,
                    crop_xy=crop_xy,
                    crop_wh=crop_wh,
                    t_anchors=self.t_anchors,
                    num_pts=self.num_pts,
                    use_full_pose=use_full_pose,
                    reader_seed=pipe_seed,
                    reader_kwargs=reader_kwargs,
                    shuffle=shuffle,
                    frame_wh=frame_wh,
                    # ---- augmentations ----
                    use_augs=use_augs,
                    p_hflip=p_hflip,
                    use_warp_perspective=use_warp_perspective,
                    p_warp_perspective=p_warp_perspective,
                    warp_angle_range=warp_angle_range,
                    warp_focal=warp_focal,
                    use_jitter=use_jitter,
                    p_jitter=p_jitter,
                    jitter_n_degree=jitter_n_degree,
                    color_jitter_brightness=color_jitter_brightness,
                    color_jitter_contrast=color_jitter_contrast,
                    color_jitter_saturation=color_jitter_saturation,
                    color_jitter_hue=color_jitter_hue,
                    # -----------------------
                    seed=pipe_seed,
                )
            )
        for pipe in self.pipes:
            pipe.build()

        self._output_map = [
            "frames",
            "frame_times_s",
            "future_poses",
            "frame_speeds",
            "reader_label",
            "reader_start_frame",
            "reader_hflip",
        ]

        self._iter = DALIGenericIterator(
            self.pipes,
            self._output_map,
            reader_name="video_reader",
            last_batch_policy=last_batch_policy,
            auto_reset=True,
        )

    def __iter__(self):
        for per_pipe in self._iter:
            batch = per_pipe[0]
            if len(per_pipe) > 1:
                batch = {key: torch.cat([pipe_batch[key] for pipe_batch in per_pipe], dim=0) for key in batch.keys()}
            yield batch

    def __len__(self):
        return len(self._iter)
