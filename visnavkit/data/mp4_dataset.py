import logging
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder

from visnavkit.data.file_list import parse_file_list_frame_ranges, resolve_path
from visnavkit.data.pose_targets import get_current_frame_idxs, get_future_poses_from_dir, target_safe_frame_ranges
from visnavkit.utils.common import build_idxs, load_npy

logger = logging.getLogger(__name__)


class Mp4WindowDataset(Dataset):
    """
    Torch counterpart of DaliDataset: consumes the same ``path label start end`` file_list,
    enumerates fixed windows every ``steps_between_samples`` frames, decodes paired frames with
    torchcodec, and emits the same batch keys (``frames`` uint8 (S, 6, h, w), ``frame_times_s``,
    ``future_poses``, ``frame_speeds``, ``target_times_s``). Target times are relative
    seconds, shape (T,) per sample and (B,T) after collation; all S frames share them.
    """

    def __init__(
        self,
        file_list,
        data_root,
        seq_len=10,
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
        # ---- augmentations (hflip + color jitter only; warp/jitter are DALI-only) ----
        use_augs=False,
        p_hflip=0.5,
        color_jitter_brightness=0.0,
        color_jitter_contrast=0.0,
        color_jitter_saturation=0.0,
        color_jitter_hue=0.0,
        **kwargs,  # tolerate DALI-only keys so both loaders can share a config
    ):
        self.data_root = Path(data_root)
        self.file_list = resolve_path(file_list, self.data_root)
        self.seq_len = seq_len
        self.frame_step = frame_step
        self.seq_step = seq_step
        self.use_full_pose = use_full_pose
        self.p_hflip = p_hflip if use_augs else 0.0
        for name, value in {
            "seq_len": seq_len,
            "frame_step": frame_step,
            "seq_step": seq_step,
            "steps_between_samples": steps_between_samples,
            "downscale_factor": downscale_factor,
        }.items():
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if plan_len_points < 2 or not np.isfinite(plan_len_seconds) or plan_len_seconds <= 0:
            raise ValueError("plan_len_points must be >= 2 and plan_len_seconds must be positive and finite")

        if offset_t_anchors:
            self.t_anchors = build_idxs(plan_len_seconds, plan_len_points + 1)[1:]
        else:
            self.t_anchors = build_idxs(plan_len_seconds, plan_len_points)
        self.num_pts = plan_len_points  # Legacy argument; the horizon is timestamp-based.

        s = downscale_factor
        self.crop_slice_h = slice(crop_xy[1] // s, (crop_xy[1] + crop_wh[1]) // s)
        self.crop_slice_w = slice(crop_xy[0] // s, (crop_xy[0] + crop_wh[0]) // s)

        jitter = (color_jitter_brightness, color_jitter_contrast, color_jitter_saturation, color_jitter_hue)
        self.color_jitter = v2.ColorJitter(*jitter) if use_augs and any(v > 0 for v in jitter) else None

        # Same span as the DALI reader: reader_seq_len = (seq_len - 1) * seq_step + 2 decoded
        # frames with stride=frame_step, windows every steps_between_samples source frames.
        span = ((seq_len - 1) * seq_step + 1) * frame_step + 1
        self.windows: list[tuple[str, int]] = []
        rows = parse_file_list_frame_ranges(self.file_list, data_root=self.data_root)
        safe_rows = target_safe_frame_ranges(rows, float(self.t_anchors[-1]))
        candidate_count = sum(len(range(start, end - span + 1, steps_between_samples)) for _, _, start, end in rows)
        for video_fp, _, start_f, end_f in safe_rows:
            for w_start in range(start_f, end_f - span + 1, steps_between_samples):
                self.windows.append((video_fp, w_start))
        logger.info(
            "Indexed %d windows; discarded %d without complete targets", len(self.windows), candidate_count - len(self)
        )

    def __getitem__(self, idx):
        video_fp, start_idx = self.windows[idx]
        sample_dir = Path(video_fp).parent

        pair_starts = [start_idx + k * self.seq_step * self.frame_step for k in range(self.seq_len)]
        frame_idxs = [i for p in pair_starts for i in (p, p + self.frame_step)]
        decoder = VideoDecoder(video_fp, device="cpu", dimension_order="NCHW")
        frames = decoder.get_frames_at(indices=frame_idxs).data  # uint8 (2S, 3, H, W)
        frames = frames[..., self.crop_slice_h, self.crop_slice_w]
        if self.color_jitter is not None:
            frames = self.color_jitter(frames)
        frames = torch.cat([frames[0::2], frames[1::2]], dim=1)  # (S, 6, h, w) prev+cur

        seq_idxs = get_current_frame_idxs(start_idx, self.frame_step, self.seq_step, self.seq_len)
        future_poses, _ = get_future_poses_from_dir(
            sample_dir, seq_idxs, self.t_anchors, self.num_pts, self.use_full_pose, strict=True, interp_to_end=False
        )
        frame_times_s = load_npy(sample_dir / "frame_times.npy") / 1e9
        frame_speeds = load_npy(sample_dir / "frame_speeds.npy")

        if torch.rand(1) < self.p_hflip:
            frames = torch.flip(frames, dims=[-1])
            future_poses[..., 1] *= -1.0

        return dict(
            frames=frames,
            frame_times_s=torch.tensor(np.asarray(frame_times_s[seq_idxs]), dtype=torch.float64),
            future_poses=torch.from_numpy(future_poses),
            target_times_s=torch.tensor(self.t_anchors, dtype=torch.float32),
            frame_speeds=torch.tensor(np.asarray(frame_speeds[seq_idxs]), dtype=torch.float32).reshape(-1, 1),
        )

    def __len__(self):
        return len(self.windows)
