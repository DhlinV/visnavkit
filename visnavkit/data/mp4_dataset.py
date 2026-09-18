import logging
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder

from visnavkit.data.file_list import parse_file_list_frame_ranges, resolve_path
from visnavkit.data.pose_targets import (
    get_current_frame_idxs,
    get_future_poses,
    goal_local_targets,
    goal_point_targets,
    past_xy_targets,
    load_pose_arrays,
    sample_goal_frame,
    target_safe_frame_ranges,
)
from visnavkit.utils.common import anchor_times, load_npy
from visnavkit.utils.orientation import yaw_from_quat

GOAL_TYPES = ("none", "point", "gps", "image", "route_image", "instruction")
EGO_FEATURES = {"speed": 1, "yaw_rate": 1, "past_xy": 2}  # name -> channels
CAMERA_INTRINSICS = "camera_intrinsics.npy"
CAMERA_EXTRINSICS = "camera_extrinsics.npy"

logger = logging.getLogger(__name__)


class Mp4WindowDataset(Dataset):
    """
    Torch counterpart of DaliDataset: consumes the same ``path label start end`` file_list,
    enumerates fixed windows every ``steps_between_samples`` frames, decodes frames with
    torchcodec, and emits the same batch keys (``vision`` uint8 (S, 3, h, w), ``frame_times_s``,
    ``future_poses``, ``frame_speeds``, ``target_times_s``). Target times are relative
    seconds, shape (T,) per sample and (B,T) after collation; all S frames share them.

    ``goal_type`` adds a ``goal`` key, and may be a list of types for policies with several goal
    encoders (``goal`` is then a list in the same order): ``point`` (S, 3) distance/cos/sin of a
    future frame sampled ``goal_horizon_s`` seconds ahead, ``gps`` (S, 2) the same goal as a local x/y offset in
    metres, ``image`` (3, h, w) uint8 crop of that
    frame, ``route_image`` (3, h, w) from the episode's ``route_images.npy`` (N, h, w, 3) sidecar
    indexed by the current frame, or ``instruction`` (E,) from ``instruction_embedding.npy``.

    ``ego_features`` adds an ``ego`` key (S, E) with the per-frame ego status: ``speed`` reads
    ``frame_speeds.npy``, ``yaw_rate`` differentiates the frame orientations, and ``past_xy``
    gives each observed frame's position in the newest frame's ego frame (past odometry).

    ``use_camera`` adds ``intrinsics`` (S, 3, 3) and ``extrinsics`` (S, 4, 4) from the episode's
    ``camera_intrinsics.npy`` / ``camera_extrinsics.npy`` sidecars (either one matrix for the
    clip or one per frame). The intrinsics are adjusted for the crop and downscale, and mirrored
    with the frames on a horizontal flip, so they describe the image the policy actually sees.
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
        uniform_t_anchors=False,
        use_full_pose=False,
        crop_xy=(0, 0),
        crop_wh=(1920, 1080),
        downscale_factor=4,
        goal_type="none",
        goal_horizon_s=(3.0, 15.0),
        ego_features=(),
        use_camera=False,
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
        self.goal_types = [goal_type] if isinstance(goal_type, str) else list(goal_type)
        for name in self.goal_types:
            if name not in GOAL_TYPES:
                raise ValueError(f"goal_type must be one of {GOAL_TYPES}, got {name!r}")
        self.goal_is_list = not isinstance(goal_type, str)
        self.ego_features = tuple(ego_features or ())
        for name in self.ego_features:
            if name not in EGO_FEATURES:
                raise ValueError(f"ego_features must be from {tuple(EGO_FEATURES)}, got {name!r}")
        self.use_camera = bool(use_camera)
        self.downscale_factor = downscale_factor
        self.goal_horizon_s = tuple(float(v) for v in goal_horizon_s)
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

        self.t_anchors = anchor_times(plan_len_seconds, plan_len_points, offset_t_anchors, uniform_t_anchors)
        self.num_pts = plan_len_points  # Legacy argument; the horizon is timestamp-based.

        s = downscale_factor
        self.crop_slice_h = slice(crop_xy[1] // s, (crop_xy[1] + crop_wh[1]) // s)
        self.crop_slice_w = slice(crop_xy[0] // s, (crop_xy[0] + crop_wh[0]) // s)

        jitter = (color_jitter_brightness, color_jitter_contrast, color_jitter_saturation, color_jitter_hue)
        self.color_jitter = v2.ColorJitter(*jitter) if use_augs and any(v > 0 for v in jitter) else None

        # Same span as the DALI reader: seq_len frames strided by seq_step * frame_step source
        # frames, with windows starting every steps_between_samples source frames.
        span = (seq_len - 1) * seq_step * frame_step + 1
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
        positions, orientations, speeds, times_s = load_pose_arrays(sample_dir)
        seq_idxs = get_current_frame_idxs(start_idx, self.frame_step, self.seq_step, self.seq_len)
        goal_idx = None
        if {"point", "gps", "image"} & set(self.goal_types):
            goal_idx = sample_goal_frame(times_s, seq_idxs[-1], self.goal_horizon_s, f"{video_fp}:{start_idx}")

        decoder = VideoDecoder(video_fp, device="cpu", dimension_order="NCHW")
        decode_idxs = list(seq_idxs) + ([goal_idx] if "image" in self.goal_types else [])
        decoded = decoder.get_frames_at(indices=decode_idxs).data  # uint8 (S[+1], 3, H, W)
        decoded = decoded[..., self.crop_slice_h, self.crop_slice_w]
        if self.color_jitter is not None:
            decoded = self.color_jitter(decoded)
        frames = decoded[: self.seq_len]  # (S, 3, h, w)

        future_poses, _ = get_future_poses(
            positions,
            orientations,
            speeds,
            times_s,
            seq_idxs,
            self.t_anchors,
            self.num_pts,
            self.use_full_pose,
            strict=True,
            interp_to_end=False,
        )
        goals = [
            self._goal(kind, sample_dir, positions, orientations, seq_idxs, goal_idx, decoded)
            for kind in self.goal_types
        ]

        flip = bool(torch.rand(1) < self.p_hflip)
        if flip:
            frames = torch.flip(frames, dims=[-1])
            future_poses[..., 1] *= -1.0
        goals = [self._flip_goal(kind, goal) if flip else goal for kind, goal in zip(self.goal_types, goals)]

        sample = dict(
            vision=frames,
            frame_times_s=torch.tensor(np.asarray(times_s[seq_idxs]), dtype=torch.float64),
            future_poses=torch.from_numpy(future_poses),
            target_times_s=torch.tensor(self.t_anchors, dtype=torch.float32),
            frame_speeds=torch.tensor(np.asarray(speeds[seq_idxs]), dtype=torch.float32).reshape(-1, 1),
        )
        present = [goal for goal in goals if goal is not None]
        if present:
            sample["goal"] = present if self.goal_is_list else present[0]
        if self.ego_features:
            sample["ego"] = self._ego(positions, orientations, speeds, times_s, seq_idxs, flip)
        if self.use_camera:
            sample["intrinsics"], sample["extrinsics"] = self._camera(sample_dir, seq_idxs, frames.shape[-1], flip)
        return sample

    def _camera(self, sample_dir, seq_idxs, width, flip):
        """Per-frame calibration for the cropped, downscaled and possibly mirrored frames."""
        intrinsics = np.asarray(load_npy(sample_dir / CAMERA_INTRINSICS), dtype=np.float32)
        extrinsics = np.asarray(load_npy(sample_dir / CAMERA_EXTRINSICS), dtype=np.float32)
        intrinsics = intrinsics[seq_idxs] if intrinsics.ndim == 3 else np.repeat(intrinsics[None], len(seq_idxs), 0)
        extrinsics = extrinsics[seq_idxs] if extrinsics.ndim == 3 else np.repeat(extrinsics[None], len(seq_idxs), 0)
        if intrinsics.shape[1:] != (3, 3) or extrinsics.shape[1:] != (4, 4):
            raise ValueError("camera_intrinsics.npy must be (3,3)/(N,3,3) and camera_extrinsics.npy (4,4)/(N,4,4)")
        intrinsics = intrinsics.copy()
        # Principal point follows the crop, then the whole matrix follows the downscale.
        intrinsics[:, 0, 2] -= self.crop_slice_w.start * self.downscale_factor
        intrinsics[:, 1, 2] -= self.crop_slice_h.start * self.downscale_factor
        intrinsics[:, :2] /= self.downscale_factor
        if flip:  # mirroring the image mirrors the x axis of the image plane
            intrinsics[:, 0, 2] = width - intrinsics[:, 0, 2]
        return torch.from_numpy(intrinsics), torch.from_numpy(extrinsics.copy())

    def _ego(self, positions, orientations, speeds, times_s, seq_idxs, flip):
        """Per-frame ego status ``(S, E)``; the encoder only needs the width, not the meanings."""
        columns = []
        for name in self.ego_features:
            if name == "speed":
                columns.append(np.asarray(speeds[seq_idxs], dtype=np.float32)[:, None])
            elif name == "yaw_rate":
                yaw = yaw_from_quat(np.asarray(orientations))
                rate = np.gradient(np.unwrap(yaw), np.asarray(times_s, dtype=np.float64))
                columns.append(np.asarray(rate[seq_idxs], dtype=np.float32)[:, None] * (-1.0 if flip else 1.0))
            else:
                past = past_xy_targets(positions, orientations, seq_idxs)
                columns.append(past * ([1.0, -1.0] if flip else [1.0, 1.0]))
        return torch.from_numpy(np.concatenate(columns, axis=-1).astype(np.float32))

    def _flip_goal(self, kind, goal):
        if goal is None:
            return None
        if kind == "point":
            goal[:, 2] *= -1.0
        elif kind == "gps":
            goal[:, 1] *= -1.0
        elif kind in ("image", "route_image"):
            goal = torch.flip(goal, dims=[-1])
        return goal

    def _goal(self, kind, sample_dir, positions, orientations, seq_idxs, goal_idx, decoded):
        if kind == "none":
            return None
        if kind == "point":
            return torch.from_numpy(goal_point_targets(positions, orientations, seq_idxs, goal_idx))
        if kind == "gps":
            return torch.from_numpy(goal_local_targets(positions, orientations, seq_idxs, goal_idx))
        if kind == "image":
            return decoded[-1]
        if kind == "route_image":
            routes = np.load(sample_dir / "route_images.npy", mmap_mode="r")
            return torch.from_numpy(np.ascontiguousarray(routes[seq_idxs[-1]])).permute(2, 0, 1)
        return torch.from_numpy(
            np.asarray(np.load(sample_dir / "instruction_embedding.npy"), dtype=np.float32).reshape(-1)
        )

    def __len__(self):
        return len(self.windows)
