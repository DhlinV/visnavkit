"""Cache each window's targets, so statistics, anchors and plots never decode a frame."""

from pathlib import Path

import numpy as np
from hydra.utils import instantiate

from visnavkit.data.mp4_dataset import Mp4WindowDataset
from visnavkit.data.pose_targets import get_current_frame_idxs, get_future_poses, load_pose_arrays
from visnavkit.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["build_dataset", "cache_path", "cache_targets", "load_cache"]


def build_dataset(cfg, split: str = "train") -> Mp4WindowDataset:
    """The split's window dataset, built from the same loader config training uses."""
    key = f"{split}_loader"
    if "dataset" not in cfg or cfg.dataset is None or key not in cfg.dataset:
        raise ValueError(f"No dataset.{key} in the config; pass dataset=torch (or another loader)")
    return instantiate(cfg.dataset[key], _target_="visnavkit.data.mp4_dataset.Mp4WindowDataset", _convert_="all")


def cache_path(output_dir, split: str) -> Path:
    return Path(output_dir) / f"targets_{split}.npz"


def cache_targets(cfg, split: str = "train", output_dir="outputs/dataset") -> Path:
    """Write ``(N, S, T, P)`` future poses and per-frame speeds for every window of a split.

    Only the pose sidecars are read: the windows come from the dataset's own index, so the cache
    describes exactly the samples training would see.
    """
    dataset = build_dataset(cfg, split)
    episodes: dict[Path, tuple] = {}
    poses, speeds, clips, starts = [], [], [], []
    for video_fp, start_idx in dataset.windows:
        sample_dir = Path(video_fp).parent
        if sample_dir not in episodes:
            episodes[sample_dir] = load_pose_arrays(sample_dir)
        positions, orientations, frame_speeds, times_s = episodes[sample_dir]
        seq_idxs = get_current_frame_idxs(start_idx, dataset.frame_step, dataset.seq_step, dataset.seq_len)
        future, _ = get_future_poses(
            positions,
            orientations,
            frame_speeds,
            times_s,
            seq_idxs,
            dataset.t_anchors,
            dataset.num_pts,
            dataset.use_full_pose,
            strict=True,
            interp_to_end=False,
        )
        poses.append(future)
        speeds.append(np.asarray(frame_speeds[seq_idxs], dtype=np.float32))
        clips.append(str(sample_dir))
        starts.append(start_idx)

    if not poses:
        raise ValueError(f"No windows in split {split!r}; check the file_list and the target horizon")
    path = cache_path(output_dir, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        future_poses=np.stack(poses).astype(np.float32),
        frame_speeds=np.stack(speeds),
        t_anchors=np.asarray(dataset.t_anchors, dtype=np.float32),
        clips=np.asarray(clips),
        window_starts=np.asarray(starts, dtype=np.int64),
    )
    logger.info(f"Cached {len(poses)} windows from {len(episodes)} clips to {path}")
    return path


def load_cache(output_dir, split: str = "train") -> dict:
    path = cache_path(output_dir, split)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run command=cache first")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
