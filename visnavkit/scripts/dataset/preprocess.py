"""Turn a directory of clips into validated manifests, before anything trains on them."""

from pathlib import Path

import numpy as np

from visnavkit.data.file_list import write_file_list_frame_ranges
from visnavkit.data.pose_targets import FRAME_TIMES_S, load_pose_arrays
from visnavkit.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["find_clips", "preprocess"]


def find_clips(data_root) -> list[Path]:
    """Clip directories under ``data_root``: a ``video.mp4`` next to its pose sidecars."""
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"data_root {root} is not a directory")
    return sorted(path.parent for path in root.rglob("video.mp4") if (path.parent / FRAME_TIMES_S).exists())


def preprocess(data_root, output_dir=None, val_fraction: float = 0.1, seed: int = 42) -> dict:
    """Validate every clip and write ``train.txt`` / ``val.txt`` covering the ones that pass.

    Validation is the same check the loaders make (shapes agree, timestamps strictly increase,
    quaternions are non-degenerate), so a corpus that passes here will not fail mid-epoch.
    Clips are split whole, never window-wise, so no episode appears in both splits.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    root = Path(data_root)
    destination = Path(output_dir) if output_dir else root
    clips = find_clips(root)
    if not clips:
        raise FileNotFoundError(f"No clips (video.mp4 + {FRAME_TIMES_S}) under {root}")

    usable, rejected = [], {}
    for clip in clips:
        try:
            *_, times = load_pose_arrays(clip)
        except (ValueError, FileNotFoundError) as error:
            rejected[str(clip)] = str(error)
            continue
        usable.append((str(clip / "video.mp4"), len(times)))

    order = np.random.default_rng(seed).permutation(len(usable))
    split_at = int(round(len(usable) * val_fraction))
    splits = {"val": order[:split_at], "train": order[split_at:]}
    destination.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, indices in splits.items():
        rows = [(usable[i][0], label, 0, usable[i][1]) for label, i in enumerate(sorted(indices))]
        if rows:
            write_file_list_frame_ranges(rows, destination / f"{name}.txt")
        counts[name] = len(rows)

    for clip, reason in rejected.items():
        logger.warning(f"Skipping {clip}: {reason}")
    logger.info(f"Wrote {counts} clips to {destination}; rejected {len(rejected)}")
    return {"counts": counts, "rejected": rejected, "destination": str(destination)}
