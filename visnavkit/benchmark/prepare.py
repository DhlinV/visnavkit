"""Prepare bounded open-loop evaluation shards from the legacy video dataset."""

import hashlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from visnavkit.benchmark.export import sha256_file
from visnavkit.data.mp4_dataset import Mp4WindowDataset
from visnavkit.data.pose_targets import FRAME_ORIENTATIONS, FRAME_POSITIONS, FRAME_SPEEDS, FRAME_TIMES_S

# load_archive materializes arrays in memory. Keep each prepared shard bounded.
MAX_PREPARED_BYTES = 512 * 1024 * 1024


def prepare_dataset(cfg, destination, limit=None):
    """Write an NPZ evaluation shard and provenance sidecar; return the metadata.

    Uses val_loader's deterministic crop and observation/target settings with augmentation
    disabled. Each sample evaluates the final sequence frame with full observation history.
    ``limit`` selects the first N valid windows in manifest order. If limit is None, all
    valid windows are included, subject to the 512 MiB uncompressed shard bound.

    Files contain uint8 vision (N,S,3,H,W), targets (N,T,D), relative target_times_s (T,),
    current_speed (N,), and unique absolute-video-path:current-frame sample_ids (N,).
    """
    destination = Path(destination)
    if destination.suffix != ".npz":
        raise ValueError("Prepared dataset destination must end with .npz")
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer or None")
    settings = OmegaConf.to_container(cfg.dataset.val_loader, resolve=True)
    settings.update(use_augs=False, shuffle=False)
    dataset = Mp4WindowDataset(**settings)
    count = len(dataset) if limit is None else min(limit, len(dataset))
    if count == 0:
        raise ValueError("No valid evaluation windows remain after observation and future-target checks")
    windows = dataset.windows[:count]
    sample_ids = np.asarray(
        [
            f"{Path(video).resolve()}:{start + dataset.frame_step + (dataset.seq_len - 1) * dataset.seq_step * dataset.frame_step}"
            for video, start in windows
        ]
    )
    if len(set(sample_ids)) != count:
        raise ValueError("The validation manifest repeats observation windows; sample IDs must be unique")

    first = dataset[0]
    frame_shape = tuple(first["vision"].shape)
    target_shape = tuple(first["future_poses"][-1].shape)
    target_times = first["target_times_s"].numpy().copy()
    estimated_bytes = (
        count * (int(np.prod(frame_shape)) + int(np.prod(target_shape)) * 4 + 4)
        + sample_ids.nbytes
        + target_times.nbytes
    )
    if estimated_bytes > MAX_PREPARED_BYTES:
        raise ValueError(
            f"Prepared shard would require {estimated_bytes / 2**20:.1f} MiB uncompressed; "
            "use a smaller limit or a smaller validation manifest (maximum 512 MiB per shard)"
        )
    frames = np.empty((count, *frame_shape), dtype=np.uint8)
    targets = np.empty((count, *target_shape), dtype=np.float32)
    speed = np.empty(count, dtype=np.float32)
    for i in range(count):
        sample = first if i == 0 else dataset[i]
        if tuple(sample["vision"].shape) != frame_shape or tuple(sample["future_poses"][-1].shape) != target_shape:
            raise ValueError("Prepared observations and targets must have consistent shapes across episodes")
        if not np.array_equal(sample["target_times_s"].numpy(), target_times):
            raise ValueError("Prepared episodes must share the same target anchor times")
        frames[i] = sample["vision"].numpy()
        targets[i] = sample["future_poses"][-1].numpy()
        speed[i] = sample["frame_speeds"][-1].item()

    source_files = []
    videos = sorted({str(Path(video).resolve()) for video, _ in windows})
    paths = {Path(video) for video in videos}
    for video in videos:
        paths.update(
            Path(video).parent / name for name in (FRAME_POSITIONS, FRAME_ORIENTATIONS, FRAME_TIMES_S, FRAME_SPEEDS)
        )
    for path in sorted(paths):
        stat = path.stat()
        record = {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if path.suffix == ".npy":
            record["sha256"] = sha256_file(path)
        source_files.append(record)
    metadata = {
        "schema_version": 1,
        "dataset_kind": "real_prepared",
        "samples": count,
        "available_windows": len(dataset),
        "requested_limit": limit,
        "selection": "first valid windows in manifest order",
        "decision_frame": "final sequence frame",
        "estimated_uncompressed_bytes": estimated_bytes,
        "source_manifest": {"path": str(dataset.file_list), "sha256": sha256_file(dataset.file_list)},
        "source_files": source_files,
        "source_fingerprint_sha256": hashlib.sha256(json.dumps(source_files, sort_keys=True).encode()).hexdigest(),
        "source_fingerprint_kind": "paths, sizes, mtimes, and pose-array content hashes; videos use size/mtime",
        "loader_settings": settings,
        "coordinate_frame": "current robot ego frame; quaternion convention wxyz",
        "target_channels": ["x_m", "y_m", "speed_m_s"] if targets.shape[-1] == 3 else ["x_m", "y_m"],
        "target_times_s": target_times.tolist(),
        "vision_shape": list(frames.shape),
        "augmentation": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        vision=frames,
        targets=targets,
        target_times_s=target_times,
        current_speed=speed,
        sample_ids=sample_ids,
    )
    metadata["archive_sha256"] = sha256_file(destination)
    destination.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
