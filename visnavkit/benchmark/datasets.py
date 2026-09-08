"""Portable prepared arrays and a generated fixture for checking the benchmark."""

import json
from pathlib import Path

import numpy as np

from visnavkit.benchmark.export import target_times


def write_fixture(path, cfg, *, samples=4, seed=42):
    if samples < 1:
        raise ValueError("samples must be positive")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    common = cfg.common
    h, w = int(common.crop_wh[1] // common.downscale_factor), int(common.crop_wh[0] // common.downscale_factor)
    shape = (samples, int(common.seq_length), 6, h, w)
    frames = rng.integers(0, 256, size=shape, dtype=np.uint8)
    times = target_times(cfg).astype(np.float32)
    speed = np.linspace(0, 2, samples, dtype=np.float32)
    dims = int(cfg.model.modules.action_decoder.plan_head.pose_size)
    targets = np.zeros((samples, len(times), dims), dtype=np.float32)
    targets[..., 0] = speed[:, None] * times
    if dims >= 3:
        targets[..., 2] = speed[:, None]
    np.savez(path, frames=frames, targets=targets, target_times_s=times, current_speed=speed,
             sample_ids=np.asarray([f"synthetic-{i:04d}" for i in range(samples)]))
    metadata = {"schema_version": 1, "dataset_kind": "synthetic_fixture",
                "purpose": "pipeline correctness only; random pixels and analytic straight trajectories",
                "samples": samples, "seed": seed}
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_archive(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    required = {"targets", "target_times_s", "sample_ids"}
    if missing := required - data.keys():
        raise ValueError(f"Missing prepared dataset keys: {sorted(missing)}")
    ids = data["sample_ids"].astype(str)
    if ids.ndim != 1 or len(ids) != len(data["targets"]) or len(set(ids)) != len(ids):
        raise ValueError("sample_ids must be unique and match targets.")
    meta_path = path.with_suffix(".metadata.json")
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {"dataset_kind": "unspecified"}
    return data, metadata
