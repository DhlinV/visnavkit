from pathlib import Path

import numpy as np


def load_npy(file: Path, mmap_mode: str | None = "r"):
    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(f"File not found: {file}")
    return np.load(file, mmap_mode=mmap_mode)


def build_idxs(max_val: float, size: int):
    """Quadratically space out points https://github.com/commaai/openpilot/blob/7d3ad941bc4ba4c923af7a1d7b48544bfc0d3e13/selfdrive/common/modeldata.h#L14-L25"""
    return np.array([max_val * ((i / (size - 1)) ** 2) for i in range(size)])


def anchor_times(plan_len_seconds: float, plan_len_points: int, offset: bool = False, uniform: bool = False):
    """Relative target times: ``build_idxs``'s quadratic grid (``offset`` drops its t = 0 anchor), or with
    ``uniform`` a fixed rate, plan_len_points steps of plan_len_seconds / plan_len_points s (t = 0 excluded)."""
    if uniform:
        return np.arange(1, plan_len_points + 1) * (plan_len_seconds / plan_len_points)
    if offset:
        return build_idxs(plan_len_seconds, plan_len_points + 1)[1:]
    return build_idxs(plan_len_seconds, plan_len_points)
