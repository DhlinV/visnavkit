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
