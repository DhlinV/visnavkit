import numpy as np


def build_idxs(max_val: float, size: int):
    """Quadratically space out points https://github.com/commaai/openpilot/blob/7d3ad941bc4ba4c923af7a1d7b48544bfc0d3e13/selfdrive/common/modeldata.h#L14-L25"""
    return np.array([max_val * ((i / (size - 1)) ** 2) for i in range(size)])
