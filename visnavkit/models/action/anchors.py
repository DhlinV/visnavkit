"""Trajectory anchors: a fixed vocabulary of candidate plans for classification-style decoders."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

__all__ = ["AnchorSet", "arc_anchors"]


def arc_anchors(num_anchors: int, t_anchors, pose_size: int = 3, max_curvature: float = 0.5, speeds=(0.5, 1.0, 1.5)):
    """Constant-curvature arcs over ``t_anchors`` seconds: ``len(speeds)`` speeds x curvature fan -> (K, T, P)."""
    if num_anchors < 1:
        raise ValueError("num_anchors must be positive")
    t = np.asarray(t_anchors, dtype=np.float64)
    speeds = np.asarray(speeds, dtype=np.float64)
    curvatures = np.linspace(-max_curvature, max_curvature, max(int(np.ceil(num_anchors / len(speeds))), 1))
    anchors = []
    for kappa in curvatures:
        for v in speeds:
            s = v * t
            if abs(kappa) < 1e-6:
                x, y = s, np.zeros_like(s)
            else:
                x, y = np.sin(kappa * s) / kappa, (1 - np.cos(kappa * s)) / kappa
            columns = [x, y] + ([np.full_like(s, v)] if pose_size == 3 else [])
            anchors.append(np.column_stack(columns))
    return np.asarray(anchors[:num_anchors], dtype=np.float32)


class AnchorSet(nn.Module):
    """``(K, T, P)`` anchor poses from ``anchors_path`` (NPZ key ``anchors``) or a synthetic arc fan.

    Data-driven anchors (k-means over a corpus) belong in the NPZ; the arc fan needs no data.
    """

    def __init__(
        self, num_anchors: int = 16, anchors_path: str | None = None, max_curvature: float = 0.5, speeds=(0.5, 1.0, 1.5)
    ):
        super().__init__()
        self.num_anchors = num_anchors
        self.anchors_path = anchors_path
        self.max_curvature = max_curvature
        self.speeds = tuple(speeds)
        self.register_buffer("poses", torch.zeros(0))

    def build(self, t_anchors, pose_size: int):
        if self.anchors_path is not None:
            anchors = np.asarray(np.load(Path(self.anchors_path))["anchors"], dtype=np.float32)
            if anchors.ndim != 3 or anchors.shape[1:] != (len(t_anchors), pose_size):
                raise ValueError(f"anchors must be (K, {len(t_anchors)}, {pose_size}), got {anchors.shape}")
            self.num_anchors = anchors.shape[0]
        else:
            anchors = arc_anchors(self.num_anchors, t_anchors, pose_size, self.max_curvature, self.speeds)
        self.poses = torch.as_tensor(anchors)
        return self

    def nearest(self, gt_poses: torch.Tensor) -> torch.Tensor:
        """``(N, T, P)`` targets -> ``(N,)`` index of the anchor with the lowest xy displacement."""
        distance = (gt_poses[:, None, :, :2] - self.poses[None, :, :, :2].to(gt_poses.dtype)).norm(dim=-1).mean(-1)
        return distance.argmin(dim=1)
