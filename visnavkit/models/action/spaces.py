"""Action spaces: what a trajectory decoder predicts per anchor time, and how it maps back to poses."""

import math

import torch
import torch.nn as nn

from visnavkit.utils.common import build_idxs

__all__ = ["ActionSpace"]

KINDS = ("waypoint", "velocity")


class ActionSpace(nn.Module):
    """Convert dataset pose targets ``(N, T, P)`` to actions ``(N, T, A)`` and back.

    - ``waypoint``: actions are the ego-frame poses themselves (``A = P``: x, y[, speed]).
    - ``velocity``: unicycle commands per anchor segment (``A = 2``: speed, yaw rate) derived
      from consecutive waypoints; ``to_poses`` integrates them back, so metrics and export
      always see poses.

    Anchor times come from the training config (``build_idxs`` grid), identical to the datasets'.
    """

    def __init__(
        self,
        kind: str = "waypoint",
        pose_size: int = 3,
        plan_len_seconds: float = 3.0,
        plan_len_points: int = 10,
        offset_t_anchors: bool = False,
    ):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        if pose_size not in (2, 3):
            raise ValueError("pose_size must be 2 (x, y) or 3 (x, y, speed)")
        if plan_len_points < 2 or not math.isfinite(plan_len_seconds) or plan_len_seconds <= 0:
            raise ValueError("plan_len_points must be >= 2 and plan_len_seconds positive and finite")
        if offset_t_anchors:
            anchors = build_idxs(plan_len_seconds, plan_len_points + 1)[1:]
        else:
            anchors = build_idxs(plan_len_seconds, plan_len_points)
        self.kind = kind
        self.pose_size = pose_size
        self.num_pts = int(plan_len_points)
        self.action_dim = pose_size if kind == "waypoint" else 2
        self.register_buffer("t_anchors", torch.as_tensor(anchors, dtype=torch.float32), persistent=False)
        deltas = torch.diff(self.t_anchors, prepend=self.t_anchors.new_zeros(1))
        self.register_buffer("dt", deltas, persistent=False)

    @property
    def flat_dim(self) -> int:
        return self.num_pts * self.action_dim

    def targets_from_poses(self, poses: torch.Tensor) -> torch.Tensor:
        """``(N, T, P)`` pose targets -> ``(N, T, A)`` actions."""
        if poses.shape[-2] != self.num_pts or poses.shape[-1] != self.pose_size:
            raise ValueError(f"Expected poses (N, {self.num_pts}, {self.pose_size}), got {tuple(poses.shape)}")
        if self.kind == "waypoint":
            return poses
        xy = poses[..., :2]
        previous = torch.cat([xy.new_zeros(*xy.shape[:-2], 1, 2), xy[..., :-1, :]], dim=-2)
        delta = xy - previous
        dt = self.dt.to(poses.dtype)
        valid = dt > 0
        safe_dt = torch.where(valid, dt, torch.ones_like(dt))
        distance = delta.norm(dim=-1)
        speed = torch.where(valid, distance / safe_dt, torch.zeros_like(distance))
        heading = torch.atan2(delta[..., 1], delta[..., 0])
        # carry the last moving heading forward so yaw rate stays defined on stationary segments
        moving = distance > 1e-6
        steps = torch.arange(self.num_pts, device=poses.device).expand_as(moving)
        last_moving = torch.cummax(torch.where(moving, steps, torch.full_like(steps, -1)), dim=-1).values
        heading = torch.where(last_moving >= 0, heading.gather(-1, last_moving.clamp_min(0)), torch.zeros_like(heading))
        previous_heading = torch.cat([heading.new_zeros(*heading.shape[:-1], 1), heading[..., :-1]], dim=-1)
        yaw_delta = torch.remainder(heading - previous_heading + math.pi, 2 * math.pi) - math.pi
        yaw_rate = torch.where(valid, yaw_delta / safe_dt, torch.zeros_like(yaw_delta))
        return torch.stack([speed, yaw_rate], dim=-1)

    def to_poses(self, actions: torch.Tensor) -> torch.Tensor:
        """``(..., T, A)`` actions -> ``(..., T, P)`` ego-frame poses."""
        if self.kind == "waypoint":
            return actions
        speed, yaw_rate = actions[..., 0], actions[..., 1]
        dt = self.dt.to(actions.dtype)
        heading = torch.cumsum(yaw_rate * dt, dim=-1)
        step = speed * dt
        x = torch.cumsum(step * torch.cos(heading), dim=-1)
        y = torch.cumsum(step * torch.sin(heading), dim=-1)
        columns = [x, y] + ([speed] if self.pose_size == 3 else [])
        return torch.stack(columns, dim=-1)
