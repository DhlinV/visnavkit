"""Average Displacement Error (ADE) for trajectory/plan predictions."""

import torch

from navigators.utils.common import build_idxs


def _compute_ade_at_timesteps(
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    point_times: torch.Tensor,
    timesteps_to_compute: list[float],
) -> dict[str, torch.Tensor]:
    out = {}
    for t in timesteps_to_compute:
        mask = point_times <= t
        if not mask.any():
            continue
        l2 = torch.norm(pred_xy[:, mask, :] - gt_xy[:, mask, :], dim=-1)
        out[t] = l2.mean(dim=1).mean()
    return out


def compute_top1_ade(
    preds: dict,
    gt_poses: torch.Tensor,
    max_val: float,
    timesteps_to_compute: list[float],
) -> dict[str, torch.Tensor]:
    """Compute ADE at specified timesteps for the highest-confidence trajectory."""
    num_pts = preds["plans"].shape[2]
    point_times = torch.from_numpy(build_idxs(max_val, num_pts)).float().to(preds["plans"].device)

    pred_xy = preds["best_plan"][..., :2]
    gt_xy = gt_poses[..., :2]
    return _compute_ade_at_timesteps(pred_xy, gt_xy, point_times, timesteps_to_compute)


def compute_min_ade(
    preds: dict,
    gt_poses: torch.Tensor,
    max_val: float,
    timesteps_to_compute: list[float],
) -> dict[str, torch.Tensor]:
    """Compute min ADE at specified timesteps: pick the min-ADE mode over the full trajectory, then report that mode's ADE at each timestep (trajectory truncated to time <= t).

    Trajectory has num_pts points quadratically spaced: point i at time max_val * (i / (num_pts - 1)) ** 2 (seconds).
    For each t in timesteps_to_compute, ADE is computed over points with time <= t using the best mode only.

    Args:
        preds: The parsed output of the planner module
        gt_poses: Ground-truth future poses, shape (N, num_pts, pose_size).
        max_val: Time horizon in seconds (last point is at max_val).
        timesteps_to_compute: Times up to which to compute ADE (e.g. [0.5, 1.0, 1.5, 3.0]).

    Returns:
        Dict mapping keys like "min_ade_0_5", "min_ade_1_0", ... (decimal as _, whole numbers with _0) to scalar tensors.
    """
    num_pts = preds["plans"].shape[2]
    point_times = torch.from_numpy(build_idxs(max_val, num_pts)).float().to(preds["plans"].device)

    pred_xy = preds["plans"][..., :2]
    gt_xy = gt_poses[..., :2]

    l2_full = torch.norm(pred_xy - gt_xy.unsqueeze(1), dim=-1)
    ade_best_mode = l2_full.mean(dim=2).argmin(dim=1)
    pred_best_xy = pred_xy[torch.arange(pred_xy.shape[0], device=preds["plans"].device), ade_best_mode]
    return _compute_ade_at_timesteps(pred_best_xy, gt_xy, point_times, timesteps_to_compute)
