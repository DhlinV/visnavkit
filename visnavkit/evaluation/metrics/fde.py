"""Final Displacement Error (FDE) for trajectory/plan predictions."""

import torch


def _compute_fde(pred_xy: torch.Tensor, gt_xy: torch.Tensor) -> torch.Tensor:
    last_idx = pred_xy.shape[-2] - 1
    return torch.norm(pred_xy[..., last_idx, :] - gt_xy[:, last_idx, :].unsqueeze(1), dim=-1)


def compute_top1_fde(
    preds: dict,
    gt_poses: torch.Tensor,
) -> torch.Tensor:
    """Compute FDE for the highest-confidence trajectory."""
    pred_xy = preds["best_plan"][..., :2]
    gt_xy = gt_poses[..., :2]
    return torch.norm(pred_xy[:, -1, :] - gt_xy[:, -1, :], dim=-1).mean()


def compute_min_fde(
    preds: dict,
    gt_poses: torch.Tensor,
) -> torch.Tensor:
    """Compute min FDE: pick the mode with min FDE at the last timestep, then report that mode's displacement error at the final point.

    Trajectory has num_pts points quadratically spaced: point i at time max_val * (i / (num_pts - 1)) ** 2 (seconds).
    FDE is the L2 displacement at the last point, using the best mode only.

    Args:
        preds: The parsed output of the planner module
        gt_poses: Ground-truth future poses, shape (N, num_pts, pose_size).

    Returns:
        Mean minimum final-point displacement error as a scalar tensor.
    """
    pred_xy = preds["plans"][..., :2]
    gt_xy = gt_poses[..., :2]
    fde_per_mode = _compute_fde(pred_xy, gt_xy)
    best_mode_idx = fde_per_mode.argmin(dim=1)
    best_fde = fde_per_mode[torch.arange(pred_xy.shape[0], device=preds["plans"].device), best_mode_idx]
    return best_fde.mean()
