"""Shared flat trajectory layout emitted by every action decoder."""

import torch
import torch.nn.functional as F

__all__ = ["parse_plan_output"]


def parse_plan_output(output, num_modes: int = 5, num_pts: int = 10, pose_size: int = 3):
    """Split ``(N, M * (2 * T * P + 1))`` into trajectories, scales, confidences, and the best plan."""
    preds = output.reshape(-1, num_modes, 2 * num_pts * pose_size + 1)
    pred_conf = F.softmax(preds[..., -1], dim=1)
    best_plan_idx = torch.argmax(pred_conf, dim=1)
    pred_trajectories = preds[..., :-1].reshape(-1, num_modes, 2, num_pts, pose_size)
    plans = pred_trajectories[:, :, 0]
    pred_scales = pred_trajectories[:, :, 1]
    best_plan = plans[torch.arange(pred_trajectories.shape[0]), best_plan_idx]
    return dict(confs=pred_conf, best_plan=best_plan, plans=plans, pred_scales=pred_scales)
