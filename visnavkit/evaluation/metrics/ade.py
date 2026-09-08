"""Average displacement error at actual target timestamps, in metres."""

import torch

from visnavkit.utils.common import build_idxs


def _resolve_point_times(preds, max_val, point_times):
    plans = preds["plans"]
    n, _, count, _ = plans.shape
    if point_times is None:
        point_times = build_idxs(max_val, count)
    times = torch.as_tensor(point_times, device=plans.device, dtype=plans.dtype)
    if times.shape == (count,):
        times = times.expand(n, -1)
    if times.shape != (n, count):
        raise ValueError(f"point_times must have shape ({count},) or ({n}, {count}), got {tuple(times.shape)}")
    if not torch.isfinite(times).all() or (times < 0).any() or (times.diff(dim=-1) <= 0).any():
        raise ValueError("point_times must be finite, nonnegative and strictly increasing")
    return times


def _mean_at_horizons(l2, point_times, timesteps_to_compute, minimum_over_modes=False):
    """Average valid points per example, then modes (if requested), then examples."""
    out = {}
    for t in timesteps_to_compute:
        mask = point_times <= t
        counts = mask.sum(dim=-1)
        if not mask.any():
            continue
        if (counts == 0).any():
            raise ValueError(f"Some examples have no target points at or before {t} seconds")
        if minimum_over_modes:
            per_mode = (l2 * mask.unsqueeze(1)).sum(dim=-1) / counts.unsqueeze(1)
            per_example = per_mode.min(dim=1).values
        else:
            per_example = (l2 * mask).sum(dim=-1) / counts
        out[t] = per_example.mean()
    return out


def compute_top1_ade(
    preds: dict,
    gt_poses: torch.Tensor,
    max_val: float,
    timesteps_to_compute: list[float],
    point_times: torch.Tensor | None = None,
) -> dict[float, torch.Tensor]:
    """ADE of the selected trajectory, using supplied (T,) or (N,T) relative seconds.

    When omitted, point_times retains the legacy quadratic grid over [0, max_val].
    """
    times = _resolve_point_times(preds, max_val, point_times)
    l2 = torch.linalg.vector_norm(preds["best_plan"][..., :2] - gt_poses[..., :2], dim=-1)
    return _mean_at_horizons(l2, times, timesteps_to_compute)


def compute_min_ade(
    preds: dict,
    gt_poses: torch.Tensor,
    max_val: float,
    timesteps_to_compute: list[float],
    point_times: torch.Tensor | None = None,
) -> dict[float, torch.Tensor]:
    """Minimum ADE over modes independently at each requested horizon.

    Each horizon averages target points at time <= horizon, selects that example's
    lowest-error mode, and averages over examples. Actual times may be (T,) or (N,T).
    Omitting point_times uses the legacy quadratic grid over [0, max_val].
    """
    times = _resolve_point_times(preds, max_val, point_times)
    l2 = torch.linalg.vector_norm(preds["plans"][..., :2] - gt_poses[..., :2].unsqueeze(1), dim=-1)
    return _mean_at_horizons(l2, times, timesteps_to_compute, minimum_over_modes=True)
