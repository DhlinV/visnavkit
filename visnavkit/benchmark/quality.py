"""Open-loop metrics in meters at declared times; no oracle top-1 selection."""

import numpy as np


def trajectory_metrics(predictions, targets, prediction_times, target_times, *, scores=None, horizons=(0.5, 1, 3), valid_mask=None):
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    pt = np.asarray(prediction_times, dtype=np.float64)
    tt = np.asarray(target_times, dtype=np.float64)
    if predictions.ndim != 4 or targets.ndim != 3:
        raise ValueError("Expected predictions (N,K,T,D) and targets (N,T,D).")
    n, candidates, points, dims = predictions.shape
    if n == 0 or candidates == 0 or targets.shape[0] != n or dims < 2 or targets.shape[-1] < 2:
        raise ValueError("Predictions and targets must contain matching nonempty XY samples.")
    if pt.shape != (points,) or np.any(np.diff(pt) <= 0) or not np.isfinite(pt).all():
        raise ValueError("Prediction timestamps must be finite, strictly increasing, and match trajectory points.")
    if tt.ndim == 1:
        tt = np.broadcast_to(tt, targets.shape[:2])
    if tt.shape != targets.shape[:2] or not np.isfinite(tt).all() or np.any(np.diff(tt, axis=1) <= 0):
        raise ValueError("Target timestamps must match the targets and be finite and strictly increasing.")
    valid = np.ones(targets.shape[:2], dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != targets.shape[:2] or np.any(np.diff(valid.astype(int), axis=1) > 0):
        raise ValueError("valid_mask must have shape (N,T) and contain a contiguous valid prefix.")
    if not np.isfinite(predictions).all() or not np.isfinite(targets[valid]).all():
        raise ValueError("Nonfinite valid trajectories cannot be scored.")
    selected = None
    if scores is not None:
        scores = np.asarray(scores)
        if scores.shape != (n, candidates) or not np.isfinite(scores).all():
            raise ValueError("Scores must be finite with shape (N,K).")
        selected = np.argmax(scores, axis=1)
    elif candidates == 1:
        selected = np.zeros(n, dtype=int)
    results = {}
    for horizon in horizons:
        horizon = float(horizon)
        if not np.isfinite(horizon) or horizon <= 0:
            raise ValueError("Metric horizons must be positive.")
        values = {}
        for i in range(n):
            sample_times = tt[i, valid[i]]
            sample_targets = targets[i, valid[i]]
            if len(sample_times) < 2:
                continue
            if horizon > min(pt[-1], sample_times[-1]) + 1e-6:
                continue
            if horizon < max(pt[0], sample_times[0]) - 1e-6:
                continue
            # A dataset-defined grid makes metrics comparable across model anchor schedules.
            grid = sample_times[(sample_times > 0) & (sample_times < horizon)]
            grid = np.append(grid, horizon)
            if grid[0] < pt[0] - 1e-6:
                continue  # Do not silently omit early evaluation points or extrapolate predictions.
            true = np.column_stack([np.interp(grid, sample_times, sample_targets[:, d]) for d in range(2)])
            pred = np.stack([
                np.column_stack([np.interp(grid, pt, predictions[i, k, :, d]) for d in range(2)])
                for k in range(candidates)
            ])
            errors = np.linalg.norm(pred - true[None], axis=-1)
            ade, fde = errors.mean(axis=-1), errors[:, -1]
            row = {"min_ade": ade.min(), "min_fde": fde.min(), "mean_sample_ade": ade.mean(),
                   "mean_sample_fde": fde.mean()}
            if selected is not None:
                row.update(top1_ade=ade[selected[i]], top1_fde=fde[selected[i]])
            if dims >= 3 and targets.shape[-1] >= 3:
                true_v = np.interp(grid, sample_times, sample_targets[:, 2])
                speed_error = np.stack([
                    np.abs(np.interp(grid, pt, predictions[i, k, :, 2]) - true_v) for k in range(candidates)
                ]).mean(axis=1)
                row["mean_sample_speed_mae"] = speed_error.mean()
                if selected is not None:
                    row["top1_speed_mae"] = speed_error[selected[i]]
            for name, value in row.items():
                values.setdefault(name, []).append(float(value))
        results[f"{horizon:g}s"] = {
            "count": len(next(iter(values.values()), [])),
            **{name: float(np.mean(samples)) for name, samples in values.items()},
        }
    return {
        "samples": n, "candidates": candidates, "units": {"displacement": "meter", "speed": "meter_per_second"},
        "definition": "point-mean ADE over positive dataset target anchors plus horizon; linearly interpolated FDE",
        "selection": "highest_score" if scores is not None else ("single_trajectory" if candidates == 1 else "unranked"),
        "horizons": results,
    }
