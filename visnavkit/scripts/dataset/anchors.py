"""k-means over the corpus trajectories: the anchor vocabulary classification decoders score."""

from pathlib import Path

import numpy as np
import torch

from visnavkit.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["fit_anchors", "kmeans"]


def kmeans(points: torch.Tensor, num_clusters: int, iters: int = 100, seed: int = 42, tol: float = 1e-6):
    """Lloyd's algorithm from a k-means++ seeding; ``(N, D)`` points -> ``(K, D)`` centres."""
    if points.ndim != 2 or points.shape[0] < num_clusters:
        raise ValueError(f"Need at least {num_clusters} points of shape (N, D), got {tuple(points.shape)}")
    generator = torch.Generator().manual_seed(seed)
    centres = points[torch.randint(points.shape[0], (1,), generator=generator)]
    for _ in range(num_clusters - 1):  # k-means++: sample the next centre far from the current ones
        distance = torch.cdist(points, centres).amin(dim=1) ** 2
        total = distance.sum()
        index = (
            torch.multinomial(distance / total, 1, generator=generator)
            if total > 0
            else torch.randint(points.shape[0], (1,), generator=generator)
        )
        centres = torch.cat([centres, points[index]])

    for _ in range(iters):
        assignment = torch.cdist(points, centres).argmin(dim=1)
        updated = centres.clone()
        for k in range(num_clusters):
            members = points[assignment == k]
            if len(members):  # keep empty clusters where they are rather than collapsing them
                updated[k] = members.mean(dim=0)
        shift = (updated - centres).norm(dim=1).max()
        centres = updated
        if shift < tol:
            break
    return centres, torch.cdist(points, centres).argmin(dim=1)


def fit_anchors(cache: dict, num_anchors: int = 16, output_dir="outputs/dataset", seed: int = 42) -> Path:
    """Cluster the cached trajectories into ``anchors`` (K, T, P), ready for ``AnchorSet``.

    Clustering runs on the xy path scaled to roughly [-1, 1] so that near and far waypoints
    count alike (FlowPilot's constrained k-means); the saved anchors are back in metres.
    """
    poses = torch.from_numpy(cache["future_poses"]).float()
    poses = poses.reshape(-1, *poses.shape[-2:])  # every decision, not just the window's last
    scale = poses[..., :2].abs().amax().clamp_min(1e-6)
    centres, assignment = kmeans((poses[..., :2] / scale).flatten(1), num_anchors, seed=seed)

    anchors = poses.new_zeros(num_anchors, poses.shape[1], poses.shape[2])
    anchors[..., :2] = centres.reshape(num_anchors, -1, 2) * scale
    if poses.shape[2] > 2:  # speed channel: the mean of each cluster's members, 0 where empty
        for k in range(num_anchors):
            members = poses[assignment == k]
            if len(members):
                anchors[k, :, 2:] = members[..., 2:].mean(dim=0)

    path = Path(output_dir) / f"anchors_{num_anchors}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, anchors=anchors.numpy().astype(np.float32), t_anchors=cache["t_anchors"])
    counts = torch.bincount(assignment, minlength=num_anchors)
    logger.info(f"Fitted {num_anchors} anchors over {len(poses)} trajectories to {path}")
    logger.info(f"Cluster sizes: min={int(counts.min())} median={int(counts.median())} max={int(counts.max())}")
    return path
