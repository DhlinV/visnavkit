"""Normalizer statistics and distribution plots, both read off the cached targets."""

from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate

from visnavkit.models.normalization import Normalizer
from visnavkit.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["cached_actions", "fit_stats", "plot_distributions"]


def cached_actions(cfg, cache: dict) -> torch.Tensor:
    """Cached poses mapped through the configured action space -> ``(N, T, A)`` actions."""
    space = instantiate(cfg.model.action_decoder.action_space)
    poses = torch.from_numpy(cache["future_poses"]).float()
    return space.targets_from_poses(poses.reshape(-1, *poses.shape[-2:]))


def fit_stats(cfg, cache: dict, mode: str = "meanstd", output_dir="outputs/dataset", name: str = "actions") -> Path:
    """Fit and save supervision statistics for ``action_decoder.normalizer.stats_path``.

    One file per corpus: a multi-dataset run fits each corpus separately and points its own
    loader at its own file, so mixing datasets never means mixing statistics.
    """
    actions = cached_actions(cfg, cache)
    normalizer = Normalizer(mode=mode).fit(actions)
    path = Path(output_dir) / f"{name}_{mode}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    normalizer.save_stats(path)
    normalized = normalizer.normalize(actions)
    logger.info(f"Fitted {mode} statistics over {len(actions)} trajectories to {path}")
    logger.info(f"Normalized range [{normalized.amin():.3f}, {normalized.amax():.3f}], std {normalized.std():.3f}")
    return path


def plot_distributions(cfg, cache: dict, output_dir="outputs/dataset", name: str = "actions") -> Path:
    """Histogram every action channel, plus the trajectory spread and the speed profile."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - depends on the optional extra
        raise ImportError("Plotting needs matplotlib: uv sync --extra plot") from error

    actions = cached_actions(cfg, cache).numpy()
    poses = cache["future_poses"].reshape(-1, *cache["future_poses"].shape[-2:])
    channels = actions.shape[-1]
    figure, axes = plt.subplots(1, channels + 2, figsize=(4 * (channels + 2), 3.2))
    for channel in range(channels):
        axes[channel].hist(actions[..., channel].reshape(-1), bins=80, color="#4477aa")
        axes[channel].set_title(f"action channel {channel}")
    sample = poses[np.random.default_rng(0).choice(len(poses), size=min(len(poses), 500), replace=False)]
    for trajectory in sample:
        axes[channels].plot(trajectory[:, 0], trajectory[:, 1], color="#4477aa", alpha=0.05)
    axes[channels].set_title(f"{len(sample)} trajectories (x, y)")
    axes[channels].set_aspect("equal", adjustable="datalim")
    axes[channels + 1].hist(cache["frame_speeds"].reshape(-1), bins=80, color="#aa7744")
    axes[channels + 1].set_title("observed speed (m/s)")
    figure.tight_layout()

    path = Path(output_dir) / f"{name}_distribution.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)
    logger.info(f"Wrote {path}")
    return path
