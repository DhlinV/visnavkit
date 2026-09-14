"""Look at what the policy will actually be fed, before spending a GPU on it."""

from pathlib import Path

import numpy as np
import torch

from visnavkit.scripts.dataset.cache import build_dataset
from visnavkit.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["describe_sample", "visualize"]


def _summary(value) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_summary(item) for item in value) + "]"
    tensor = torch.as_tensor(value)
    numeric = tensor.float()
    return (
        f"{tuple(tensor.shape)} {str(tensor.dtype).removeprefix('torch.')} [{numeric.min():.3f}, {numeric.max():.3f}]"
    )


def describe_sample(sample: dict, index: int, dataset) -> str:
    """Every entry the loader emits, with the shape, dtype and range the policy will see."""
    lines = [f"sample {index}: {dataset.windows[index][0]} @ frame {dataset.windows[index][1]}"]
    lines += [f"  goal_types={dataset.goal_types} ego_features={list(dataset.ego_features)}"]
    lines += [f"  {key:<14} {_summary(value)}" for key, value in sample.items()]
    final = torch.as_tensor(sample["future_poses"])[-1]
    lines.append(f"  target x {final[:, 0].tolist()}")
    lines.append(f"  target y {final[:, 1].tolist()}")
    return "\n".join(lines)


def visualize(cfg, split: str = "train", samples: int = 4, output_dir="outputs/dataset", stride: int = 0) -> Path:
    """Print each sample's inputs and draw its frames next to the trajectory it is supervised on."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - depends on the optional extra
        raise ImportError("Visualization needs matplotlib: uv sync --extra plot") from error

    dataset = build_dataset(cfg, split)
    count = min(samples, len(dataset))
    stride = stride or max(len(dataset) // max(count, 1), 1)
    indices = [min(i * stride, len(dataset) - 1) for i in range(count)]

    figure, axes = plt.subplots(count, dataset.seq_len + 1, figsize=(2.2 * (dataset.seq_len + 1), 2.4 * count))
    axes = np.atleast_2d(axes)
    for row, index in enumerate(indices):
        sample = dataset[index]
        print(describe_sample(sample, index, dataset))
        frames = sample["vision"].permute(0, 2, 3, 1).numpy()
        for column in range(dataset.seq_len):
            axes[row, column].imshow(frames[column])
            axes[row, column].set_title(f"t-{dataset.seq_len - 1 - column}", fontsize=8)
        target = sample["future_poses"][-1].numpy()
        axes[row, -1].plot(target[:, 1], target[:, 0], "-o", color="#4477aa", markersize=3)
        axes[row, -1].set_title("target (y left, x fwd)", fontsize=8)
        axes[row, -1].set_aspect("equal", adjustable="datalim")
        axes[row, -1].grid(alpha=0.3)
    for axis in axes[:, :-1].reshape(-1):
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()

    path = Path(output_dir) / f"samples_{split}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)
    logger.info(f"Wrote {path}")
    return path
