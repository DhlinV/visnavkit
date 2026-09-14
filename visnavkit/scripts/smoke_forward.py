"""Forward smoke test on random inputs, shapes derived from the experiment config.

uv run python -m visnavkit.scripts.smoke_forward [experiment | hydra overrides ...]
"""

import sys

import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate

from visnavkit.models.lit_model import disable_pretrained_downloads


def _shapes(value):
    if value is None:
        return None
    return [tuple(v.shape) if v is not None else None for v in value] if isinstance(value, list) else tuple(value.shape)


def main(argv=None):
    """Run the configured policy on random vision/goal/ego inputs without downloads."""
    args = sys.argv[1:] if argv is None else argv
    overrides = [a if "=" in a else f"experiment={a}" for a in args]
    with initialize_config_module(version_base=None, config_module="visnavkit.configs"):
        cfg = compose(config_name="train", overrides=overrides)
    disable_pretrained_downloads(cfg.model)

    batch, frames = 2, cfg.common.seq_length
    width, height = (d // cfg.common.downscale_factor for d in cfg.common.crop_wh)
    model = instantiate(cfg.model).eval()
    vision, goal, ego, intrinsics, extrinsics = model.example_batch(batch, frames, (height, width))
    with torch.no_grad():
        y = model(vision, goal=goal, ego=ego, intrinsics=intrinsics, extrinsics=extrinsics)

    print(
        f"{cfg.exp_name}: vision {tuple(vision.shape)} goal {_shapes(goal)} ego {_shapes(ego)} "
        f"camera {_shapes(intrinsics)}/{_shapes(extrinsics)}"
    )
    print("plan:", {k: tuple(v.shape) for k, v in y.plan.items() if torch.is_tensor(v)})
    print("tokens:", tuple(y.vision.tokens.shape), "speed:", _shapes(y.vision.speed))


if __name__ == "__main__":
    main()
