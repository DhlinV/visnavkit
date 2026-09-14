"""Forward smoke test on random inputs, shapes derived from the experiment config.

uv run python -m visnavkit.scripts.smoke_forward [experiment | hydra overrides ...]
"""

import sys

import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate


def main(argv=None):
    """Run the configured policy on random RGB pairs (and a synthetic goal) without downloads."""
    args = sys.argv[1:] if argv is None else argv
    overrides = [a if "=" in a else f"experiment={a}" for a in args]
    with initialize_config_module(version_base=None, config_module="visnavkit.configs"):
        cfg = compose(config_name="train", overrides=overrides)
    cfg.model.vision_encoder.pretrained = False
    if "pretrained" in cfg.model.goal_encoder:
        cfg.model.goal_encoder.pretrained = False

    batch, frames = 2, cfg.common.seq_length
    width, height = (d // cfg.common.downscale_factor for d in cfg.common.crop_wh)
    model = instantiate(cfg.model).eval()
    x = torch.rand(batch, frames, 6, height, width)
    goal_encoder = model.goal_encoder
    goal = None
    if goal_encoder.num_tokens:
        goal = goal_encoder.example_input(batch * frames if goal_encoder.per_frame else batch, image_hw=(height, width))
        if goal_encoder.per_frame:
            goal = goal.reshape(batch, frames, -1)
    with torch.no_grad():
        y = model(x, goal=goal)

    print(f"{cfg.exp_name}: x {tuple(x.shape)}" + (f" goal {tuple(goal.shape)}" if goal is not None else ""))
    print("plan:", {k: tuple(v.shape) for k, v in y.plan.items() if torch.is_tensor(v)})
    print("pose:", tuple(y.vision.pose.shape), "tokens:", tuple(y.vision.tokens.shape))


if __name__ == "__main__":
    main()
