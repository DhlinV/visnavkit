"""Forward smoke test on random inputs, shapes derived from the experiment config.

    uv run python -m visnavkit.scripts.smoke_forward [experiment]  # default: base model, no experiment
"""

import sys

import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate

def main(argv=None):
    """Run configured components on random RGB pairs without downloading weights."""
    args = sys.argv[1:] if argv is None else argv
    overrides = [a if "=" in a else f"experiment={a}" for a in args]
    with initialize_config_module(version_base=None, config_module="visnavkit.configs"):
        cfg = compose(config_name="train", overrides=overrides)
    cfg.model.modules.vision_encoder.pretrained = False

    batch, frames = 2, cfg.common.seq_length
    width, height = (d // cfg.common.downscale_factor for d in cfg.common.crop_wh)

    model = instantiate(cfg.model).eval()
    x = torch.rand(batch, frames, cfg.model.modules.vision_encoder.in_chans, height, width)
    with torch.no_grad():
        y = model(x)

    print(f"{cfg.exp_name}: x {tuple(x.shape)}")
    print("plan:", {k: tuple(v.shape) for k, v in y["action"]["plan"].items()})
    print("pose:", tuple(y["vision"]["pose"].shape))


if __name__ == "__main__":
    main()
