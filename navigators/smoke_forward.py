"""Forward smoke test on random inputs, shapes derived from the experiment config.

    uv run python -m navigators.smoke_forward [experiment]  # default: base model, no experiment
"""

import sys

import torch
from hydra import compose, initialize
from hydra.utils import instantiate

# args are hydra overrides (a bare first arg means experiment=<arg>), e.g.
#   uv run python -m navigators.smoke_forward experiment=baseline model=base
overrides = [a if "=" in a else f"experiment={a}" for a in sys.argv[1:]]
with initialize(version_base=None, config_path="configs"):
    cfg = compose(config_name="train", overrides=overrides)

B, S = 2, cfg.common.seq_length
w, h = (d // cfg.common.downscale_factor for d in cfg.common.crop_wh)

model = instantiate(cfg.model).eval()
x = torch.rand(B, S, cfg.model.modules.vision_encoder.in_chans, h, w)
with torch.no_grad():
    y = model(x)

print(f"{cfg.exp_name}: x {tuple(x.shape)}")
print("plan:", {k: tuple(v.shape) for k, v in y["action"]["plan"].items()})
print("pose:", tuple(y["vision"]["pose"].shape))
