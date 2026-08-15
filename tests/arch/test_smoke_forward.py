import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate


def test_base_model_forward():
    with initialize_config_module(version_base=None, config_module="navigators.configs"):
        cfg = compose(config_name="train", overrides=["model.modules.vision.pretrained=false"])

    model = instantiate(cfg.model).eval()
    B, S = 2, cfg.common.seq_length
    w, h = (d // cfg.common.downscale_factor for d in cfg.common.crop_wh)
    x = torch.rand(B, S, cfg.model.modules.vision.in_chans, h, w)
    with torch.no_grad():
        y = model(x)

    ph = cfg.model.modules.policy.plan_head
    assert y["policy"]["plan"]["plans"].shape == (B * S, ph.num_modes * (2 * ph.num_pts * ph.pose_size + 1))
    assert y["vision"]["pose"].shape == (B * S, 1)
