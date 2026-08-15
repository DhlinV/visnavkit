import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate


def _forward(overrides):
    with initialize_config_module(version_base=None, config_module="navigators.configs"):
        cfg = compose(config_name="train", overrides=["model.modules.vision_encoder.pretrained=false", *overrides])

    model = instantiate(cfg.model).eval()
    B, S = 2, cfg.common.seq_length
    w, h = (d // cfg.common.downscale_factor for d in cfg.common.crop_wh)
    x = torch.rand(B, S, cfg.model.modules.vision_encoder.in_chans, h, w)
    with torch.no_grad():
        y = model(x)
    return cfg, model, x, y


def _assert_shapes(cfg, y, B=2):
    S = cfg.common.seq_length
    ph = cfg.model.modules.action_decoder.plan_head
    assert y["action"]["plan"]["plans"].shape == (B * S, ph.num_modes * (2 * ph.num_pts * ph.pose_size + 1))
    assert y["vision"]["pose"].shape == (B * S, 1)


def test_base_model_forward():
    cfg, _, _, y = _forward([])
    _assert_shapes(cfg, y)


def test_dinov3_encoder_forward():
    cfg, _, _, y = _forward(["model=dinov3"])
    _assert_shapes(cfg, y)


def test_diffusion_plan_head():
    cfg, model, x, y = _forward(["model=diffusion"])
    _assert_shapes(cfg, y)  # eval: DDIM-sampled plans in the flat MHP layout

    # train: loss comes from noise prediction, sampling skipped
    model.train()
    y = model(x)
    B_S = y["vision"]["pose"].shape[0]
    targets = dict(
        vision=dict(frame_speeds=torch.rand(B_S, 1)),
        action=dict(future_poses=torch.rand(B_S, cfg.plan_len_points, 3)),
    )
    loss_dict, _ = model.get_losses(y, targets)
    assert torch.isfinite(loss_dict["loss"])
    assert loss_dict["action_cls"] == 0
