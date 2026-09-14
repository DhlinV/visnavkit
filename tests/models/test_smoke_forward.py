import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate


def _forward(overrides):
    with initialize_config_module(version_base=None, config_module="visnavkit.configs"):
        cfg = compose(config_name="train", overrides=["model.vision_encoder.pretrained=false", *overrides])
    if "pretrained" in cfg.model.goal_encoder:
        cfg.model.goal_encoder.pretrained = False
    model = instantiate(cfg.model).eval()
    B, S = 2, cfg.common.seq_length
    w, h = (d // cfg.common.downscale_factor for d in cfg.common.crop_wh)
    x = torch.rand(B, S, 6, h, w)
    encoder = model.goal_encoder
    goal = None
    if encoder.num_tokens:
        goal = (
            encoder.example_input(B * S, image_hw=(h, w)).reshape(B, S, -1)
            if encoder.per_frame
            else encoder.example_input(B, image_hw=(h, w))
        )
    with torch.no_grad():
        y = model(x, goal=goal)
    return cfg, model, x, goal, y


def _assert_shapes(cfg, model, y, B=2):
    S = cfg.common.seq_length
    assert y.plan.plans.shape == (B * S, model.action_decoder.flat_size)
    assert y.vision.pose.shape == (B * S, 1)


def test_base_model_forward():
    cfg, model, _, _, y = _forward([])
    _assert_shapes(cfg, model, y)


@pytest.mark.parametrize(
    "recipe", ["gnm", "vint", "nomad", "citywalker", "s2e", "mimic", "dinov3", "diffusion", "flow_dit", "anchor"]
)
def test_recipe_forward(recipe):
    cfg, model, _, _, y = _forward([f"model={recipe}"])
    _assert_shapes(cfg, model, y)


@pytest.mark.parametrize("recipe", ["gnm", "nomad"])
def test_recipe_training_loss(recipe):
    cfg, model, x, goal, _ = _forward([f"model={recipe}"])
    model.train()
    y = model(x, goal=goal)
    n = y.vision.pose.shape[0]
    targets = dict(
        vision=dict(frame_speeds=torch.rand(n, 1)), action=dict(future_poses=torch.rand(n, cfg.plan_len_points, 3))
    )
    loss_dict, _ = model.get_losses(y, targets)
    assert torch.isfinite(loss_dict["loss"])
    assert loss_dict["action_cls"] == 0
