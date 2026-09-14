import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate

from visnavkit.models.lit_model import disable_pretrained_downloads


def _forward(overrides):
    with initialize_config_module(version_base=None, config_module="visnavkit.configs"):
        cfg = compose(config_name="train", overrides=list(overrides))
    disable_pretrained_downloads(cfg.model)
    model = instantiate(cfg.model).eval()
    B, S = 2, cfg.common.seq_length
    w, h = (d // cfg.common.downscale_factor for d in cfg.common.crop_wh)
    inputs = model.example_batch(B, S, (h, w))
    with torch.no_grad():
        y = model(inputs[0], goal=inputs[1], ego=inputs[2], intrinsics=inputs[3], extrinsics=inputs[4])
    return cfg, model, inputs, y


def _assert_shapes(cfg, model, y, B=2):
    S = cfg.common.seq_length
    assert y.plan.plans.shape == (B * S, model.action_decoder.flat_size)
    assert y.vision.tokens.shape == (B * S, model.vision_tokens, model.feat_size)
    assert (y.vision.speed is None) == (not model.vision_encoder.has_speed_head)


def test_base_model_forward():
    cfg, model, _, y = _forward([])
    _assert_shapes(cfg, model, y)


@pytest.mark.parametrize(
    "recipe", ["gnm", "vint", "nomad", "citywalker", "s2e", "mimic", "dinov3", "diffusion", "flow_dit", "anchor"]
)
def test_recipe_forward(recipe):
    cfg, model, _, y = _forward([f"model={recipe}"])
    _assert_shapes(cfg, model, y)


def test_side_inputs_are_opt_in():
    """Ego status, calibration and the speed head are per-recipe, and each adds its own tokens."""
    cfg, model, inputs, y = _forward([])
    assert inputs[2] is None and inputs[3] is None and inputs[4] is None
    assert y.ego_tokens is None and y.camera_tokens is None and model.num_tokens == model.vision_tokens

    cfg, model, (_, _, ego, intrinsics, extrinsics), y = _forward(
        [
            "model/ego_encoder=state",
            "model/camera_encoder=pinhole",
            "model.vision_encoder.speed_head=true",
            "common.ego_features=[speed]",
            "common.use_camera=true",
        ]
    )
    frames = cfg.common.seq_length
    assert ego.shape == (2, frames, 1)
    assert intrinsics.shape == (2, frames, 3, 3) and extrinsics.shape == (2, frames, 4, 4)
    assert y.ego_tokens.shape == y.camera_tokens.shape == (2, frames, 1, model.feat_size)
    assert model.num_tokens == model.vision_tokens + 2
    _assert_shapes(cfg, model, y)


@pytest.mark.parametrize("recipe", ["gnm", "nomad"])
def test_recipe_training_loss(recipe):
    cfg, model, inputs, _ = _forward([f"model={recipe}", "model.vision_encoder.speed_head=true"])
    model.train()
    y = model(inputs[0], goal=inputs[1], ego=inputs[2])
    n = y.vision.speed.shape[0]
    targets = dict(
        vision=dict(frame_speeds=torch.rand(n, 1)), action=dict(future_poses=torch.rand(n, cfg.plan_len_points, 3))
    )
    loss_dict, _ = model.get_losses(y, targets)
    assert torch.isfinite(loss_dict["loss"])
    assert loss_dict["action_cls"] == 0
