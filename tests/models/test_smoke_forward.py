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
    vision, goal, modalities = model.example_batch(B, S, (h, w))
    with torch.no_grad():
        y = model(vision, goal=goal, **modalities)
    return cfg, model, (vision, goal, modalities), y


def _assert_shapes(cfg, model, y, B=2):
    S = cfg.common.seq_length
    assert y.plan.plans.shape == (B * S, model.action_decoder.flat_size)
    assert y.vision.tokens.shape == (B * S, model.vision_tokens, model.feat_size)
    assert (y.vision.speed is None) == (not model.vision_encoder.has_speed_head)


def test_base_model_forward():
    cfg, model, _, y = _forward([])
    _assert_shapes(cfg, model, y)


@pytest.mark.parametrize(
    "recipe",
    [
        "gnm",
        "vint",
        "nomad",
        "citywalker",
        "mbra",
        "navdp",
        "s2e",
        "socialnav",
        "internvla_n1",
        "mimic",
        "flowpilot",
    ],
)
def test_recipe_forward(recipe):
    cfg, model, _, y = _forward([f"model={recipe}"])
    _assert_shapes(cfg, model, y)


def test_modalities_and_speed_head_are_opt_in():
    """Modalities and the auxiliary speed head are per-recipe; each modality adds its tokens."""
    cfg, model, (_, _, modalities), y = _forward([])
    assert modalities == {} and y.modality_tokens is None and model.num_tokens == model.vision_tokens

    cfg, model, (_, _, modalities), y = _forward(
        [
            "model/modality_encoder=ego_camera",
            "model.vision_encoder.speed_head=true",
            "common.ego_features=[speed]",
            "common.use_camera=true",
        ]
    )
    frames = cfg.common.seq_length
    assert model.modality_input_names == ["ego", "intrinsics", "extrinsics"]
    assert modalities["ego"].shape == (2, frames, 1)
    assert modalities["intrinsics"].shape == (2, frames, 3, 3)
    assert modalities["extrinsics"].shape == (2, frames, 4, 4)
    assert {name: tuple(t.shape) for name, t in y.modality_tokens.items()} == {
        "ego": (2, frames, 1, model.feat_size),
        "camera": (2, frames, 1, model.feat_size),
    }
    assert model.num_tokens == model.vision_tokens + 2
    _assert_shapes(cfg, model, y)


@pytest.mark.parametrize("recipe", ["gnm", "nomad"])
def test_recipe_training_loss(recipe):
    cfg, model, (vision, goal, modalities), _ = _forward([f"model={recipe}", "model.vision_encoder.speed_head=true"])
    model.train()
    y = model(vision, goal=goal, **modalities)
    n = y.vision.speed.shape[0]
    targets = dict(
        vision=dict(frame_speeds=torch.rand(n, 1)), action=dict(future_poses=torch.rand(n, cfg.plan_len_points, 3))
    )
    loss_dict, _ = model.get_losses(y, targets)
    assert torch.isfinite(loss_dict["loss"])
    assert loss_dict["action_cls"] == 0
