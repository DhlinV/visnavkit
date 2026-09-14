"""Recipes compose from independent nested Hydra groups under model/."""

import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import OmegaConf

from visnavkit.models.lit_model import disable_pretrained_downloads

SMALL = [
    "common.seq_length=3",
    "common.crop_wh=[32,32]",
    "common.downscale_factor=1",
    "model.feat_size=8",
    "plan_len_points=4",
    "model.vision_encoder.pretrained=false",
    "model.vision_encoder.img_embed_size=16",
    "model.vision_encoder.neck_cfg.n_res_blocks=0",
    "model.temporal_encoder.num_heads=2",
    "model.temporal_encoder.ff_dim=16",
]


def _compose(*overrides):
    with initialize_config_module(version_base=None, config_module="visnavkit.configs"):
        return compose(config_name="train", overrides=list(overrides))


def _targets(cfg):
    model = cfg.model
    return (
        model.vision_encoder._target_.rsplit(".", 1)[1],
        model.temporal_encoder._target_.rsplit(".", 1)[1],
        model.goal_encoder.goal_type,
        model.action_decoder._target_.rsplit(".", 1)[1],
    )


@pytest.mark.parametrize(
    "recipe, expected, layers",
    [
        ("base", ("TimmCNNEncoder", "CausalTemporalEncoder", "none", "MHPDecoder"), 1),
        ("mimic", ("TimmCNNEncoder", "CausalTemporalEncoder", "none", "MHPDecoder"), 1),
        ("resnet18", ("TimmCNNEncoder", "IdentityTemporalEncoder", "none", "RegressionDecoder"), 0),
        ("gnm", ("TimmCNNEncoder", "IdentityTemporalEncoder", "image", "RegressionDecoder"), 0),
        ("vint", ("TimmCNNEncoder", "CausalTemporalEncoder", "image", "RegressionDecoder"), 4),
        ("nomad", ("TimmCNNEncoder", "CausalTemporalEncoder", "image", "GenerativeDecoder"), 4),
        ("citywalker", ("TimmViTEncoder", "CausalTemporalEncoder", "point", "RegressionDecoder"), 4),
        ("s2e", ("TimmViTEncoder", "CausalTemporalEncoder", "point", "MHPDecoder"), 1),
        ("dinov2", ("TimmViTEncoder", "CausalTemporalEncoder", "none", "MHPDecoder"), 1),
        ("dinov3", ("TimmViTEncoder", "CausalTemporalEncoder", "none", "MHPDecoder"), 1),
        ("diffusion", ("TimmCNNEncoder", "CausalTemporalEncoder", "none", "GenerativeDecoder"), 1),
        ("flow_dit", ("TimmCNNEncoder", "CausalTemporalEncoder", "none", "GenerativeDecoder"), 1),
        ("anchor", ("TimmCNNEncoder", "CausalTemporalEncoder", "none", "AnchorDecoder"), 1),
        ("flowpilot", ("TimmCNNEncoder", "CausalTemporalEncoder", "point", "GenerativeDecoder"), 1),
        ("mbra", ("TimmCNNEncoder", "CausalTemporalEncoder", "point", "RegressionDecoder"), 4),
        ("navdp", ("TimmViTEncoder", "CausalTemporalEncoder", "point", "GenerativeDecoder"), 4),
    ],
)
def test_recipes_select_expected_components(recipe, expected, layers):
    cfg = _compose(f"model={recipe}")
    assert _targets(cfg) == expected
    assert cfg.model.temporal_encoder.get("num_layers", 1) == layers
    assert cfg.model.vision_encoder.feat_size == cfg.model.feat_size
    assert cfg.model.action_decoder.feat_size == cfg.model.feat_size
    assert cfg.model.action_decoder.action_space.plan_len_points == cfg.plan_len_points
    OmegaConf.to_container(cfg.model, resolve=True, throw_on_missing=True)


def test_dataset_follows_the_goal_encoder_and_ego_features():
    cfg = _compose("dataset=torch", "model=s2e")
    assert cfg.dataset.train_loader.goal_type == "point"
    assert cfg.dataset.val_loader.goal_horizon_s == cfg.common.goal_horizon_s
    assert list(cfg.dataset.train_loader.ego_features) == []
    cfg = _compose("dataset=torch", "model=gnm", "model/goal_encoder=none", "common.ego_features=[speed,yaw_rate]")
    assert cfg.dataset.train_loader.goal_type == "none"
    assert list(cfg.dataset.train_loader.ego_features) == ["speed", "yaw_rate"]


@pytest.mark.parametrize(
    "overrides, expected",
    [
        (
            ["model=vint", "model/temporal_encoder=identity", "model/action_decoder=mhp"],
            ("TimmCNNEncoder", "IdentityTemporalEncoder", "image", "MHPDecoder"),
        ),
        (
            [
                "model=gnm",
                "model/vision_encoder=dinov3_s",
                "model/goal_encoder=point",
                "model/action_decoder=flow_unet",
            ],
            ("TimmViTEncoder", "IdentityTemporalEncoder", "point", "GenerativeDecoder"),
        ),
        (
            [
                "model=nomad",
                "model/vision_encoder=resnet18",
                "model/temporal_encoder=bidirectional",
                "model/goal_encoder=instruction",
            ],
            ("TimmCNNEncoder", "BidirectionalTemporalEncoder", "instruction", "GenerativeDecoder"),
        ),
    ],
)
def test_component_overrides_replace_recipe_defaults(overrides, expected):
    cfg = _compose(*overrides)
    assert _targets(cfg) == expected
    decoder = cfg.model.action_decoder
    if expected[3] != "MHPDecoder":
        assert "mode_selection" not in decoder
    if expected[3] != "GenerativeDecoder":
        assert "denoiser" not in decoder and "scheduler" not in decoder
    encoder = cfg.model.vision_encoder
    if expected[0] == "TimmViTEncoder":
        assert "out_indices" not in encoder
    else:
        assert "backbone_name" in encoder and "out_indices" in encoder


DECODERS = [
    "regression",
    "mhp",
    "anchor",
    "diffusion_mlp",
    "diffusion_dit",
    "diffusion_unet",
    "flow_mlp",
    "flow_dit",
    "flow_unet",
    "anchor_diffusion_dit",
    "anchor_flow_dit",
]


def _small_decoder(name):
    overrides = [f"model/action_decoder={name}", "model/vision_encoder=resnet18"]
    if name in ("regression", "mhp", "anchor"):
        return overrides + ["model.action_decoder.hidden=16"]
    overrides.append("model.action_decoder.sample_steps=2")
    if "dit" in name:
        overrides += [
            "model.action_decoder.denoiser.hidden=16",
            "model.action_decoder.denoiser.depth=1",
            "model.action_decoder.denoiser.num_heads=2",
        ]
    elif "unet" in name:
        overrides += ["model.action_decoder.denoiser.down_dims=[8,16]", "model.action_decoder.denoiser.n_groups=4"]
    else:
        overrides += ["model.action_decoder.denoiser.hidden=16"]
    return overrides


@pytest.mark.parametrize("name", DECODERS)
def test_every_action_decoder_group_runs(name):
    cfg = _compose(*SMALL, *_small_decoder(name))
    model = instantiate(cfg.model).eval()
    with torch.no_grad():
        out = model(torch.rand(2, 3, 3, 32, 32))
    assert out.plan.plans.shape == (6, model.action_decoder.flat_size)
    assert torch.isfinite(out.plan.plans).all()
    if name.startswith("anchor"):
        assert model.action_decoder.num_modes == 16


@pytest.mark.parametrize("name", ["none", "point", "image", "route_image", "instruction"])
def test_every_goal_encoder_group_runs(name):
    cfg = _compose(
        *SMALL, f"model/goal_encoder={name}", "model/vision_encoder=resnet18", "model.action_decoder.hidden=16"
    )
    disable_pretrained_downloads(cfg.model)
    model = instantiate(cfg.model).eval()
    vision, goal, modalities = model.example_batch(2, 3, (32, 32))
    with torch.no_grad():
        out = model(vision, goal=goal, **modalities)
    assert out.plan.plans.shape[0] == 6
    assert model.export_input_names() == ["vision", "feature_buffer"] + (["goal"] if name != "none" else [])


@pytest.mark.parametrize("name", ["identity", "causal", "causal_4layer", "bidirectional"])
@pytest.mark.parametrize("token_mode", ["global", "fused"])
def test_temporal_groups_and_token_modes(name, token_mode):
    cfg = _compose(
        *SMALL,
        f"model/temporal_encoder={name}",
        "model/vision_encoder=resnet18",
        "model.action_decoder.hidden=16",
        f"model.vision_encoder.token_mode={token_mode}",
        "model.vision_encoder.patch_grid=[2,2]",
    )
    model = instantiate(cfg.model).eval()
    with torch.no_grad():
        out = model(torch.rand(2, 3, 3, 32, 32))
    decisions = 2 if name == "bidirectional" else 6
    assert out.plan.plans.shape[0] == decisions
    assert out.vision.tokens.shape[1] == (1 if token_mode == "global" else 5)


VISION_PRESETS = [
    "fastvit_t8",
    "resnet18",
    "resnet50",
    "efficientnet_b0",
    "mobilenetv2",
    "mobilenetv3",
    "mobilenetv4",
    "convnext_tiny",
    "convnextv2_nano",
    "regnety_008",
    "repvit_m1",
    "efficientvit_b0",
    "dinov2_s",
    "dinov3_s",
    "vit_s",
    "deit_s",
    "eva02_s",
]


@pytest.mark.parametrize("name", VISION_PRESETS)
def test_vision_presets_run_a_small_forward(name):
    cfg = _compose(*SMALL, f"model/vision_encoder={name}", "model.action_decoder.hidden=16")
    model = instantiate(cfg.model).eval()
    with torch.no_grad():
        out = model(torch.rand(2, 3, 3, 32, 32))
    assert out.vision.tokens.shape == (6, 1, 8) and torch.isfinite(out.vision.tokens).all()


@pytest.mark.parametrize("name", ["dinov2_b", "dinov3_b", "siglip_b", "clip_b", "fastvit_t12"])
def test_large_vision_presets_compose(name):
    cfg = _compose(*SMALL, f"model/vision_encoder={name}")
    OmegaConf.to_container(cfg.model, resolve=True, throw_on_missing=True)
    assert cfg.model.vision_encoder.backbone_name


@pytest.mark.parametrize(
    "option, expected",
    [
        ("none", []),
        ("ego", ["ego"]),
        ("camera", ["intrinsics", "extrinsics"]),
        ("ego_camera", ["ego", "intrinsics", "extrinsics"]),
    ],
)
def test_modality_encoder_group_is_an_open_set(option, expected):
    cfg = _compose(*SMALL, f"model/modality_encoder={option}", "model/vision_encoder=resnet18")
    model = instantiate(cfg.model).eval()
    assert model.modality_input_names == expected
    assert model.num_tokens == model.vision_tokens + len(cfg.model.modality_encoders)
    # A further slot is just another key, no policy or group file needed.
    extra = _compose(
        *SMALL,
        f"model/modality_encoder={option}",
        "model/vision_encoder=resnet18",
        "+model.modality_encoders.imu._target_=visnavkit.models.modality.vector.VectorEncoder",
        "+model.modality_encoders.imu.feat_size=${model.feat_size}",
        "+model.modality_encoders.imu.key=imu",
        "+model.modality_encoders.imu.in_dim=6",
    )
    assert instantiate(extra.model).modality_input_names == [*expected, "imu"]


def test_flowpilot_recipe_follows_the_paper_knobs():
    """Anchored rectified flow with Beta(1.5, 1) times and the displacement auxiliary loss."""
    cfg = _compose("model=flowpilot")
    decoder = cfg.model.action_decoder
    assert cfg.model.vision_encoder.speed_head is True
    assert decoder.anchors._target_.endswith("AnchorSet")
    assert decoder.scheduler.time_sampling == "beta"
    assert (decoder.scheduler.beta_alpha, decoder.scheduler.beta_beta) == (1.5, 1.0)


def test_ema_group_is_optional_and_builds_a_trainer_callback():
    from visnavkit.scripts.train import create_trainer
    from visnavkit.utils.ema import EMACallback

    assert _compose("model=base").get("ema") is None
    cfg = _compose("model=base", "ema=default", "logger=null", "+trainer.kwargs.accelerator=cpu")
    assert isinstance(instantiate(cfg.ema), EMACallback)
    assert any(isinstance(callback, EMACallback) for callback in create_trainer(cfg).callbacks)
