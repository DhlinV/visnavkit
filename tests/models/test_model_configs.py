"""Architecture recipes remain composable through independent Hydra groups."""

import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict


def _compose(*overrides):
    with initialize_config_module(version_base=None, config_module="visnavkit.configs"):
        return compose(config_name="train", overrides=list(overrides))


def _assert_components(cfg, vision, temporal, head, layers):
    modules = cfg.model.modules
    assert modules.vision_encoder._target_ == f"visnavkit.models.spatial_encoders.vision_encoders.{vision}"
    assert modules.action_decoder._target_ == "visnavkit.models.action_decoders.base.ActionDecoder"
    assert modules.action_decoder.temporal_encoder._target_ == f"visnavkit.models.temporal_encoders.{temporal}"
    assert modules.action_decoder.plan_head._target_ == f"visnavkit.models.action_decoders.{head}"
    assert modules.action_decoder.temporal_encoder.get("num_layers", 1) == layers
    assert modules.vision_encoder.neck_cfg.dim == cfg.model.feat_size
    assert modules.action_decoder.temporal_encoder.embed_dim == cfg.model.feat_size
    assert modules.action_decoder.plan_head.feat_size == cfg.model.feat_size
    assert modules.action_decoder.plan_head.num_pts == cfg.plan_len_points
    # Resolve the whole model, so missing interpolation targets cannot hide in unused fields.
    OmegaConf.to_container(cfg.model, resolve=True, throw_on_missing=True)


@pytest.mark.parametrize(
    "recipe, vision, temporal, head, layers",
    [
        ("base", "vit_fastvit.FastViTEncoder", "causal.CausalTemporalEncoder", "mhp.PlanHead", 1),
        ("mimic", "vit_fastvit.FastViTEncoder", "causal.CausalTemporalEncoder", "mhp.PlanHead", 1),
        ("diffusion", "vit_fastvit.FastViTEncoder", "causal.CausalTemporalEncoder", "diffusion.DiffusionPlanHead", 1),
        ("dinov2", "vit_dinov2.DINOv2Encoder", "causal.CausalTemporalEncoder", "mhp.PlanHead", 1),
        ("dinov3", "vit_dinov3.DINOv3Encoder", "causal.CausalTemporalEncoder", "mhp.PlanHead", 1),
        ("s2e", "vit_dinov3.DINOv3Encoder", "causal.CausalTemporalEncoder", "mhp.PlanHead", 1),
        ("gnm", "cnn_mobilenet.MobileNetEncoder", "identity.IdentityTemporalEncoder", "waypoint.WaypointHead", 0),
        ("resnet18", "cnn_resnet.ResNetEncoder", "identity.IdentityTemporalEncoder", "waypoint.WaypointHead", 0),
        ("vint", "cnn_efficientnet.EfficientNetEncoder", "causal.CausalTemporalEncoder", "waypoint.WaypointHead", 4),
        ("citywalker", "vit_dinov2.DINOv2Encoder", "causal.CausalTemporalEncoder", "waypoint.WaypointHead", 4),
        ("nomad", "cnn_efficientnet.EfficientNetEncoder", "causal.CausalTemporalEncoder", "diffusion.DiffusionPlanHead", 4),
    ],
)
def test_existing_recipes_select_canonical_components(recipe, vision, temporal, head, layers):
    cfg = _compose(f"model={recipe}")
    _assert_components(cfg, vision, temporal, head, layers)
    assert cfg.model.modules.action_decoder.temporal_encoder.reduction == "none"
    assert cfg.model.modules.action_decoder.plan_head.num_modes == (1 if head == "waypoint.WaypointHead" else 5)


_CROSS_OVERRIDES = [
    (
        ["model=vint", "temporal_encoder=identity", "action_decoder=mhp"],
        "cnn_efficientnet.EfficientNetEncoder", "identity.IdentityTemporalEncoder", "mhp.PlanHead", 0,
    ),
    (
        ["model=gnm", "vision_encoder=vit_dinov3", "temporal_encoder=causal", "action_decoder=diffusion"],
        "vit_dinov3.DINOv3Encoder", "causal.CausalTemporalEncoder", "diffusion.DiffusionPlanHead", 1,
    ),
    (
        ["model=nomad", "vision_encoder=cnn_resnet", "temporal_encoder=bidirectional", "action_decoder=waypoint"],
        "cnn_resnet.ResNetEncoder", "bidirectional.BidirectionalTemporalEncoder", "waypoint.WaypointHead", 1,
    ),
]


@pytest.mark.parametrize("overrides, vision, temporal, head, layers", _CROSS_OVERRIDES)
def test_component_overrides_replace_recipe_defaults_without_stale_options(overrides, vision, temporal, head, layers):
    cfg = _compose(*overrides)
    _assert_components(cfg, vision, temporal, head, layers)
    encoder = cfg.model.modules.vision_encoder
    plan_head = cfg.model.modules.action_decoder.plan_head
    if vision.startswith("vit_dino"):
        assert "out_indices" not in encoder and "act_layer" not in encoder
    else:
        assert "freeze_backbone" not in encoder
    if head != "mhp.PlanHead":
        assert "mode_selection" not in plan_head and "loss_cls_alpha" not in plan_head
    if head != "diffusion.DiffusionPlanHead":
        assert "sample_steps" not in plan_head and "train_timesteps" not in plan_head
    assert plan_head.num_modes == (1 if head == "waypoint.WaypointHead" else 5)
    expected_reduction = "last" if temporal.startswith("bidirectional") else "none"
    assert cfg.model.modules.action_decoder.temporal_encoder.reduction == expected_reduction


@pytest.mark.parametrize("overrides, vision, temporal, head, layers", _CROSS_OVERRIDES)
def test_composed_component_overrides_run_a_small_forward(overrides, vision, temporal, head, layers):
    cfg = _compose(*overrides)
    cfg.common.seq_length = 3
    cfg.model.feat_size = 8
    cfg.plan_len_points = 3
    encoder = cfg.model.modules.vision_encoder
    encoder.pretrained = False
    encoder.img_embed_size = 16
    encoder.neck_cfg.n_res_blocks = 0
    temporal_cfg = cfg.model.modules.action_decoder.temporal_encoder
    temporal_cfg.num_heads = 2
    temporal_cfg.ff_dim = 16
    plan_head = cfg.model.modules.action_decoder.plan_head
    with open_dict(plan_head):
        plan_head.hidden = 16
    if head == "diffusion.DiffusionPlanHead":
        plan_head.time_embed_dim = 8
        plan_head.train_timesteps = 5
        plan_head.sample_steps = 2
    model = instantiate(cfg.model).eval()
    with torch.no_grad():
        output = model(torch.rand(2, 3, 6, 32, 32))
    batch = 6 if temporal_cfg.reduction == "none" else 2
    assert output["vision"]["pose"].shape == (6, 1)
    assert output["action"]["plan"]["plans"].shape == (batch, plan_head.num_modes * 19)
    assert torch.isfinite(output["action"]["plan"]["plans"]).all()
