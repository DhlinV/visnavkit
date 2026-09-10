import copy

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from visnavkit.benchmark import export as benchmark_export
from visnavkit.models.compatibility import normalize_model_config

VISION = "visnavkit.models.spatial_encoders.vision_encoders."
TEMPORAL = "visnavkit.models.temporal_encoders."
ACTION = "visnavkit.models.action_decoders."
LEGACY_VISION = "visnavkit.models.encoders.vision_encoder.VisionEncoder"
LEGACY_DINO = "visnavkit.models.encoders.dino_encoder.DinoEncoder"


def _model(vision, *, temporal=None, head=None, canonical=False):
    return OmegaConf.create(
        {
            "_target_": "visnavkit.models.e2e_model.E2EModel",
            "feat_size": 8,
            "modules": {
                "vision_encoder": {"in_chans": 6, "pretrained": True, **vision},
                "action_decoder": {
                    "_target_": ACTION + "base.ActionDecoder"
                    if canonical
                    else "visnavkit.models.action_decoder.ActionDecoder",
                    "temporal_encoder": temporal
                    or {
                        "_target_": TEMPORAL
                        + ("causal.CausalTemporalEncoder" if canonical else "temporal_encoder.TemporalEncoder"),
                        "embed_dim": 8,
                        "reduction": "none",
                    },
                    "plan_head": head
                    or {
                        "_target_": ACTION + "mhp.PlanHead"
                        if canonical
                        else "visnavkit.models.heads.plan_head.PlanHead",
                        "num_modes": 3,
                        "num_pts": 4,
                        "pose_size": 3,
                    },
                },
            },
        }
    )


@pytest.mark.parametrize(
    "legacy_vision, canonical_vision",
    [
        (
            {"_target_": LEGACY_VISION, "backbone_name": "fastvit_t8"},
            {"_target_": VISION + "vit_fastvit.FastViTEncoder", "backbone_name": "fastvit_t8"},
        ),
        ({"_target_": LEGACY_VISION}, {"_target_": VISION + "timm.TimmVisionEncoder"}),
        ({"_target_": LEGACY_VISION}, {"_target_": VISION + "FastViTEncoder"}),
        (
            {"_target_": LEGACY_VISION, "backbone_name": "resnet18", "out_indices": [2, 3, 4], "act_layer": None},
            {"_target_": VISION + "cnn_resnet.ResNetEncoder"},
        ),
        (
            {
                "_target_": LEGACY_VISION,
                "backbone_name": "efficientnet_b0",
                "out_indices": [2, 3, 4],
                "act_layer": None,
            },
            {"_target_": VISION + "cnn_efficientnet.EfficientNetEncoder"},
        ),
        (
            {
                "_target_": LEGACY_VISION,
                "backbone_name": "mobilenetv2_100",
                "out_indices": [2, 3, 4],
                "act_layer": None,
            },
            {"_target_": VISION + "cnn_mobilenet.MobileNetEncoder"},
        ),
        (
            {"_target_": LEGACY_DINO, "backbone_name": "vit_small_patch14_dinov2"},
            {"_target_": VISION + "vit_dinov2.DINOv2Encoder"},
        ),
        ({"_target_": LEGACY_DINO}, {"_target_": VISION + "vit_dinov3.DINOv3Encoder"}),
    ],
)
def test_known_vision_migrations_preserve_architecture(legacy_vision, canonical_vision):
    legacy = _model(legacy_vision)
    canonical = _model(canonical_vision, canonical=True)
    original = copy.deepcopy(legacy)
    assert normalize_model_config(legacy) == normalize_model_config(canonical)
    assert legacy == original


@pytest.mark.parametrize(
    "legacy_head, canonical_head",
    [
        ("plan_head.PlanHead", "mhp.PlanHead"),
        ("waypoint_head.WaypointHead", "waypoint.WaypointHead"),
        ("diffusion_plan_head.DiffusionPlanHead", "diffusion.DiffusionPlanHead"),
    ],
)
def test_identity_and_action_head_aliases(legacy_head, canonical_head):
    legacy = _model(
        {"_target_": LEGACY_VISION},
        temporal={"_target_": TEMPORAL + "temporal_encoder.TemporalEncoder", "num_layers": 0},
        head={"_target_": "visnavkit.models.heads." + legacy_head},
    )
    canonical = _model(
        {"_target_": VISION + "vit_fastvit.FastViTEncoder", "pretrained": False},
        temporal={"_target_": TEMPORAL + "identity.IdentityTemporalEncoder"},
        head={"_target_": ACTION + canonical_head},
        canonical=True,
    )
    assert normalize_model_config(legacy) == normalize_model_config(canonical)


@pytest.mark.parametrize("target", ["waypoint.WaypointHead", "diffusion.DiffusionPlanHead"])
def test_removed_ignored_mhp_options_do_not_change_other_head_architectures(target):
    original = _model(
        {"_target_": LEGACY_VISION},
        head={
            "_target_": ACTION + target,
            "mode_selection": "angle",
            "loss_cls_alpha": 1,
            "weights": "unused.pt",
            "angle_deg_tiebreak_threshold": 5.0,
            "log_b_min": -1.609,
        },
    )
    cleaned = _model({"_target_": LEGACY_VISION}, head={"_target_": ACTION + target})
    assert normalize_model_config(original) == normalize_model_config(cleaned)


@pytest.mark.parametrize(
    "path, value",
    [
        ("modules.vision_encoder._target_", "external.CustomVisionEncoder"),
        ("modules.vision_encoder.backbone_name", "resnet18"),
        ("modules.vision_encoder.out_indices", [0, 1, 2]),
        ("modules.vision_encoder.act_layer", None),
        ("modules.action_decoder.temporal_encoder._target_", TEMPORAL + "bidirectional.BidirectionalTemporalEncoder"),
        ("modules.action_decoder.temporal_encoder.num_layers", 0),
        ("modules.action_decoder.plan_head._target_", ACTION + "diffusion.DiffusionPlanHead"),
        ("modules.action_decoder.plan_head.num_modes", 5),
        ("modules.action_decoder.plan_head.mode_selection", "ade"),
        ("feat_size", 16),
    ],
)
def test_architecture_changes_remain_distinct(path, value):
    legacy = _model({"_target_": LEGACY_VISION})
    canonical = _model({"_target_": VISION + "vit_fastvit.FastViTEncoder"}, canonical=True)
    OmegaConf.update(canonical, path, value)
    assert normalize_model_config(legacy) != normalize_model_config(canonical)


def test_cnn_and_dino_family_defaults_are_not_treated_as_identical():
    # A generic encoder with omitted stage/activation settings differs from a native CNN wrapper.
    generic = _model({"_target_": LEGACY_VISION, "backbone_name": "resnet18"})
    resnet = _model({"_target_": VISION + "cnn_resnet.ResNetEncoder"}, canonical=True)
    assert normalize_model_config(generic) != normalize_model_config(resnet)
    dinov2 = _model({"_target_": VISION + "vit_dinov2.DINOv2Encoder"}, canonical=True)
    dinov3 = _model({"_target_": VISION + "vit_dinov3.DINOv3Encoder"}, canonical=True)
    assert normalize_model_config(dinov2) != normalize_model_config(dinov3)


def test_native_export_accepts_legacy_checkpoint_config(tmp_path, monkeypatch):
    saved = OmegaConf.create({"model": _model({"_target_": LEGACY_VISION})})
    requested = OmegaConf.create({"model": _model({"_target_": VISION + "vit_fastvit.FastViTEncoder"}, canonical=True)})
    checkpoint = tmp_path / "legacy.ckpt"
    torch.save({"hyper_parameters": {"cfg": OmegaConf.to_container(saved, resolve=True)}}, checkpoint)

    class ModelLoadingReached(Exception):
        pass

    def load_model(cfg, checkpoint_path):
        assert cfg == saved
        assert checkpoint_path == checkpoint
        raise ModelLoadingReached

    monkeypatch.setattr(benchmark_export, "load_native_model", load_model)
    with pytest.raises(ModelLoadingReached):
        benchmark_export.export_native(requested, tmp_path / "accepted.onnx", checkpoint=checkpoint)


def test_complete_checkpoint_does_not_load_component_initialization_weights(tmp_path, monkeypatch):
    cfg = OmegaConf.create({"model": _model({"_target_": LEGACY_VISION, "weights": "old-vision.pt"})})
    cfg.model.modules.action_decoder.plan_head.weights = "old-head.pt"
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"state_dict": {}}, checkpoint)

    def instantiate(model_cfg):
        assert model_cfg.modules.vision_encoder.pretrained is False
        assert model_cfg.modules.vision_encoder.weights is None
        assert model_cfg.modules.action_decoder.plan_head.weights is None
        return torch.nn.Identity()

    monkeypatch.setattr(benchmark_export, "instantiate", instantiate)
    model = benchmark_export.load_native_model(cfg, checkpoint)
    assert not model.training
    assert cfg.model.modules.vision_encoder.weights == "old-vision.pt"
    assert cfg.model.modules.action_decoder.plan_head.weights == "old-head.pt"


def test_legacy_numpy_plan_parser_retains_keys_and_xy_shapes():
    from visnavkit.scripts.export import parse_plan_output

    means = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    logits = np.array([-2, 2], dtype=np.float32)
    flat = np.concatenate([means.reshape(2, -1), -means.reshape(2, -1), logits[:, None]], axis=1).reshape(1, -1)
    parsed = parse_plan_output(flat, M=2, num_pts=3, pose_width=3)
    assert set(parsed) == {"pred_logits", "pred_confs", "pred_plans", "best_plan"}
    assert all(isinstance(value, np.ndarray) for value in parsed.values())
    np.testing.assert_array_equal(parsed["pred_logits"], logits)
    np.testing.assert_allclose(
        parsed["pred_confs"], np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
    )
    np.testing.assert_array_equal(parsed["pred_plans"], means[:, :, :2])
    np.testing.assert_array_equal(parsed["best_plan"], means[1, :, :2])
