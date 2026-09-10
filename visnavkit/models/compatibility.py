"""Architecture-preserving configuration aliases for existing checkpoints."""

from omegaconf import OmegaConf

__all__ = ["normalize_model_config"]


def normalize_model_config(model_cfg):
    """Compare known import migrations by effective architecture, retaining all other fields."""
    model = OmegaConf.to_container(model_cfg, resolve=True)
    modules = model["modules"]
    vision = modules["vision_encoder"]
    decoder = modules["action_decoder"]
    temporal = decoder["temporal_encoder"]
    head = decoder["plan_head"]
    vision_prefix = "visnavkit.models.spatial_encoders.vision_encoders."
    temporal_prefix = "visnavkit.models.temporal_encoders."
    action_prefix = "visnavkit.models.action_decoders."
    aliases = {
        "visnavkit.models.encoders.vision_encoder.VisionEncoder": vision_prefix + "timm.TimmVisionEncoder",
        "visnavkit.models.encoders.VisionEncoder": vision_prefix + "timm.TimmVisionEncoder",
        "visnavkit.models.encoders.dino_encoder.DinoEncoder": vision_prefix + "vit_dino.DinoEncoder",
        "visnavkit.models.encoders.DinoEncoder": vision_prefix + "vit_dino.DinoEncoder",
        temporal_prefix + "temporal_encoder.TemporalEncoder": temporal_prefix + "causal.CausalTemporalEncoder",
        temporal_prefix + "TemporalEncoder": temporal_prefix + "causal.CausalTemporalEncoder",
        "visnavkit.models.action_decoder.ActionDecoder": action_prefix + "base.ActionDecoder",
        "visnavkit.models.heads.plan_head.PlanHead": action_prefix + "mhp.PlanHead",
        "visnavkit.models.heads.waypoint_head.WaypointHead": action_prefix + "waypoint.WaypointHead",
        "visnavkit.models.heads.diffusion_plan_head.DiffusionPlanHead": action_prefix + "diffusion.DiffusionPlanHead",
    }
    # Public package exports are aliases of these exact implementation classes.
    for prefix, targets in (
        (
            vision_prefix,
            (
                "timm.TimmVisionEncoder",
                "vit_dino.DinoEncoder",
                "vit_dinov2.DINOv2Encoder",
                "vit_dinov3.DINOv3Encoder",
                "vit_fastvit.FastViTEncoder",
                "cnn_resnet.ResNetEncoder",
                "cnn_efficientnet.EfficientNetEncoder",
                "cnn_mobilenet.MobileNetEncoder",
            ),
        ),
        (
            temporal_prefix,
            (
                "causal.CausalTemporalEncoder",
                "bidirectional.BidirectionalTemporalEncoder",
                "identity.IdentityTemporalEncoder",
            ),
        ),
        (action_prefix, ("base.ActionDecoder", "mhp.PlanHead", "waypoint.WaypointHead", "diffusion.DiffusionPlanHead")),
    ):
        aliases.update({prefix + target.rsplit(".", 1)[1]: prefix + target for target in targets})
    for component in (vision, decoder, temporal, head):
        target = component.get("_target_")
        if target in aliases:
            component["_target_"] = aliases[target]

    vision_target = (
        vision["_target_"].removeprefix(vision_prefix) if vision["_target_"].startswith(vision_prefix) else None
    )
    timm_defaults = {
        "timm.TimmVisionEncoder": ("fastvit_t12", [1, 2, 3], "gelu_tanh"),
        "vit_fastvit.FastViTEncoder": ("fastvit_t12", [1, 2, 3], "gelu_tanh"),
        "cnn_resnet.ResNetEncoder": ("resnet18", [2, 3, 4], None),
        "cnn_efficientnet.EfficientNetEncoder": ("efficientnet_b0", [2, 3, 4], None),
        "cnn_mobilenet.MobileNetEncoder": ("mobilenetv2_100", [2, 3, 4], None),
    }
    dino_defaults = {
        "vit_dino.DinoEncoder": "vit_small_patch16_dinov3",
        "vit_dinov2.DINOv2Encoder": "vit_small_patch14_dinov2",
        "vit_dinov3.DINOv3Encoder": "vit_small_patch16_dinov3",
    }
    if vision_target in timm_defaults:
        backbone, stages, activation = timm_defaults[vision_target]
        vision.setdefault("backbone_name", backbone)
        vision.setdefault("out_indices", stages)
        vision.setdefault("act_layer", activation)
        vision["_target_"] = vision_prefix + "timm.TimmVisionEncoder"
    elif vision_target in dino_defaults:
        vision.setdefault("backbone_name", dino_defaults[vision_target])
        vision["_target_"] = vision_prefix + "vit_dino.DinoEncoder"
    # Backbone initialization is irrelevant once complete checkpoint weights are loaded.
    vision["pretrained"] = False

    if temporal["_target_"] in {
        temporal_prefix + "causal.CausalTemporalEncoder",
        temporal_prefix + "identity.IdentityTemporalEncoder",
    }:
        default_layers = 0 if temporal["_target_"] == temporal_prefix + "identity.IdentityTemporalEncoder" else 1
        temporal.setdefault("num_layers", default_layers)
        if temporal["num_layers"] == 0:
            temporal["_target_"] = temporal_prefix + "identity.IdentityTemporalEncoder"
    if head["_target_"] in {action_prefix + "waypoint.WaypointHead", action_prefix + "diffusion.DiffusionPlanHead"}:
        # Old recipe inheritance passed MHP-only options through these heads' **_ignored kwargs.
        for key in ("loss_cls_alpha", "weights", "mode_selection", "angle_deg_tiebreak_threshold", "log_b_min"):
            head.pop(key, None)
    return model
