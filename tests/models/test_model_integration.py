"""Exercise model composition across training, saved configurations, and deployment."""

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


def _model_config(temporal="causal", head="mhp", *, reduction="none", seq_len=3, seq_step=1):
    temporal_class = {
        "causal": "CausalTemporalEncoder",
        "bidirectional": "BidirectionalTemporalEncoder",
        "identity": "IdentityTemporalEncoder",
    }[temporal]
    head_class = {"mhp": "PlanHead", "waypoint": "WaypointHead", "diffusion": "DiffusionPlanHead"}[head]
    plan_head = {
        "_target_": f"visnavkit.models.action_decoders.{head}.{head_class}",
        "feat_size": 8,
        "hidden": 16,
        "num_modes": 1 if head == "waypoint" else 2,
        "num_pts": 3,
        "pose_size": 3,
    }
    if head == "diffusion":
        plan_head.update(train_timesteps=5, sample_steps=2, time_embed_dim=8)
    return OmegaConf.create(
        {
            "_target_": "visnavkit.models.e2e_model.E2EModel",
            "feat_size": 8,
            "loss_cfg": {"vision_weight": 1, "action_weight": 2},
            "export_cfg": {"seq_len": seq_len, "seq_step": seq_step},
            "modules": {
                "vision_encoder": {
                    "_target_": "visnavkit.models.spatial_encoders.vision_encoders.cnn_resnet.ResNetEncoder",
                    "_recursive_": False,
                    "backbone_name": "resnet18",
                    "pretrained": False,
                    "in_chans": 6,
                    "out_indices": [2, 3, 4],
                    "act_layer": None,
                    "img_embed_size": 16,
                    "img_embed_drop": 0,
                    "p_drop_prev_img": 0,
                    "neck_cfg": {"dim": 8, "n_res_blocks": 0, "dropout": 0},
                    "heads": {},
                    "loss_pose_weight": 1,
                },
                "action_decoder": {
                    "_target_": "visnavkit.models.action_decoders.base.ActionDecoder",
                    "temporal_encoder": {
                        "_target_": f"visnavkit.models.temporal_encoders.{temporal}.{temporal_class}",
                        "embed_dim": 8,
                        "num_heads": 2,
                        "ff_dim": 16,
                        "seq_len": seq_len,
                        "dropout": 0,
                        "mask_p": 0,
                        "reduction": reduction,
                    },
                    "plan_head": plan_head,
                },
            },
        }
    )


@pytest.mark.parametrize(
    "temporal, head, reduction",
    [
        ("causal", "mhp", "none"),
        ("causal", "diffusion", "none"),
        ("bidirectional", "waypoint", "last"),
        ("identity", "waypoint", "none"),
    ],
)
def test_composed_model_trains_with_noncontiguous_frames(temporal, head, reduction):
    torch.manual_seed(31)
    model = instantiate(_model_config(temporal, head, reduction=reduction)).train()
    source = torch.rand(3, 2, 6, 32, 32, requires_grad=True)
    frames = source.transpose(0, 1)
    assert not frames.is_contiguous()
    predictions = model(frames)
    plan_batch = 6 if reduction == "none" else 2
    targets = {
        "vision": {"frame_speeds": torch.rand(6, 1)},
        "action": {"future_poses": torch.rand(plan_batch, 3, 3)},
    }
    losses, debug = model.get_losses(predictions, targets)
    assert predictions["vision"]["pose"].shape == (6, 1)
    assert predictions["action"]["plan"]["plans"].shape == (plan_batch, model.action_decoder.plan_head.flat_size)
    assert debug["action_loss_debug"]["imitation_loss_per_sample"].shape == (plan_batch,)
    assert all(torch.isfinite(value).all() for value in losses.values())
    losses["loss"].backward()
    assert source.grad is not None and torch.isfinite(source.grad).all()
    assert source.grad.abs().sum() > 0
    for component in (model.vision_encoder.backbone, model.action_decoder.plan_head):
        gradients = [parameter.grad for parameter in component.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(gradient.abs().sum() > 0 for gradient in gradients)


@pytest.mark.parametrize(
    "temporal, head, reduction, seq_len, seq_step",
    [
        ("causal", "mhp", "last", 3, 2),
        ("bidirectional", "waypoint", "last", 3, 2),
        ("bidirectional", "waypoint", "avg", 3, 2),
        ("identity", "waypoint", "last", 1, 1),
        ("causal", "diffusion", "last", 3, 1),
    ],
)
def test_feature_buffer_matches_full_observation_window(temporal, head, reduction, seq_len, seq_step):
    torch.manual_seed(32)
    model = instantiate(
        _model_config(temporal, head, reduction=reduction, seq_len=seq_len, seq_step=seq_step)
    ).eval()
    history_size = (seq_len - 1) * seq_step
    frames = torch.rand(2, history_size + 1, 6, 32, 32)
    with torch.no_grad():
        encoded = model.vision_encoder(frames.flatten(0, 1))["feat_out"].reshape(2, history_size + 1, 8)
        # Use identical noise draws for the stochastic diffusion head in both paths.
        torch.manual_seed(33)
        sequence_output = model(frames[:, ::seq_step])
        torch.manual_seed(33)
        exported_plan, exported_pose, exported_token = model(frames[:, -1], encoded[:, :-1])

    assert model.get_export_output_names() == ["plan", "pose", "feat_out"]
    torch.testing.assert_close(exported_plan, sequence_output["action"]["plan"]["plans"], atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(exported_pose, sequence_output["vision"]["pose"].reshape(2, seq_len, 1)[:, -1])
    torch.testing.assert_close(exported_token, encoded[:, -1])


def test_saved_legacy_hydra_config_and_checkpoint_still_load(tmp_path):
    config = _model_config(reduction="last")
    legacy = OmegaConf.create(OmegaConf.to_container(config))
    legacy.modules.vision_encoder._target_ = "visnavkit.models.encoders.vision_encoder.VisionEncoder"
    legacy.modules.action_decoder._target_ = "visnavkit.models.action_decoder.ActionDecoder"
    legacy.modules.action_decoder.temporal_encoder._target_ = (
        "visnavkit.models.temporal_encoders.temporal_encoder.TemporalEncoder"
    )
    legacy.modules.action_decoder.plan_head._target_ = "visnavkit.models.heads.plan_head.PlanHead"
    config_path = tmp_path / "saved-config.yaml"
    weights_path = tmp_path / "saved-weights.pt"
    OmegaConf.save(legacy, config_path)
    original = instantiate(OmegaConf.load(config_path)).eval()
    torch.save(original.state_dict(), weights_path)
    restored = instantiate(config).eval()
    restored.load_state_dict(torch.load(weights_path, weights_only=True), strict=True)
    frames = torch.rand(2, 3, 6, 32, 32)
    with torch.no_grad():
        before, after = original(frames), restored(frames)
    torch.testing.assert_close(after["vision"]["pose"], before["vision"]["pose"], rtol=0, atol=0)
    torch.testing.assert_close(after["vision"]["feat_out"], before["vision"]["feat_out"], rtol=0, atol=0)
    torch.testing.assert_close(after["action"]["plan"]["plans"], before["action"]["plan"]["plans"], rtol=0, atol=0)


def test_partial_training_preserves_parent_and_selected_module_modes():
    config = _model_config()
    config.trainable_modules = ["action_decoder"]
    model = instantiate(config)
    for _ in range(2):
        assert model.training
        assert not model.vision_encoder.training
        assert model.action_decoder.training
        assert all(not parameter.requires_grad for parameter in model.vision_encoder.parameters())
        assert all(parameter.requires_grad for parameter in model.action_decoder.parameters())
        model.eval()
        assert not any(module.training for module in model.modules())
        model.train()


@pytest.mark.parametrize("reduction", ["none", "last", "avg", "sum"])
@pytest.mark.parametrize("shared_times", [False, True])
def test_training_targets_follow_action_reduction(reduction, shared_times):
    pytest.importorskip("lightning")
    from visnavkit.models.lit_model import build_targets

    future_poses = torch.arange(54, dtype=torch.float32).reshape(2, 3, 3, 3)
    speeds = torch.arange(6, dtype=torch.float32).reshape(2, 3, 1)
    times = torch.tensor([0.5, 1.0, 2.0]) if shared_times else torch.tensor([[0.5, 1.0, 2.0], [0.4, 1.2, 2.1]])
    targets = build_targets(
        {"future_poses": future_poses, "frame_speeds": speeds, "target_times_s": times},
        action_reduction=reduction,
    )
    torch.testing.assert_close(targets["vision"]["frame_speeds"], speeds.flatten(0, 1))
    torch.testing.assert_close(
        targets["action"]["future_poses"], future_poses.flatten(0, 1) if reduction == "none" else future_poses[:, -1]
    )
    expected_times = times.repeat_interleave(3, dim=0) if reduction == "none" and not shared_times else times
    torch.testing.assert_close(targets["action"]["target_times_s"], expected_times)
