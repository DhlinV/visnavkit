"""Policy composition across training, feature-buffer deployment, goals, and partial training."""

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

TEMPORAL = {
    "causal": "causal.CausalTemporalEncoder",
    "bidirectional": "bidirectional.BidirectionalTemporalEncoder",
    "identity": "identity.IdentityTemporalEncoder",
}
GOAL = {
    "none": {"_target_": "visnavkit.models.goal.none.NoGoalEncoder"},
    "point": {"_target_": "visnavkit.models.goal.point.PointGoalEncoder", "hidden": 16},
    "image": {
        "_target_": "visnavkit.models.goal.image.ImageGoalEncoder",
        "backbone_name": "resnet18",
        "pretrained": False,
        "stack_observation": True,
    },
    "instruction": {
        "_target_": "visnavkit.models.goal.instruction.InstructionGoalEncoder",
        "embed_dim": 16,
        "hidden": 16,
    },
}
DECODER = {
    "regression": {"_target_": "visnavkit.models.action.regression.RegressionDecoder", "hidden": 16},
    "mhp": {"_target_": "visnavkit.models.action.mhp.MHPDecoder", "num_modes": 2, "hidden": 16},
    "anchor": {
        "_target_": "visnavkit.models.action.anchor.AnchorDecoder",
        "hidden": 16,
        "anchors": {"_target_": "visnavkit.models.action.anchors.AnchorSet", "num_anchors": 6},
    },
    "flow_dit": {
        "_target_": "visnavkit.models.action.generative.GenerativeDecoder",
        "num_modes": 2,
        "sample_steps": 2,
        "denoiser": {
            "_target_": "visnavkit.models.action.denoisers.dit.DiTDenoiser",
            "_partial_": True,
            "hidden": 16,
            "depth": 1,
            "num_heads": 2,
            "time_dim": 8,
        },
        "scheduler": {"_target_": "visnavkit.models.action.schedulers.flow.FlowMatchingScheduler"},
    },
    "diffusion_unet": {
        "_target_": "visnavkit.models.action.generative.GenerativeDecoder",
        "num_modes": 2,
        "sample_steps": 2,
        "denoiser": {
            "_target_": "visnavkit.models.action.denoisers.unet.UNet1DDenoiser",
            "_partial_": True,
            "down_dims": [8, 16],
            "n_groups": 4,
            "time_dim": 8,
        },
        "scheduler": {"_target_": "visnavkit.models.action.schedulers.ddim.DDIMScheduler", "train_timesteps": 5},
    },
}


def policy_config(
    temporal="causal",
    goal="none",
    decoder="mhp",
    *,
    reduction="none",
    seq_len=3,
    seq_step=1,
    token_mode="global",
    kind="waypoint",
):
    return OmegaConf.create(
        {
            "_target_": "visnavkit.models.policy.NavigationPolicy",
            "feat_size": 8,
            "loss_cfg": {"vision_weight": 1, "action_weight": 2},
            "export_cfg": {"seq_len": seq_len, "seq_step": seq_step},
            "vision_encoder": {
                "_target_": "visnavkit.models.vision.timm_cnn.TimmCNNEncoder",
                "_recursive_": False,
                "backbone_name": "resnet18",
                "pretrained": False,
                "out_indices": [2, 3, 4],
                "feat_size": 8,
                "token_mode": token_mode,
                "patch_grid": [2, 2],
                "img_embed_size": 16,
                "img_embed_drop": 0,
                "p_drop_prev_img": 0,
                "neck_cfg": {"n_res_blocks": 0, "dropout": 0},
            },
            "temporal_encoder": {
                "_target_": f"visnavkit.models.temporal.{TEMPORAL[temporal]}",
                "embed_dim": 8,
                "num_heads": 2,
                "ff_dim": 16,
                "seq_len": seq_len,
                "dropout": 0,
                "mask_p": 0,
                "reduction": reduction,
                **({"num_layers": 0} if temporal == "identity" else {}),
            },
            "goal_encoder": {"feat_size": 8, **GOAL[goal]},
            "action_decoder": {
                "feat_size": 8,
                "action_space": {
                    "_target_": "visnavkit.models.action.spaces.ActionSpace",
                    "kind": kind,
                    "pose_size": 3,
                    "plan_len_seconds": 3,
                    "plan_len_points": 3,
                },
                **DECODER[decoder],
            },
        }
    )


def synthetic_goal(model, batch, frames, image_hw=(32, 32)):
    encoder = model.goal_encoder
    if encoder.num_tokens == 0:
        return None
    if encoder.per_frame:
        return encoder.example_input(batch * frames, image_hw=image_hw).reshape(batch, frames, -1)
    return encoder.example_input(batch, image_hw=image_hw)


@pytest.mark.parametrize(
    "temporal, goal, decoder, reduction, token_mode",
    [
        ("causal", "none", "mhp", "none", "global"),
        ("causal", "point", "regression", "none", "fused"),
        ("bidirectional", "image", "flow_dit", "last", "patch"),
        ("identity", "instruction", "anchor", "last", "global"),
        ("causal", "point", "diffusion_unet", "none", "global"),
    ],
)
def test_policy_trains_end_to_end(temporal, goal, decoder, reduction, token_mode):
    torch.manual_seed(31)
    model = instantiate(policy_config(temporal, goal, decoder, reduction=reduction, token_mode=token_mode)).train()
    source = torch.rand(3, 2, 6, 32, 32, requires_grad=True)
    frames = source.transpose(0, 1)  # noncontiguous (B=2, F=3)
    goal_batch = synthetic_goal(model, 2, 3)
    out = model(frames, goal=goal_batch)
    decisions = 6 if reduction == "none" else 2
    targets = {"vision": {"frame_speeds": torch.rand(6, 1)}, "action": {"future_poses": torch.rand(decisions, 3, 3)}}
    losses, debug = model.get_losses(out, targets)
    assert out.vision.pose.shape == (6, 1)
    assert out.vision.tokens.shape == (6, model.num_tokens, 8)
    assert out.plan.plans.shape == (decisions, model.action_decoder.flat_size)
    assert debug["action_loss_debug"]["imitation_loss_per_sample"].shape == (decisions,)
    assert all(torch.isfinite(v).all() for v in losses.values())
    losses["loss"].backward()
    assert source.grad is not None and torch.isfinite(source.grad).all() and source.grad.abs().sum() > 0
    if goal != "none" and decoder != "flow_dit":  # DiT adaLN-Zero gates start at zero, so upstream gradients do too
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.goal_encoder.parameters())


@pytest.mark.parametrize(
    "temporal, goal, decoder, reduction, seq_len, seq_step",
    [
        ("causal", "none", "mhp", "last", 3, 2),
        ("causal", "point", "regression", "none", 3, 2),
        ("bidirectional", "image", "flow_dit", "last", 3, 1),
        ("identity", "instruction", "anchor", "last", 1, 1),
        ("bidirectional", "none", "regression", "avg", 3, 2),
    ],
)
def test_feature_buffer_matches_full_observation_window(temporal, goal, decoder, reduction, seq_len, seq_step):
    torch.manual_seed(32)
    model = instantiate(
        policy_config(
            temporal, goal, decoder, reduction=reduction, seq_len=seq_len, seq_step=seq_step, token_mode="fused"
        )
    ).eval()
    history = (seq_len - 1) * seq_step
    frames = torch.rand(2, history + 1, 6, 32, 32)
    window = frames[:, ::seq_step]
    goal_batch = synthetic_goal(model, 2, seq_len)
    noise = model.action_decoder.example_noise(2) if model.action_decoder.uses_noise else None
    with torch.no_grad():
        encoded = model.encode_frames(frames).tokens.reshape(2, history + 1, model.token_dim)
        full = model(window, goal=goal_batch, noise=noise)
        expected = full.plan.plans if reduction != "none" else full.plan.plans.reshape(2, seq_len, -1)[:, -1]
        last_goal = None if goal_batch is None else (goal_batch[:, -1] if model.goal_encoder.per_frame else goal_batch)
        plan, pose, token = model.predict(frames[:, -1], encoded[:, :-1], goal=last_goal, noise=noise)
    assert model.export_output_names() == ["plan", "pose", "feat_out"]
    assert model.export_input_names() == ["input", "feature_buffer"] + (["goal"] if goal != "none" else []) + (
        ["noise"] if noise is not None else []
    )
    torch.testing.assert_close(plan, expected, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(pose, full.vision.pose.reshape(2, seq_len, 1)[:, -1])
    torch.testing.assert_close(token, encoded[:, -1])


def test_goal_shape_validation_and_null_goal():
    model = instantiate(policy_config("causal", "point", "regression")).eval()
    frames = torch.rand(2, 3, 6, 32, 32)
    with pytest.raises(ValueError, match="per-frame goals"):
        model(frames, goal=torch.zeros(2, 4, 3))
    out = model(frames)  # missing goal -> learned null token
    assert out.goal_tokens.shape == (6, 1, 8)
    image = instantiate(policy_config("bidirectional", "image", "regression", reduction="last")).eval()
    with pytest.raises(ValueError, match="one goal per window"):
        image(frames, goal=torch.rand(3, 3, 32, 32))


def test_partial_training_uses_module_prefixes():
    config = policy_config()
    config.trainable_modules = ["action_decoder"]
    model = instantiate(config)
    for _ in range(2):
        assert model.training and not model.vision_encoder.training and model.action_decoder.training
        assert all(not p.requires_grad for p in model.vision_encoder.parameters())
        assert all(p.requires_grad for p in model.action_decoder.parameters())
        model.eval()
        assert not any(module.training for module in model.modules())
        model.train()
    config = policy_config()
    config.frozen_modules = ["vision_encoder.backbone"]
    model = instantiate(config)
    assert all(not p.requires_grad for p in model.vision_encoder.backbone.parameters())
    assert model.vision_encoder.final_linear.weight.requires_grad


def test_velocity_action_space_end_to_end():
    model = instantiate(policy_config(decoder="regression", kind="velocity")).train()
    out = model(torch.rand(2, 3, 6, 32, 32))
    losses, _ = model.get_losses(
        out, {"vision": {"frame_speeds": torch.rand(6, 1)}, "action": {"future_poses": torch.rand(6, 3, 3)}}
    )
    losses["loss"].backward()
    parsed = model.action_decoder.parse_output(out.plan.plans)
    assert parsed["plans"].shape == (6, 1, 3, 3)


@pytest.mark.parametrize("reduction", ["none", "last", "avg", "sum"])
@pytest.mark.parametrize("shared_times", [False, True])
def test_training_targets_follow_action_reduction(reduction, shared_times):
    from visnavkit.models.lit_model import build_targets

    future_poses = torch.arange(54, dtype=torch.float32).reshape(2, 3, 3, 3)
    speeds = torch.arange(6, dtype=torch.float32).reshape(2, 3, 1)
    times = torch.tensor([0.5, 1.0, 2.0]) if shared_times else torch.tensor([[0.5, 1.0, 2.0], [0.4, 1.2, 2.1]])
    targets = build_targets(
        {"future_poses": future_poses, "frame_speeds": speeds, "target_times_s": times}, action_reduction=reduction
    )
    torch.testing.assert_close(targets["vision"]["frame_speeds"], speeds.flatten(0, 1))
    torch.testing.assert_close(
        targets["action"]["future_poses"], future_poses.flatten(0, 1) if reduction == "none" else future_poses[:, -1]
    )
    expected_times = times.repeat_interleave(3, dim=0) if reduction == "none" and not shared_times else times
    torch.testing.assert_close(targets["action"]["target_times_s"], expected_times)
