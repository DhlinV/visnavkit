"""Policy composition across training, feature-buffer deployment, goals, ego status, freezing."""

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
    ego=False,
    camera=False,
    speed_head=False,
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
                "speed_head": speed_head,
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
            "goal_encoder": (
                [{"feat_size": 8, **GOAL[name]} for name in goal]
                if isinstance(goal, list)
                else {"feat_size": 8, **GOAL[goal]}
            ),
            "ego_encoder": (
                {"_target_": "visnavkit.models.ego.state.EgoStateEncoder", "feat_size": 8, "in_dim": 2, "hidden": 16}
                if ego
                else {"_target_": "visnavkit.models.ego.none.NoEgoEncoder", "feat_size": 8}
            ),
            "camera_encoder": (
                {"_target_": "visnavkit.models.camera.pinhole.PinholeCameraEncoder", "feat_size": 8, "hidden": 16}
                if camera
                else {"_target_": "visnavkit.models.camera.none.NoCameraEncoder", "feat_size": 8}
            ),
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


@pytest.mark.parametrize(
    "temporal, goal, decoder, reduction, token_mode, ego, camera",
    [
        ("causal", "none", "mhp", "none", "global", False, False),
        ("causal", "point", "regression", "none", "fused", True, False),
        ("bidirectional", "image", "flow_dit", "last", "patch", False, True),
        ("identity", "instruction", "anchor", "last", "global", True, True),
        ("causal", "point", "diffusion_unet", "none", "global", False, False),
        ("causal", ["point", "image", "none"], "mhp", "last", "global", True, True),
    ],
)
def test_policy_trains_end_to_end(temporal, goal, decoder, reduction, token_mode, ego, camera):
    torch.manual_seed(31)
    model = instantiate(
        policy_config(
            temporal,
            goal,
            decoder,
            reduction=reduction,
            token_mode=token_mode,
            ego=ego,
            camera=camera,
            speed_head=True,
        )
    ).train()
    vision, goal_batch, ego_batch, intrinsics, extrinsics = model.example_batch(2, 3, (32, 32))
    vision.requires_grad_(True)
    out = model(vision, goal=goal_batch, ego=ego_batch, intrinsics=intrinsics, extrinsics=extrinsics)
    decisions = 6 if reduction == "none" else 2
    goal_names = goal if isinstance(goal, list) else [goal]
    expected_goal_tokens = sum(name != "none" for name in goal_names)
    targets = {"vision": {"frame_speeds": torch.rand(6, 1)}, "action": {"future_poses": torch.rand(decisions, 3, 3)}}
    losses, debug = model.get_losses(out, targets)
    assert out.vision.speed.shape == (6, 1)
    assert out.vision.tokens.shape == (6, model.vision_tokens, 8)
    assert model.num_tokens == model.vision_tokens + (1 if ego else 0) + (1 if camera else 0)
    assert (out.ego_tokens is None) == (not ego)
    assert (out.camera_tokens is None) == (not camera)
    assert out.goal_tokens is None or out.goal_tokens.shape == (decisions, expected_goal_tokens, 8)
    assert (out.goal_tokens is None) == (expected_goal_tokens == 0)
    assert out.plan.plans.shape == (decisions, model.action_decoder.flat_size)
    assert debug["action_loss_debug"]["imitation_loss_per_sample"].shape == (decisions,)
    assert all(torch.isfinite(v).all() for v in losses.values())
    losses["loss"].backward()
    assert vision.grad is not None and torch.isfinite(vision.grad).all() and vision.grad.abs().sum() > 0
    if expected_goal_tokens and decoder != "flow_dit":  # DiT adaLN-Zero gates start at zero, so gradients do too
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.goal_encoders.parameters())


@pytest.mark.parametrize(
    "temporal, goal, decoder, reduction, seq_len, seq_step, ego, camera",
    [
        ("causal", "none", "mhp", "last", 3, 2, False, False),
        ("causal", "point", "regression", "none", 3, 2, True, False),
        ("bidirectional", "image", "flow_dit", "last", 3, 1, False, True),
        ("identity", "instruction", "anchor", "last", 1, 1, True, True),
        ("bidirectional", "none", "regression", "avg", 3, 2, False, False),
        ("causal", ["point", "image"], "mhp", "last", 3, 1, True, True),
    ],
)
def test_feature_buffer_matches_full_observation_window(
    temporal, goal, decoder, reduction, seq_len, seq_step, ego, camera
):
    torch.manual_seed(32)
    model = instantiate(
        policy_config(
            temporal,
            goal,
            decoder,
            reduction=reduction,
            seq_len=seq_len,
            seq_step=seq_step,
            token_mode="fused",
            ego=ego,
            camera=camera,
            speed_head=True,
        )
    ).eval()
    history = (seq_len - 1) * seq_step
    dense, goal_batch, ego_batch, intrinsics, extrinsics = model.example_batch(2, history + 1, (32, 32))
    window = dense[:, ::seq_step]
    goals = goal_batch if isinstance(goal_batch, list) else [goal_batch]
    window_goals = [
        None if g is None else (g[:, ::seq_step] if e.per_frame else g) for e, g in zip(model.goal_encoders, goals)
    ]
    newest = [None if g is None else (g[:, -1] if e.per_frame else g) for e, g in zip(model.goal_encoders, goals)]
    if not isinstance(goal_batch, list):
        window_goals, newest = window_goals[0], newest[0]
    noise = model.action_decoder.example_noise(2) if model.action_decoder.uses_noise else None
    goal_names = goal if isinstance(goal, list) else [goal]
    live = [name for name in goal_names if name != "none"]
    with torch.no_grad():
        encoded = model._per_frame_tokens(
            model.encode_frames(dense).tokens.reshape(2, history + 1, model.vision_tokens, 8),
            model.encode_ego(ego_batch, 2, history + 1),
            model.encode_camera(intrinsics, extrinsics, 2, history + 1, (32, 32)),
            dim=2,
        ).flatten(2)
        strided = {
            "ego": None if not ego else ego_batch[:, ::seq_step],
            "intrinsics": None if not camera else intrinsics[:, ::seq_step],
            "extrinsics": None if not camera else extrinsics[:, ::seq_step],
        }
        full = model(window, goal=window_goals, **strided, noise=noise)
        expected = full.plan.plans if reduction != "none" else full.plan.plans.reshape(2, seq_len, -1)[:, -1]
        plan, token, speed = model.predict(
            dense[:, -1],
            encoded[:, :-1],
            goal=newest,
            ego=None if not ego else ego_batch[:, -1],
            intrinsics=None if not camera else intrinsics[:, -1],
            extrinsics=None if not camera else extrinsics[:, -1],
            noise=noise,
        )
    assert model.export_output_names() == ["plan", "feat_out", "speed"]
    goal_inputs = ["goal"] if len(live) == 1 else [f"goal_{i}" for i in range(len(live))]
    assert model.export_input_names() == ["vision", "feature_buffer", *goal_inputs] + (["ego"] if ego else []) + (
        ["intrinsics", "extrinsics"] if camera else []
    ) + (["noise"] if noise is not None else [])
    torch.testing.assert_close(plan, expected, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(speed, full.vision.speed.reshape(2, seq_len, 1)[:, -1])
    torch.testing.assert_close(token, encoded[:, -1])


def test_goal_shape_validation_and_null_goal():
    model = instantiate(policy_config("causal", "point", "regression")).eval()
    frames = torch.rand(2, 3, 3, 32, 32)
    with pytest.raises(ValueError, match="per-frame goals"):
        model(frames, goal=torch.zeros(2, 4, 3))
    out = model(frames)  # missing goal -> learned null token
    assert out.goal_tokens.shape == (6, 1, 8)
    torch.testing.assert_close(out.goal_tokens, model.null_goal_tokens(6))
    image = instantiate(policy_config("bidirectional", "image", "regression", reduction="last")).eval()
    with pytest.raises(ValueError, match="one goal per window"):
        image(frames, goal=torch.rand(3, 3, 32, 32))
    pair = instantiate(policy_config("causal", ["point", "image"], "regression", reduction="last")).eval()
    with pytest.raises(ValueError, match="2 goal encoders expect 2 goals"):
        pair(frames, goal=torch.zeros(2, 3, 3))


def test_camera_validation_and_null_token():
    model = instantiate(policy_config("causal", "none", "regression", reduction="last", camera=True)).eval()
    frames = torch.rand(2, 3, 3, 32, 32)
    assert model.camera_tokens == 1 and model.num_tokens == model.vision_tokens + 1
    with pytest.raises(ValueError, match=r"Intrinsics must be \(B, F, 3, 3\)"):
        model(frames, intrinsics=torch.eye(3).repeat(2, 3, 1, 1)[..., :2], extrinsics=torch.eye(4).repeat(2, 3, 1, 1))
    out = model(frames)  # missing calibration -> learned null token
    torch.testing.assert_close(out.camera_tokens, model.camera_encoder.null_token.repeat(2, 3, 1, 1))
    # Intrinsics are normalized by the live frame size, so the same K at twice the resolution
    # (and twice the focal length / principal point) yields the same tokens.
    small = model.camera_encoder.example_input(2, 3, (32, 32))
    large = model.camera_encoder.example_input(2, 3, (64, 64))
    torch.testing.assert_close(
        model(frames, intrinsics=small[0], extrinsics=small[1]).camera_tokens,
        model(torch.rand(2, 3, 3, 64, 64), intrinsics=large[0], extrinsics=large[1]).camera_tokens,
    )


def test_ego_status_validation_and_null_token():
    model = instantiate(policy_config("causal", "none", "regression", reduction="last", ego=True)).eval()
    frames = torch.rand(2, 3, 3, 32, 32)
    assert model.ego_tokens == 1 and model.num_tokens == model.vision_tokens + 1
    with pytest.raises(ValueError, match=r"\(B, F, 2\)"):
        model(frames, ego=torch.zeros(2, 3, 5))
    out = model(frames)  # missing ego status -> learned null token
    torch.testing.assert_close(out.ego_tokens, model.ego_encoder.null_token.repeat(2, 3, 1, 1))


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
    out = model(torch.rand(2, 3, 3, 32, 32))
    losses, _ = model.get_losses(out, {"vision": {}, "action": {"future_poses": torch.rand(6, 3, 3)}})
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
    assert build_targets({"future_poses": future_poses}, action_reduction=reduction)["vision"] == {}
    torch.testing.assert_close(
        targets["action"]["future_poses"], future_poses.flatten(0, 1) if reduction == "none" else future_poses[:, -1]
    )
    expected_times = times.repeat_interleave(3, dim=0) if reduction == "none" and not shared_times else times
    torch.testing.assert_close(targets["action"]["target_times_s"], expected_times)
