import functools
import math

import numpy as np
import pytest
import torch

from visnavkit.models.action import (
    ActionNormalizer,
    ActionSpace,
    AnchorDecoder,
    AnchorSet,
    DDIMScheduler,
    DiTDenoiser,
    FlowMatchingScheduler,
    GenerativeDecoder,
    MHPDecoder,
    MLPDenoiser,
    RegressionDecoder,
    UNet1DDenoiser,
    arc_anchors,
    parse_plan_output,
)

D, T, P = 8, 4, 3


def space(kind="waypoint", offset=False):
    return ActionSpace(kind=kind, pose_size=P, plan_len_seconds=3.0, plan_len_points=T, offset_t_anchors=offset)


DENOISERS = {
    "mlp": functools.partial(MLPDenoiser, hidden=16, time_dim=8),
    "dit": functools.partial(DiTDenoiser, hidden=16, depth=1, num_heads=2, time_dim=8),
    "unet": functools.partial(UNet1DDenoiser, down_dims=(8, 16), n_groups=4, time_dim=8),
}
SCHEDULERS = {"ddim": lambda: DDIMScheduler(train_timesteps=5), "flow": FlowMatchingScheduler}


def decoders(action_space):
    yield "regression", RegressionDecoder(action_space, feat_size=D, hidden=16)
    yield "mhp", MHPDecoder(action_space, feat_size=D, num_modes=3, hidden=16)
    yield "anchor", AnchorDecoder(action_space, feat_size=D, anchors=AnchorSet(num_anchors=6), hidden=16)
    for d_name, denoiser in DENOISERS.items():
        for s_name, scheduler in SCHEDULERS.items():
            yield (
                f"{s_name}_{d_name}",
                GenerativeDecoder(action_space, denoiser, scheduler(), feat_size=D, num_modes=2, sample_steps=2),
            )
    yield (
        "anchor_flow_dit",
        GenerativeDecoder(
            action_space,
            DENOISERS["dit"],
            FlowMatchingScheduler(),
            feat_size=D,
            sample_steps=2,
            anchors=AnchorSet(num_anchors=6),
        ),
    )


# ---- action spaces and normalization -------------------------------------------------------
@pytest.mark.parametrize("offset", [False, True])
def test_velocity_space_round_trips_waypoints(offset):
    torch.manual_seed(0)
    vel = space("velocity", offset)
    poses = torch.rand(5, T, P)
    if not offset:
        poses[:, 0, :2] = 0  # the t=0 anchor is the ego origin in real targets
    actions = vel.targets_from_poses(poses)
    assert actions.shape == (5, T, 2) and vel.action_dim == 2 and vel.pose_size == 3
    back = vel.to_poses(actions)
    torch.testing.assert_close(back[..., :2], poses[..., :2], atol=1e-5, rtol=1e-4)
    if not offset:  # t=0 anchor: no motion is representable on the degenerate first segment
        assert torch.all(actions[:, 0] == 0)
    assert space().targets_from_poses(poses) is poses


def test_velocity_space_carries_the_last_heading_through_stationary_segments():
    """A stopped segment keeps the previous heading, so its yaw rate is 0 and not a jump back."""
    vel = space("velocity", offset=True)
    poses = torch.zeros(1, T, P)
    poses[0, 1:, 1] = 1.0  # one step along +y, then stationary
    actions = vel.targets_from_poses(poses)
    assert torch.all(actions[0, 2:] == 0)
    assert actions[0, 1, 1] > 0  # the turn onto +y happens once


def test_generative_decoder_warns_when_clipping_unnormalized_actions(monkeypatch):
    from visnavkit.models.action import generative

    warnings = []
    monkeypatch.setattr(generative.logger, "warning", warnings.append)
    GenerativeDecoder(space(), DENOISERS["mlp"], DDIMScheduler(train_timesteps=5), feat_size=D)
    assert len(warnings) == 1 and "clip_sample" in warnings[0]

    warnings.clear()
    GenerativeDecoder(space(), DENOISERS["mlp"], DDIMScheduler(train_timesteps=5, clip_sample=None), feat_size=D)
    GenerativeDecoder(space(), DENOISERS["mlp"], FlowMatchingScheduler(), feat_size=D)
    GenerativeDecoder(
        space(),
        DENOISERS["mlp"],
        DDIMScheduler(train_timesteps=5),
        normalizer=ActionNormalizer(mode="minmax"),
        feat_size=D,
    )
    assert warnings == []


def test_velocity_space_matches_unicycle_kinematics():
    vel = space("velocity", offset=True)
    dt = vel.dt
    actions = torch.tensor([[1.0, 0.0]] * T)[None]  # 1 m/s straight
    poses = vel.to_poses(actions)[0]
    torch.testing.assert_close(poses[:, 0], torch.cumsum(dt, 0))
    torch.testing.assert_close(poses[:, 1], torch.zeros(T))
    turning = torch.tensor([[1.0, math.pi / 2]] * T)[None]
    heading = torch.cumsum(dt * math.pi / 2, 0)
    torch.testing.assert_close(vel.to_poses(turning)[0, :, 1], torch.cumsum(dt * torch.sin(heading), 0))


@pytest.mark.parametrize("mode", ["meanstd", "minmax"])
def test_normalizer_fit_save_load_round_trip(tmp_path, mode):
    torch.manual_seed(1)
    actions = torch.randn(64, T, 2) * 3 + 1
    normalizer = ActionNormalizer(mode=mode).fit(actions)
    normalized = normalizer.normalize(actions)
    if mode == "meanstd":
        torch.testing.assert_close(normalized.mean(0), torch.zeros(T, 2), atol=1e-5, rtol=0)
    else:
        assert normalized.amin() >= -1 - 1e-6 and normalized.amax() <= 1 + 1e-6
    torch.testing.assert_close(normalizer.unnormalize(normalized), actions)
    normalizer.save_stats(tmp_path / "stats.npz")
    loaded = ActionNormalizer(mode=mode, stats_path=tmp_path / "stats.npz")
    torch.testing.assert_close(loaded.normalize(actions), normalized)
    assert ActionNormalizer().identity
    with pytest.raises(ValueError, match="mode"):
        ActionNormalizer(mode="zscore")


def test_arc_anchors_and_nearest():
    t = space().t_anchors.numpy()
    anchors = arc_anchors(6, t, pose_size=3, speeds=(1.0, 2.0))
    assert anchors.shape == (6, T, 3)
    straight = anchors[np.abs(anchors[:, -1, 1]).argmin()]
    np.testing.assert_allclose(straight[:, 0], straight[:, 2] * t, atol=1e-6)
    anchor_set = AnchorSet(num_anchors=6, speeds=(1.0, 2.0)).build(t.tolist(), 3)
    label = anchor_set.nearest(torch.from_numpy(anchors[[4, 1]]))
    assert label.tolist() == [4, 1]


# ---- decoders -----------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["waypoint", "velocity"])
@pytest.mark.parametrize("name, decoder", list(decoders(space())), ids=lambda x: x if isinstance(x, str) else "")
def test_forward_loss_backward_and_parse(kind, name, decoder):
    torch.manual_seed(11)
    if kind == "velocity":
        decoder = dict(decoders(space("velocity")))[name]
    context = torch.randn(4, 3, D, requires_grad=True)
    goal = torch.randn(4, 1, D)
    decoder.train()
    out = decoder(context, goal)
    assert out.plans.shape == (4, decoder.flat_size)
    losses, debug = decoder.get_losses(out, {"future_poses": torch.rand(4, T, P)})
    assert set(losses) == {"total", "reg", "cls"}
    assert all(torch.isfinite(v).all() for v in losses.values())
    assert debug["imitation_loss_per_sample"].shape == (4,)
    losses["total"].backward()
    assert context.grad is not None and torch.isfinite(context.grad).all()
    if "dit" not in name:  # adaLN-Zero gates are zero at init, so DiT context gradients start at zero
        assert context.grad.abs().sum() > 0
    decoder.eval()
    with torch.no_grad():
        out = decoder(context, goal)
    parsed = decoder.parse_output(out.plans)
    assert parsed["plans"].shape == (4, decoder.num_modes, T, P)
    assert torch.isfinite(parsed["best_plan"]).all()
    if name.startswith("anchor"):
        assert decoder.num_modes == 6 and out.logits.shape == (4, 6)
    if name in ("regression",) or name.startswith(("ddim", "flow")):
        torch.testing.assert_close(parsed["confs"], torch.full((4, decoder.num_modes), 1 / decoder.num_modes))


def test_pooling_paths_and_goal_type_embedding():
    decoder = RegressionDecoder(space(), feat_size=D, hidden=16, pooling="mean").eval()
    single = torch.randn(2, 1, D)
    tokens, cond = decoder.condition(single)
    assert tokens is single and torch.equal(cond, single[:, 0])
    tokens, cond = decoder.condition(single, torch.randn(2, 2, D))
    assert tokens.shape == (2, 3, D) and torch.equal(cond, tokens.mean(1))
    attention = RegressionDecoder(space(), feat_size=D, hidden=16).eval()
    assert attention.condition(torch.randn(2, 3, D))[1].shape == (2, D)
    with pytest.raises(ValueError, match="context must be"):
        attention(torch.randn(2, D))


def test_generative_noise_is_explicit_and_deterministic():
    decoder = GenerativeDecoder(
        space(), DENOISERS["mlp"], FlowMatchingScheduler(), feat_size=D, num_modes=2, sample_steps=3
    ).eval()
    context = torch.randn(2, 1, D)
    noise = decoder.example_noise(2)
    assert noise.shape == (2, 2, T, P) and decoder.uses_noise
    first = decoder(context, noise=noise).plans
    torch.testing.assert_close(first, decoder(context, noise=noise).plans, rtol=0, atol=0)
    assert not torch.allclose(first, decoder(context, noise=-noise).plans)
    with pytest.raises(ValueError, match="noise must be"):
        decoder(context, noise=torch.randn(2, 3, T, P))
    decoder.train()
    training = decoder(context)
    assert torch.count_nonzero(training.plans) == 0 and training.cond is not None


@pytest.mark.parametrize("scheduler", [DDIMScheduler(train_timesteps=10), FlowMatchingScheduler()])
def test_schedulers_walk_from_noise_to_clean(scheduler):
    x0 = torch.randn(3, T, 2)
    noise = torch.randn_like(x0)
    times = scheduler.step_times(4)
    assert times[0] == 1.0 and times[-1] == 0.0 and len(times) == 5
    torch.testing.assert_close(scheduler.add_noise(x0, noise, torch.zeros(3)), x0)
    at_one = scheduler.add_noise(x0, noise, torch.ones(3))
    assert torch.allclose(at_one, noise, atol=0.3)
    # An oracle prediction recovers x0 exactly in one step.
    t = torch.full((3,), 0.6)
    x_t = scheduler.add_noise(x0, noise, t)
    torch.testing.assert_close(
        scheduler.step(scheduler.target(x0, noise, t), x_t, t, torch.zeros(3)), x0, atol=1e-5, rtol=1e-4
    )
    assert scheduler.sample_t(5).shape == (5,)


@pytest.mark.parametrize("sampling", ["uniform", "logit_normal", "beta"])
def test_flow_time_sampling_stays_in_the_unit_interval(sampling):
    torch.manual_seed(0)
    t = FlowMatchingScheduler(time_sampling=sampling).sample_t(4096)
    assert t.shape == (4096,) and t.min() >= 0.0 and t.max() <= 1.0
    # Beta(1.5, 1) weights the noisy end, where the trajectory is still undecided (openpi pi0).
    assert (t.mean() > 0.55) == (sampling == "beta")


def test_flow_shift_moves_sampling_steps_toward_high_noise():
    plain = FlowMatchingScheduler().step_times(4)
    shifted = FlowMatchingScheduler(shift=3.0).step_times(4)
    assert shifted[0] == plain[0] == 1.0 and shifted[-1] == plain[-1] == 0.0
    assert all(later < earlier for earlier, later in zip(shifted, shifted[1:]))
    assert all(s >= p for s, p in zip(shifted, plain)) and shifted[2] > plain[2]


def test_schedulers_reject_invalid_settings():
    with pytest.raises(ValueError):
        FlowMatchingScheduler(time_sampling="cosine")
    with pytest.raises(ValueError):
        FlowMatchingScheduler(shift=0.0)
    with pytest.raises(ValueError):
        DDIMScheduler(train_timesteps=1)


def test_mhp_mode_selection_variants_and_layout():
    torch.manual_seed(2)
    gt = torch.rand(4, T, P)
    for selection in ("ade", "angle", "angle-with-tie-break", "cosine-similarity"):
        decoder = MHPDecoder(space(), feat_size=D, num_modes=3, hidden=16, mode_selection=selection)
        preds = decoder.select_mode(torch.rand(4, 3, T, P), gt)
        assert preds.shape == (4,) and preds.max() < 3
    with pytest.raises(ValueError, match="mode_selection"):
        MHPDecoder(space(), feat_size=D, mode_selection="random")
    means = torch.arange(24, dtype=torch.float32).reshape(2, 2, 3, 2)
    logits = torch.tensor([[0.0, 3.0], [2.0, 0.0]])
    output = torch.cat([means.flatten(2), -means.flatten(2), logits.unsqueeze(-1)], dim=-1).flatten(1)
    parsed = parse_plan_output(output, num_modes=2, num_pts=3, pose_size=2)
    torch.testing.assert_close(parsed["plans"], means)
    torch.testing.assert_close(parsed["best_plan"], torch.stack([means[0, 1], means[1, 0]]))


def test_pack_unnormalizes_and_carries_laplace_scales():
    normalizer = ActionNormalizer(mode="meanstd").fit(torch.randn(16, T, P) * 2 + 1)
    decoder = MHPDecoder(space(), normalizer, feat_size=D, num_modes=1, hidden=16)
    mu = torch.zeros(1, 1, T, P)
    log_b = torch.zeros(1, 1, T, P)
    parsed = decoder.parse_output(decoder.pack(mu, log_b, torch.zeros(1, 1)))
    torch.testing.assert_close(parsed["plans"][0, 0], normalizer.offset.expand(T, P))
    torch.testing.assert_close(parsed["pred_scales"][0, 0], torch.log(normalizer.scale).expand(T, P))
