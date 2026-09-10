import pytest
import torch

from visnavkit.models.action_decoder import ActionDecoder as LegacyActionDecoder
from visnavkit.models.action_decoders import ActionDecoder, DiffusionPlanHead, PlanHead, WaypointHead, parse_plan_output
from visnavkit.models.heads.diffusion_plan_head import DiffusionPlanHead as LegacyDiffusionPlanHead
from visnavkit.models.heads.plan_head import PlanHead as LegacyPlanHead
from visnavkit.models.heads.plan_head import parse_plan_output as legacy_parse_plan_output
from visnavkit.models.heads.waypoint_head import WaypointHead as LegacyWaypointHead
from visnavkit.models.temporal_encoders import CausalTemporalEncoder


def test_legacy_imports_use_canonical_implementations():
    assert LegacyActionDecoder is ActionDecoder
    assert LegacyPlanHead is PlanHead
    assert LegacyWaypointHead is WaypointHead
    assert LegacyDiffusionPlanHead is DiffusionPlanHead
    assert legacy_parse_plan_output is parse_plan_output


def test_shared_output_layout_selects_highest_confidence_mode():
    means = torch.arange(24, dtype=torch.float32).reshape(2, 2, 3, 2)
    scales = -means
    logits = torch.tensor([[0.0, 3.0], [2.0, 0.0]])
    output = torch.cat([means.flatten(2), scales.flatten(2), logits.unsqueeze(-1)], dim=-1).flatten(1)
    parsed = parse_plan_output(output, num_modes=2, num_pts=3, pose_size=2)
    torch.testing.assert_close(parsed["plans"], means)
    torch.testing.assert_close(parsed["pred_scales"], scales)
    torch.testing.assert_close(parsed["confs"], logits.softmax(1))
    torch.testing.assert_close(parsed["best_plan"], torch.stack([means[0, 1], means[1, 0]]))


@pytest.mark.parametrize("head_class", [PlanHead, WaypointHead, DiffusionPlanHead])
@pytest.mark.parametrize("reduction", ["none", "last"])
def test_action_forward_loss_and_backward(head_class, reduction):
    torch.manual_seed(11)
    kwargs = dict(feat_size=8, num_modes=1 if head_class is WaypointHead else 2, num_pts=3, pose_size=2, hidden=16)
    if head_class is DiffusionPlanHead:
        kwargs.update(train_timesteps=5, sample_steps=2, time_embed_dim=8)
    decoder = ActionDecoder(
        CausalTemporalEncoder(embed_dim=8, num_heads=2, ff_dim=16, dropout=0, reduction=reduction),
        head_class(**kwargs),
    )
    frames = torch.randn(2, 3, 8, requires_grad=True)
    batch = 6 if reduction == "none" else 2
    predictions = decoder(frames)
    assert predictions["plan"]["plans"].shape == (batch, kwargs["num_modes"] * 13)
    losses, debug = decoder.get_losses(predictions, {"future_poses": torch.randn(batch, 3, 2)})
    assert set(losses) == {"total", "reg", "cls"}
    assert all(torch.isfinite(value).all() for value in losses.values())
    assert debug["imitation_loss_per_sample"].shape == (batch,)
    losses["total"].backward()
    assert frames.grad is not None and torch.isfinite(frames.grad).all()
    assert frames.grad.abs().sum() > 0
    assert any(key.startswith("temporal_encoder.tformer.") for key in decoder.state_dict())
    assert any(key.startswith("plan_head.") for key in decoder.state_dict())

    decoder.eval()
    with torch.no_grad():
        evaluation = decoder(frames)
    parsed = decoder.plan_head.parse_output(evaluation["plan"]["plans"])
    assert parsed["best_plan"].shape == (batch, 3, 2)
    assert torch.isfinite(parsed["plans"]).all()


def test_diffusion_training_skips_sampling_and_fixed_noise_sampling_repeats():
    head = DiffusionPlanHead(
        feat_size=8,
        hidden=16,
        num_modes=2,
        num_pts=3,
        pose_size=2,
        train_timesteps=5,
        sample_steps=3,
        time_embed_dim=8,
    )
    features = torch.randn(2, 8)
    training = head(features)
    assert training["cond"] is features
    assert torch.count_nonzero(training["plans"]) == 0

    head.eval()
    noise = torch.randn(4, 6)
    first = head._sample(features, noise=noise)
    torch.testing.assert_close(first, head._sample(features, noise=noise), rtol=0, atol=0)
    assert not torch.allclose(first, head._sample(features, noise=-noise))
    parsed = head.parse_output(first)
    torch.testing.assert_close(parsed["confs"], torch.full((2, 2), 0.5))
    assert torch.count_nonzero(parsed["pred_scales"]) == 0
