import pytest
import torch

from visnavkit.models.goal import (
    ImageGoalEncoder,
    InstructionGoalEncoder,
    NoGoalEncoder,
    PointGoalEncoder,
    RouteImageGoalEncoder,
)

D = 8


def encoders():
    return [
        PointGoalEncoder(D, hidden=16),
        ImageGoalEncoder(D, backbone_name="resnet18", pretrained=False),
        ImageGoalEncoder(D, backbone_name="resnet18", pretrained=False, stack_observation=True),
        RouteImageGoalEncoder(D, channels=(8, 16)),
        InstructionGoalEncoder(D, embed_dim=16, hidden=16, num_tokens=2),
    ]


def test_goal_free_encoder_emits_zero_tokens():
    encoder = NoGoalEncoder(D)
    assert encoder.num_tokens == 0 and encoder.goal_type == "none"
    assert encoder(None, batch_size=3).shape == (3, 0, D)
    assert encoder.example_input(3) is None
    assert not list(encoder.parameters())


@pytest.mark.parametrize(
    "encoder",
    encoders(),
    ids=lambda e: f"{type(e).__name__}-{e.stack_observation}" if hasattr(e, "stack_observation") else type(e).__name__,
)
def test_goal_tokens_null_token_and_dropout(encoder):
    encoder.eval()
    goal = encoder.example_input(3, image_hw=(32, 32))
    observation = torch.rand(3, 6, 32, 32)
    tokens = encoder(goal, observation=observation)
    assert tokens.shape == (3, encoder.num_tokens, D)
    assert torch.isfinite(tokens).all()
    null = encoder(None, batch_size=3)
    assert null.shape == tokens.shape
    torch.testing.assert_close(null[0], encoder.null_token[0])
    encoder.p_drop = 1.0
    encoder.train()
    dropped = encoder(goal, observation=observation)
    torch.testing.assert_close(dropped, null.expand_as(dropped))
    encoder.p_drop = 0.0
    kept = encoder(goal, observation=observation)
    assert not torch.allclose(kept, null.expand_as(kept))


def test_point_goal_distance_is_clipped_and_uint8_images_accepted():
    point = PointGoalEncoder(D, hidden=16, max_distance=10.0).eval()
    near, far = torch.tensor([[10.0, 1.0, 0.0]]), torch.tensor([[50.0, 1.0, 0.0]])
    torch.testing.assert_close(point(near), point(far))
    with pytest.raises(ValueError, match="\\(N, 3\\)"):
        point(torch.zeros(2, 2))
    image = ImageGoalEncoder(D, backbone_name="resnet18", pretrained=False).eval()
    goal = (torch.rand(2, 3, 32, 32) * 255).to(torch.uint8)
    torch.testing.assert_close(image(goal), image(goal.float() / 255))
    with pytest.raises(ValueError, match="stack_observation"):
        ImageGoalEncoder(D, backbone_name="resnet18", pretrained=False, stack_observation=True)(
            torch.rand(2, 3, 32, 32)
        )
    with pytest.raises(ValueError, match="Instruction"):
        InstructionGoalEncoder(D, embed_dim=16, hidden=16)(torch.zeros(2, 8))
    assert (
        ImageGoalEncoder(D, backbone_name="resnet18", pretrained=False, freeze_backbone=True).train().backbone.training
        is False
    )
