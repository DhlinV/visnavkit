import pytest
import torch

from visnavkit.models.temporal import (
    BidirectionalTemporalEncoder,
    CausalTemporalEncoder,
    IdentityTemporalEncoder,
    generate_causal_mask,
)

KW = dict(embed_dim=8, num_heads=2, ff_dim=16, seq_len=4, dropout=0, mask_p=0, reduction="none")


@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("tokens", [1, 3])
def test_attention_direction(num_layers, tokens):
    torch.manual_seed(7)
    causal = CausalTemporalEncoder(num_layers=num_layers, **KW).eval()
    bidirectional = BidirectionalTemporalEncoder(num_layers=num_layers, allow_offline=True, **KW).eval()
    bidirectional.load_state_dict(causal.state_dict(), strict=True)
    frames = torch.randn(2, 4, tokens, 8)
    changed_future = frames.clone()
    changed_future[:, 2:] = torch.randn(2, 2, tokens, 8) * 10
    torch.testing.assert_close(causal(frames)[:, :2], causal(changed_future)[:, :2], rtol=0, atol=0)
    assert not torch.allclose(bidirectional(frames)[:, :2], bidirectional(changed_future)[:, :2])


def test_tokens_within_a_frame_attend_to_each_other():
    torch.manual_seed(3)
    encoder = CausalTemporalEncoder(**KW).eval()
    frames = torch.randn(1, 2, 2, 8)
    changed = frames.clone()
    changed[:, 1, 1, 0] += 5  # one feature: a constant shift would be erased by LayerNorm
    assert not torch.allclose(encoder(frames)[:, 1, 0], encoder(changed)[:, 1, 0])


@pytest.mark.parametrize(
    "reduction, shape", [("last", (2, 1, 3, 8)), ("avg", (2, 1, 3, 8)), ("sum", (2, 1, 3, 8)), ("none", (2, 4, 3, 8))]
)
def test_reductions_keep_the_token_axis(reduction, shape):
    frames = torch.arange(2 * 4 * 3 * 8, dtype=torch.float32).reshape(2, 4, 3, 8)
    encoder = IdentityTemporalEncoder(embed_dim=8, reduction=reduction)
    expected = {
        "last": frames[:, -1:],
        "avg": frames.mean(1, keepdim=True),
        "sum": frames.sum(1, keepdim=True),
        "none": frames,
    }[reduction]
    out = encoder(frames)
    assert out.shape == shape
    torch.testing.assert_close(out, expected)
    assert not encoder.state_dict()


@pytest.mark.parametrize("encoder_class", [CausalTemporalEncoder, BidirectionalTemporalEncoder])
def test_eval_is_deterministic(encoder_class):
    encoder = encoder_class(embed_dim=8, num_heads=2, ff_dim=16, dropout=0.5, mask_p=0.9).eval()
    frames = torch.randn(2, 4, 1, 8)
    torch.testing.assert_close(encoder(frames), encoder(frames), rtol=0, atol=0)


@pytest.mark.parametrize("mask_p", [0, 0.5, 1])
def test_causal_mask_never_drops_self_or_reveals_future(mask_p):
    mask = generate_causal_mask(5, mask_p=mask_p, device="cpu")
    assert torch.all(mask.diag() == 0)
    assert torch.all(torch.isneginf(mask[torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)]))
    if mask_p == 0:
        assert torch.all(mask[torch.tril(torch.ones(5, 5, dtype=torch.bool))] == 0)
    if mask_p == 1:
        assert torch.all(torch.isneginf(mask[~torch.eye(5, dtype=torch.bool)]))


@pytest.mark.parametrize(
    "shape, message", [((2, 8), "Expected"), ((2, 0, 1, 8), "at least one"), ((2, 3, 1, 7), "8 features")]
)
def test_invalid_inputs(shape, message):
    with pytest.raises(ValueError, match=message):
        CausalTemporalEncoder(embed_dim=8, num_heads=2)(torch.randn(*shape))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"seq_len": 0}, "seq_len"),
        ({"num_layers": -1}, "num_layers"),
        ({"reduction": "median"}, "reduction"),
        ({"mask_p": 1.1}, "mask_p"),
        ({"embed_dim": 7, "num_heads": 2}, "divisible"),
    ],
)
def test_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        CausalTemporalEncoder(**kwargs)


def test_window_limit_identity_and_offline_guard():
    with pytest.raises(ValueError, match="capacity"):
        CausalTemporalEncoder(embed_dim=8, num_heads=2, seq_len=2)(torch.randn(2, 3, 1, 8))
    with pytest.raises(ValueError, match="num_layers=0"):
        IdentityTemporalEncoder(num_layers=1)
    with pytest.raises(ValueError, match="allow_offline"):
        BidirectionalTemporalEncoder(embed_dim=8, num_heads=2, reduction="none")
    assert BidirectionalTemporalEncoder(embed_dim=8, num_heads=2).reduction == "last"
