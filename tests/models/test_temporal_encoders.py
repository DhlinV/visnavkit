import pytest
import torch

from visnavkit.models.temporal_encoders import (
    BidirectionalTemporalEncoder,
    CausalTemporalEncoder,
    IdentityTemporalEncoder,
    generate_causal_mask,
)
from visnavkit.models.temporal_encoders.temporal_encoder import TemporalEncoder


@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("training", [False, True])
def test_attention_direction(num_layers, training):
    torch.manual_seed(7)
    kwargs = dict(embed_dim=8, num_heads=2, ff_dim=16, seq_len=4, dropout=0, mask_p=0, reduction="none")
    causal = CausalTemporalEncoder(num_layers=num_layers, **kwargs).train(training)
    bidirectional = BidirectionalTemporalEncoder(num_layers=num_layers, **kwargs).train(training)
    bidirectional.load_state_dict(causal.state_dict(), strict=True)
    frames = torch.randn(2, 4, 8)
    changed_future = frames.clone()
    changed_future[:, 2:] = torch.randn(2, 2, 8) * 10

    torch.testing.assert_close(causal(frames)[:, :2], causal(changed_future)[:, :2], rtol=0, atol=0)
    assert not torch.allclose(bidirectional(frames)[:, :2], bidirectional(changed_future)[:, :2])


@pytest.mark.parametrize("encoder_class", [CausalTemporalEncoder, BidirectionalTemporalEncoder])
def test_eval_is_deterministic(encoder_class):
    encoder = encoder_class(embed_dim=8, num_heads=2, ff_dim=16, dropout=0.5, mask_p=0.9).eval()
    frames = torch.randn(2, 4, 8)
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


@pytest.mark.parametrize("reduction", ["last", "avg", "sum", "none"])
def test_identity_matches_legacy_zero_layer_reductions(reduction):
    frames = torch.arange(48, dtype=torch.float32).reshape(2, 3, 8)
    # Identity does not need positional embeddings or impose a maximum window length.
    encoder = IdentityTemporalEncoder(embed_dim=6, route_embed_dim=2, seq_len=1, reduction=reduction)
    legacy = TemporalEncoder(embed_dim=6, route_embed_dim=2, seq_len=1, reduction=reduction, num_layers=0)
    expected = {
        "last": frames[:, -1],
        "avg": frames.mean(dim=1),
        "sum": frames.sum(dim=1),
        "none": frames,
    }[reduction]
    torch.testing.assert_close(encoder(frames), expected)
    torch.testing.assert_close(legacy(frames), expected)
    assert not encoder.state_dict()


@pytest.mark.parametrize(
    "encoder_class", [CausalTemporalEncoder, BidirectionalTemporalEncoder, IdentityTemporalEncoder]
)
@pytest.mark.parametrize(
    "shape, message", [((2, 8), "Expected"), ((2, 0, 8), "at least one"), ((2, 3, 7), "8 features")]
)
def test_invalid_inputs(encoder_class, shape, message):
    encoder = encoder_class(embed_dim=8, num_heads=2)
    with pytest.raises(ValueError, match=message):
        encoder(torch.randn(*shape))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"seq_len": 0}, "seq_len"),
        ({"num_layers": -1}, "num_layers"),
        ({"reduction": "median"}, "reduction"),
        ({"mask_p": 1.1}, "mask_p"),
        ({"mask_p": -0.1}, "mask_p"),
        ({"embed_dim": 0}, "embed_dim"),
        ({"route_embed_dim": -1}, "route_embed_dim"),
        ({"embed_dim": 7, "num_heads": 2}, "divisible"),
        ({"num_heads": 0}, "num_heads"),
    ],
)
def test_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        CausalTemporalEncoder(**kwargs)


def test_transformer_window_limit_and_identity_configuration():
    encoder = CausalTemporalEncoder(embed_dim=8, num_heads=2, seq_len=2)
    with pytest.raises(ValueError, match="capacity"):
        encoder(torch.randn(2, 3, 8))
    with pytest.raises(ValueError, match="num_layers=0"):
        IdentityTemporalEncoder(num_layers=1)


def test_legacy_import_and_checkpoint_hierarchy():
    assert TemporalEncoder is CausalTemporalEncoder
    single = CausalTemporalEncoder(embed_dim=8, num_heads=2, ff_dim=16)
    stack = CausalTemporalEncoder(embed_dim=8, num_heads=2, ff_dim=16, num_layers=2)
    layer_keys = {
        "self_attn.in_proj_weight",
        "self_attn.in_proj_bias",
        "self_attn.out_proj.weight",
        "self_attn.out_proj.bias",
        "linear1.weight",
        "linear1.bias",
        "linear2.weight",
        "linear2.bias",
        "norm1.weight",
        "norm1.bias",
        "norm2.weight",
        "norm2.bias",
    }
    assert set(single.state_dict()) == {"pos_embedding.weight"} | {f"tformer.{key}" for key in layer_keys}
    assert set(stack.state_dict()) == {"pos_embedding.weight"} | {
        f"tformer.layers.{layer}.{key}" for layer in range(2) for key in layer_keys
    }
