import pytest
import torch

from visnavkit.models.vision import TimmCNNEncoder, TimmViTEncoder

SMALL = dict(pretrained=False, feat_size=8, img_embed_size=16, neck_cfg={"n_res_blocks": 0, "dropout": 0})


def _cnn(**kwargs):
    return TimmCNNEncoder(backbone_name="resnet18", out_indices=(2, 3, 4), **SMALL, **kwargs)


def _vit(**kwargs):
    return TimmViTEncoder(backbone_name="vit_small_patch14_dinov2", **SMALL, **kwargs)


@pytest.mark.parametrize("factory", [_cnn, _vit])
@pytest.mark.parametrize("token_mode, expected", [("global", 1), ("patch", 6), ("fused", 7)])
def test_token_modes(factory, token_mode, expected):
    model = factory(token_mode=token_mode, patch_grid=(2, 3)).eval()
    assert model.num_tokens == expected
    with torch.no_grad():
        out = model(torch.rand(2, 3, 32, 48))
    assert out.tokens.shape == (2, expected, 8)
    assert out.speed is None and not model.has_speed_head
    assert torch.isfinite(out.tokens).all()


@pytest.mark.parametrize(
    "backbone, out_indices, reductions",
    [
        ("resnet18", (2, 3, 4), [8, 16, 32]),
        ("efficientnet_b0", (2, 3, 4), [8, 16, 32]),
        ("fastvit_t8", (1, 2, 3), [8, 16, 32]),
        ("convnext_tiny", (-3, -2, -1), [8, 16, 32]),
    ],
)
def test_feature_pyramid_backbones(backbone, out_indices, reductions):
    model = TimmCNNEncoder(
        backbone_name=backbone,
        out_indices=out_indices,
        act_layer="gelu_tanh" if backbone.startswith("fastvit") else None,
        **SMALL,
    ).eval()
    with torch.no_grad():
        out = model(torch.rand(2, 3, 64, 96))
    assert out.tokens.shape == (2, 1, 8)
    assert model.backbone.feature_info.reduction() == reductions


@pytest.mark.parametrize("factory", [_cnn, _vit])
def test_single_frames_normalized_without_mutating_input(factory, monkeypatch):
    model = factory().train()
    seen, encode = [], model._encode

    def spy(x):
        seen.append(x.detach().clone())
        return encode(x)

    monkeypatch.setattr(model, "_encode", spy)
    x = torch.rand(2, 3, 32, 42, requires_grad=True)
    original = x.detach().clone()
    result = model(x)
    torch.testing.assert_close(x, original)
    torch.testing.assert_close(seen[0], model.normalize_frame_transform(original))
    assert result.tokens.shape == (2, 1, 8)


def test_optional_speed_head_is_the_only_auxiliary_loss():
    """The speed head is a per-recipe auxiliary loss, so an encoder without it has no losses."""
    plain, supervised = _cnn().eval(), _cnn(speed_head=True, loss_speed_weight=2.0).eval()
    assert not plain.has_speed_head and supervised.has_speed_head
    with torch.no_grad():
        assert plain(torch.rand(2, 3, 32, 48)).speed is None
    assert plain.get_losses(plain(torch.rand(2, 3, 32, 48)), {"frame_speeds": torch.ones(2, 1)}) == {}

    out = supervised(torch.rand(2, 3, 32, 48))
    assert out.speed.shape == (2, 1)
    losses = supervised.get_losses(out, {"frame_speeds": torch.ones(2, 1)})
    assert set(losses) == {"speed", "total"} and torch.isfinite(losses["total"])
    torch.testing.assert_close(losses["total"], 2.0 * losses["speed"])
    losses["total"].backward()
    assert supervised.speed_head.head[0].weight.grad is not None


@pytest.mark.parametrize("backbone", ["vit_small_patch14_dinov2", "vit_small_patch16_dinov3"])
def test_frozen_vit_training_and_patch_padding(backbone):
    model = TimmViTEncoder(backbone_name=backbone, token_mode="fused", patch_grid=(2, 2), speed_head=True, **SMALL)
    assert not model.backbone.training
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    model.train()
    assert model.training and not model.backbone.training
    result = model(torch.rand(2, 3, 29, 43))
    assert result.tokens.shape == (2, 5, 8)
    loss = model.get_losses(result, {"frame_speeds": torch.ones(2, 1)})["total"]
    loss.backward()
    assert model.final_linear.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.backbone.parameters())


def test_vit_backbone_can_be_trained():
    model = _vit(freeze_backbone=False)
    assert model.backbone.training
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())
    model.eval().train()
    assert model.backbone.training


def test_invalid_inputs_and_options():
    model = _cnn().eval()
    with pytest.raises(ValueError, match=r"\(N, 3, H, W\)"):
        model(torch.rand(2, 6, 32, 48))
    with pytest.raises(TypeError, match="floating-point"):
        model(torch.zeros(2, 3, 32, 48, dtype=torch.uint8))
    with pytest.raises(ValueError, match="token_mode"):
        _cnn(token_mode="cls")
    with pytest.raises(ValueError, match="spatial heads"):
        _vit(heads={"depth": {}})


@pytest.mark.parametrize("image_size", [(32, 32), (29, 43), (28, 56)])
def test_dinov2_export_preparation_preserves_outputs_and_original_weights(image_size):
    model = _vit(speed_head=True).eval()
    original_positions = model.backbone.pos_embed.detach().clone()
    prepared = model.prepare_for_export(image_size)
    assert prepared is not model
    assert model.backbone.dynamic_img_size
    torch.testing.assert_close(model.backbone.pos_embed, original_positions, rtol=0, atol=0)
    frames = torch.rand(2, 3, *image_size)
    with torch.no_grad():
        expected, actual = model(frames), prepared(frames)
    torch.testing.assert_close(actual.tokens, expected.tokens, rtol=0, atol=0)
    torch.testing.assert_close(actual.speed, expected.speed, rtol=0, atol=0)


def test_dinov2_padded_nonsquare_onnx_export(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    model = _vit(token_mode="fused", patch_grid=(2, 2)).eval()
    frames = torch.rand(2, 3, 32, 48)
    prepared = model.prepare_for_export((32, 48))
    path = tmp_path / "dinov2.onnx"

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = prepared

        def forward(self, x):
            return self.encoder(x).tokens

    with torch.no_grad():
        expected = model(frames)
        torch.onnx.export(
            Wrapper(), (frames,), path, input_names=["vision"], output_names=["tokens"], opset_version=17, dynamo=False
        )
    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (tokens,) = session.run(["tokens"], {"vision": frames.numpy()})
    torch.testing.assert_close(torch.from_numpy(tokens), expected.tokens, rtol=2e-3, atol=2e-4)
