import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from visnavkit.models.encoders.dino_encoder import DinoEncoder as LegacyDinoEncoder
from visnavkit.models.encoders.vision_encoder import VisionEncoder
from visnavkit.models.spatial_encoders.vision_encoders import (
    DINOv2Encoder,
    DINOv3Encoder,
    DinoEncoder,
    EfficientNetEncoder,
    FastViTEncoder,
    MobileNetEncoder,
    ResNetEncoder,
    TimmVisionEncoder,
)


def _encoder(encoder_cls, **kwargs):
    return encoder_cls(
        pretrained=False,
        img_embed_size=16,
        neck_cfg={"dim": 8, "n_res_blocks": 0, "dropout": 0},
        **kwargs,
    )


def test_legacy_targets_and_checkpoint_names():
    assert VisionEncoder is TimmVisionEncoder
    assert LegacyDinoEncoder is DinoEncoder
    cfg = OmegaConf.create(
        {
            "_target_": "visnavkit.models.encoders.vision_encoder.VisionEncoder",
            "backbone_name": "resnet18",
            "pretrained": False,
            "act_layer": None,
            "out_indices": [2, 3, 4],
            "img_embed_size": 16,
            "neck_cfg": {"dim": 8, "n_res_blocks": 0, "dropout": 0},
        }
    )
    legacy = instantiate(cfg).eval()
    canonical = _encoder(ResNetEncoder).eval()
    canonical.load_state_dict(legacy.state_dict(), strict=True)
    assert {key.split(".")[0] for key in canonical.state_dict()} == {
        "backbone",
        "final_linear",
        "embed_norm",
        "action_neck",
        "feat_norm",
        "pose_neck",
        "pose_head",
    }
    x = torch.rand(2, 6, 32, 48)
    with torch.no_grad():
        expected, actual = legacy(x), canonical(x, export_heads=[])
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name])


@pytest.mark.parametrize("encoder_cls", [FastViTEncoder, ResNetEncoder, EfficientNetEncoder, MobileNetEncoder])
def test_feature_pyramid_variants(encoder_cls):
    model = _encoder(encoder_cls).eval()
    with torch.no_grad():
        result = model(torch.rand(2, 6, 64, 96), export_heads=[])
    assert result["pose"].shape == (2, 1)
    assert result["feat_out"].shape == (2, 8)
    assert result["prev_img_mask"].all()
    assert model.backbone.feature_info.reduction() == [8, 16, 32]
    assert model.get_head_output_names([]) == []
    if encoder_cls is ResNetEncoder:
        assert isinstance(model.backbone.act1, torch.nn.ReLU)
    elif encoder_cls is EfficientNetEncoder:
        assert any(isinstance(module, torch.nn.SiLU) for module in model.backbone.modules())
    elif encoder_cls is MobileNetEncoder:
        assert any(isinstance(module, torch.nn.ReLU6) for module in model.backbone.modules())


@pytest.mark.parametrize("out_indices", [(4,), (1, 2, 3, 4)])
def test_variable_number_of_feature_stages(out_indices):
    model = _encoder(ResNetEncoder, out_indices=out_indices).eval()
    with torch.no_grad():
        result = model(torch.rand(2, 6, 32, 48))
    assert len(model.embed_dims) == len(out_indices)
    assert result["feat_out"].shape == (2, 8)


@pytest.mark.parametrize("p_drop_prev_img", [0, 1])
def test_rgb_pairs_normalized_without_mutating_input(p_drop_prev_img):
    model = _encoder(ResNetEncoder, p_drop_prev_img=p_drop_prev_img).train()
    seen = []
    model.backbone.register_forward_pre_hook(lambda _, args: seen.append(args[0].detach().clone()))
    x = torch.rand(2, 6, 32, 48, requires_grad=True)
    original = x.detach().clone()
    result = model(x)
    torch.testing.assert_close(x, original)
    prev = torch.zeros_like(original[:, :3]) if p_drop_prev_img else original[:, :3]
    expected = torch.cat(
        (model.normalize_frame_transform(prev), model.normalize_frame_transform(original[:, 3:])), dim=1
    )
    torch.testing.assert_close(seen[0], expected)
    assert result["prev_img_mask"].tolist() == [not p_drop_prev_img] * 2
    losses = model.get_losses(result, {"frame_speeds": torch.ones(2, 1)})
    assert torch.isfinite(losses["total"])
    if p_drop_prev_img:
        assert losses["total"].item() == 0
    losses["total"].backward()
    assert x.grad is not None


@pytest.mark.parametrize("encoder_cls", [DINOv2Encoder, DINOv3Encoder])
def test_frozen_dino_training_and_patch_padding(encoder_cls):
    model = _encoder(encoder_cls, p_drop_prev_img=0)
    assert not model.backbone.training
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    model.train()
    assert model.training and not model.backbone.training
    x = torch.rand(2, 6, 29, 43)
    result = model(x, export_heads=[])
    assert result["pose"].shape == (2, 1)
    assert result["feat_out"].shape == (2, 8)
    loss = model.get_losses(result, {"frame_speeds": torch.ones(2, 1)})["total"]
    assert torch.isfinite(loss)
    loss.backward()
    assert model.final_linear.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.backbone.parameters())


def test_dino_backbone_can_be_trained():
    model = _encoder(DINOv3Encoder, freeze_backbone=False)
    assert model.backbone.training
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())
    model.eval().train()
    assert model.backbone.training


def test_default_neck_and_invalid_frame_pairs():
    model = ResNetEncoder(pretrained=False, img_embed_size=16).eval()
    with torch.no_grad():
        assert model(torch.rand(2, 6, 32, 48))["feat_out"].shape == (2, 256)
    with pytest.raises(ValueError, match="RGB pair"):
        model(torch.rand(2, 3, 32, 48))
    with pytest.raises(TypeError, match="floating-point"):
        model(torch.zeros(2, 6, 32, 48, dtype=torch.uint8))
    with pytest.raises(ValueError, match="in_chans=6"):
        _encoder(ResNetEncoder, in_chans=3)
    with pytest.raises(ValueError, match="extra heads"):
        _encoder(DINOv2Encoder, heads={"depth": {}})


@pytest.mark.parametrize("image_size", [(32, 32), (29, 43), (28, 56)])
def test_dinov2_export_preparation_preserves_outputs_and_original_weights(image_size):
    model = _encoder(DINOv2Encoder).eval()
    original_positions = model.backbone.pos_embed.detach().clone()
    prepared = model.prepare_for_export(image_size)
    assert prepared is not model
    assert model.backbone.dynamic_img_size
    torch.testing.assert_close(model.backbone.pos_embed, original_positions, rtol=0, atol=0)
    frames = torch.rand(2, 6, *image_size)
    with torch.no_grad():
        expected, actual = model(frames), prepared(frames)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_dinov2_padded_nonsquare_onnx_export(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    model = _encoder(DINOv2Encoder).eval()
    frames = torch.rand(2, 6, 32, 48)
    prepared = model.prepare_for_export((32, 48))
    path = tmp_path / "dinov2.onnx"
    names = ["pose", "feat_out", "prev_img_mask"]
    with torch.no_grad():
        expected = model(frames)
        torch.onnx.export(
            prepared,
            (frames,),
            path,
            input_names=["frames"],
            output_names=names,
            opset_version=17,
            dynamo=False,
        )
    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(names, {"frames": frames.numpy()})
    for name, output in zip(names, actual):
        torch.testing.assert_close(torch.from_numpy(output), expected[name], rtol=2e-3, atol=2e-4)
