"""Deployment export: presence-driven inputs, ONNX Runtime parity, checkpoint round trip."""

import numpy as np
import onnxruntime as ort
import pytest
import torch
from hydra import compose, initialize_config_module

from visnavkit.scripts.export import export_policy, parse_plan_output

SMALL = [
    "common.seq_length=2",
    "common.crop_wh=[32,32]",
    "common.downscale_factor=1",
    "model.feat_size=8",
    "plan_len_points=4",
    "model.vision_encoder.pretrained=false",
    "model.vision_encoder.img_embed_size=16",
    "model.vision_encoder.neck_cfg.n_res_blocks=0",
    "model.temporal_encoder.num_heads=2",
    "model.temporal_encoder.ff_dim=16",
]


def _cfg(*overrides):
    with initialize_config_module(version_base=None, config_module="visnavkit.configs"):
        return compose(config_name="export", overrides=[*SMALL, *overrides])


def _inputs(path):
    return [node.name for node in ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).get_inputs()]


@pytest.mark.parametrize(
    "overrides, inputs",
    [
        (["model/vision_encoder=resnet18", "model.action_decoder.hidden=16"], ["vision", "feature_buffer"]),
        (
            [
                "model/vision_encoder=resnet18",
                "model/goal_encoder=point",
                "model/action_decoder=flow_dit",
                "model.action_decoder.sample_steps=2",
                "model.action_decoder.denoiser.hidden=16",
                "model.action_decoder.denoiser.depth=1",
                "model.action_decoder.denoiser.num_heads=2",
                "model.vision_encoder.token_mode=fused",
                "model.vision_encoder.patch_grid=[2,2]",
            ],
            ["vision", "feature_buffer", "goal", "noise"],
        ),
        (
            [
                "model/vision_encoder=resnet18",
                "model/goal_encoder=image",
                "model/action_decoder=diffusion_unet",
                "model.action_decoder.sample_steps=2",
                "model.action_decoder.denoiser.down_dims=[8,16]",
                "model.action_decoder.denoiser.n_groups=4",
                "model.goal_encoder.pretrained=false",
            ],
            ["vision", "feature_buffer", "goal", "noise"],
        ),
    ],
)
def test_untrained_export_has_presence_driven_inputs_and_parity(tmp_path, overrides, inputs):
    torch.set_num_threads(1)
    path = tmp_path / "policy.onnx"
    errors = export_policy(_cfg(*overrides), path, half=False)
    assert _inputs(path) == inputs
    assert set(errors) == {"plan", "feat_out"}
    # These are absolute errors on untrained outputs; export_policy itself applies the relative
    # check (rtol 2e-3), and an untrained denoiser amplifies float noise over its sampling loop.
    assert all(error < 5e-3 for error in errors.values()), errors


def test_checkpoint_round_trip_enforces_parity(tmp_path):
    lightning = pytest.importorskip("lightning")
    from visnavkit.models.lit_model import LitModel

    torch.set_num_threads(1)
    cfg = _cfg(
        "model/vision_encoder=resnet18",
        "model/action_decoder=regression",
        "model.action_decoder.hidden=16",
        "model/goal_encoder=point",
    )
    model = LitModel(cfg)
    checkpoint = tmp_path / "last.ckpt"
    trainer = lightning.Trainer(logger=False, enable_checkpointing=False, accelerator="cpu", devices=1, max_steps=0)
    trainer.strategy.connect(model)
    trainer.save_checkpoint(checkpoint)
    restored = LitModel.load_from_checkpoint(checkpoint, cfg=cfg)
    assert not any(p.requires_grad is None for p in restored.parameters())
    errors = export_policy(cfg, tmp_path / "trained.onnx", half=False, checkpoint=str(checkpoint))
    assert errors["plan"] < 2e-4


def test_numpy_plan_parser_keeps_xy_layout():
    means = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    logits = np.array([-2, 2], dtype=np.float32)
    flat = np.concatenate([means.reshape(2, -1), -means.reshape(2, -1), logits[:, None]], axis=1).reshape(1, -1)
    parsed = parse_plan_output(flat, M=2, num_pts=3, pose_width=3)
    assert set(parsed) == {"pred_logits", "pred_confs", "pred_plans", "best_plan"}
    np.testing.assert_array_equal(parsed["pred_logits"], logits)
    np.testing.assert_array_equal(parsed["pred_plans"], means[:, :, :2])
    np.testing.assert_array_equal(parsed["best_plan"], means[1, :, :2])
