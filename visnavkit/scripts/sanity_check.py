"""Check a configured model and print its pipeline using synthetic RGB pairs.

python -m visnavkit.scripts.sanity_check
python -m visnavkit.scripts.sanity_check vision_encoder=cnn_resnet temporal_encoder=bidirectional
python -m visnavkit.scripts.sanity_check --onnx action_decoder=diffusion
"""

import argparse
import copy
from pathlib import Path

import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate

from visnavkit.models.lit_model import build_targets


def _pipeline(model, frames, output, backbone_name):
    batch, sequence, channels, height, width = frames.shape
    vision, action = model.vision_encoder, model.action_decoder
    features = output["vision"]["feat_out"].reshape(batch, sequence, model.feat_size)
    temporal = action.temporal_encoder(features)
    plans = output["action"]["plan"]["plans"]
    trajectories = action.plan_head.parse_output(plans)["plans"]
    return "\n".join(
        [
            type(model).__name__,
            "|",
            f"+-- RGB frame pairs {list(frames.shape)}  [batch, frames, prev+current RGB, height, width]",
            f"|   `-- flatten frames -> {[batch * sequence, channels, height, width]}",
            f"+-- vision_encoder: {type(vision).__name__} ({backbone_name})",
            f"|   +-- pose -> {list(output['vision']['pose'].shape)}",
            f"|   `-- feat_out -> {list(output['vision']['feat_out'].shape)} -> {list(features.shape)}",
            f"`-- action_decoder: {type(action).__name__}",
            f"    +-- temporal_encoder: {type(action.temporal_encoder).__name__}",
            f"    |   `-- reduction={action.temporal_encoder.reduction} -> {list(temporal.shape)}",
            f"    `-- plan_head: {type(action.plan_head).__name__}",
            f"        +-- plans (flat) -> {list(plans.shape)}",
            f"        `-- trajectories -> {list(trajectories.shape)}  [decisions, modes, points, pose dimensions]",
        ]
    )


@torch.no_grad()
def _check_feature_buffer(model, frames, seq_step):
    exported = copy.deepcopy(model).eval()
    temporal = exported.action_decoder.temporal_encoder
    if temporal.reduction == "none":
        temporal.reduction = "last"
    batch, sequence = frames.shape[:2]
    torch.manual_seed(43)
    expected = exported(frames)
    features = expected["vision"]["feat_out"].reshape(batch, sequence, model.feat_size)
    history_size = (sequence - 1) * seq_step
    buffer = torch.randn(batch, history_size, model.feat_size)
    # Unselected slots represent intervening frames and must not affect the decision.
    buffer[:, ::seq_step] = features[:, :-1]
    torch.manual_seed(43)  # Match diffusion noise between full-window and buffered inference.
    plans, pose, token, *_ = exported(frames[:, -1], buffer)
    torch.testing.assert_close(plans, expected["action"]["plan"]["plans"], rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(pose, expected["vision"]["pose"].reshape(batch, sequence, -1)[:, -1])
    torch.testing.assert_close(token, features[:, -1])
    return history_size, temporal.reduction


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. model=gnm action_decoder=mhp")
    parser.add_argument("--onnx", action="store_true", help="Also export ONNX and verify ONNX Runtime numerical parity")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sanity"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2, help="CPU threads for small synthetic checks")
    args = parser.parse_intermixed_args(argv)
    if args.batch_size < 1 or args.threads < 1:
        parser.error("--batch-size and --threads must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    with initialize_config_module(version_base=None, config_module="visnavkit.configs"):
        cfg = compose(
            config_name="train",
            overrides=["common.seq_length=3", "common.crop_wh=[64,64]", "common.downscale_factor=1", *args.overrides],
        )
    encoder_cfg = cfg.model.modules.vision_encoder
    encoder_cfg.pretrained = False
    for component in (encoder_cfg, cfg.model.modules.action_decoder.plan_head):
        if "weights" in component:
            component.weights = None
    model = instantiate(cfg.model).cpu().eval()
    if model.route_encoder is not None:
        raise ValueError("This synthetic check requires a goal-free model without route_encoder.")
    batch, sequence = args.batch_size, int(cfg.common.seq_length)
    width, height = (int(value // cfg.common.downscale_factor) for value in cfg.common.crop_wh)
    if batch * sequence < 2:
        parser.error("Training BatchNorm requires batch-size * common.seq_length >= 2")
    frames = torch.rand(batch, sequence, int(encoder_cfg.in_chans), height, width)
    head = model.action_decoder.plan_head
    reduction = model.action_decoder.temporal_encoder.reduction
    decisions = batch * sequence if reduction == "none" else batch

    with torch.no_grad():
        output = model(frames)
        expected_shapes = {
            "pose": (batch * sequence, 1),
            "feat_out": (batch * sequence, model.feat_size),
        }
        for name, shape in expected_shapes.items():
            tensor = output["vision"][name]
            assert tuple(tensor.shape) == shape, f"{name}: expected {shape}, got {tuple(tensor.shape)}"
            assert torch.isfinite(tensor).all(), f"Nonfinite {name}"
        plans = output["action"]["plan"]["plans"]
        assert tuple(plans.shape) == (decisions, head.flat_size), "Unexpected action output shape"
        assert torch.isfinite(plans).all(), "Nonfinite action outputs"
        diagram = _pipeline(model, frames, output, encoder_cfg.backbone_name)
    print(diagram)
    print("\n[PASS] Forward shapes and finite outputs")

    model.train()
    targets = build_targets(
        {
            "frame_speeds": torch.rand(batch, sequence, 1),
            "future_poses": torch.rand(batch, sequence, head.num_pts, head.pose_size),
        },
        action_reduction=reduction,
    )
    model.zero_grad(set_to_none=True)
    losses, _ = model.get_losses(model(frames), targets)
    assert all(torch.isfinite(value).all() for value in losses.values()), "Nonfinite training loss"
    losses["loss"].backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients), "Missing or nonfinite gradients"
    assert any(gradient.abs().sum() > 0 for gradient in gradients), "All gradients are zero"
    print(f"[PASS] Training loss={losses['loss'].item():.6f}; backward gradients finite")
    model.zero_grad(set_to_none=True)
    model.eval()
    history_size, export_reduction = _check_feature_buffer(model, frames, int(cfg.model.export_cfg.seq_step))
    print(f"[PASS] Feature-buffer parity (history={history_size}, reduction={export_reduction})")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = args.output_dir / "pipeline.txt"
    pipeline_path.write_text(diagram + "\n")
    print(f"Pipeline: {pipeline_path}")

    if args.onnx:
        from visnavkit.benchmark.export import export_native

        model_path = args.output_dir / "model.onnx"
        metadata = export_native(cfg, model_path, batch_size=batch, model_id="sanity")
        errors = ", ".join(f"{name}={value['max_abs_error']:.3g}" for name, value in metadata["parity"].items())
        print(f"[PASS] ONNX Runtime parity (maximum absolute errors: {errors})")
        print(f"ONNX: {model_path}")


if __name__ == "__main__":
    main()
