"""Check a configured policy and print its pipeline using synthetic inputs.

python -m visnavkit.scripts.sanity_check
python -m visnavkit.scripts.sanity_check model/vision_encoder=resnet18 model/temporal_encoder=bidirectional
python -m visnavkit.scripts.sanity_check --onnx model/action_decoder=flow_dit model/goal_encoder=point
"""

import argparse
import copy
from pathlib import Path

import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate

from visnavkit.models.lit_model import build_targets, disable_pretrained_downloads


def _synthetic_goal(model, batch, sequence, image_hw):
    encoder = model.goal_encoder
    if encoder.num_tokens == 0:
        return None
    if encoder.per_frame:
        return encoder.example_input(batch * sequence, image_hw=image_hw).reshape(batch, sequence, -1)
    return encoder.example_input(batch, image_hw=image_hw)


def _pipeline(model, frames, goal, output, cfg):
    batch, sequence, channels, height, width = frames.shape
    tokens = output.vision.tokens
    context = model.temporal_encoder(tokens.reshape(batch, sequence, model.num_tokens, model.feat_size))
    plans = output.plan.plans
    trajectories = model.action_decoder.parse_output(plans)["plans"]
    decoder = model.action_decoder
    goal_line = (
        "(no goal tokens)" if goal is None else f"goal {list(goal.shape)} -> tokens {list(output.goal_tokens.shape)}"
    )
    lines = [
        type(model).__name__,
        "|",
        f"+-- RGB frame pairs {list(frames.shape)}  [batch, frames, prev+current RGB, height, width]",
        f"|   `-- flatten frames -> {[batch * sequence, channels, height, width]}",
        f"+-- vision_encoder: {type(model.vision_encoder).__name__} ({cfg.model.vision_encoder.backbone_name}, {model.vision_encoder.token_mode})",
        f"|   +-- pose -> {list(output.vision.pose.shape)}",
        f"|   `-- tokens -> {list(tokens.shape)}  [frames, tokens per frame, feat_size]",
        f"+-- temporal_encoder: {type(model.temporal_encoder).__name__}",
        f"|   `-- reduction={model.temporal_encoder.reduction} -> {list(context.shape)}  [batch, decisions, tokens, feat_size]",
        f"+-- goal_encoder: {type(model.goal_encoder).__name__} ({model.goal_encoder.goal_type}) {goal_line}",
        f"`-- action_decoder: {type(decoder).__name__} ({decoder.action_space.kind}, {decoder.num_modes} modes)",
    ]
    if hasattr(decoder, "denoiser"):
        lines.append(
            f"    +-- denoiser: {type(decoder.denoiser).__name__} x {decoder.sample_steps} steps ({type(decoder.scheduler).__name__})"
        )
    lines += [
        f"    +-- plans (flat) -> {list(plans.shape)}",
        f"    `-- trajectories -> {list(trajectories.shape)}  [decisions, modes, points, pose dimensions]",
    ]
    return "\n".join(lines)


@torch.no_grad()
def _check_feature_buffer(model, frames, goal, seq_step):
    exported = copy.deepcopy(model).eval()
    temporal = exported.temporal_encoder
    if temporal.reduction == "none":
        temporal.reduction = "last"
    batch, sequence = frames.shape[:2]
    noise = exported.action_decoder.example_noise(batch) if exported.action_decoder.uses_noise else None
    expected = exported(frames, goal=goal, noise=noise)
    features = expected.vision.tokens.reshape(batch, sequence, model.token_dim)
    history_size = (sequence - 1) * seq_step
    buffer = torch.randn(batch, history_size, model.token_dim)
    # Unselected slots represent intervening frames and must not affect the decision.
    buffer[:, ::seq_step] = features[:, :-1]
    last_goal = None if goal is None else (goal[:, -1] if exported.goal_encoder.per_frame else goal)
    plans, pose, token, *_ = exported.predict(frames[:, -1], buffer, goal=last_goal, noise=noise)
    torch.testing.assert_close(plans, expected.plan.plans, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(pose, expected.vision.pose.reshape(batch, sequence, -1)[:, -1])
    torch.testing.assert_close(token, features[:, -1])
    return history_size, temporal.reduction


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. model=gnm model/action_decoder=mhp")
    parser.add_argument(
        "--onnx", action="store_true", help="Also export the deployment ONNX graph and verify ONNX Runtime parity"
    )
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
    disable_pretrained_downloads(cfg.model)
    model = instantiate(cfg.model).cpu().eval()
    batch, sequence = args.batch_size, int(cfg.common.seq_length)
    width, height = (int(value // cfg.common.downscale_factor) for value in cfg.common.crop_wh)
    if batch * sequence < 2:
        parser.error("Training BatchNorm requires batch-size * common.seq_length >= 2")
    frames = torch.rand(batch, sequence, 6, height, width)
    goal = _synthetic_goal(model, batch, sequence, (height, width))
    decoder = model.action_decoder
    reduction = model.temporal_encoder.reduction
    decisions = batch * sequence if reduction == "none" else batch

    with torch.no_grad():
        output = model(frames, goal=goal)
        assert tuple(output.vision.pose.shape) == (batch * sequence, 1), "Unexpected pose shape"
        assert tuple(output.vision.tokens.shape) == (batch * sequence, model.num_tokens, model.feat_size), (
            "Unexpected token shape"
        )
        assert torch.isfinite(output.vision.tokens).all() and torch.isfinite(output.vision.pose).all(), (
            "Nonfinite vision outputs"
        )
        plans = output.plan.plans
        assert tuple(plans.shape) == (decisions, decoder.flat_size), "Unexpected action output shape"
        assert torch.isfinite(plans).all(), "Nonfinite action outputs"
        diagram = _pipeline(model, frames, goal, output, cfg)
    print(diagram)
    print("\n[PASS] Forward shapes and finite outputs")

    model.train()
    targets = build_targets(
        {
            "frame_speeds": torch.rand(batch, sequence, 1),
            "future_poses": torch.rand(batch, sequence, decoder.num_pts, decoder.pose_size),
        },
        action_reduction=reduction,
    )
    model.zero_grad(set_to_none=True)
    losses, _ = model.get_losses(model(frames, goal=goal), targets)
    assert all(torch.isfinite(value).all() for value in losses.values()), "Nonfinite training loss"
    losses["loss"].backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients), "Missing or nonfinite gradients"
    assert any(gradient.abs().sum() > 0 for gradient in gradients), "All gradients are zero"
    print(f"[PASS] Training loss={losses['loss'].item():.6f}; backward gradients finite")
    model.zero_grad(set_to_none=True)
    model.eval()
    history_size, export_reduction = _check_feature_buffer(model, frames, goal, int(cfg.model.export_cfg.seq_step))
    print(f"[PASS] Feature-buffer parity (history={history_size}, reduction={export_reduction})")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = args.output_dir / "pipeline.txt"
    pipeline_path.write_text(diagram + "\n")
    print(f"Pipeline: {pipeline_path}")

    if args.onnx:
        from visnavkit.scripts.export import export_policy

        model_path = args.output_dir / "model.onnx"
        parity = export_policy(cfg, model_path, half=False, checkpoint=None, batch_size=batch)
        errors = ", ".join(f"{name}={value:.3g}" for name, value in parity.items())
        print(f"[PASS] ONNX Runtime parity (maximum absolute errors: {errors})")
        print(f"ONNX: {model_path}")


if __name__ == "__main__":
    main()
