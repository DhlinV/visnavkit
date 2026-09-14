"""Export the full observation window with an explicit trajectory output contract."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from visnavkit.models.action.outputs import parse_plan_output
from visnavkit.models.lit_model import disable_pretrained_downloads
from visnavkit.utils.common import build_idxs


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_times(cfg):
    count = int(cfg.plan_len_points)
    if cfg.common.get("offset_t_anchors", False):
        return build_idxs(float(cfg.plan_len_seconds), count + 1)[1:]
    return build_idxs(float(cfg.plan_len_seconds), count)


def architecture_config(model_cfg):
    """Resolved model config without initialization-only fields, for checkpoint/recipe comparison."""
    model_cfg = disable_pretrained_downloads(copy.deepcopy(model_cfg))
    return OmegaConf.to_container(model_cfg, resolve=True)


def load_native_model(cfg, checkpoint=None):
    cfg = copy.deepcopy(cfg)
    if checkpoint is not None:
        # Loading a complete checkpoint must not fetch backbone initialization weights.
        disable_pretrained_downloads(cfg.model)
    model = instantiate(cfg.model)
    if checkpoint is not None:
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        weights = loaded.get("state_dict", loaded)
        if any(key.startswith("model.") for key in weights):
            weights = {key.removeprefix("model."): value for key, value in weights.items() if key.startswith("model.")}
        model.load_state_dict(weights, strict=True)
    return model.cpu().eval()


class SequencePolicy(nn.Module):
    """One decision per independent history window; no hidden feature cache.

    Goal-conditioned recipes run with their learned null goal token (goal-free inference, as
    in NoMaD exploration); the metadata labels this ``null_goal_token``.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.decoder = model.action_decoder

    @property
    def conditioning(self) -> str:
        return "goal_free" if self.model.goal_encoder.num_tokens == 0 else "null_goal_token"

    def forward(self, frames, initial_noise=None):
        batch, history = frames.shape[:2]
        vision = self.model.encode_frames(frames)
        tokens = vision.tokens.reshape(batch, history, self.model.num_tokens, self.model.feat_size)
        context = self.model.temporal_encoder(tokens)[:, -1]
        goal_tokens = (
            None if self.model.goal_encoder.num_tokens == 0 else self.model.goal_encoder(None, batch_size=batch)
        )
        flat = self.decoder(context, goal_tokens, initial_noise).plans
        parsed = parse_plan_output(
            flat, num_modes=self.decoder.num_modes, num_pts=self.decoder.num_pts, pose_size=self.decoder.pose_size
        )
        speed = vision.pose.reshape(batch, history, -1)[:, -1]
        return parsed["plans"], parsed["confs"], speed


def export_native(cfg, output, *, checkpoint=None, seed=42, batch_size=1, model_id="base"):
    """FP32 full-window export plus metadata and nonzero-input numerical parity."""
    import onnx

    from visnavkit.benchmark.runtime import create_session

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(cfg)
    checkpoint_path = Path(checkpoint) if checkpoint else None
    if checkpoint_path:
        stored = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_cfg = stored.get("hyper_parameters", {}).get("cfg")
        if saved_cfg is not None:
            saved_cfg = OmegaConf.create(saved_cfg) if isinstance(saved_cfg, dict) else saved_cfg
            if architecture_config(cfg.model) != architecture_config(saved_cfg.model):
                raise ValueError(
                    "Checkpoint model config differs from the requested recipe. Compose its original model/config to avoid mislabeled benchmarks."
                )
            cfg = saved_cfg
    torch.manual_seed(seed)
    model = load_native_model(cfg, checkpoint_path)
    parameters_total = sum(p.numel() for p in model.parameters())
    parameters_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size = cfg.common
    h, w = int(size.crop_wh[1] // size.downscale_factor), int(size.crop_wh[0] // size.downscale_factor)
    model.vision_encoder = model.vision_encoder.prepare_for_export((h, w))
    wrapper = SequencePolicy(model).eval()
    frames = torch.rand(batch_size, int(size.seq_length), 6, h, w)
    head = model.action_decoder
    is_diffusion = head.uses_noise
    inputs = (frames,)
    names = ["frames"]
    if is_diffusion:
        inputs += (head.example_noise(batch_size),)
        names += ["initial_noise"]
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.no_grad():
            expected = wrapper(*inputs)
            with FlopCounterMode(display=False) as counter:
                wrapper(*inputs)
            counted_flops = int(counter.get_total_flops())
            torch.onnx.export(
                wrapper,
                inputs,
                str(output),
                input_names=names,
                output_names=["trajectories", "scores", "speed"],
                opset_version=17,
                dynamo=False,
                # Fixed batch/context dimensions make the benchmark contract explicit.
            )
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
    onnx.checker.check_model(str(output))
    session = create_session(output)
    feeds = {name: tensor.numpy() for name, tensor in zip(names, inputs)}
    actual = session.run(None, feeds)
    parity = {}
    for name, reference, observed in zip(["trajectories", "scores", "speed"], expected, actual):
        target = reference.detach().numpy()
        np.testing.assert_allclose(observed, target, rtol=2e-3, atol=2e-4, err_msg=name)
        parity[name] = {"max_abs_error": float(np.max(np.abs(observed - target)))}
    has_checkpoint = checkpoint_path is not None
    meta = {
        "schema_version": 1,
        "model_id": model_id,
        "implementation_kind": "architecture_adaptation",
        "weights": "checkpoint" if has_checkpoint else "untrained_policy",
        "checkpoint_sha256": sha256_file(checkpoint_path) if has_checkpoint else None,
        "onnx_sha256": sha256_file(output),
        "inference_mode": "full_context",
        "conditioning": wrapper.conditioning,
        "precision": "float32",
        "input_shapes": {name: list(value.shape) for name, value in feeds.items()},
        "target_times_s": target_times(cfg).tolist(),
        "trajectory_units": ["meter", "meter"] + (["meter_per_second"] if head.pose_size == 3 else []),
        "selection": "unranked_samples" if is_diffusion else "highest_score",
        "num_candidates": head.num_modes,
        "denoising_steps": head.sample_steps if is_diffusion else 0,
        "action_space": head.action_space.kind,
        "parameters_total": parameters_total,
        "parameters_trainable": parameters_trainable,
        "parameter_scope": "source model, including auxiliary heads; not inferred from ONNX constants",
        "torch_counted_flops": counted_flops,
        "flop_scope": "full-context decision at exported batch size; PyTorch registered operators only, not a total",
        "parity": parity,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "seed": seed,
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")
    np.savez(output.with_suffix(".inputs.npz"), **feeds)
    return meta
