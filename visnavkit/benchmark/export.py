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

from visnavkit.models.action_decoders.diffusion import DiffusionPlanHead
from visnavkit.models.action_decoders.outputs import parse_plan_output
from visnavkit.models.compatibility import normalize_model_config
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


def load_native_model(cfg, checkpoint=None):
    cfg = copy.deepcopy(cfg)
    if checkpoint is not None:
        # Loading a complete checkpoint must not fetch backbone initialization weights.
        cfg.model.modules.vision_encoder.pretrained = False
        for component in (cfg.model.modules.vision_encoder, cfg.model.modules.action_decoder.plan_head):
            if "weights" in component:
                component.weights = None
    model = instantiate(cfg.model)
    if checkpoint is not None:
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        weights = loaded.get("state_dict", loaded)
        if any(key.startswith("model.") for key in weights):
            weights = {key.removeprefix("model."): value for key, value in weights.items() if key.startswith("model.")}
        model.load_state_dict(weights, strict=True)
    return model.cpu().eval()


class SequencePolicy(nn.Module):
    """One decision per independent history window; no hidden feature cache."""

    def __init__(self, model):
        super().__init__()
        if model.route_encoder is not None:
            raise ValueError(
                "Native benchmark export currently requires a goal-free recipe; use explicit external feeds."
            )
        self.model = model
        self.plan_head = model.action_decoder.plan_head

    def forward(self, frames, initial_noise=None):
        batch, history, channels, height, width = frames.shape
        vision = self.model.vision_encoder(frames.reshape(batch * history, channels, height, width))
        features = vision["feat_out"].reshape(batch, history, self.model.feat_size)
        temporal = self.model.action_decoder.temporal_encoder(features)
        if temporal.ndim == 3:
            temporal = temporal[:, -1]
        if isinstance(self.plan_head, DiffusionPlanHead):
            flat = self.plan_head._sample(temporal, noise=initial_noise)
        else:
            flat = self.plan_head(temporal)["plans"]
        parsed = parse_plan_output(
            flat, num_modes=self.plan_head.num_modes, num_pts=self.plan_head.num_pts, pose_size=self.plan_head.pose_size
        )
        speed = vision["pose"].reshape(batch, history, -1)[:, -1]
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
            requested = normalize_model_config(cfg.model)
            restored = normalize_model_config(saved_cfg.model)
            if requested != restored:
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
    prepare_vision = getattr(model.vision_encoder, "prepare_for_export", None)
    if prepare_vision is not None:
        model.vision_encoder = prepare_vision((h, w))
    wrapper = SequencePolicy(model).eval()
    frames = torch.rand(batch_size, int(size.seq_length), int(cfg.model.modules.vision_encoder.in_chans), h, w)
    head = model.action_decoder.plan_head
    is_diffusion = isinstance(head, DiffusionPlanHead)
    inputs = (frames,)
    names = ["frames"]
    if is_diffusion:
        inputs += (torch.randn(batch_size * head.num_modes, head.traj_dim),)
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
        "conditioning": "goal_free",
        "precision": "float32",
        "input_shapes": {name: list(value.shape) for name, value in feeds.items()},
        "target_times_s": target_times(cfg).tolist(),
        "trajectory_units": ["meter", "meter"] + (["meter_per_second"] if head.pose_size == 3 else []),
        "selection": "unranked_samples" if is_diffusion else "highest_score",
        "num_candidates": head.num_modes,
        "denoising_steps": head.sample_steps if is_diffusion else 0,
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
