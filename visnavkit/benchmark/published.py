"""Published NoMaD sampling with full-loop ONNX Runtime timing.

This is a performance adapter producing normalized delta actions. Camera
preprocessing and metric-space trajectory conversion remain separate.

Scheduler source: UCLA-VAIL/Navigation-Model-Zoo-Public, revision
9c1f523ef8dddbccb6dee1d3d588112f754bc8f2, NoMaD_GL_Official/inference.py.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from visnavkit.benchmark import runtime
from visnavkit.benchmark.catalog import HF_REVISION, get_model

_DENOISING_STEPS = 10


def _positive_int(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def cosine_betas(steps=_DENOISING_STEPS):
    """Discretize the squaredcos_cap_v2 schedule with beta capped at 0.999."""
    _positive_int(steps, "steps")

    def alpha_bar(t):
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    return np.asarray(
        [min(1 - alpha_bar((t + 1) / steps) / alpha_bar(t / steps), 0.999) for t in range(steps)],
        dtype=np.float64,
    )


def ddpm_step(sample, predicted_noise, timestep, betas, variance_noise):
    """DDPM epsilon-prediction posterior update with clipped x0 and explicit noise."""
    cumulative = np.cumprod(1.0 - betas)
    if not isinstance(timestep, int) or not 0 <= timestep < len(betas):
        raise ValueError("timestep is outside the training schedule")
    if sample.shape != predicted_noise.shape or sample.shape != variance_noise.shape:
        raise ValueError("sample, predicted_noise and variance_noise must have identical shapes")
    alpha = cumulative[timestep]
    clean = np.clip((sample - np.sqrt(1.0 - alpha) * predicted_noise) / np.sqrt(alpha), -1, 1)
    if timestep == 0:
        return clean.astype(np.float32)
    previous = cumulative[timestep - 1]
    beta = betas[timestep]
    clean_weight = np.sqrt(previous) * beta / (1.0 - alpha)
    sample_weight = np.sqrt(1.0 - beta) * (1.0 - previous) / (1.0 - alpha)
    variance = beta * (1.0 - previous) / (1.0 - alpha)
    return (clean_weight * clean + sample_weight * sample + np.sqrt(variance) * variance_noise).astype(np.float32)


class NoMaDSampler:
    """Session-compatible adapter executing every step of normalized-action sampling.

    The fixed noise tape is generated at construction and reused for timing.
    Inputs follow the exported encoder contract; no image processing is implied.
    Distance prediction is optional because the published trajectory wrapper
    does not execute its distance graph.
    """

    def __init__(self, vision, denoiser, *, batch_size=1, num_samples=8, seed=0, distance=None):
        _positive_int(batch_size, "batch_size")
        _positive_int(num_samples, "num_samples")
        expected = {
            "vision": {"obs_img", "goal_img", "goal_mask"},
            "denoiser": {"sample", "timestep", "global_cond"},
        }
        for label, session in (("vision", vision), ("denoiser", denoiser)):
            actual = {item.name for item in session.get_inputs()}
            if actual != expected[label]:
                raise ValueError(f"NoMaD {label} input contract differs: {sorted(actual)}")
        if distance is not None and len(distance.get_inputs()) != 1:
            raise ValueError("NoMaD distance component must accept one conditioning input")
        self.vision = vision
        self.denoiser = denoiser
        self.distance = distance
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.betas = cosine_betas()
        rng = np.random.default_rng(seed)
        shape = (batch_size * num_samples, 8, 2)
        self.initial_noise = rng.standard_normal(shape).astype(np.float32)
        self.variance_noise = rng.standard_normal((_DENOISING_STEPS, *shape)).astype(np.float32)

    def __getattr__(self, name):
        # Runtime metadata and encoder input descriptors share the session API.
        return getattr(self.vision, name)

    def run(self, output_names, feeds):
        if output_names is not None:
            raise ValueError("NoMaDSampler returns all outputs; output_names must be None")
        condition = self.vision.run(None, feeds)[0]
        if condition.ndim != 2 or condition.shape[0] != self.batch_size:
            raise ValueError("NoMaD conditioning output must have shape (batch_size, embedding)")
        repeated = np.repeat(condition, self.num_samples, axis=0)
        sample = self.initial_noise.copy()
        for timestep in reversed(range(_DENOISING_STEPS)):
            prediction = self.denoiser.run(
                None,
                {"sample": sample, "timestep": np.asarray(timestep, dtype=np.int64), "global_cond": repeated},
            )[0]
            sample = ddpm_step(sample, prediction, timestep, self.betas, self.variance_noise[timestep])
        outputs = [sample.reshape(self.batch_size, self.num_samples, 8, 2)]
        if self.distance is not None:
            name = self.distance.get_inputs()[0].name
            outputs.append(self.distance.run(None, {name: condition})[0])
        return outputs


def benchmark_nomad(
    bundle_root,
    *,
    provider="CPUExecutionProvider",
    threads=1,
    batch_size=1,
    num_samples=8,
    warmup=5,
    iterations=30,
    seed=0,
    include_distance=False,
    feeds=None,
):
    """Profile pinned artifacts under the catalog downloader's destination.

    No downloads occur. Prepared encoder feeds are optional; otherwise the
    benchmark generates repeatable nonzero synthetic input tensors.
    """
    _positive_int(batch_size, "batch_size")
    _positive_int(num_samples, "num_samples")
    spec = get_model("nomad")
    root = Path(bundle_root)
    paths = {
        "vision": root / "NoMaD_GL_Official/nomad_vision_encoder.onnx",
        "denoiser": root / "NoMaD_GL_Official/nomad_noise_pred.onnx",
        "distance": root / "NoMaD_GL_Official/nomad_dist_pred.onnx",
    }
    reports = {name: runtime.inspect_graph(path) for name, path in paths.items()}
    files = {Path(item["path"]).resolve(): item for report in reports.values() for item in report["artifact_files"]}
    for artifact in spec.artifacts:
        path = (root / artifact.path).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("NoMaD artifact path escapes the bundle directory")
        if path.stat().st_size != artifact.size_bytes or runtime._sha256(path) != artifact.sha256:
            raise ValueError(f"NoMaD bundle differs from pinned revision {HF_REVISION}: {artifact.path}")
    vision = runtime.create_session(paths["vision"], provider=provider, threads=threads)
    denoiser = runtime.create_session(paths["denoiser"], provider=provider, threads=threads)
    distance = (
        runtime.create_session(paths["distance"], provider=provider, threads=threads) if include_distance else None
    )
    input_kind = "prepared_inputs" if feeds is not None else "synthetic"
    if feeds is None:
        feeds = runtime.make_inputs(
            vision,
            seed=seed,
            shapes={
                "obs_img": [batch_size, 12, 96, 96],
                "goal_img": [batch_size, 3, 96, 96],
                "goal_mask": [batch_size],
            },
        )
    expected = {item.name for item in vision.get_inputs()}
    if set(feeds) != expected or any(not isinstance(value, np.ndarray) for value in feeds.values()):
        raise ValueError("NoMaD encoder feeds must contain exactly the graph's NumPy inputs")
    if any(not np.isfinite(value).all() for value in feeds.values()):
        raise ValueError("NoMaD encoder feeds must be finite")
    if feeds["obs_img"].shape[0] != batch_size:
        raise ValueError("Observation batch size differs from batch_size")
    sampler = NoMaDSampler(
        vision, denoiser, batch_size=batch_size, num_samples=num_samples, seed=seed, distance=distance
    )
    outputs = sampler.run(None, feeds)
    if any(not np.isfinite(value).all() for value in outputs):
        raise ValueError("NoMaD sampler produced nonfinite outputs")
    timing = runtime.benchmark_session(sampler, feeds, warmup=warmup, iterations=iterations)
    timing.update(
        {
            "num_samples": num_samples,
            "denoising_steps": _DENOISING_STEPS,
            "include_distance": include_distance,
            "scheduler": {
                "name": "DDPM",
                "beta_schedule": "squaredcos_cap_v2",
                "train_timesteps": 10,
                "prediction_type": "epsilon",
                "clip_sample": True,
                "seed": seed,
                "noise_policy": "fixed initial and variance noise reused for each measured invocation",
            },
            "timing_scope": "complete normalized-action sampler: encoder + ten denoiser calls + host DDPM updates; "
            "synchronous host feeds/outputs and transfers included; optional distance head declared",
            "throughput_unit": "observation decisions per second; each decision produces num_samples candidates",
            "outputs": [{"shape": list(value.shape), "dtype": str(value.dtype)} for value in outputs],
        }
    )
    timing["excluded_from_timing"].extend(["noise generation", "metric trajectory decoding", "controller"])

    condition = vision.run(None, feeds)[0]
    component_feeds = {
        "vision": feeds,
        "denoiser": {
            "sample": sampler.initial_noise,
            "timestep": np.asarray(9, dtype=np.int64),
            "global_cond": np.repeat(condition, num_samples, axis=0),
        },
    }
    if distance is not None:
        component_feeds["distance"] = {distance.get_inputs()[0].name: condition}
    for name, values in component_feeds.items():
        reports[name] = runtime.inspect_graph(paths[name], values)
    counts = {"vision": 1, "denoiser": _DENOISING_STEPS, "distance": int(include_distance)}
    counted = sum(reports[name]["flops"]["counted_flops"] * count for name, count in counts.items())
    graph = {
        "artifact_bytes": sum(item["bytes"] for item in files.values()),
        "bundle_bytes": spec.download_bytes,
        "bundle_scope": "all released files, including unreferenced external-data companions",
        "artifact_files": list(files.values()),
        "initializer_elements": sum(report["initializer_elements"] for report in reports.values()),
        "initializer_scope": "stored initializers across all three graphs, including unused distance head; "
        "not a model parameter count",
        "flops": {
            "counted_flops": counted,
            "total_flops": None,
            "complete": False,
            "convention": "2 FLOPs per multiply-accumulate; component count multiplied by invocations",
            "scope": "partial neural-network subtotal at executed shapes; host DDPM arithmetic is uncounted",
            "component_invocations": counts,
            "components": {name: report["flops"] for name, report in reports.items()},
        },
        "components": reports,
    }
    return {
        "schema_version": 1,
        "command": "profile",
        "model_id": "nomad",
        "implementation_kind": "published_export_variant",
        "measurement_scope": "complete_normalized_action_sampler",
        "artifact": str(paths["vision"]),
        "metadata": {
            "parameters_total": None,
            "parameters_trainable": None,
            "output_semantics": "normalized delta actions; metric conversion not validated",
            "revision": HF_REVISION,
            "input_kind": input_kind,
        },
        "catalog": asdict(spec),
        "runtime": timing,
        "graph": graph,
    }
