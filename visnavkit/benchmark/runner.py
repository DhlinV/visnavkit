"""Benchmark orchestration; model construction stays in Hydra and predictor registries."""

import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from visnavkit.benchmark import predictors as _predictors  # noqa: F401 - registers built-in predictors
from visnavkit.benchmark.catalog import download_model, get_model, list_models
from visnavkit.benchmark.datasets import load_archive, write_fixture
from visnavkit.benchmark.export import export_native, sha256_file
from visnavkit.benchmark.quality import trajectory_metrics
from visnavkit.benchmark.registry import PREDICTORS
from visnavkit.benchmark.runtime import benchmark_session, create_session, inspect_graph, make_inputs


def _metadata(path):
    metadata_path = Path(path).with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if metadata.get("onnx_sha256") and metadata["onnx_sha256"] != sha256_file(path):
        raise ValueError(
            "ONNX artifact differs from its metadata. Re-export or supply the matching artifact and sidecar."
        )
    return metadata


def _artifact(cfg, model_id):
    if cfg.artifact:
        return Path(cfg.artifact)
    if cfg.source == "native":
        destination = Path(cfg.output_dir) / f"{model_id}.onnx"
        export_native(
            cfg, destination, checkpoint=cfg.checkpoint, seed=cfg.seed, batch_size=cfg.batch_size, model_id=model_id
        )
        return destination
    if cfg.source != "published":
        raise ValueError("source must be native or published")
    spec = get_model(model_id)
    graphs = [artifact for artifact in spec.artifacts if artifact.path.endswith(".onnx")]
    if len(graphs) != 1:
        raise ValueError(
            f"{spec.name} has {len(graphs)} graphs. A complete registered policy runner is required; "
            "individual graph latency is not policy latency. Use artifact=... for explicitly labeled component profiling."
        )
    path = Path(cfg.artifact_dir) / graphs[0].path
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run command=download model_id={model_id} first.")
    return path


def _feeds(session, cfg, path):
    archive = cfg.input_archive
    if archive is None:
        sibling = path.with_suffix(".inputs.npz")
        archive = sibling if sibling.exists() else None
    if archive is not None:
        with np.load(archive, allow_pickle=False) as source:
            return {name: source[name] for name in source.files}
    shapes = OmegaConf.to_container(cfg.runtime.shapes, resolve=True)
    if cfg.source == "published":
        for item in session.get_inputs():
            if item.name not in shapes and item.shape and not isinstance(item.shape[0], int):
                shapes[item.name] = [int(cfg.batch_size), *item.shape[1:]]
    return make_inputs(session, seed=cfg.seed, shapes=shapes)


def profile(cfg, model_id):
    if cfg.source == "published" and model_id == "nomad" and cfg.artifact is None:
        from visnavkit.benchmark.published import benchmark_nomad

        feeds = None
        if cfg.input_archive is not None:
            with np.load(cfg.input_archive, allow_pickle=False) as source:
                feeds = {name: source[name] for name in source.files}
        measured = benchmark_nomad(
            cfg.artifact_dir,
            provider=cfg.runtime.provider,
            threads=cfg.runtime.threads,
            batch_size=cfg.batch_size,
            num_samples=cfg.sampling.num_candidates,
            warmup=cfg.runtime.warmup,
            iterations=cfg.runtime.iterations,
            seed=cfg.seed,
            feeds=feeds,
        )
        return {
            "schema_version": 1,
            "command": "profile",
            "model_id": model_id,
            "catalog": asdict(get_model(model_id)),
            **measured,
        }
    path = _artifact(cfg, model_id)
    session = create_session(path, provider=cfg.runtime.provider, threads=cfg.runtime.threads)
    feeds = _feeds(session, cfg, path)
    timing = benchmark_session(session, feeds, warmup=cfg.runtime.warmup, iterations=cfg.runtime.iterations)
    graph = inspect_graph(path, feeds)
    meta = _metadata(path)
    spec = asdict(get_model(model_id)) if cfg.source == "published" else None
    if spec is not None:
        expected_hashes = {item["sha256"] for item in spec["artifacts"]}
        actual_hashes = {item["sha256"] for item in graph["artifact_files"]}
        if not actual_hashes or not actual_hashes.issubset(expected_hashes):
            raise ValueError("Published model artifact hash does not match the pinned catalog identity.")
    return {
        "schema_version": 1,
        "command": "profile",
        "model_id": model_id,
        "implementation_kind": meta.get("implementation_kind", "external_onnx" if cfg.artifact else cfg.source),
        "measurement_scope": meta.get("inference_mode", "supplied_graph"),
        "artifact": str(path),
        "metadata": meta,
        "catalog": spec,
        "runtime": timing,
        "graph": graph,
    }


def evaluate(cfg, model_id):
    if cfg.dataset_archive is None:
        raise ValueError("dataset_archive is required. Use command=fixture for a pipeline-check dataset.")
    data, dataset_meta = load_archive(cfg.dataset_archive)
    adapter = cfg.adapter
    is_control = adapter.name in ("stationary", "constant_velocity")
    meta = {}
    if is_control:
        times = np.asarray(data["target_times_s"])
        if times.ndim != 1:
            raise ValueError("Control baseline currently requires common one-dimensional target timestamps.")
        predictor = PREDICTORS.get(adapter.name)()
        model_id = adapter.name
    else:
        path = _artifact(cfg, model_id)
        meta = _metadata(path)
        times = adapter.prediction_times_s
        if times is None:
            times = meta.get("target_times_s")
        if times is None:
            raise ValueError(
                "External evaluation requires explicit adapter.prediction_times_s and metric-space scaling."
            )
        if meta.get("weights") == "untrained_policy" and dataset_meta.get("dataset_kind") != "synthetic_fixture":
            if not cfg.allow_untrained:
                raise ValueError(
                    "Untrained-policy evaluation requires allow_untrained=true; scores are diagnostic only."
                )
        session = create_session(path, provider=cfg.runtime.provider, threads=cfg.runtime.threads)
        if any(
            item.shape and isinstance(item.shape[0], int) and item.shape[0] != 1
            for item in session.get_inputs()
            if item.name != "initial_noise"
        ):
            raise ValueError("Independent-sample evaluation requires batch-one export.")
        noise = next((item for item in session.get_inputs() if item.name == "initial_noise"), None)
        if noise is not None and "input__initial_noise" not in data:
            rng = np.random.default_rng(cfg.seed)
            data["input__initial_noise"] = rng.standard_normal((len(data["targets"]), *noise.shape)).astype(np.float32)
        scores_name = adapter.scores_name
        if meta.get("selection") == "highest_score" and scores_name is None:
            scores_name = "scores"
        predictor = PREDICTORS.get(adapter.name)(
            session,
            output_name=adapter.output_name,
            scores_name=scores_name,
            xy_scale=adapter.xy_scale,
            layout=adapter.layout,
            speed_index=2 if meta.get("trajectory_units", [None])[-1] == "meter_per_second" else adapter.speed_index,
        )
    predictions, scores = predictor.predict(data, np.asarray(times))
    metrics = trajectory_metrics(
        predictions,
        data["targets"],
        times,
        data["target_times_s"],
        scores=scores,
        horizons=cfg.horizons,
        valid_mask=data.get("valid_mask"),
    )
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(
        output / "predictions.npz",
        trajectories=predictions,
        prediction_times_s=np.asarray(times),
        sample_ids=data["sample_ids"],
        **({"scores": scores} if scores is not None else {}),
    )
    synthetic = dataset_meta.get("dataset_kind") == "synthetic_fixture"
    return {
        "schema_version": 1,
        "command": "evaluate",
        "model_id": model_id,
        "result_kind": "pipeline_check" if synthetic or meta.get("weights") == "untrained_policy" else "open_loop",
        "dataset": dataset_meta,
        "dataset_sha256": sha256_file(cfg.dataset_archive),
        "metadata": meta,
        "adapter": OmegaConf.to_container(adapter, resolve=True),
        "metrics": metrics,
        "seed": cfg.seed,
    }


def run(cfg, model_id):
    command = cfg.command
    if command == "list":
        return {
            "schema_version": 1,
            "command": command,
            "models": [asdict(model) for model in list_models()],
            "native_recipes": list(cfg.suite.models),
            "predictors": PREDICTORS.names(),
        }
    if command == "download":
        files = download_model(model_id, cfg.artifact_dir)
        return {"schema_version": 1, "command": command, "model_id": model_id, "files": [str(path) for path in files]}
    if command == "prepare":
        from visnavkit.benchmark.prepare import prepare_dataset

        path = cfg.dataset_archive or str(Path(cfg.output_dir) / "dataset.npz")
        return {
            "schema_version": 1,
            "command": command,
            "path": path,
            "dataset": prepare_dataset(cfg, path, limit=cfg.samples),
        }
    if command == "fixture":
        path = cfg.dataset_archive or str(Path(cfg.output_dir) / "fixture.npz")
        return {
            "schema_version": 1,
            "command": command,
            "path": path,
            "dataset": write_fixture(path, cfg, samples=cfg.samples, seed=cfg.seed),
        }
    if command == "export":
        path = Path(cfg.artifact) if cfg.artifact else Path(cfg.output_dir) / f"{model_id}.onnx"
        if cfg.source != "native":
            raise ValueError("command=export supports native recipes; command=download retrieves published artifacts.")
        return {
            "schema_version": 1,
            "command": command,
            "artifact": str(path),
            "metadata": export_native(
                cfg, path, checkpoint=cfg.checkpoint, seed=cfg.seed, batch_size=cfg.batch_size, model_id=model_id
            ),
        }
    if command == "profile":
        return profile(cfg, model_id)
    if command == "evaluate":
        return evaluate(cfg, model_id)
    if command == "smoke":
        if cfg.source != "native":
            raise ValueError("smoke uses native models and generated observations.")
        local = copy.deepcopy(cfg)
        local.dataset_archive = str(Path(cfg.output_dir) / "fixture.npz")
        write_fixture(local.dataset_archive, local, samples=local.samples, seed=local.seed)
        measured = profile(local, model_id)
        local.artifact = measured["artifact"]
        scored = evaluate(local, model_id)
        return {"schema_version": 1, "command": command, "result_kind": "pipeline_check", "results": [measured, scored]}
    if command == "suite":
        results = []
        root = Path(cfg.output_dir)
        for name in cfg.suite.models:
            try:
                if cfg.source == "native":
                    # Compose each actual recipe, retaining experiment/runtime overrides.
                    if GlobalHydra.instance().is_initialized():
                        recipe = compose(config_name="train", overrides=[f"model={name}"])
                    else:
                        with initialize_config_module(version_base=None, config_module="visnavkit.configs"):
                            recipe = compose(config_name="train", overrides=[f"model={name}"])
                    local = copy.deepcopy(cfg)
                    local.model = recipe.model
                    local.model.vision_encoder.pretrained = False
                else:
                    local = copy.deepcopy(cfg)
                local.output_dir = str(root / name)
                local.checkpoint = cfg.suite.checkpoints.get(name)
                result = profile(local, name)
                if local.dataset_archive:
                    local.artifact = result["artifact"]
                    result["quality"] = evaluate(local, name)
                results.append(result)
            except Exception as error:
                if cfg.suite.fail_fast:
                    raise
                results.append({"model_id": name, "status": "failed", "error": str(error)})
        return {"schema_version": 1, "command": command, "results": results}
    raise ValueError(f"Unknown command {command!r}")
