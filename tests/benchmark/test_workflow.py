import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from visnavkit.benchmark.export import export_native
from visnavkit.benchmark.hooks import ReportHook, emit
from visnavkit.benchmark.registry import Registry
from visnavkit.benchmark.runner import _metadata, run


def _config(tmp_path, model="gnm"):
    with initialize_config_module(version_base=None, config_module="visnavkit.configs"):
        return compose(
            config_name="benchmark",
            overrides=[
                f"model={model}",
                "common.crop_wh=[32,32]",
                "common.downscale_factor=1",
                "common.seq_length=2",
                "runtime.iterations=2",
                "runtime.warmup=1",
                "samples=2",
                f"output_dir={tmp_path}",
            ],
        )


def test_native_smoke_writes_traceable_results_and_onnx_parity(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cfg = _config(tmp_path)
        cfg.command = "smoke"
        result = run(cfg, "gnm")
    finally:
        torch.set_num_threads(previous_threads)
    assert result["result_kind"] == "pipeline_check"
    profile, quality = result["results"]
    assert profile["runtime"]["iterations"] == 2
    assert profile["metadata"]["parameters_total"] > 0
    assert profile["metadata"]["parity"]["trajectories"]["max_abs_error"] < 2e-4
    assert quality["metrics"]["samples"] == 2
    assert quality["result_kind"] == "pipeline_check"
    emit([ReportHook()], "on_result", result=result, output_dir=tmp_path)
    assert json.loads((tmp_path / "result.json").read_text())["command"] == "smoke"
    assert (tmp_path / "results.csv").is_file()
    assert (tmp_path / "predictions.npz").is_file()


def test_registry_extensions_and_hook_errors_are_explicit():
    registry = Registry("test_predictor")

    @registry.register("custom")
    class Custom:
        pass

    assert registry.get("custom") is Custom
    assert registry.names() == ("custom",)
    with pytest.raises(ValueError, match="already contains"):
        registry.register("custom")(object)
    with pytest.raises(ValueError, match="choices"):
        registry.get("missing")

    class Observer:
        def __init__(self):
            self.seen = []

        def on_result(self, result, output_dir):
            self.seen.append((result, output_dir))

    observer = Observer()
    emit([observer], "on_start", config={})
    emit([observer], "on_result", result={"ok": True}, output_dir="unused")
    assert observer.seen == [({"ok": True}, "unused")]

    class BrokenHook:
        def on_result(self, **kwargs):
            raise RuntimeError("hook failed")

    with pytest.raises(RuntimeError, match="hook failed"):
        emit([BrokenHook()], "on_result", result={}, output_dir="unused")


def test_checkpoint_config_cannot_be_labeled_as_another_recipe(tmp_path):
    requested = _config(tmp_path, model="gnm")
    saved = _config(tmp_path, model="vint")
    checkpoint = tmp_path / "wrong_model.ckpt"
    torch.save({"hyper_parameters": {"cfg": OmegaConf.to_container(saved, resolve=True)}, "state_dict": {}}, checkpoint)
    with pytest.raises(ValueError, match="Checkpoint model config differs"):
        export_native(requested, tmp_path / "mislabeled.onnx", checkpoint=checkpoint, model_id="gnm")
    assert not (tmp_path / "mislabeled.onnx").exists()


def test_stale_export_metadata_is_rejected(tmp_path):
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"replaced graph")
    artifact.with_suffix(".metadata.json").write_text(json.dumps({"onnx_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="differs from its metadata"):
        _metadata(artifact)


def test_published_nomad_profile_uses_supplied_inputs(tmp_path, monkeypatch):
    from visnavkit.benchmark import published

    cfg = _config(tmp_path)
    cfg.command = "profile"
    cfg.source = "published"
    cfg.input_archive = str(tmp_path / "observations.npz")
    frames = np.full((1, 12, 96, 96), 0.25, dtype=np.float32)
    np.savez(cfg.input_archive, obs_img=frames)

    def benchmark(bundle_root, **kwargs):
        np.testing.assert_array_equal(kwargs["feeds"]["obs_img"], frames)
        return {"runtime": {"used_prepared_inputs": True}}

    monkeypatch.setattr(published, "benchmark_nomad", benchmark)
    result = run(cfg, "nomad")
    assert result["runtime"]["used_prepared_inputs"]


def test_cli_smoke_and_analytic_control(tmp_path):
    def cli(*overrides):
        completed = subprocess.run(
            [sys.executable, "-m", "visnavkit.scripts.benchmark", *overrides],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr

    smoke = tmp_path / "smoke"
    cli("command=smoke", "model=gnm", "common.crop_wh=[32,32]", "common.downscale_factor=1",
        "common.seq_length=2", "samples=2", "runtime.warmup=1", "runtime.iterations=2", f"output_dir={smoke}")
    report = json.loads((smoke / "result.json").read_text())
    assert report["benchmark_config"]["command"] == "smoke"
    assert "revision" in report["code"]
    assert report["results"][1]["metrics"]["samples"] == 2
    control = tmp_path / "control"
    cli("command=evaluate", "adapter.name=constant_velocity", f"dataset_archive={smoke / 'fixture.npz'}",
        f"output_dir={control}")
    report = json.loads((control / "result.json").read_text())
    assert report["result_kind"] == "pipeline_check"
    assert (control / "results.csv").is_file()
    for horizon in report["metrics"]["horizons"].values():
        assert horizon["count"] == 2
        assert horizon["top1_ade"] == pytest.approx(0, abs=1e-6)
        assert horizon["top1_fde"] == pytest.approx(0, abs=1e-6)
