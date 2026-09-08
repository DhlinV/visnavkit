"""Hydra entry point for model profiling, open-loop metrics, and benchmark suites."""

import subprocess

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from visnavkit.benchmark.hooks import emit
from visnavkit.benchmark.runner import run


@hydra.main(version_base=None, config_path="../configs", config_name="benchmark")
def main(cfg: DictConfig):
    model_id = cfg.model_id or HydraConfig.get().runtime.choices.get("model", "base")
    hooks = [instantiate(item) for item in cfg.hooks]
    emit(hooks, "on_start", config=cfg)
    try:
        result = run(cfg, model_id)
        result["benchmark_config"] = OmegaConf.to_container(cfg, resolve=True)
        try:
            revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
            status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
            result["code"] = {"revision": revision.stdout.strip(), "dirty": bool(status.stdout.strip())}
        except (OSError, subprocess.CalledProcessError):
            result["code"] = {"revision": None, "dirty": None}
        emit(hooks, "on_result", result=result, output_dir=cfg.output_dir)
    except Exception as error:
        emit(hooks, "on_error", error=error, config=cfg)
        raise
    if cfg.command == "list":
        for model in result["models"]:
            print(f"{model['id']:12} {model['status']:22} {model['name']}")
    print(f"Benchmark report: {cfg.output_dir}/result.json")


if __name__ == "__main__":
    main()
