"""Corpus tooling entry point.

    uv run visnavkit-dataset command=preprocess                     # validate clips, write manifests
    uv run visnavkit-dataset command=cache    dataset=torch         # window targets, no video decode
    uv run visnavkit-dataset command=stats    dataset=torch         # normalizer NPZ + distribution plot
    uv run visnavkit-dataset command=anchors  dataset=torch         # k-means anchor vocabulary
    uv run visnavkit-dataset command=visualize dataset=torch        # what the policy is actually fed

``stats``, ``anchors`` and ``plot`` reuse the cache and build it on demand. Every output is named
after ``name``, which defaults to the dataset config, so several corpora keep separate statistics.
"""

import hydra
from omegaconf import DictConfig

from visnavkit.scripts.dataset.anchors import fit_anchors
from visnavkit.scripts.dataset.cache import cache_path, cache_targets, load_cache
from visnavkit.scripts.dataset.preprocess import preprocess
from visnavkit.scripts.dataset.stats import fit_stats, plot_distributions
from visnavkit.scripts.dataset.visualize import visualize

COMMANDS = ("preprocess", "cache", "stats", "anchors", "plot", "visualize")


def _cache(cfg):
    if not cache_path(cfg.output_dir, cfg.split).exists():
        cache_targets(cfg, cfg.split, cfg.output_dir)
    return load_cache(cfg.output_dir, cfg.split)


def run(cfg: DictConfig):
    command = cfg.command
    if command not in COMMANDS:
        raise ValueError(f"command must be one of {COMMANDS}, got {command!r}")
    if command == "preprocess":
        return preprocess(cfg.common.data_root, cfg.get("manifest_dir"), cfg.val_fraction, cfg.seed)
    if command == "cache":
        return cache_targets(cfg, cfg.split, cfg.output_dir)
    if command == "visualize":
        return visualize(cfg, cfg.split, cfg.samples, cfg.output_dir)
    cache = _cache(cfg)
    if command == "anchors":
        return fit_anchors(cache, cfg.num_anchors, cfg.output_dir, cfg.seed)
    if command == "plot":
        return plot_distributions(cfg, cache, cfg.output_dir, cfg.name)
    path = fit_stats(cfg, cache, cfg.normalizer_mode, cfg.output_dir, cfg.name)
    plot_distributions(cfg, cache, cfg.output_dir, cfg.name)
    return path


@hydra.main(version_base=None, config_path="../../configs", config_name="dataset_tools")
def main(cfg: DictConfig):
    return run(cfg)


if __name__ == "__main__":
    main()
