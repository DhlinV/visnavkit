"""Benchmark the DALI vs torch dataloaders on the same dataset semantics.

Usage:
    uv run python -m navigators.scripts.benchmark_dataloader [--datasets dali torch] [--warmup 5] [--batches 50] \
        [hydra overrides, e.g. common.data_root=/data/nav_clips dataset.batch_size=32]

Overrides are applied to every benchmarked dataset config. Throughput for the torch loader
includes the H2D copy (DALI batches already land on GPU) so numbers are comparable.
"""

import argparse
import time

import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate


def iterate_forever(loader):
    while True:
        yield from loader


def bench(dataset_name: str, overrides: list[str], warmup: int, batches: int, device: str):
    with initialize_config_module(version_base=None, config_module="navigators.configs"):
        cfg = compose(config_name="train", overrides=[f"dataset={dataset_name}", *overrides])
    datamodule = instantiate(cfg.dataset)
    datamodule.setup("fit")

    it = iterate_forever(datamodule.train_dataloader())
    t0 = time.perf_counter()
    first = next(it)
    ttfb = time.perf_counter() - t0
    frames = first["frames"]
    for _ in range(warmup):
        next(it)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_samples = 0
    for _ in range(batches):
        x = next(it)["frames"]
        if device == "cuda" and x.device.type == "cpu":
            x = x.to(device, non_blocking=True)
        n_samples += x.shape[0]
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    print(
        f"{dataset_name:>6}: {n_samples / dt:8.1f} samples/s | {batches / dt:6.2f} batches/s | "
        f"first batch {ttfb:5.2f}s | frames {tuple(frames.shape)} {frames.dtype} on {frames.device}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=["dali", "torch"], help="dataset config names to benchmark")
    parser.add_argument("--warmup", type=int, default=5, help="untimed warmup batches")
    parser.add_argument("--batches", type=int, default=50, help="timed batches")
    args, overrides = parser.parse_known_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name in args.datasets:
        bench(name, overrides, args.warmup, args.batches, device)


if __name__ == "__main__":
    main()
