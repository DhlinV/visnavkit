"""preprocess -> cache -> stats -> anchors -> visualize, on a synthetic two-clip corpus."""

import shutil
import subprocess

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module

pytest.importorskip("torchcodec")

from visnavkit.scripts.dataset.anchors import fit_anchors, kmeans
from visnavkit.scripts.dataset.cache import cache_targets, load_cache
from visnavkit.scripts.dataset.cli import run
from visnavkit.scripts.dataset.preprocess import find_clips, preprocess
from visnavkit.scripts.dataset.stats import fit_stats
from visnavkit.scripts.dataset.visualize import describe_sample

SMALL = [
    "common.seq_length=3",
    "common.crop_wh=[32,24]",
    "common.downscale_factor=1",
    "plan_len_points=4",
    "dataset.train_loader.steps_between_samples=20",
    "dataset.val_loader.steps_between_samples=20",
]


def _clip(root, name, fps=20, curve=0.0, speed=1.0):
    """A clip with the four sidecars; `curve` bends the path and `speed` sets its pace (m/s)."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    count = 5 * fps + 1
    times = np.arange(count, dtype=np.float64) / fps
    np.save(directory / "frame_times.npy", np.round(times * 1e9).astype(np.int64))
    np.save(
        directory / "frame_positions.npy",
        np.column_stack((speed * times, curve * times**2, np.zeros(count))),
    )
    np.save(directory / "frame_orientations.npy", np.tile([1.0, 0.0, 0.0, 0.0], (count, 1)))
    np.save(directory / "frame_speeds.npy", np.full(count, speed))
    (directory / "video.mp4").touch()
    return directory


def _render(directory, size="32x24"):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required to decode frames")
    subprocess.run(
        # fmt: off
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={size}:rate=20:duration=6",
            "-pix_fmt",
            "yuv420p",
            str(directory / "video.mp4"),
        ],
        # fmt: on
        check=True,
        capture_output=True,
    )


def _cfg(root, *overrides):
    with initialize_config_module(version_base=None, config_module="visnavkit.configs"):
        return compose(
            config_name="dataset_tools",
            overrides=["dataset=torch", f"common.data_root={root}", *SMALL, *overrides],
        )


def test_preprocess_validates_clips_and_writes_manifests(tmp_path):
    _clip(tmp_path, "clip_a")
    _clip(tmp_path, "clip_b", curve=0.05)
    broken = _clip(tmp_path, "clip_bad")
    np.save(broken / "frame_times.npy", np.zeros(101, dtype=np.int64))  # not strictly increasing

    assert len(find_clips(tmp_path)) == 3
    report = preprocess(tmp_path, val_fraction=0.5, seed=0)
    assert report["counts"] == {"train": 1, "val": 1}
    assert list(report["rejected"]) == [str(broken)]
    rows = (tmp_path / "train.txt").read_text().split()
    assert rows[1] == "0" and rows[2] == "0" and rows[3] == "101"

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No clips"):
        preprocess(empty)
    with pytest.raises(FileNotFoundError, match="not a directory"):
        preprocess(tmp_path / "missing")


def test_cache_stats_and_anchors_chain(tmp_path):
    _clip(tmp_path, "clip_a")
    _clip(tmp_path, "clip_b", curve=0.05)
    preprocess(tmp_path, val_fraction=0.0, seed=0)
    (tmp_path / "val.txt").write_text((tmp_path / "train.txt").read_text())
    cfg = _cfg(tmp_path, f"output_dir={tmp_path / 'out'}")

    path = cache_targets(cfg, "train", cfg.output_dir)
    cache = load_cache(cfg.output_dir, "train")
    assert path.exists()
    windows = cache["future_poses"].shape[0]
    assert cache["future_poses"].shape == (windows, 3, 4, 3)  # windows, frames, anchors, (x, y, v)
    assert cache["frame_speeds"].shape == (windows, 3) and len(set(cache["clips"].tolist())) == 2
    # Straight clip at 1 m/s: the furthest anchor sits ~3 s ahead.
    assert cache["future_poses"][..., -1, 0].max() == pytest.approx(3.0, abs=0.2)

    stats = fit_stats(cfg, cache, "meanstd", cfg.output_dir, "corpus_a")
    with np.load(stats) as loaded:
        assert loaded["mean"].shape == (4, 3) and (loaded["std"] > 0).all()

    anchors = fit_anchors(cache, num_anchors=4, output_dir=cfg.output_dir, seed=0)
    with np.load(anchors) as loaded:
        assert loaded["anchors"].shape == (4, 4, 3)
        assert np.isfinite(loaded["anchors"]).all()


def test_stats_are_per_corpus(tmp_path):
    """Two corpora keep separate statistics, which is what multi-dataset training needs."""
    fast, slow = tmp_path / "fast", tmp_path / "slow"
    _clip(fast, "clip", speed=2.0)
    _clip(slow, "clip", speed=0.5)
    means = {}
    for root in (fast, slow):
        preprocess(root, val_fraction=0.0, seed=0)
        (root / "val.txt").write_text((root / "train.txt").read_text())
        cfg = _cfg(root, f"output_dir={root / 'out'}")
        cache_targets(cfg, "train", cfg.output_dir)
        stats = fit_stats(cfg, load_cache(cfg.output_dir, "train"), "meanstd", cfg.output_dir, root.name)
        with np.load(stats) as loaded:
            means[root.name] = loaded["mean"]
    # 2 m/s reaches four times as far as 0.5 m/s at the same anchor time.
    assert means["fast"][-1, 0] == pytest.approx(4 * means["slow"][-1, 0], rel=0.05)
    assert (fast / "out" / "fast_meanstd.npz").exists() and (slow / "out" / "slow_meanstd.npz").exists()


def test_kmeans_recovers_separated_clusters():
    torch.manual_seed(0)
    points = torch.cat(
        [torch.randn(64, 2) * 0.05 + centre for centre in (torch.tensor([-3.0, 0.0]), torch.tensor([3.0, 0.0]))]
    )
    centres, assignment = kmeans(points, 2, seed=0)
    assert sorted(int(c) for c in torch.bincount(assignment)) == [64, 64]
    assert sorted(round(float(c[0])) for c in centres) == [-3, 3]
    with pytest.raises(ValueError, match="at least 5 points"):
        kmeans(torch.randn(3, 2), 5)


def test_visualize_and_cli_describe_every_input(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    _clip(tmp_path, "clip_a")
    _render(tmp_path / "clip_a")
    preprocess(tmp_path, val_fraction=0.0, seed=0)
    (tmp_path / "val.txt").write_text((tmp_path / "train.txt").read_text())
    cfg = _cfg(
        tmp_path,
        f"output_dir={tmp_path / 'out'}",
        "command=visualize",
        "samples=2",
        "model/goal_encoder=gps",
        "common.ego_features=[speed]",
    )
    path = run(cfg)
    assert path.exists()
    output = capsys.readouterr().out
    assert "goal_types=['gps'] ego_features=['speed']" in output
    assert "vision         (3, 3, 24, 32)" in output and "ego            (3, 1)" in output

    from visnavkit.scripts.dataset.cache import build_dataset

    dataset = build_dataset(cfg, "train")
    assert "sample 0" in describe_sample(dataset[0], 0, dataset)
