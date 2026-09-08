import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip("torchcodec")

from visnavkit.data.file_list import parse_file_list_frame_ranges, write_file_list_frame_ranges
from visnavkit.data.mp4_dataset import Mp4WindowDataset
from visnavkit.data.pose_targets import target_safe_frame_ranges
from visnavkit.data.torch_datamodule import TorchDataModule


def make_episode(tmp_path, fps=20, end=None, start=0):
    count = 5 * fps + 1
    times = np.arange(count, dtype=np.float64) / fps
    np.save(tmp_path / "frame_times.npy", np.round(times * 1e9).astype(np.int64))
    np.save(tmp_path / "frame_positions.npy", np.column_stack((times, np.zeros((count, 2)))))
    np.save(tmp_path / "frame_orientations.npy", np.tile([1.0, 0.0, 0.0, 0.0], (count, 1)))
    np.save(tmp_path / "frame_speeds.npy", np.ones(count))
    (tmp_path / "video.mp4").touch()
    (tmp_path / "val.txt").write_text(f"video.mp4 0 {start} {count if end is None else end}\n")
    return dict(
        file_list="val.txt",
        data_root=str(tmp_path),
        seq_len=3,
        plan_len_seconds=3,
        plan_len_points=4,
        steps_between_samples=20,
        downscale_factor=1,
        crop_wh=(32, 24),
        use_augs=False,
    )


@pytest.mark.parametrize("fps, expected", [(20, [0, 20]), (30, [0, 20, 40])])
def test_window_index_excludes_incomplete_targets(tmp_path, fps, expected):
    ds = Mp4WindowDataset(**make_episode(tmp_path, fps))
    assert [start for _, start in ds.windows] == expected


def test_existing_observation_caps_and_start_offsets_preserved(tmp_path):
    ds = Mp4WindowDataset(**make_episode(tmp_path, start=10, end=35))
    assert [start for _, start in ds.windows] == [10, 30]


def test_invalid_metadata_fails_at_index_construction(tmp_path):
    config = make_episode(tmp_path)
    np.save(tmp_path / "frame_orientations.npy", np.zeros((101, 4)))
    with pytest.raises(ValueError, match="degenerate quaternion"):
        Mp4WindowDataset(**config)


def test_normalized_manifest_preserves_paths_labels_and_capped_ranges(tmp_path):
    make_episode(tmp_path, start=10, end=35)
    rows = parse_file_list_frame_ranges("val.txt", data_root=tmp_path)
    safe = target_safe_frame_ranges(rows, 3.0)
    destination = tmp_path / "resolved.txt"
    write_file_list_frame_ranges(safe, destination)
    assert destination.read_text() == f"{tmp_path / 'video.mp4'} 0 10 35\n"
    assert parse_file_list_frame_ranges(destination) == rows


def test_validation_retains_small_and_partial_batches():
    config = OmegaConf.create({"file_list": "unused"})
    module = TorchDataModule(train_loader=config, val_loader=config, batch_size=32, num_workers=0, pin_memory=False)
    module._per_gpu_batch_size = 32
    module.val_dataset = list(range(37))
    batches = list(module.val_dataloader())
    assert [len(batch) for batch in batches] == [32, 5]
    assert torch.cat(batches).tolist() == list(range(37))
    module.val_dataset = list(range(5))
    assert len(list(module.val_dataloader())[0]) == 5


def test_real_video_decoding_and_target_alignment(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required for the real-video integration test")
    config = make_episode(tmp_path)
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=32x24:r=20",
            "-frames:v",
            "101",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(tmp_path / "video.mp4"),
        ],
        check=True,
        capture_output=True,
    )
    ds = Mp4WindowDataset(**config)
    sample = ds[-1]
    assert sample["frames"].shape == (3, 6, 24, 32)
    assert sample["frames"].dtype == torch.uint8
    torch.testing.assert_close(sample["future_poses"][:, -1, 0], torch.full((3,), 3.0))
    torch.testing.assert_close(sample["frame_times_s"], torch.tensor([1.05, 1.1, 1.15], dtype=torch.float64))
    torch.testing.assert_close(sample["target_times_s"], torch.tensor([0, 1 / 3, 4 / 3, 3]))
    assert Path(ds.windows[0][0]).is_absolute()


def test_dali_and_torch_relative_manifest_target_and_tail_parity(tmp_path):
    pytest.importorskip("nvidia.dali")
    if not torch.cuda.is_available():
        pytest.skip("DALI video decoding requires a CUDA GPU")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required for the real-video integration test")
    from visnavkit.data.dali_datamodule import DaliDataModule

    config = make_episode(tmp_path)
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:r=20",
            "-frames:v",
            "101",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(tmp_path / "video.mp4"),
        ],
        check=True,
        capture_output=True,
    )
    expected = Mp4WindowDataset(**config)
    loader_config = OmegaConf.create({**config, "num_threads": 1, "frame_wh": [64, 64]})
    module = DaliDataModule(train_loader=loader_config, val_loader=loader_config, batch_size=3)
    module.setup("validate")
    batches = list(module.val_dataloader())
    assert len(batches) == 1
    batch = batches[0]
    assert batch["frames"].shape == (2, 3, 6, 24, 32)
    assert batch["reader_start_frame"].flatten().tolist() == [0, 20]
    torch.testing.assert_close(
        batch["future_poses"].cpu(), torch.stack([expected[i]["future_poses"] for i in range(2)])
    )
    torch.testing.assert_close(
        batch["target_times_s"].cpu(), torch.stack([expected[i]["target_times_s"] for i in range(2)])
    )


def test_prepare_dataset_keeps_final_context_target_and_provenance(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required for the real-video integration test")
    from visnavkit.benchmark.datasets import load_archive
    from visnavkit.benchmark.prepare import prepare_dataset

    config = make_episode(tmp_path)
    config["offset_t_anchors"] = True
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=32x24:r=20",
            "-frames:v",
            "101",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(tmp_path / "video.mp4"),
        ],
        check=True,
        capture_output=True,
    )
    cfg = OmegaConf.create({"dataset": {"val_loader": config}})
    destination = tmp_path / "prepared.npz"
    metadata = prepare_dataset(cfg, destination, limit=1)
    data, sidecar = load_archive(destination)
    assert data["frames"].shape == (1, 3, 6, 24, 32)
    assert data["targets"].shape == (1, 4, 2)
    np.testing.assert_allclose(data["target_times_s"], [3 / 16, 3 / 4, 27 / 16, 3])
    np.testing.assert_allclose(data["targets"][0, :, 0], data["target_times_s"])
    np.testing.assert_allclose(data["current_speed"], 1)
    assert data["sample_ids"].tolist() == [f"{tmp_path / 'video.mp4'}:3"]
    assert metadata == sidecar
    assert metadata["dataset_kind"] == "real_prepared"
    assert metadata["available_windows"] == 2
    assert metadata["samples"] == 1
    assert len(metadata["archive_sha256"]) == 64
    assert len(metadata["source_files"]) == 5


def test_prepare_dataset_rejects_empty_data_and_invalid_limits(tmp_path):
    from visnavkit.benchmark.prepare import prepare_dataset

    config = make_episode(tmp_path, start=90)
    cfg = OmegaConf.create({"dataset": {"val_loader": config}})
    with pytest.raises(ValueError, match="No valid evaluation windows"):
        prepare_dataset(cfg, tmp_path / "empty.npz")
    with pytest.raises(ValueError, match="limit"):
        prepare_dataset(cfg, tmp_path / "empty.npz", limit=0)
