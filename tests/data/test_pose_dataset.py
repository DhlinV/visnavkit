"""Pose-only windows: the reference 20 Hz slot grid without frames."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from visnavkit.data.pose_dataset import PoseDataModule, PoseWindowDataset, translate
from visnavkit.scripts.dataset.preprocess import find_clips, preprocess

ASSETS = Path(__file__).resolve().parents[2] / "assets" / "datasets"


def make_clip(tmp_path, fps=20, seconds=8.0, yaw_rate=0.0):
    """1 m/s from t = 0: along +x, or on a circle of radius 1 / yaw_rate turning left."""
    times = np.arange(int(seconds * fps) + 1, dtype=np.float64) / fps
    yaw = yaw_rate * times
    x, y = (np.sin(yaw) / yaw_rate, (1 - np.cos(yaw)) / yaw_rate) if yaw_rate else (times, np.zeros_like(times))
    np.save(tmp_path / "frame_times.npy", np.round(times * 1e9).astype(np.int64))
    np.save(tmp_path / "frame_positions.npy", np.column_stack((x, y, np.zeros_like(x))))
    np.save(tmp_path / "frame_orientations.npy", np.column_stack((np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2))))
    np.save(tmp_path / "frame_speeds.npy", np.ones_like(times))
    (tmp_path / "video.mp4").touch()
    (tmp_path / "train.txt").write_text(f"video.mp4 0 0 {len(times)}\n")
    return dict(
        file_list="train.txt", data_root=str(tmp_path), seq_len=20, hz=20, plan_len_seconds=4, plan_len_points=80
    )


def render_video(tmp_path, frames, fps, size="32x24"):
    """A blue h264 clip of ``frames`` frames at ``fps`` next to the sidecars, or skip without ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required for the frame tests")
    source = f"color=c=blue:s={size}:r={fps}"
    args = ["-v", "error", "-y", "-f", "lavfi", "-i", source, "-frames:v", str(frames), "-c:v", "libx264"]
    subprocess.run([ffmpeg, *args, "-pix_fmt", "yuv420p", str(tmp_path / "video.mp4")], check=True, capture_output=True)


def at_time(dataset, current):
    """The dataset's first clip with a single window whose current frame is ``current``."""
    dataset.windows = [(dataset.windows[0][0], current)]
    return dataset[0]


def test_windows_need_the_past_second_inside_the_row_and_four_seconds_of_future(tmp_path):
    ds = PoseWindowDataset(**make_clip(tmp_path))  # 8 s at 20 fps: t = 0.95, 1.95, 2.95, 3.95
    assert [current for _, current in ds.windows] == [19, 39, 59, 79]
    sample = ds[0]
    torch.testing.assert_close(sample["frame_times_s"], torch.arange(20, dtype=torch.float64) / 20)
    torch.testing.assert_close(sample["target_times_s"], torch.arange(1, 81, dtype=torch.float32) / 20)
    assert [current for _, current in PoseWindowDataset(**make_clip(tmp_path, fps=5)).windows] == [5, 10, 15, 20]


@pytest.mark.parametrize("fps", [20, 5, 2])
def test_straight_motion_is_frame_rate_independent(tmp_path, fps):
    sample = at_time(PoseWindowDataset(**make_clip(tmp_path, fps=fps)), 2 * fps)  # t = 2 s at every rate
    ego, future = sample["ego"], sample["future_poses"]
    assert ego.shape == (20, 5) and future.shape == (20, 80, 3)
    torch.testing.assert_close(ego[:, 0], torch.arange(-19, 1, dtype=torch.float32) / 20)  # past_xy x
    torch.testing.assert_close(
        ego[:, 1:], torch.tensor([[0.0, 0.0, 1.0, 0.0]]).expand(20, 4)
    )  # y, yaw, speed, yaw rate
    torch.testing.assert_close(future[-1, :, 0], torch.arange(1, 81, dtype=torch.float32) / 20)
    torch.testing.assert_close(future[..., 1], torch.zeros(20, 80))
    torch.testing.assert_close(future[..., 2], torch.ones(20, 80))
    torch.testing.assert_close(sample["frame_speeds"], torch.ones(20, 1))


def test_turns_give_heading_and_yaw_rate_and_mirror_with_the_flip(tmp_path):
    config = make_clip(tmp_path, yaw_rate=0.5)  # left turn, radius 2 m
    sample = at_time(PoseWindowDataset(**config), 40)
    ego, future = sample["ego"], sample["future_poses"]
    torch.testing.assert_close(ego[:, 2], torch.arange(-19, 1, dtype=torch.float32) / 20 * 0.5)  # heading
    torch.testing.assert_close(ego[:, 4], torch.full((20,), 0.5), atol=1e-3, rtol=0)  # yaw rate
    assert torch.all(ego[:-1, 0] < 0) and torch.all(ego[:-1, 1] > 0)  # the circle lies behind and to the left
    assert torch.all(future[..., 0] > 0) and torch.all(future[..., 1] > 0)

    flipped = at_time(PoseWindowDataset(**config, p_hflip=1.0), 40)
    torch.testing.assert_close(flipped["ego"][:, [1, 2, 4]], -ego[:, [1, 2, 4]])
    torch.testing.assert_close(flipped["ego"][:, [0, 3]], ego[:, [0, 3]])
    torch.testing.assert_close(flipped["future_poses"][..., 1], -future[..., 1])
    torch.testing.assert_close(flipped["future_poses"][..., [0, 2]], future[..., [0, 2]])
    kept = at_time(PoseWindowDataset(**config, p_hflip=1.0, no_flip=["video"]), 40)
    torch.testing.assert_close(kept["ego"], ego)


@pytest.mark.parametrize("goal_type", ["gps", "point"])
def test_goals_follow_the_goal_type(tmp_path, goal_type):
    config = make_clip(tmp_path, seconds=20.0)
    ds = PoseWindowDataset(**config, goal_type=goal_type, goal_horizon_s=(3.0, 5.0))
    goal = at_time(ds, 19)["goal"]  # straight ahead, 3-5 s past the current frame, 0.95 s farther for slot 0
    if goal_type == "gps":
        assert goal.shape == (20, 2)
        assert torch.all(goal[:, 0] >= 3.0 - 1e-5) and torch.all(goal[:, 0] <= 5.95 + 1e-5)
        torch.testing.assert_close(goal[:, 1], torch.zeros(20))
    else:
        assert goal.shape == (20, 3)
        torch.testing.assert_close(goal[:, 1:], torch.tensor([[1.0, 0.0]]).expand(20, 2))
    assert torch.equal(goal, ds[0]["goal"])  # deterministic per window
    assert "goal" not in PoseWindowDataset(**config)[0]
    with pytest.raises(ValueError, match="goal_type"):
        PoseWindowDataset(**config, goal_type="image")


def test_frames_fill_the_slots_that_hold_a_source_frame(tmp_path):
    config = make_clip(tmp_path, fps=5)  # 41 frames: slot k of the window at t = 4 s holds frame (k - 19) / 4 + 20
    render_video(tmp_path, 41, 5)
    labels = np.zeros((41, 8, 8), np.uint8)
    labels[:, :, 0] = (np.arange(41) % 3)[:, None]  # asymmetric: the flip test sees it
    np.save(tmp_path / "route_labels.npy", labels)
    sample = at_time(PoseWindowDataset(**config, frames=True, frame_wh=(16, 12), route_hw=(8, 8)), 20)
    expected = torch.zeros(20, dtype=torch.bool)
    expected[[3, 7, 11, 15, 19]] = True
    assert torch.equal(sample["frame_mask"], expected) and torch.equal(sample["route_mask"], expected)
    assert sample["vision"].shape == (20, 3, 12, 16) and sample["vision"].dtype == torch.uint8
    assert torch.all(sample["vision"][~expected] == 0) and torch.all(
        sample["vision"][expected].float().mean((1, 2, 3)) > 0
    )
    assert sample["route_patch"].shape == (20, 8, 8)
    torch.testing.assert_close(sample["route_patch"][expected, 0, 0], torch.tensor([16.0, 17, 18, 19, 20]) % 3)
    assert torch.all(sample["route_patch"][~expected] == 0)

    flipped = at_time(PoseWindowDataset(**config, frames=True, route_hw=(8, 8), p_hflip=1.0), 20)
    torch.testing.assert_close(flipped["route_patch"][expected, 0, -1], torch.tensor([16.0, 17, 18, 19, 20]) % 3)
    assert flipped["vision"].shape == (20, 3, 24, 32)
    (tmp_path / "route_labels.npy").unlink()
    np.savez_compressed(tmp_path / "route_labels.npz", labels=labels)  # the compressed sidecar reads the same
    compressed = at_time(PoseWindowDataset(**config, route_hw=(8, 8)), 20)
    assert torch.equal(compressed["route_patch"], sample["route_patch"])
    assert torch.equal(compressed["route_mask"], expected)
    (tmp_path / "route_labels.npz").unlink()
    plain = at_time(PoseWindowDataset(**make_clip(tmp_path), frames=True), 40)  # 20 fps: every slot
    assert plain["frame_mask"].all() and "route_patch" not in plain
    bare = at_time(PoseWindowDataset(**config, route_hw=(8, 8)), 20)  # no sidecar: zeros, never real
    assert not bare["route_mask"].any() and torch.all(bare["route_patch"] == 0) and "vision" not in bare


def test_pose_size_five_adds_heading_and_yaw_rate(tmp_path):
    config = make_clip(tmp_path, yaw_rate=0.5)
    future = at_time(PoseWindowDataset(**config, pose_size=5), 40)["future_poses"]
    assert future.shape == (20, 80, 5)
    torch.testing.assert_close(future[-1, :, 2], torch.arange(1, 81) / 20 * 0.5)  # heading relative to the slot
    torch.testing.assert_close(future[..., 3], torch.ones(20, 80))
    torch.testing.assert_close(future[..., 4], torch.full((20, 80), 0.5), atol=1e-3, rtol=0)
    flipped = at_time(PoseWindowDataset(**config, pose_size=5, p_hflip=1.0), 40)["future_poses"]
    torch.testing.assert_close(flipped[..., [1, 2, 4]], -future[..., [1, 2, 4]])
    assert at_time(PoseWindowDataset(**config, pose_size=2), 40)["future_poses"].shape == (20, 80, 2)
    with pytest.raises(ValueError, match="pose_size"):
        PoseWindowDataset(**config, pose_size=4)


def test_embodiment_and_bounds_follow_the_corpus_directory(tmp_path):
    corpus = tmp_path / "robot_a"
    corpus.mkdir()
    make_clip(corpus)
    (tmp_path / "train.txt").write_text("robot_a/video.mp4 0 0 161\n")
    bounds = {"robot_a": [[-0.1, -0.1, -0.1, 0.0, -1.0], [0.3, 0.1, 0.1, 3.0, 1.0]]}
    (tmp_path / "action_bounds.json").write_text(json.dumps(bounds))
    config = dict(file_list="train.txt", data_root=str(tmp_path), action_bounds="action_bounds.json")
    sample = PoseWindowDataset(**config, embodiment_ids={"robot_a": 3})[0]
    assert sample["embodiment_id"].item() == 3 and sample["embodiment_id"].dtype == torch.long
    torch.testing.assert_close(sample["action_bounds"], torch.tensor(bounds["robot_a"]))
    with pytest.raises(ValueError, match="robot_a"):
        PoseWindowDataset(**config, embodiment_ids={"robot_b": 0})


def test_datamodule_batches(tmp_path):
    loader = OmegaConf.create({**make_clip(tmp_path), "shuffle": False})
    module = PoseDataModule(train_loader=loader, val_loader=loader, batch_size=4, num_workers=0, pin_memory=False)
    module.setup("fit")
    batch = next(iter(module.val_dataloader()))
    assert batch["ego"].shape == (4, 20, 5) and batch["future_poses"].shape == (4, 20, 80, 3)
    assert batch["target_times_s"].shape == (4, 80) and batch["frame_speeds"].shape == (4, 20, 1)


def _corpora() -> list[Path]:
    if not ASSETS.is_dir():
        return []
    return sorted(path for path in ASSETS.iterdir() if path.is_dir() and find_clips(path))


@pytest.mark.parametrize("corpus", _corpora() or [None], ids=lambda p: p.name if p else "none-bundled")
def test_bundled_corpus_yields_finite_windows(corpus, tmp_path):
    if corpus is None:
        pytest.skip(f"No corpora under {ASSETS}; drop one in and this covers it")
    preprocess(corpus, output_dir=tmp_path, val_fraction=0.0)
    ds = PoseWindowDataset(file_list="train.txt", data_root=tmp_path, goal_type="gps", frames=True, frame_wh=(96, 54))
    assert len(ds) > 0, f"{corpus.name} produced no windows"
    sample = ds[len(ds) // 2]
    assert sample["ego"].shape == (20, 5) and sample["future_poses"].shape == (20, 80, 3)
    assert sample["vision"].shape == (20, 3, 54, 96) and sample["frame_mask"][-1]  # the current frame is real
    assert all(torch.isfinite(value).all() for value in sample.values())


def test_translate_moves_the_content_by_the_offset_with_zeros_outside():
    frames = torch.zeros(1, 3, 6, 8, dtype=torch.uint8)
    frames[..., 2, 3] = 200
    moved = translate(frames, 1.0, -2.0)  # out(x, y) = in(x + 1, y - 2): the dot moves to (2, 4)
    assert moved[0, 0].nonzero().tolist() == [[4, 2]] and int(moved[0, 0, 4, 2]) == 200
    half = translate(frames, 0.5, 0.0)  # bilinear: split over x = 2, 3
    assert int(half[0, 0, 2, 2]) == 100 and int(half[0, 0, 2, 3]) == 100
    assert torch.equal(translate(frames, 0.0, 0.0), frames)
    assert not translate(frames, 9.0, 0.0).any()


def test_camera_is_scaled_to_the_frame_and_the_calibration_recentres_the_frames(tmp_path):
    config = make_clip(tmp_path)
    render_video(tmp_path, 161, 20, size="32x24")
    camera = [20.0, 20.0, 16.0, 12.0, -0.03, 0.003, 0.0, 0.0, 0.456, 1.0]
    meta = {"camera": camera, "width": 32, "height": 24, "principal_point_delta": [2.0, -1.0]}
    (tmp_path / "camera.json").write_text(json.dumps(meta))
    kw = dict(config, frames=True, frame_wh=(16, 12))
    plain = at_time(PoseWindowDataset(**kw, camera=True), 40)  # the image keeps its offset: cx, cy follow it
    torch.testing.assert_close(plain["camera"], torch.tensor([10.0, 10.0, 9.0, 5.5, -0.03, 0.003, 0, 0, 0.456, 1.0]))
    calibrated = at_time(PoseWindowDataset(**kw, camera=True, principal_point_calibration=True), 40)
    torch.testing.assert_close(calibrated["camera"][:4], torch.tensor([10.0, 10.0, 8.0, 6.0]))  # nominal
    torch.testing.assert_close(calibrated["vision"], translate(plain["vision"], 1.0, -0.5))  # the offset at 16 x 12
    flipped = at_time(PoseWindowDataset(**kw, camera=True, p_hflip=1.0), 40)
    assert float(flipped["camera"][2]) == 16 - 9.0
    torch.testing.assert_close(plain["past_poses"][:, 0], torch.arange(-0.95, 0.01, 0.05, dtype=torch.float32))
    with pytest.raises(ValueError, match="frame_wh"):
        PoseWindowDataset(**config, camera=True)
