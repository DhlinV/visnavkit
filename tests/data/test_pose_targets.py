import numpy as np
import pytest

from visnavkit.data.pose_targets import get_future_poses


def episode(times):
    times = np.asarray(times, dtype=np.float64)
    positions = np.column_stack((times - times[0], np.zeros((len(times), 2))))
    quats = np.tile([1.0, 0.0, 0.0, 0.0], (len(times), 1))
    return positions, quats, np.ones(len(times)), times


def targets(arrays, indices=(0,), anchors=(0.0, 3.0), **kwargs):
    return get_future_poses(
        *arrays,
        np.asarray(indices),
        np.asarray(anchors) if anchors is not None else None,
        num_pts=kwargs.pop("num_pts", 60),
        use_full_pose=kwargs.pop("use_full_pose", False),
        strict=kwargs.pop("strict", True),
        interp_to_end=kwargs.pop("interp_to_end", False),
        **kwargs,
    )


@pytest.mark.parametrize("fps", [20, 30])
def test_three_second_target_independent_of_frame_rate(fps):
    arrays = episode(np.arange(5 * fps + 1) / fps)
    poses, valid = targets(arrays, indices=(3, 10), num_pts=2)
    np.testing.assert_allclose(poses[:, -1, 0], 3.0, atol=1e-6)
    np.testing.assert_allclose(poses[:, 0], 0.0, atol=1e-6)
    assert valid == 2


def test_irregular_timestamps_include_bracketing_frame():
    poses, _ = targets(episode([0.0, 0.3, 1.2, 2.2, 3.3, 4.9]), anchors=[0.2, 0.9, 3.0], num_pts=1)
    np.testing.assert_allclose(poses[0, :, 0], [0.2, 0.9, 3.0], atol=1e-6)


def test_scaled_quaternion_is_normalized_and_local_frame_is_correct():
    arrays = list(episode(np.arange(5.0)))
    # Heading +90 degrees: global +x is local -y.
    arrays[1][:] = np.array([np.sqrt(0.5), 0, 0, np.sqrt(0.5)]) * 2
    poses, _ = targets(arrays, use_full_pose=True)
    np.testing.assert_allclose(poses[0, -1], [0.0, -3.0, 1.0], atol=1e-6)


@pytest.mark.parametrize(
    "invalid", ["zero_quat", "nan_position", "nan_speed", "duplicate_time", "backward_time", "length"]
)
def test_invalid_episode_metadata_rejected(invalid):
    arrays = list(episode(np.arange(5.0)))
    if invalid == "zero_quat":
        arrays[1][0] = 0
    elif invalid == "nan_position":
        arrays[0][0, 0] = np.nan
    elif invalid == "nan_speed":
        arrays[2][0] = np.nan
    elif invalid == "duplicate_time":
        arrays[3][1] = 0
    elif invalid == "backward_time":
        arrays[3][1] = -1
    else:
        arrays[2] = arrays[2][:-1]
    with pytest.raises(ValueError):
        targets(arrays)


def test_insufficient_future_raises_or_returns_shaped_valid_prefix():
    arrays = episode(np.arange(5.0))
    with pytest.raises(ValueError, match="insufficient future"):
        targets(arrays, indices=(0, 2))
    poses, valid = targets(arrays, indices=(0, 2), strict=False)
    assert poses.shape == (1, 2, 2)
    assert valid == 1
    poses, valid = targets(arrays, indices=(2,), strict=False)
    assert poses.shape == (0, 2, 2)
    assert valid == 0


def test_interpolate_to_end_starts_at_current_frame():
    poses, valid = targets(episode(np.arange(5.0)), indices=(2,), anchors=None, interp_to_end=True, num_pts=3)
    np.testing.assert_allclose(poses[0, :, 0], [0.0, 1.0, 2.0])
    assert valid == 1


def test_epoch_timestamp_endpoint_rounding():
    poses, _ = targets(episode(1_700_000_000 + np.arange(101) / 20), indices=(40,))
    assert poses[0, -1, 0] == pytest.approx(3.0, abs=1e-6)


@pytest.mark.parametrize("anchors", [[-1, 3], [1, 1], [2, 1], [0, np.nan]])
def test_invalid_target_times_rejected(anchors):
    with pytest.raises(ValueError, match="t_anchors"):
        targets(episode(np.arange(5.0)), anchors=anchors)


def test_goal_frame_sampling_is_deterministic_and_bounded():
    from visnavkit.data.pose_targets import goal_point_targets, sample_goal_frame

    times = np.arange(0, 30, 0.5)
    first = sample_goal_frame(times, 10, (3.0, 10.0), "clip:10")
    assert first == sample_goal_frame(times, 10, (3.0, 10.0), "clip:10")
    assert 3.0 <= times[first] - times[10] <= 10.0
    assert sample_goal_frame(times, 55, (3.0, 10.0), "clip:55") == 59  # falls back to the final frame
    with pytest.raises(ValueError, match="goal_horizon_s"):
        sample_goal_frame(times, 0, (5.0, 1.0), "clip")
    positions = np.column_stack([times, np.zeros_like(times), np.zeros_like(times)])
    orientations = np.tile([1.0, 0.0, 0.0, 0.0], (len(times), 1))
    goal = goal_point_targets(positions, orientations, [10, 12], 20)
    np.testing.assert_allclose(goal, [[5.0, 1.0, 0.0], [4.0, 1.0, 0.0]])
    # yaw 90 degrees: a goal straight ahead in odometry is to the right in the ego frame
    turned = np.tile([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)], (len(times), 1))
    np.testing.assert_allclose(goal_point_targets(positions, turned, [10], 20), [[5.0, 0.0, -1.0]], atol=1e-6)
