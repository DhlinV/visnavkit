"""Timestamp-aligned ego-frame targets for the legacy MP4/NumPy episode format."""

import logging
from pathlib import Path

import numpy as np

from visnavkit.utils.common import load_npy
from visnavkit.utils.orientation import rot_from_quat

logger = logging.getLogger(__name__)

FRAME_POSITIONS = "frame_positions.npy"
FRAME_TIMES_S = "frame_times.npy"
FRAME_ORIENTATIONS = "frame_orientations.npy"
FRAME_SPEEDS = "frame_speeds.npy"


def validate_pose_arrays(frame_positions, frame_orientations, frame_speeds, frame_times_s):
    """Validate positions (metres), wxyz quaternions, speeds and timestamps (seconds)."""
    arrays = tuple(np.asarray(value) for value in (frame_positions, frame_orientations, frame_speeds, frame_times_s))
    positions, orientations, speeds, times = arrays
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("frame_times_s must be a nonempty 1D array")
    n = len(times)
    if positions.shape != (n, 3) or orientations.shape != (n, 4) or speeds.shape != (n,):
        raise ValueError("pose array lengths/shapes must agree: positions (N,3), orientations (N,4), speeds (N,)")
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("pose arrays must contain only finite values")
    if np.any(np.diff(times) <= 0):
        raise ValueError("frame_times_s must be strictly increasing")
    if np.any(np.linalg.norm(orientations, axis=1) <= 1e-6):
        raise ValueError("frame_orientations contains a degenerate quaternion")
    return arrays


def load_pose_arrays(sample_dir: Path):
    """Load and validate an episode; frame_times.npy contains nanoseconds."""
    sample_dir = Path(sample_dir)
    return validate_pose_arrays(
        load_npy(sample_dir / FRAME_POSITIONS),
        load_npy(sample_dir / FRAME_ORIENTATIONS),
        load_npy(sample_dir / FRAME_SPEEDS),
        np.asarray(load_npy(sample_dir / FRAME_TIMES_S), dtype=np.float64) / 1e9,
    )


def _timestamp_tolerance(times):
    # Allow only rounding incurred by converting absolute epoch timestamps to seconds.
    return max(1e-9, float(np.spacing(np.max(np.abs(times)))) * 4)


def target_safe_frame_ranges(rows, horizon_s):
    """Cap observation ranges to frames with complete future labels.

    Existing manifest ends remain observation caps. Future labels may use the rest of
    the episode, supporting manifests already manually trimmed for target availability.
    """
    if not np.isfinite(horizon_s) or horizon_s < 0:
        raise ValueError("horizon_s must be finite and nonnegative")
    episode_ends = {}
    safe_rows = []
    for video_fp, label, start, end in rows:
        sample_dir = Path(video_fp).parent
        if sample_dir not in episode_ends:
            *_, times = load_pose_arrays(sample_dir)
            relative_times = times - times[0]
            safe_end = np.searchsorted(
                relative_times, relative_times[-1] - horizon_s + _timestamp_tolerance(times), side="right"
            )
            episode_ends[sample_dir] = (len(times), int(safe_end))
        n_frames, safe_end = episode_ends[sample_dir]
        if start < 0 or end <= start or end > n_frames:
            raise ValueError(f"Invalid frame range [{start}, {end}) for {video_fp}: metadata has {n_frames} frames")
        safe_rows.append((video_fp, label, start, min(end, safe_end)))
    return safe_rows


def get_current_frame_idxs(start_idx, frame_step, seq_step, sequence_length):
    """Current frame indices for stacked previous/current frame pairs."""
    return [start_idx + frame_step + k * seq_step * frame_step for k in range(sequence_length)]


def get_future_poses(
    frame_positions: np.ndarray,
    frame_orientations: np.ndarray,
    frame_speeds: np.ndarray,
    frame_times_s: np.ndarray,
    seq_idxs: np.ndarray,
    t_anchors: np.ndarray | None,
    num_pts: int,
    use_full_pose: bool,
    strict: bool,
    interp_to_end: bool,
):
    """Interpolate ego-frame targets at relative times, including the final anchor.

    ``num_pts`` is retained for compatibility and controls output count only with
    ``interp_to_end=True``. A fixed frame count never limits the time horizon.
    ``strict=False`` returns the valid prefix and its length when an index or horizon
    is unsupported; malformed episode data always raises. Quaternions use wxyz order.
    Outputs have shape (valid_indices, target_points, 3 if use_full_pose else 2), with
    channels (x, y, speed) or (x, y). interp_to_end starts at the current frame.
    """
    frame_positions, frame_orientations, frame_speeds, frame_times_s = validate_pose_arrays(
        frame_positions, frame_orientations, frame_speeds, frame_times_s
    )
    if not interp_to_end and t_anchors is None:
        raise ValueError("t_anchors must be non-null if interp_to_end=False")
    if interp_to_end:
        if t_anchors is not None:
            logger.warning("interp_to_end=True overrides t_anchors")
        if not isinstance(num_pts, (int, np.integer)) or num_pts < 1:
            raise ValueError("num_pts must be a positive integer for interp_to_end")
        output_points = num_pts
    else:
        t_anchors = np.asarray(t_anchors, dtype=np.float64)
        if (
            t_anchors.ndim != 1
            or t_anchors.size == 0
            or not np.isfinite(t_anchors).all()
            or np.any(t_anchors < 0)
            or np.any(np.diff(t_anchors) <= 0)
        ):
            raise ValueError("t_anchors must be finite, nonnegative, strictly increasing relative seconds")
        output_points = len(t_anchors)
    seq_idxs = np.asarray(seq_idxs)
    if seq_idxs.ndim != 1 or (seq_idxs.size and not np.issubdtype(seq_idxs.dtype, np.integer)):
        raise ValueError("seq_idxs must be a 1D array of integer frame indices")

    future_poses = []
    max_idx = len(seq_idxs)
    n_frames = len(frame_times_s)
    tolerance = _timestamp_tolerance(frame_times_s)
    for i, seq_idx in enumerate(seq_idxs):
        supported = 0 <= seq_idx < n_frames and (
            interp_to_end or t_anchors[-1] <= frame_times_s[-1] - frame_times_s[seq_idx] + tolerance
        )
        if not supported:
            if strict:
                raise ValueError(
                    f"frame {seq_idx} is out of range or has insufficient future timestamps for the anchors"
                )
            max_idx = i
            break
        quat = frame_orientations[seq_idx]
        local_from_odom = rot_from_quat(quat / np.linalg.norm(quat)).T
        if interp_to_end:
            end_idx = n_frames - 1
            anchors = np.linspace(0.0, frame_times_s[-1] - frame_times_s[seq_idx], num_pts)
        else:
            end_idx = min(int(np.searchsorted(frame_times_s, frame_times_s[seq_idx] + t_anchors[-1])), n_frames - 1)
            anchors = t_anchors
        interval = slice(seq_idx, end_idx + 1)
        local_positions = np.einsum("ij,kj->ki", local_from_odom, frame_positions[interval] - frame_positions[seq_idx])
        local_times = frame_times_s[interval] - frame_times_s[seq_idx]
        columns = [np.interp(anchors, local_times, local_positions[:, axis]) for axis in (0, 1)]
        if use_full_pose:
            columns.append(np.interp(anchors, local_times, frame_speeds[interval]))
        future_poses.append(np.column_stack(columns))
    poses = np.asarray(future_poses, dtype=np.float32).reshape(-1, output_points, 3 if use_full_pose else 2)
    return poses, max_idx


def get_future_poses_from_dir(
    sample_dir: Path,
    seq_idxs: np.ndarray,
    t_anchors: np.ndarray | None,
    num_pts: int,
    use_full_pose: bool,
    strict: bool,
    interp_to_end: bool,
):
    """Load an episode and call get_future_poses with the same semantics."""
    return get_future_poses(
        *load_pose_arrays(sample_dir), seq_idxs, t_anchors, num_pts, use_full_pose, strict, interp_to_end
    )


def dali_pose_target_loader(
    video_files: list[str],
    labels: np.ndarray,
    start_frame_num: np.ndarray,
    do_hflip: np.ndarray,
    sequence_length: int,
    seq_step: int,
    frame_step: int,
    t_anchors: np.ndarray,
    num_pts: int,
    use_full_pose: bool,
):
    """Return sequence timestamps, ego-frame targets and speeds for a DALI sample."""
    flip_sample = bool(int(do_hflip.reshape(-1)[0]))
    start_idx = int(start_frame_num.reshape(-1)[0])
    sample_dir = Path(video_files[int(labels.reshape(-1)[0])]).parent
    arrays = load_pose_arrays(sample_dir)
    seq_idxs = get_current_frame_idxs(start_idx, frame_step, seq_step, sequence_length)
    future_poses, _ = get_future_poses(
        *arrays, seq_idxs, t_anchors, num_pts, use_full_pose, strict=True, interp_to_end=False
    )
    if flip_sample:
        future_poses[..., 1] *= -1.0
    frame_times_s = np.asarray(arrays[3][seq_idxs], dtype=np.float64)
    frame_speeds = np.asarray(arrays[2][seq_idxs], dtype=np.float32).reshape(-1, 1)
    return frame_times_s, future_poses, frame_speeds
