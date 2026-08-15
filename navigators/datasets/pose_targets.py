import logging
from pathlib import Path

import numpy as np

from navigators.utils.common import load_npy
from navigators.utils.orientation import rot_from_quat

logger = logging.getLogger(__name__)

# expected file names
FRAME_POSITIONS = "frame_positions.npy"
FRAME_TIMES_S = "frame_times.npy"
FRAME_ORIENTATIONS = "frame_orientations.npy"
FRAME_SPEEDS = "frame_speeds.npy"


def get_current_frame_idxs(start_idx, frame_step, seq_step, sequence_length):
    """Return target frame indices aligned to the current frame of each stacked frame pair."""
    first_seq_idx = start_idx + frame_step
    seq_idxs = [first_seq_idx + (k * seq_step * frame_step) for k in range(sequence_length)]

    return seq_idxs


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
    """Computes the future GT trajectories from a sequence of frame poses, orientations, and speeds.

    Args:
        frame_positions: Array of raw frame position data
        frame_orientations: Array of raw frame orientation data
        frame_speeds: Array of raw frame speed data, in m/s
        frame_times_s: Array of raw frame times data, in seconds
        seq_idxs: Array of indices to compute future poses for
        t_anchors: Array of timestamps to compute poses for, or None if interp_to_end == True
        num_pts: Number of points to compute poses for per timestamp
        use_full_pose: If True, returns 3D pose (x, y, v), else returns 2D pose (x, y)
        strict: If True, will raise an error if the seq_idx is invalid (out of range), otherwise will just skip invalid samples
        interp_to_end: If True, will NOT use t_anchors and instead interpolate results to the end of frame_times_s

    Returns:
        future_poses: Array of the computed poses
        max_idx: The last index of the sequence that was valid, given strict == False
    """
    if not interp_to_end and t_anchors is None:
        raise ValueError("t_anchors must be non-null if 'interp_to_end == False'")
    elif interp_to_end and t_anchors is not None:
        logger.warning(f"Setting 'interp_to_end == True' will overwrite specified t_anchor values of {t_anchors}")

    future_poses = []
    max_idx = len(seq_idxs)
    n_frames = frame_times_s.shape[0]
    for i, seq_idx in enumerate(seq_idxs):
        max_idx_allowed = seq_idx if interp_to_end else seq_idx + num_pts
        if seq_idx < 0 or max_idx_allowed >= n_frames:
            if strict:
                raise ValueError(f"frame {seq_idx} invalid - (need {seq_idx + num_pts} <= {n_frames} frames).")
            max_idx = i
            break
        # frame_orientations occasionally stores non-unit / degenerate quaternions. rot_from_quat
        # (quat2rot) does NOT normalize, so a quaternion of norm s produces R ~= s^2 * R_unit: the
        # whole future (x, y) target is silently SCALED (or, for s -> 0, COLLAPSED to the origin --
        # a fake "standstill" label while the vehicle is actually moving). Speed is rotation- and
        # scale-invariant, so this corruption is invisible to speed-based cleaning. L2-normalizing
        # recovers the intended rotation for scaled quats and for most degenerate ones.
        # CAVEAT: a genuinely near-zero quaternion carries no reliable heading, and normalizing that
        # noise can yield a WRONG (even backward) direction. Normalization is a safety net, not a
        # substitute for dropping unrecoverable clips (|norm - 1| > ~0.1) at the split level.
        quat = frame_orientations[seq_idx]
        norm = np.linalg.norm(quat)
        if norm > 1e-6:
            quat = quat / norm
        odom_from_local = rot_from_quat(quat)
        local_from_odom = odom_from_local.T

        frame_positions_local = np.einsum(
            "ij,kj->ki",
            local_from_odom,
            frame_positions - frame_positions[seq_idx],
        ).astype(np.float32)

        t_local = frame_times_s[seq_idx : seq_idx + num_pts] - frame_times_s[seq_idx]
        if interp_to_end:
            t_local = frame_times_s - frame_times_s[seq_idx]
            t_anchors = np.linspace(float(t_local[0]), float(t_local[-1]), num_pts)
            x_interp = np.interp(t_anchors, t_local, frame_positions_local[:, 0])
            y_interp = np.interp(t_anchors, t_local, frame_positions_local[:, 1])
            v_interp = np.interp(t_anchors, t_local, frame_speeds)
        else:
            x_interp = np.interp(t_anchors, t_local, frame_positions_local[seq_idx : seq_idx + num_pts, 0])
            y_interp = np.interp(t_anchors, t_local, frame_positions_local[seq_idx : seq_idx + num_pts, 1])
            v_interp = np.interp(t_anchors, t_local, frame_speeds[seq_idx : seq_idx + num_pts])

        if use_full_pose:
            interp_positions = np.column_stack((x_interp, y_interp, v_interp))
        else:
            interp_positions = np.column_stack((x_interp, y_interp))
        future_poses.append(interp_positions)

    future_poses = np.asarray(future_poses, dtype=np.float32)
    return future_poses, max_idx


def get_future_poses_from_dir(
    sample_dir: Path,
    seq_idxs: np.ndarray,
    t_anchors: np.ndarray | None,
    num_pts: int,
    use_full_pose: bool,
    strict: bool,
    interp_to_end: bool,
):
    """
    Wrapper around get_future_poses that loads raw frame positions, orientations, speeds, and times from the given directory
    before calling get_future_poses. See get_future_poses for details.
    """
    frame_positions = load_npy(sample_dir / FRAME_POSITIONS)
    frame_orientations = load_npy(sample_dir / FRAME_ORIENTATIONS)
    frame_speeds = load_npy(sample_dir / FRAME_SPEEDS)
    frame_times_s = load_npy(sample_dir / FRAME_TIMES_S) / 1e9

    return get_future_poses(
        frame_positions,
        frame_orientations,
        frame_speeds,
        frame_times_s,
        seq_idxs,
        t_anchors,
        num_pts,
        use_full_pose,
        strict,
        interp_to_end,
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
    """
    DALI-side wrapper: per-sequence-frame future ego pose in the local frame of the current sequence frame (not the first frame).
    This means that if seq_step > 1, the future poses will reflect the fact that we are skipping frames in the future trajectory when determining the future poses.
    Horizontal flipping mirrors lateral motion in the local frame.

    Returns:
        frame_times_s: Array of frame times in seconds for each sequence frame.
        future_poses: Array of future poses for each sequence frame.
        frame_speeds: Array of frame speeds for each sequence frame.
    """
    flip_sample = bool(int(do_hflip.reshape(-1)[0]))
    start_idx = int(start_frame_num.reshape(-1)[0])
    video_idx = int(labels.reshape(-1)[0])
    video_fp = Path(video_files[video_idx])
    sample_dir = video_fp.parent
    frame_times_s = load_npy(sample_dir / FRAME_TIMES_S) / 1e9
    frame_speeds = load_npy(sample_dir / FRAME_SPEEDS)

    seq_idxs = get_current_frame_idxs(start_idx, frame_step, seq_step, sequence_length)
    future_poses, _ = get_future_poses_from_dir(
        sample_dir, seq_idxs, t_anchors, num_pts, use_full_pose, strict=True, interp_to_end=False
    )

    # Horizontal image flip mirrors lateral motion in the local frame.
    if flip_sample:
        future_poses[..., 1] *= -1.0

    frame_times_s = np.asarray(frame_times_s[seq_idxs], dtype=np.float32)
    frame_speeds = np.asarray(frame_speeds[seq_idxs], dtype=np.float32).reshape(-1, 1)

    return frame_times_s, future_poses, frame_speeds
