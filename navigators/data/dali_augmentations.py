import numpy as np
import nvidia.dali as dali
import nvidia.dali.fn as fn
from nvidia.dali.pipeline import do_not_convert

IDENTITY_HOMOGRAPHY_FLAT = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


###############################
######### RANDOM CROP #########
###############################
def apply_crop(videos, crop_xy: tuple[int, int], crop_wh: tuple[int, int]):
    """Crop videos to the specified xy and wh."""
    crop_x, crop_y = crop_xy
    crop_w, crop_h = crop_wh
    return fn.slice(videos, axes=[1, 2], start=[crop_y, crop_x], shape=[crop_h, crop_w])


###############################
####### HORIZONTAL FLIP #######
###############################
def apply_horizontal_flip(videos, enabled: bool, probability: float):
    """Randomly horizontally flip videos with the given probability if enabled."""
    do_hflip = fn.random.coin_flip(probability=probability if enabled else 0.0, dtype=dali.types.INT32)
    return fn.flip(videos, horizontal=do_hflip), do_hflip


###############################
######### IMAGE JITTER ########
###############################
def apply_jitter(videos, reader_sequence_length: int, enabled: bool, probability: float, n_degree: int):
    """Randomly apply DALI jitter to each frame in the sequence with the given probability if enabled."""
    videos = fn.reshape(videos, layout="FHWC")
    if enabled & fn.random.coin_flip(probability=probability):
        jittered_frames = []
        for i in range(reader_sequence_length):
            frame_i = fn.slice(videos, axes=[0], start=[i], shape=[1])
            frame_hwc = fn.reshape(frame_i, src_dims=[1, 2, 3], rel_shape=[1, 1, 1])
            jittered = fn.jitter(frame_hwc, nDegree=n_degree, fill_value=0)
            jittered_frames.append(fn.expand_dims(jittered, axes=0))
        videos = fn.reshape(fn.cat(*jittered_frames, axis=0), layout="FHWC")
    return fn.reshape(videos, layout="FHWC")


###############################
######### COLOR JITTER ########
###############################
def sample_uniform_jitter(delta: float, center: float, min_value: float | None = None):
    """Sample a jitter factor uniformly from [center - delta, center + delta], optionally ensuring a minimum value."""
    if delta <= 0.0:
        return center
    low = center - delta
    if min_value is not None:
        low = max(min_value, low)
    return fn.random.uniform(range=[low, center + delta])


def apply_color_jitter(videos, brightness: float, contrast: float, saturation: float, hue: float):
    """Apply DALI color jitter to videos with the given brightness, contrast, saturation, and hue deltas."""
    if any(value > 0.0 for value in (brightness, contrast, saturation, hue)):
        videos = fn.color_twist(
            videos,
            brightness=sample_uniform_jitter(brightness, center=1.0, min_value=0.0),
            contrast=sample_uniform_jitter(contrast, center=1.0, min_value=0.0),
            saturation=sample_uniform_jitter(saturation, center=1.0, min_value=0.0),
            hue=sample_uniform_jitter(360.0 * hue, center=0.0),
        )
    return videos


###############################
####### WARP PERSPECTIVE ######
###############################
def _homography_from_roll_pitch_yaw(
    roll_deg: float, pitch_deg: float, yaw_deg: float, focal: float, cx: float, cy: float
) -> np.ndarray:
    """Build 3x3 homography H = A @ R @ A^{-1} from roll, pitch, yaw (degrees)."""
    r = np.deg2rad(np.asarray(roll_deg, dtype=np.float64).reshape(-1))
    p = np.deg2rad(np.asarray(pitch_deg, dtype=np.float64).reshape(-1))
    y = np.deg2rad(np.asarray(yaw_deg, dtype=np.float64).reshape(-1))
    n = len(r)
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy_a, sy = np.cos(y), np.sin(y)
    # Rx(roll)
    # Ry(pitch)
    # Rz(yaw)
    R = np.zeros((n, 3, 3), dtype=np.float32)
    R[:, 0, 0] = cy_a * cp
    R[:, 0, 1] = cy_a * sp * sr - sy * cr
    R[:, 0, 2] = cy_a * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp
    R[:, 1, 1] = sy * sp * sr + cy_a * cr
    R[:, 1, 2] = sy * sp * cr - cy_a * sr
    R[:, 2, 0] = -sp
    R[:, 2, 1] = cp * sr
    R[:, 2, 2] = cp * cr
    # A, A_inv
    A = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float32)
    A_inv = np.array(
        [[1 / focal, 0, -cx / focal], [0, 1 / focal, -cy / focal], [0, 0, 1]],
        dtype=np.float32,
    )
    H = np.zeros((n, 3, 3), dtype=np.float32)
    for i in range(n):
        H[i] = A @ R[i] @ A_inv
    return H.reshape(n, 9)


def _batch_to_flat(batch_list: list[np.ndarray]) -> np.ndarray:
    """Convert DALI batch_processing input (list of arrays) to single flat numpy array."""
    return np.concatenate([np.asarray(x, dtype=np.float64).reshape(-1) for x in batch_list])


@do_not_convert
def _make_warp_perspective_homography_fn(focal: float, cx: float, cy: float):
    """Factory for batch homography function; must stay outside pipeline for conditionals."""

    def _batch_fn(roll_b: list[np.ndarray], pitch_b: list[np.ndarray], yaw_b: list[np.ndarray]) -> list[np.ndarray]:
        roll_flat = _batch_to_flat(roll_b)
        pitch_flat = _batch_to_flat(pitch_b)
        yaw_flat = _batch_to_flat(yaw_b)
        H = _homography_from_roll_pitch_yaw(roll_flat, pitch_flat, yaw_flat, focal, cx, cy)
        # DALI batch_processing expects list of per-sample arrays (one output)
        return [H[i].copy() for i in range(len(roll_flat))]

    return _batch_fn


def apply_warp_perspective(
    videos,
    frame_wh: tuple[int, int],
    enabled: bool,
    probability: float,
    angle_range: tuple[float, float],
    focal: float | None,
):
    """
    Randomly apply warp perspective to videos with the given angle range and focal length parameters if enabled,
    returning the warped videos and the homography used for each sample in the batch. The homography is returned
    as a flat 9-element array in row-major order for each sample, with the identity homography [1, 0, 0, 0, 1, 0, 0, 0, 1]
    used when warping is not applied.
    """
    homography_flat = fn.constant(fdata=IDENTITY_HOMOGRAPHY_FLAT, shape=[9], device="cpu")
    if enabled & fn.random.coin_flip(probability=probability) & (frame_wh is not None):
        focal = float(focal if focal is not None else max(frame_wh[0], frame_wh[1]))
        cx = frame_wh[0] / 2.0
        cy = frame_wh[1] / 2.0
        roll = fn.random.uniform(range=angle_range)
        pitch = fn.random.uniform(range=angle_range)
        yaw = fn.random.uniform(range=angle_range)
        homography_fn = _make_warp_perspective_homography_fn(focal, cx, cy)
        homography_flat = fn.python_function(
            roll.cpu(),
            pitch.cpu(),
            yaw.cpu(),
            function=homography_fn,
            device="cpu",
            num_outputs=1,
            batch_processing=True,
        )
        homography_3x3 = fn.reshape(homography_flat.gpu(), shape=[3, 3])
        videos = fn.experimental.warp_perspective(
            videos,
            homography_3x3,
            fill_value=0,
            inverse_map=True,
            border_mode="constant",
        )
    return fn.reshape(videos, layout="FHWC"), homography_flat
