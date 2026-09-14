import numpy as np


def quat2rot(quats):
    quats = np.array(quats)
    input_shape = quats.shape
    quats = np.atleast_2d(quats)
    Rs = np.zeros((quats.shape[0], 3, 3))
    q0 = quats[:, 0]
    q1 = quats[:, 1]
    q2 = quats[:, 2]
    q3 = quats[:, 3]
    Rs[:, 0, 0] = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
    Rs[:, 0, 1] = 2 * (q1 * q2 - q0 * q3)
    Rs[:, 0, 2] = 2 * (q0 * q2 + q1 * q3)
    Rs[:, 1, 0] = 2 * (q1 * q2 + q0 * q3)
    Rs[:, 1, 1] = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3
    Rs[:, 1, 2] = 2 * (q2 * q3 - q0 * q1)
    Rs[:, 2, 0] = 2 * (q1 * q3 - q0 * q2)
    Rs[:, 2, 1] = 2 * (q0 * q1 + q2 * q3)
    Rs[:, 2, 2] = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3

    if len(input_shape) < 2:
        return Rs[0]
    else:
        return Rs


rot_from_quat = quat2rot


def yaw_from_quat(quats):
    """Heading (rotation about z) of wxyz quaternions, in radians."""
    rotations = np.atleast_3d(quat2rot(quats))
    return np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0])
