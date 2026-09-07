"""Quaternion / rotation helpers ported from sim2real-jaka (numpy-only).

All quaternions use the ``(w, x, y, z)`` scalar-first convention unless a
function name states otherwise. These match the conventions used by the
Jaka MF observation classes.
"""

from __future__ import annotations

import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def yaw_from_quat(quat: np.ndarray) -> np.ndarray:
    import numpy as _np

    q = _np.asarray(quat)
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    yaw = _np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return yaw[..., None]


def yaw_quat(quat: np.ndarray) -> np.ndarray:
    """Extract the yaw component of a quaternion.

    Args:
        quat: orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        A quaternion with only the yaw component, same shape.
    """
    shape = quat.shape
    quat_yaw = quat.copy().reshape(-1, 4)
    qw = quat_yaw[:, 0]
    qx = quat_yaw[:, 1]
    qy = quat_yaw[:, 2]
    qz = quat_yaw[:, 3]
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw[:] = 0.0
    quat_yaw[:, 3] = np.sin(yaw / 2)
    quat_yaw[:, 0] = np.cos(yaw / 2)
    quat_yaw = normalize(quat_yaw)
    return quat_yaw.reshape(shape)


def quat_from_yaw(yaw: np.ndarray) -> np.ndarray:
    """Build quaternions in (w, x, y, z) from yaw angles."""
    yaw = np.asarray(yaw)
    shape = yaw.shape
    yaw = yaw.reshape(-1)
    quat = np.zeros((yaw.shape[0], 4), dtype=yaw.dtype)
    quat[:, 0] = np.cos(yaw / 2.0)
    quat[:, 3] = np.sin(yaw / 2.0)
    return quat.reshape(shape + (4,))


def quat_rotate_inverse_numpy(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) by the inverse of quaternion(s).

    ``q`` and ``v`` share batch dimensions. ``q`` is (..., 4), ``v`` is (..., 3).
    """
    shape = v.shape
    q = q.reshape(-1, 4)
    v = v.reshape(-1, 3)
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]
    b = np.cross(q_vec, v) * q_w[:, np.newaxis] * 2.0
    dot_product = np.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * dot_product * 2.0
    return (a - b + c).reshape(shape)


def quat_rotate_numpy(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) by quaternion(s).

    ``q`` and ``v`` share batch dimensions. ``q`` is (..., 4), ``v`` is (..., 3).
    """
    shape = v.shape
    q = q.reshape(-1, 4)
    v = v.reshape(-1, 3)
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]
    b = np.cross(q_vec, v) * q_w[:, np.newaxis] * 2.0
    dot_product = np.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * dot_product * 2.0
    return (a + b + c).reshape(shape)


def wrap_to_pi(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi


def matrix_from_quat(quaternions: np.ndarray) -> np.ndarray:
    """Convert quaternions (w, x, y, z) to rotation matrices (..., 3, 3)."""
    original_shape = quaternions.shape[:-1]
    quaternions = quaternions.reshape(-1, 4)
    r, i, j, k = (
        quaternions[:, 0],
        quaternions[:, 1],
        quaternions[:, 2],
        quaternions[:, 3],
    )
    two_s = 2.0 / np.sum(quaternions * quaternions, axis=-1)
    o = np.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        axis=-1,
    )
    return o.reshape(original_shape + (3, 3))


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (w, x, y, z). Shape is (..., 4)."""
    if q1.shape != q2.shape:
        raise ValueError(
            f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        )
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return np.stack([w, x, y, z], axis=-1).reshape(shape)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of a quaternion (w, x, y, z). Shape is (..., 4)."""
    shape = q.shape
    q = q.reshape(-1, 4)
    return np.concatenate((q[:, 0:1], -q[:, 1:]), axis=-1).reshape(shape)


def quat_normalize(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Batch-normalize quaternions (w, x, y, z) to unit norm."""
    denom = np.linalg.norm(q, axis=-1, keepdims=True)
    denom = np.clip(denom, eps, None)
    return np.asarray(q, dtype=np.float32) / denom


def quat_slerp_batch(
    q0_wxyz: np.ndarray,
    q1_wxyz: np.ndarray,
    alpha,
    *,
    normalize_inputs: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """Batch spherical linear interpolation between quaternions.

    Args:
        q0_wxyz: (..., 4) quaternions.
        q1_wxyz: (..., 4) quaternions.
        alpha: interpolation factor, broadcast against batch dims.
    """
    q0 = np.asarray(q0_wxyz)
    q1 = np.asarray(q1_wxyz)
    if normalize_inputs:
        q0 = quat_normalize(q0, eps=eps)
        q1 = quat_normalize(q1, eps=eps)

    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    flip_mask = dot < 0.0
    q1 = np.where(flip_mask, -q1, q1)
    dot = np.where(flip_mask, -dot, dot)
    dot = np.clip(dot, -1.0, 1.0)

    alpha_arr = np.asarray(alpha, dtype=q0.dtype)
    while alpha_arr.ndim < dot.ndim:
        alpha_arr = np.expand_dims(alpha_arr, axis=-1)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha_arr

    safe_denom = np.where(sin_theta_0 > eps, sin_theta_0, 1.0)
    s0 = np.sin(theta_0 - theta) / safe_denom
    s1 = np.sin(theta) / safe_denom
    slerp_out = s0 * q0 + s1 * q1

    nlerp_out = (1.0 - alpha_arr) * q0 + alpha_arr * q1
    out = np.where(dot > 0.9995, nlerp_out, slerp_out)
    return quat_normalize(out, eps=eps)


__all__ = [
    "normalize",
    "yaw_from_quat",
    "yaw_quat",
    "quat_from_yaw",
    "quat_rotate_inverse_numpy",
    "quat_rotate_numpy",
    "wrap_to_pi",
    "matrix_from_quat",
    "quat_mul",
    "quat_conjugate",
    "quat_normalize",
    "quat_slerp_batch",
]
