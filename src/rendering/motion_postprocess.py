"""Optional filters for completed motion exports."""

import math

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
import torch


def _smooth_rotations(rotations: Rotation, sigma_frames: float) -> Rotation:
    quaternions = rotations.as_quat()
    # q and -q represent the same orientation. Unwrap before averaging so a
    # representation change cannot cancel two otherwise identical rotations.
    signs = np.ones(len(quaternions))
    signs[1:] = np.where(np.sum(quaternions[1:] * quaternions[:-1], axis=1) < 0, -1, 1)
    quaternions *= np.cumprod(signs)[:, None]
    smoothed = gaussian_filter1d(quaternions, sigma_frames, axis=0, mode="nearest", truncate=3)
    lengths = np.linalg.norm(smoothed, axis=1, keepdims=True)
    valid = lengths[:, 0] > 1e-8
    smoothed[valid] /= lengths[valid]
    smoothed[~valid] = quaternions[~valid]
    return Rotation.from_quat(smoothed)


def stabilize_neck_motion(
    body_params: torch.Tensor, *, fps: float = 30.0, smoothing_seconds: float = 0.05
) -> torch.Tensor:
    """Smooth neck/head jitter in a completed (T, 91) wxyz motion sequence.

    This centered filter uses future frames and is intended only for offline
    export. It changes local neck/head rotations (SMPL joints 12 and 15), leaving
    translation, root rotation, torso and limbs byte-for-byte unchanged. The
    combined neck/head orientation is smoothed as well, preserving compensating
    neck/head movements and ordinary gaze changes instead of fixing the head.
    Inputs are not mutated; no random stream is consumed. Keep the raw motion.
    """
    if body_params.ndim != 2 or body_params.shape[1] != 91:
        raise ValueError("body_params must have shape (T, 91)")
    if not body_params.is_floating_point():
        raise TypeError("body_params must be floating point")
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(smoothing_seconds) or smoothing_seconds < 0:
        raise ValueError("fps must be positive and smoothing_seconds nonnegative; both must be finite")
    result = body_params.clone()
    if len(body_params) < 2 or smoothing_seconds == 0:
        return result

    neck_slice, head_slice = slice(51, 55), slice(63, 67)
    neck_q = body_params[:, neck_slice].detach().cpu().double().numpy()[:, [1, 2, 3, 0]]
    head_q = body_params[:, head_slice].detach().cpu().double().numpy()[:, [1, 2, 3, 0]]
    if not np.isfinite(neck_q).all() or not np.isfinite(head_q).all():
        raise ValueError("neck and head quaternions must be finite")
    neck, head = Rotation.from_quat(neck_q), Rotation.from_quat(head_q)
    sigma_frames = fps * smoothing_seconds
    smooth_neck = _smooth_rotations(neck, sigma_frames)
    smooth_combined = _smooth_rotations(neck * head, sigma_frames)
    smooth_head = smooth_neck.inv() * smooth_combined
    for target, rotations in ((neck_slice, smooth_neck), (head_slice, smooth_head)):
        result[:, target] = torch.as_tensor(
            rotations.as_quat()[:, [3, 0, 1, 2]], dtype=body_params.dtype, device=body_params.device
        )
    return result
