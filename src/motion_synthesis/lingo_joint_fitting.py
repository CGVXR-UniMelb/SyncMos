"""Compatibility boundary for native external LINGO fitting/interpolation.

No fitting algorithm or pose constants live here. Cache only the isolated code
module, never motion/model/SMPL state. Keep existing signatures and legacy re-exports.
"""

from functools import lru_cache
import os

from src.motion_synthesis.lingo_external import load_external_joint_fitting

_load_helpers = lru_cache(maxsize=1)(load_external_joint_fitting)


def optimize_smpl(pose_pred, joints, joints_ind, hand_pca=45):
    return _load_helpers(os.environ.get("SYNCMOS_LINGO_CHECKOUT")).optimize_smpl(
        pose_pred, joints, joints_ind, hand_pca
    )


def joints_to_smpl(model, joints, joints_ind, interp_s):
    return _load_helpers(os.environ.get("SYNCMOS_LINGO_CHECKOUT")).joints_to_smpl(model, joints, joints_ind, interp_s)


def interpolate_joints(joints, scale):
    return _load_helpers(os.environ.get("SYNCMOS_LINGO_CHECKOUT")).interpolate_joints(joints, scale)
