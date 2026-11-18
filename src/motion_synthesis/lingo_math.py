"""Lazy, differentiable forwarding to trusted native LINGO math helpers.

Only code/constants are cached; inputs, gradients and request state are not.
Frames/units and quaternion packing policy remain with the callers.
"""

from functools import lru_cache
import os

from src.motion_synthesis.lingo_external import load_external_math

_load_helpers = lru_cache(maxsize=1)(load_external_math)


def transform_points(x, mat):
    return _load_helpers(os.environ.get("SYNCMOS_LINGO_CHECKOUT")).transform_points(x, mat)


def axis_angle_to_quaternion(axis_angle):
    return _load_helpers(os.environ.get("SYNCMOS_LINGO_CHECKOUT")).T.axis_angle_to_quaternion(axis_angle)


def zup_to_yup(coord):
    return _load_helpers(os.environ.get("SYNCMOS_LINGO_CHECKOUT")).zup_to_yup(coord)


def yup_to_zup(coord):
    return _load_helpers(os.environ.get("SYNCMOS_LINGO_CHECKOUT")).yup_to_zup(coord)
