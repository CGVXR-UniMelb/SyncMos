"""Lazy native pelvis-frame alignment; no model, motion or transform cache."""

from functools import lru_cache
import os

from src.motion_synthesis.lingo_external import load_external_geometry

_load_helpers = lru_cache(maxsize=1)(load_external_geometry)


def get_mat(cfg, points):
    return _load_helpers(os.environ.get("SYNCMOS_LINGO_CHECKOUT")).get_mat(cfg, points)
