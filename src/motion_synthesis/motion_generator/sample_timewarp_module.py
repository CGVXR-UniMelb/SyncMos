"""Compatibility import for the integrated LINGO time-warp implementation."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("src.motion_synthesis.time_warp")
