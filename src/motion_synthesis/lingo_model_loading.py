"""Exact forwarding of native model setup; cache code, never models or weights."""

from functools import lru_cache
import os

from src.motion_synthesis.lingo_external import load_external_math

# Model setup uses the same unchanged constants/transforms/utils source closure.
_load_helpers = lru_cache(maxsize=1)(load_external_math)


def init_model(model_cfg, device, eval, load_state_dict=False, need_ddp=True):
    return _load_helpers(os.environ.get("SYNCMOS_LINGO_CHECKOUT")).init_model(
        model_cfg, device, eval, load_state_dict, need_ddp
    )


def load_state_dict_eval(model, state_dict_path, map_location="cuda:0", device="cuda"):
    return _load_helpers(os.environ.get("SYNCMOS_LINGO_CHECKOUT")).load_state_dict_eval(
        model, state_dict_path, map_location, device
    )
