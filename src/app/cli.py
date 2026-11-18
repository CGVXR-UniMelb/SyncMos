"""One command for interactive playback, generation, and retained experiments."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from importlib import import_module

DEFAULT_CONFIG = "./configs/sample.yaml"
_MODES = {
    "viewer": "src.app.interactive",
    "generate": "src.app.generation",
    "parallel": "src.app.experiments.parallel",
    "handover": "src.app.experiments.handover",
    "bbox": "src.app.experiments.bbox",
    "scene": "src.app.experiments.scene",
}


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch a mode while forwarding only configuration overrides to OmegaConf."""
    parser = argparse.ArgumentParser(
        description="Load, generate, and continue motion from one application entrypoint.",
        epilog="Configuration overrides use key=value, e.g. scene_name=House num_characters=2.",
        allow_abbrev=False,
    )
    parser.add_argument("--mode", choices=tuple(_MODES), default="viewer", help="Default: viewer.")
    parser.add_argument(
        "--config", help=f"Configuration file (default: {DEFAULT_CONFIG}; bbox retains its Qwen preset)."
    )
    parser.add_argument("--motion-dir", help="Output directory for generate mode (default: motion).")
    parser.add_argument("--max-frames", type=int, help="Optional frame limit for generate mode.")
    args, overrides = parser.parse_known_args(argv)
    for override in overrides:
        if override.startswith("-") or "=" not in override or not override.partition("=")[0]:
            parser.error(f"Expected a key=value configuration override, got {override!r}.")
    if args.mode != "generate" and (args.motion_dir is not None or args.max_frames is not None):
        parser.error("--motion-dir and --max-frames apply only to --mode generate.")

    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    os.environ.setdefault("ROOT_DIR", os.getcwd())
    if args.mode != "generate":
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

    # Import only the selected mode. --help requires no GUI, Torch, models or resources.
    runner = import_module(_MODES[args.mode])
    from omegaconf import OmegaConf
    from src.utils import misc

    OmegaConf.register_new_resolver("times", lambda x, y: int(x) * int(y), replace=True)
    config_path = args.config or ("./configs/experiments/qwen.yaml" if args.mode == "bbox" else DEFAULT_CONFIG)
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *overrides]
        cfg = misc.load_cfg(config_path)
    finally:
        sys.argv = original_argv

    if args.mode == "generate":
        runner.run(
            cfg, motion_dir=args.motion_dir if args.motion_dir is not None else "motion", max_frames=args.max_frames
        )
    else:
        runner.run(cfg)
