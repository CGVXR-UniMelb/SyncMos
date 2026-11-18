"""Prepare or check generation resources from the SyncMos repository root."""

import argparse
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import load_dotenv

from src.scene.download_hssd import SCENES, download_hssd, mesh_paths
from src.scene.resource_archive import copy_resource, extract_resources

REVISION = "6de5ea649ca244e0f64bc23eacfe722e2b613034"
ARCHIVE_ID = "1L2V8RlPMAhWF93o_RpIznO_bacjSSLqu"
LINGO_FILES = tuple(
    Path(name)
    for name in (
        "ckpts/model/bs128__ws16__scene_occ_two.pth",
        "ckpts/scheduler/scheduler.pth",
        "ckpts/joints2smplx/train_joint2smpl__input_84__hidden_128__all__last.pth",
        "dataset/human_orient.npy",
        "dataset/human_joints_aligned.npy",
        "dataset/clip_features.npy",
        "dataset/text2features_idx.pkl",
        "dataset/norm_inter_and_loco__16frames.npy",
        "dataset/language_motion_dict/language_motion_dict__inter_and_loco__16.pkl",
    )
)
BODY_FILES = (Path("smpl_models/smplx/SMPLX_MALE.npz"), Path("data/SMPL/smplx/SMPLX_MALE.npz"))
CLIP_FILE = Path.home() / ".cache/clip/ViT-L-14-336px.pt"


def ready(path):
    return path.is_file() and path.stat().st_size > 0


def check_checkout(path):
    if not (path / ".git").exists():
        raise ValueError(f"Missing LINGO checkout at {path}. Run preparation without --check.")
    revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(f"Expected LINGO revision {REVISION} at {path}, found {revision}.")


def prepare_checkout(path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=path.parent) as temporary:
            clone = Path(temporary) / "lingo"
            subprocess.run(["git", "clone", "https://github.com/mileret/lingo-release.git", str(clone)], check=True)
            subprocess.run(["git", "-C", str(clone), "checkout", "--detach", REVISION], check=True)
            clone.rename(path)
    check_checkout(path)


def prepare_body():
    for target in BODY_FILES:
        if ready(target):
            continue
        configured = os.environ.get("SMPLX_MALE_FILE")
        candidates = [Path(configured).expanduser()] if configured else []
        source = next((p for p in [*candidates, *BODY_FILES] if ready(p)), None)
        if source is None:
            raise ValueError(
                "Download and extract the licensed SMPLX_MALE.npz from SMPL-X, then set SMPLX_MALE_FILE in .env."
            )
        with source.open("rb") as stream:
            copy_resource(stream, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=(*SCENES, "all"), default="House")
    parser.add_argument(
        "--check", action="store_true", help="Check local files, CUDA and API key presence without downloading."
    )
    args = parser.parse_args()
    if not Path("configs/sample.yaml").is_file():
        parser.error("Run from the SyncMos repository root.")
    load_dotenv(Path(".env"))
    scenes = SCENES if args.scene == "all" else (args.scene,)
    checkout = Path(os.environ.get("SYNCMOS_LINGO_CHECKOUT") or "external/lingo-release").expanduser()
    if not args.check:
        prepare_body()
        prepare_checkout(checkout)
        download_hssd(scenes)
        extract_resources(LINGO_FILES, ARCHIVE_ID, os.environ.get("LINGO_UTILS_ZIP"))
        from clip.clip import _download, _MODELS

        _download(_MODELS["ViT-L/14@336px"], str(CLIP_FILE.parent))
    check_checkout(checkout)
    required = [*LINGO_FILES, *BODY_FILES, CLIP_FILE, *(Path("data/HSSD") / p for p in mesh_paths(scenes))]
    missing = [str(p) for p in required if not ready(p)]
    if missing:
        parser.exit(1, "Missing files:\n" + "\n".join(missing) + "\n")
    if not os.environ.get("OPENAI_API_KEY"):
        parser.exit(1, "Set OPENAI_API_KEY in .env for live planning.\n")
    import torch

    if not torch.cuda.is_available():
        parser.exit(1, "Motion generation requires a working CUDA device.\n")
    print(
        f"Ready for {', '.join(scenes)}: {len(mesh_paths(scenes))} unique meshes; local files, GPU and API key presence checked."
    )


if __name__ == "__main__":
    main()
