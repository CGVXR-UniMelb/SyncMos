"""Download only the HSSD meshes referenced by the selected scenes."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

SCENES = ("House", "Office", "Restaurant")


def mesh_paths(scenes):
    paths = set()
    for scene in scenes:
        objects = json.loads(Path(f"configs/scenes/{scene}/scene_cfg.json").read_text(encoding="utf-8"))
        for obj in objects.values():
            model_id = obj["model_id"]
            paths.add(Path("objects") / model_id[0] / f"{model_id}.glb")
    return sorted(paths)


def download_hssd(scenes=SCENES):
    load_dotenv()
    folder = Path("data/HSSD")
    missing = [
        path for path in mesh_paths(scenes) if not (folder / path).is_file() or not (folder / path).stat().st_size
    ]
    if missing:
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if not token:
            raise ValueError("Set HUGGINGFACE_HUB_TOKEN in .env after accepting the HSSD dataset terms.")
        for path in missing:
            hf_hub_download(
                "hssd/hssd-models", filename=path.as_posix(), repo_type="dataset", local_dir=folder, token=token
            )


if __name__ == "__main__":
    download_hssd()
