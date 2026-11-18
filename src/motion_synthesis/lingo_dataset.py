"""Inference-only HSSD initialization bridge over the native external Dataset.

The external constructor loads motion/language/normalization; its original scene
loader is intentionally disabled. SyncMOS injects HSSD occupancy later. The small
valid-index/batch-id initialization retains the previous mixed-provenance setup;
this adapter is not a claim of independent authorship or redistribution clearance.
"""

from pathlib import Path

import numpy as np
import torch


class DatasetSceneAdapter:
    """Adjust construction only; all Dataset processing methods remain native."""

    def __init__(self, *, valid_indices_dir, **options):
        if options.get("train", True):
            raise ValueError("The external SyncMOS Dataset adapter supports inference only (train=false).")
        load_scene, vis = options.get("load_scene", True), options.get("vis", True)
        # Native scene loading is incompatible with HSSD injection. Defer the
        # native visualization filter so its relative asset path cannot use cwd.
        super().__init__(**{**options, "load_scene": False, "vis": False})
        self.load_scene, self.vis = load_scene, vis
        if self.load_language and vis and self.start_type in ("stand", "sit"):
            valid_idx = np.load(Path(valid_indices_dir) / f"valid_idx_{self.start_type}.npy", allow_pickle=False)
            # Preserve the old partial filtering, including unfiltered flags.
            self.start_ind = self.start_ind[valid_idx]
            self.end_ind = self.end_ind[valid_idx]
            self.text = [self.text[idx] for idx in valid_idx]
        if load_scene:
            self.mesh_grid, self.nb_voxels = options["mesh_grid"], options["nb_voxels"]
            self.batch_id = (
                torch.linspace(0, self.batch_size - 1, self.batch_size)
                .tile((self.nb_voxels[0] * self.nb_voxels[1] * self.nb_voxels[2], 1))
                .T.reshape(-1, 1)
                .to(device=self.device, dtype=torch.long)
            )
