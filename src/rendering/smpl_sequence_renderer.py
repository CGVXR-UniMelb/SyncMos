from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch

from src.utils import smpl


@dataclass
class CharacterMesh:
    name: str
    vertices: np.ndarray
    faces: np.ndarray


@dataclass
class MeshFrame:
    frame_id: int
    character_meshes: dict[str, CharacterMesh]


class SMPLSequenceRenderer:
    """Converts per-character body_params frames into SMPL mesh frames."""

    def __init__(self, device: str = "cpu", body_model=None):
        self.device = device
        self._body_model = body_model
        self._stats_started_at = time.time()
        self._smpl_frames = 0

    @property
    def bm(self):
        if self._body_model is None:
            self._body_model = smpl.load_body_model(device=self.device)
        return self._body_model

    def render_frame(
        self,
        body_params_by_character: dict[str, torch.Tensor],
        use_origin: bool = False,
        frame_id: int = 0,
    ) -> MeshFrame:
        return self.render_batch(body_params_by_character, frame_id=frame_id, first_frame_only=True)

    def render_batch(
        self,
        body_params_by_character: dict[str, torch.Tensor],
        frame_id: int = 0,
        first_frame_only: bool = False,
    ) -> MeshFrame:
        """Render many character/frame body_params in one SMPL call.

        When first_frame_only is False, vertices for each character have shape
        (T, V, 3). When True, each CharacterMesh stores a single (V, 3) frame.
        """
        names = []
        lengths = []
        params_list = []
        for character_name, body_params in body_params_by_character.items():
            if body_params is None or body_params.shape[0] == 0:
                continue
            params = body_params[:1] if first_frame_only else body_params
            names.append(character_name)
            lengths.append(params.shape[0])
            params_list.append(params)

        meshes = {}
        if not params_list:
            return MeshFrame(frame_id=frame_id, character_meshes=meshes)

        batched_params = torch.cat(params_list, dim=0)
        batched_vertices, faces = self.render_body_params_batch(batched_params)
        cursor = 0
        for character_name, length in zip(names, lengths):
            vertices = batched_vertices[cursor : cursor + length]
            cursor += length
            if first_frame_only:
                vertices = vertices[0]
            meshes[character_name] = CharacterMesh(character_name, vertices, faces)
        return MeshFrame(frame_id=frame_id, character_meshes=meshes)

    def render_character(self, body_params: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        if body_params is None or body_params.shape[0] == 0:
            return np.zeros((10475, 3), dtype=np.float32), self._faces()
        vertices, faces = self.render_body_params_batch(body_params)
        return vertices[0], faces

    def render_body_params_batch(self, body_params: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            if body_params is None or body_params.shape[0] == 0:
                return np.zeros((0, 10475, 3), dtype=np.float32), self._faces()

            body_params = body_params.to(self.device)
            batch_size = body_params.shape[0]
            transl = body_params[:, :3]
            root_quat = body_params[:, 3:7]
            joints_quat = body_params[:, 7:].reshape(batch_size, 21, 4)

            body_params_dict = {
                "transl": transl,
                "global_orient": smpl.quaternion_to_axis_angle(root_quat),
                "body_pose": smpl.quaternion_to_axis_angle(joints_quat).reshape(batch_size, 63),
            }

            for key in self.bm.required_keys:
                if key not in body_params_dict:
                    body_params_dict[key] = torch.zeros(
                        (batch_size, self.bm.body_param_dims[key]),
                        dtype=torch.float32,
                        device=self.device,
                    )

            smpl_output = self.bm(**body_params_dict)
            vertices = smpl_output.vertices.detach().cpu().numpy()
            self._smpl_frames += int(batch_size)
            return vertices, self._faces()

    def stats_snapshot(self) -> dict:
        elapsed = max(1e-6, time.time() - self._stats_started_at)
        return {
            "total_smpl_frames": self._smpl_frames,
            "smpl_per_sec": self._smpl_frames / elapsed,
        }

    def _faces(self) -> np.ndarray:
        if hasattr(self.bm, "bm") and hasattr(self.bm.bm, "faces"):
            return self.bm.bm.faces
        return self.bm.faces
