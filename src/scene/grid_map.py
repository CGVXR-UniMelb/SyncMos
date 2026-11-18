import json
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import trimesh
import trimesh.creation
import os
import math
from src.motion_synthesis.lingo_math import zup_to_yup, yup_to_zup
from ..type import Object
from ..utils import misc


class GridMap:
    def __init__(
        self,
        scene_name: str,
        scene_mesh: trimesh.Trimesh,
        objects: List[Object],
        cfg,
    ) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.scene_name = scene_name
        self.objects = objects
        self.obj_meshes = [obj.mesh for obj in objects]
        self.obj_convex_hulls = [obj.mesh.convex_hull for obj in objects]
        self.scene_mesh_raw = trimesh.util.concatenate([scene_mesh] + self.obj_meshes)
        self.scene_mesh = trimesh.util.concatenate([scene_mesh] + self.obj_convex_hulls)
        self.bbox = self._load_scene_bounds(self.scene_name)
        try:
            self.update_grid_map(*self._load_grid_map_settings(self.scene_name))
        except FileNotFoundError:
            pass  # grid map will be updated later

    def __getitem__(self, idx):
        if not (isinstance(idx, (tuple, np.ndarray)) and len(idx) == 2):
            raise TypeError("Invalid argument type")
        x, y = idx
        return self.grid_points[x][y]

    def update_grid_map(
        self,
        grid_step,
        x_margin,
        y_margin,
        z_margin,
        cylinder_radius,
        cylinder_height,
        user_modifications=None,
    ):
        self.grid_step = grid_step
        self.x_margin = x_margin
        self.y_margin = y_margin
        self.z_margin = z_margin
        self.cylinder_radius = cylinder_radius
        self.cylinder_height = cylinder_height
        self.cylinder_min_z = self.bbox[0, 2] + self.z_margin
        (
            self.template_ray_origins,
            self.template_ray_directions,
        ) = self._create_template_rays(self.cylinder_radius, self.cylinder_height, self.cylinder_min_z)
        self.num_template_rays = len(self.template_ray_origins)

        min_x, min_y = self.bbox[0, :2]
        max_x, max_y = self.bbox[1, :2]
        X, Y, Z = torch.meshgrid(
            torch.arange(
                min_x + self.x_margin,
                max_x - self.x_margin,
                self.grid_step,
                device=self.device,
            ),
            torch.arange(
                min_y + self.y_margin,
                max_y - self.y_margin,
                self.grid_step,
                device=self.device,
            ),
            torch.tensor(0.0, device=self.device),
            indexing="ij",
        )

        self.num_x, self.num_y, _ = X.shape
        self.grid_points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
        self.num_grid_points = len(self.grid_points)

        origins = self.template_ray_origins + self.grid_points[:, None].cpu().numpy()
        origins = origins.reshape(-1, 3)
        directions = np.tile(self.template_ray_directions, (self.num_grid_points, 1, 1))
        directions = directions.reshape(-1, 3)
        _, ray_idx, intersections = self.scene_mesh.ray.intersects_id(
            ray_origins=origins,
            ray_directions=directions,
            multiple_hits=False,
            return_locations=True,
        )
        distances = np.full((self.num_grid_points * self.num_template_rays,), np.inf)
        distances[ray_idx] = np.linalg.norm(intersections - origins[ray_idx], axis=1)
        distances = distances.reshape(self.num_grid_points, -1)

        contained = np.full((self.num_grid_points,), False, dtype=bool)
        for convex_hull in self.obj_convex_hulls:
            contained |= convex_hull.contains(origins).reshape(self.num_grid_points, -1).any(axis=-1)
        intersected = (distances <= cylinder_radius).any(axis=-1)

        self.occupied = contained | intersected
        self.occupied = self.occupied.reshape(self.num_x, self.num_y)
        self.grid_points = self.grid_points.reshape(self.num_x, self.num_y, 3).cpu().numpy()

        self._apply_user_modifications(user_modifications)

    def _create_template_rays(
        self,
        cylinder_radius,
        cylinder_height,
        cylinder_min_z,
        cylinder_sections=8,
        num_interpolate_steps=3,
    ):
        cylinder = trimesh.creation.cylinder(
            radius=cylinder_radius,
            height=cylinder_height,
            sections=cylinder_sections,
        )
        cylinder.apply_translation([0, 0, cylinder_height / 2 + cylinder_min_z])  # make bottom points lie in z = min_z

        bottom_points = np.concatenate(
            [
                cylinder.vertices[1:2],
                cylinder.vertices[:1],
                cylinder.vertices[4::2],
            ],
            axis=0,
        )
        top_points = np.concatenate(
            [
                cylinder.vertices[2:4],
                cylinder.vertices[5::2],
            ],
            axis=0,
        )
        surface_points = (
            bottom_points + (top_points - bottom_points) * np.linspace(0.0, 1.0, num_interpolate_steps)[:, None, None]
        )
        origins = np.mean(surface_points, axis=1, keepdims=True)
        origins = np.repeat(origins, surface_points.shape[1], axis=1)
        directions = surface_points - origins
        return (
            origins.reshape(-1, 3),
            directions.reshape(-1, 3),
        )

    def _apply_user_modifications(self, user_modifications=None):
        if user_modifications is None:
            return

        for x, y in user_modifications["occupied"]:
            self.occupied[x, y] = True
        for x, y in user_modifications["unoccupied"]:
            self.occupied[x, y] = False

    def is_penetration_occured(self, position):
        if isinstance(position, torch.Tensor):
            position = position.cpu().numpy()
        if position.shape[-1] == 2:
            position = misc.add_z(position, z=0)

        origins = self.template_ray_origins + position
        directions = self.template_ray_directions
        for convex_hull in self.obj_convex_hulls:
            if convex_hull.contains(origins).any():
                return True

        _, ray_idx, intersections = self.scene_mesh.ray.intersects_id(
            ray_origins=origins,
            ray_directions=directions,
            multiple_hits=False,
            return_locations=True,
        )
        distances = np.full((self.num_template_rays,), np.inf)
        distances[ray_idx] = np.linalg.norm(intersections - origins[ray_idx], axis=1)
        if (distances <= self.cylinder_radius).any():
            return True
        return False

    def is_off_boundary(self, position):
        if not isinstance(position, np.ndarray):
            position = position.cpu().numpy()
        if position.ndim == 2:
            position = position[0]

        min_bound = self.bbox[0, :2] + np.array([self.x_margin, self.y_margin])
        max_bound = self.bbox[1, :2] - np.array([self.x_margin, self.y_margin])
        if np.any(position[:2] < min_bound) or np.any(position[:2] > max_bound):
            return True
        return False

    def get_closest_grid_position(self, position, exclude_occupied=True, return_tensor=False):
        if not isinstance(position, np.ndarray):
            position = position.cpu().numpy()

        distances = np.linalg.norm(self.grid_points - position, axis=-1)
        if exclude_occupied:
            distances[self.occupied] = np.inf
        idx = np.argmin(distances)
        x, y = idx // self.num_y, idx % self.num_y
        if return_tensor:
            return torch.tensor(
                self.grid_points[x, y],
                dtype=torch.float32,
                device=self.device,
            ).reshape(1, 3)
        return int(x), int(y)

    def get_random_position(self, return_tensor=False):
        unoccupied_grid_points = self.grid_points[~self.occupied]
        position = random.choice(unoccupied_grid_points)
        if return_tensor:
            return torch.tensor(position, dtype=torch.float32, device=self.device)
        return position

    def get_grid_points(self, grid_path, required_num_points=None):
        if required_num_points is not None:
            grid_path = grid_path[:required_num_points]
            grid_path += [(0, 0)] * (required_num_points - len(grid_path))

        indices = (
            np.array([p[0] for p in grid_path]),
            np.array([p[1] for p in grid_path]),
        )
        return self.grid_points[indices]  # (N, 3)

    def _load_grid_map_settings(self, scene_name):
        grid_map_setting_file = Path(f"./configs/scenes/{scene_name}/grid_map.json")

        # create default scene setting file if not exists
        if not grid_map_setting_file.exists():
            default_grid_map_setting = {
                "grid_step": 0.3,
                "x_margin": 0.1,
                "y_margin": 0.1,
                "z_margin": 0.1,
                "cylinder_radius": 0.1,
                "cylinder_height": 0.7,
                "user_modifications": {
                    "occupied": [],
                    "unoccupied": [],
                },
            }
            with open(grid_map_setting_file, "w") as f:
                json.dump(default_grid_map_setting, f, indent=2)

        # load scene settings
        with open(grid_map_setting_file, "r") as f:
            grid_map_settings = json.load(f)

        user_modifications = grid_map_settings.get("user_modifications", None)

        self.grid_step = grid_map_settings["grid_step"]
        self.x_margin = grid_map_settings["x_margin"]
        self.y_margin = grid_map_settings["y_margin"]
        self.z_margin = grid_map_settings["z_margin"]
        self.cylinder_radius = grid_map_settings["cylinder_radius"]
        self.cylinder_height = grid_map_settings["cylinder_height"]

        if user_modifications is None:
            return (
                grid_map_settings["grid_step"],
                grid_map_settings["x_margin"],
                grid_map_settings["y_margin"],
                grid_map_settings["z_margin"],
                grid_map_settings["cylinder_radius"],
                grid_map_settings["cylinder_height"],
                None,
            )

        user_modifications["occupied"] = set(tuple(p) for p in user_modifications["occupied"])
        user_modifications["unoccupied"] = set(tuple(p) for p in user_modifications["unoccupied"])
        return (
            grid_map_settings["grid_step"],
            grid_map_settings["x_margin"],
            grid_map_settings["y_margin"],
            grid_map_settings["z_margin"],
            grid_map_settings["cylinder_radius"],
            grid_map_settings["cylinder_height"],
            user_modifications,
        )

    def _load_scene_bounds(self, scene_name):
        transform_data_file = Path(f"./configs/scenes/{scene_name}/scene.json")
        with open(transform_data_file, "r") as f:
            data = json.load(f)
            bbox = np.array(data["bbox"], dtype=np.float32).reshape(2, 3)
        return bbox

    def build_scene_occ_from_bbox(self, resolution_per_meter=50):
        """
        Generate 1 meter based on the bbox and scene grid=Create a three-dimensional occupancy field with resolution_per_meter and save it:
        - {out_path}.npy  -> Boolean value [nx, ny, nz] Occupancy per voxel
        Convention:x,z is the ground plane,y is the vertical direction (consistent with your motion code):y=Height).
        """
        # --- 1) Reading/Extended bbox ---
        bbox = self.bbox
        bb_min = bbox[0].copy()
        bb_max = bbox[1].copy()

        # --- 2) Calculate the voxel grid dimension ---
        RES = int(resolution_per_meter)  # Grid/Meter
        pitch = 1.0 / RES  # Number of meters per cell
        size_m = bb_max - bb_min  # (dx, dy, dz)
        nx = int(math.ceil(size_m[0] * RES))
        ny = int(math.ceil(size_m[1] * RES))
        nz = int(math.ceil(size_m[2] * RES))  # Vertical direction (height)

        # Ensure the upper bound falls exactly on the cell boundaries to avoid exceeding the right boundary
        bb_max_snap = bb_min + np.array([nx, ny, nz], dtype=np.float32) * pitch

        print(f"[INFO] bbox (min): {bb_min}, (max): {bb_max}, snapped_max: {bb_max_snap}")
        print(f"[INFO] resolution: {RES} vox/m | pitch: {pitch} m/vox | grid: {nx}x{ny}x{nz}")

        self.bbox[1] = bb_max_snap

        out_path = Path(f"./configs/scenes/{self.scene_name}.npy")
        if os.path.exists(out_path):
            scene_occ = np.load(out_path)
            scene_occ = torch.from_numpy(scene_occ).to(device=self.device, dtype=bool)
        else:

            # --- 3) Voxelizing the scene ---
            # Note: Voxelize has requirements for grid size. If the scene is large, it can be voxelized in chunks or directly for the entire scene
            vg = self.scene_mesh.voxelized(pitch=pitch)  # trimesh.voxel.VoxelGrid
            dense = vg.matrix.astype(bool)  # (vx, vy, vz) , Origin at vg.origin

            # --- 4) Place the voxelgrid into the scene_occ large array aligned with the bbox ---
            scene_occ = np.zeros((nx, ny, nz), dtype=bool)
            # Calculate the starting index offset of the voxelgrid within our bbox grid
            # vg.origin Is dense[0,0,0] World coordinates
            offset_m = bb_min - bb_min  # Currently no offset
            offset_idx = np.floor(offset_m * RES + 1e-6).astype(int)  # Prevent floating-point errors
            ox, oy, oz = offset_idx.tolist()

            vx, vy, vz = dense.shape
            # Target placement area (clipping to prevent out-of-bounds)
            tx0, tx1 = max(0, ox), min(nx, ox + vx)
            ty0, ty1 = max(0, oy), min(ny, oy + vy)
            tz0, tz1 = max(0, oz), min(nz, oz + vz)

            sx0, sx1 = max(0, -ox), max(0, -ox) + (tx1 - tx0)
            sy0, sy1 = max(0, -oy), max(0, -oy) + (ty1 - ty0)
            sz0, sz1 = max(0, -oz), max(0, -oz) + (tz1 - tz0)

            if tx0 < tx1 and ty0 < ty1 and tz0 < tz1:
                scene_occ[tx0:tx1, ty0:ty1, tz0:tz1] |= dense[sx0:sx1, sy0:sy1, sz0:sz1]
            else:
                print("[WARN] Voxel grid does not overlap bbox—check transforms/bbox/resolution.")

            #   z up to y up
            scene_occ = np.transpose(scene_occ, (0, 2, 1))  # Swap y and z axes
            # Reverse the z-axis (Because y-> -z), Needs to be flipped once
            scene_occ = np.flip(scene_occ, axis=2)
            # --- 5) Save ---
            np.save(out_path, scene_occ)
            scene_occ = torch.from_numpy(scene_occ.copy()).to(device=self.device, dtype=bool)

        # Compatible with scene_grid_np used in your motion dataset / scene_grid_torch Style

        center = (bb_min + bb_max_snap) / 2.0
        bb_min = bb_min - center
        bb_max_snap = bb_max_snap - center
        scene_grid_np = np.array(
            [bb_min[0], 0, bb_min[1], bb_max_snap[0], bb_max_snap[2] * 2, bb_max_snap[1], nx, nz, ny], dtype=np.float32
        )
        scene_grid_torch = torch.tensor(
            [bb_min[0], 0, bb_min[1], bb_max_snap[0], bb_max_snap[2] * 2, bb_max_snap[1], nx, nz, ny]
        ).to(self.device)

        # Need to convert event coordinates to lingo coordinates

        #  (If your dataset requires it scene_name.pkl List, written according to your data format)
        # with open(os.path.join(out_dir, "scene_name.pkl"), "wb") as f:
        #     import pickle as pkl
        #     pkl.dump([scene_name], f)

        print(f"[OK] Saved Scene_vis/{self.scene_name}.npy with shape {scene_occ.shape}")
        print(f"[OK] scene_grid_np = {scene_grid_np.tolist()} (1m = {RES} grids)")
        self.scene_grid_np = scene_grid_np
        self.scene_grid_torch = scene_grid_torch
        return scene_occ, scene_grid_np, scene_grid_torch

    def build_scene_occ_from_mesh_contains(self, resolution_per_meter=50, batch_size=200000):
        """
        Use the actual mesh’s contains() Calculate voxel occupancy to avoid simplifying complex structures in voxelized data.
        Note: The mesh must be as watertight as possible; otherwise, the results may be unstable.
        Output coordinates that are consistent with build_scene_occ_from_bbox/Save in a format, but with a file name ending in _mesh.
        """
        bbox = self.bbox
        bb_min = bbox[0].copy()
        bb_max = bbox[1].copy()

        RES = int(resolution_per_meter)
        pitch = 1.0 / RES
        size_m = bb_max - bb_min
        nx = int(math.ceil(size_m[0] * RES))
        ny = int(math.ceil(size_m[1] * RES))
        nz = int(math.ceil(size_m[2] * RES))

        bb_max_snap = bb_min + np.array([nx, ny, nz], dtype=np.float32) * pitch
        self.bbox[1] = bb_max_snap

        print(f"[INFO] bbox (min): {bb_min}, (max): {bb_max}, snapped_max: {bb_max_snap}")
        print(f"[INFO] resolution: {RES} vox/m | pitch: {pitch} m/vox | grid: {nx}x{ny}x{nz}")

        out_path = Path(f"./configs/scenes/{self.scene_name}_mesh.npy")
        if os.path.exists(out_path):
            scene_occ = np.load(out_path)
            scene_occ = torch.from_numpy(scene_occ).to(device=self.device, dtype=bool)
        else:
            xs = bb_min[0] + (np.arange(nx, dtype=np.float32) + 0.5) * pitch
            ys = bb_min[1] + (np.arange(ny, dtype=np.float32) + 0.5) * pitch
            zs = bb_min[2] + (np.arange(nz, dtype=np.float32) + 0.5) * pitch

            X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
            points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

            scene_occ = np.zeros((nx, ny, nz), dtype=bool)
            mesh = self.scene_mesh_raw
            total = points.shape[0]
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                mask = mesh.contains(points[start:end])
                scene_occ.reshape(-1)[start:end] = mask

            # z up to y up
            scene_occ = np.transpose(scene_occ, (0, 2, 1))
            scene_occ = np.flip(scene_occ, axis=2)

            np.save(out_path, scene_occ)
            scene_occ = torch.from_numpy(scene_occ.copy()).to(device=self.device, dtype=bool)

        center = (bb_min + bb_max_snap) / 2.0
        bb_min = bb_min - center
        bb_max_snap = bb_max_snap - center
        scene_grid_np = np.array(
            [bb_min[0], 0, bb_min[1], bb_max_snap[0], bb_max_snap[2] * 2, bb_max_snap[1], nx, nz, ny], dtype=np.float32
        )
        scene_grid_torch = torch.tensor(
            [bb_min[0], 0, bb_min[1], bb_max_snap[0], bb_max_snap[2] * 2, bb_max_snap[1], nx, nz, ny]
        ).to(self.device)

        print(f"[OK] Saved Scene_vis/{self.scene_name}_mesh.npy with shape {scene_occ.shape}")
        print(f"[OK] scene_grid_np = {scene_grid_np.tolist()} (1m = {RES} grids)")
        self.scene_grid_np = scene_grid_np
        self.scene_grid_torch = scene_grid_torch
        return scene_occ, scene_grid_np, scene_grid_torch

    def event_to_lingo(self, p_b):
        """
        event (Z-up, The origin is located at the smallest angle of the bbox) -> lingo (Y-up, World coordinates)
        Use the library function zup_to_yup;Take the negative of the z component of mins.
        """
        if isinstance(p_b, torch.Tensor):
            mins = self.scene_grid_torch[:3].to(p_b.device).type_as(p_b).clone()
            mins[..., 2] = -mins[..., 2]  # Only change the sign of z
            p_a_rel = zup_to_yup(p_b)  # Depending on the implementation of your library
            return p_a_rel + mins
        else:
            mins = self.scene_grid_np[:3].copy()
            mins[..., 2] = -mins[..., 2]
            # Ensure that p_b is a numpy array
            if not isinstance(p_b, np.ndarray):
                p_b = np.array(p_b)
            p_a_rel = zup_to_yup(p_b)
            return p_a_rel + mins

    def lingo_to_event(self, p_a):
        """
        lingo (Y-up, World coordinates) -> event (Z-up, The origin is located at the smallest angle of the bbox)
        Subtract the same value first“Take the negative of z”Mins of,Then yup_to_zup.
        """
        if isinstance(p_a, torch.Tensor):
            mins = self.scene_grid_torch[:3].to(p_a.device).type_as(p_a).clone()
            mins[..., 2] = -mins[..., 2]
            rel = p_a - mins
            return yup_to_zup(rel)
        else:
            mins = self.scene_grid_np[:3].copy()
            mins[..., 2] = -mins[..., 2]
            rel = p_a - mins
            return yup_to_zup(rel)
