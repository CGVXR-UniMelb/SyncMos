import json
from pathlib import Path
from typing import List

import numpy as np
import trimesh
from tqdm import tqdm

from ..type import Object
from .grid_map import GridMap
from .scene_graph import SceneGraph


class Scene:
    def __init__(self, scene_name, cfg):
        self.cfg = cfg
        self.scene_name = scene_name
        self.objects = self._get_objects(scene_name)
        self.scene_mesh = self._get_scene_mesh(self.objects)
        self.grid_map = GridMap(scene_name, self.scene_mesh, self.objects, cfg)
        self.scene_graph = SceneGraph(scene_name, self.cfg, self.objects)

    def _get_objects(self, scene_name) -> List[Object]:
        meshes = self._load_object_meshes(scene_name)
        annotations = self._load_annotations(scene_name)

        objects = []
        for id_, mesh in enumerate(meshes):
            anno = annotations[id_]
            objects.append(
                Object(
                    id=anno["id"],
                    label=anno["label"],
                    idx=anno["idx"],
                    aa_extents=np.array(anno["aa_extents"], dtype=np.float32),
                    orientation=np.array(anno["orientation"], dtype=np.float32),
                    has_orientation=anno["has_orientation"],
                    width=anno["width"],
                    depth=anno["depth"],
                    height=anno["height"],
                    center=np.array(anno["center"], dtype=np.float32),
                    bbox=np.array(anno["bbox"], dtype=np.float32),
                    mesh=mesh,
                    sit_offset_horizontal=anno["sit_offset_horizontal"],
                    sit_offset_vertical=anno["sit_offset_vertical"],
                )
            )
        return sorted(objects, key=lambda obj: obj.label)

    def _get_scene_mesh(self, objects: List[Object]):
        meshes = [obj.mesh.copy() for obj in objects]
        for mesh in meshes:
            if hasattr(mesh.visual, "to_color"):
                mesh.visual = mesh.visual.to_color()
        return trimesh.util.concatenate(meshes)

    def _load_object_meshes(self, scene_name):
        scene_cfg_file = Path(f"./configs/scenes/{scene_name}/scene_cfg.json")
        scene_cfg = json.load(scene_cfg_file.open())
        scene_transform = self._load_scene_transform(scene_name)

        meshes = []
        for id_, (_, data) in tqdm(enumerate(scene_cfg.items()), desc=f"Loading {scene_name}", total=len(scene_cfg)):
            assert id_ == data["id"], f"ID mismatch: {id_} != {data['id']}"
            model_id = data["model_id"]
            location = data["location"]
            rotation = data["rotation"]
            scale = data["scale"]

            # load mesh and apply transformations
            mesh = self._load_obj_mesh(model_id)
            transform = trimesh.transformations.compose_matrix(
                translate=location,
                angles=rotation,
                scale=scale,
            )
            mesh.apply_transform(transform)
            mesh.apply_transform(scene_transform)

            meshes.append(mesh)

        return meshes

    def _load_scene_transform(self, scene_name):
        transform_data_file = Path(f"./configs/scenes/{scene_name}/scene.json")
        with open(transform_data_file, "r") as f:
            data = json.load(f)
            transform = np.array(data["transform"], dtype=np.float32).reshape(4, 4)
        return transform

    def _load_obj_mesh(self, model_id: str):
        obj_file = f"./data/HSSD/objects/{model_id[0]}/{model_id}.glb"
        obj_mesh = trimesh.load(obj_file)
        obj_mesh = obj_mesh.to_geometry()
        obj_mesh = obj_mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]),
        )  # y-up to z-up
        return obj_mesh

    def _load_annotations(self, scene_name):
        annotation_file = Path(f"./configs/scenes/{scene_name}/objects.json")
        with annotation_file.open() as f:
            annotations = json.load(f)

        return {anno["id"]: {k: v for k, v in anno.items()} for anno in annotations}

    def move_object(
        self,
        object_name: str,
        new_center: np.ndarray,
        new_orientation: np.ndarray = None,
        *,
        move_mesh: bool = False,
        update_scene_graph: bool = True,
        update_grid_map: bool = False,
    ):
        """
        Move objects in the scene to new positions

        Args:
            object_name: Object name, in the format of "label_idx",Such as "bottle_1"
            new_center: New center position [x, y, z]
            new_orientation: New direction vector [x, y, z] (Optional)
            move_mesh: Whether to synchronize translation geometry (keep convex hull)/Ray detection is consistent)
            update_scene_graph: Whether to refresh the relationship graph (lightweight)
            update_grid_map: Whether to refresh the grid map (relatively time-consuming)
        """
        # Find objects
        target_object = None
        for obj in self.objects:
            if f"{obj.label}_{obj.idx}" == object_name:
                target_object = obj
                break

        if target_object is None:
            print(f"Warning: Object '{object_name}' not found in scene")
            return False

        old_center = target_object.center.copy()
        # Update object positions
        target_object.center = np.array(new_center, dtype=np.float32)

        # Update direction (if provided)
        if new_orientation is not None:
            target_object.orientation = np.array(new_orientation, dtype=np.float32)
            target_object.has_orientation = True

        # Synchronous movement of geometry (optional)
        if move_mesh:
            delta = target_object.center - old_center
            self._translate_object_mesh(target_object, delta)

        # Recalculate bounding box
        self._update_object_bbox(target_object)

        # Refresh dependencies
        if update_scene_graph:
            self.refresh_scene_graph()
        if update_grid_map:
            self.refresh_grid_map()

        # print(f"Moved {object_name} to position {new_center}")
        return True

    def _update_object_bbox(self, obj: Object):
        """Recalculate bounding box based on object center position and size"""
        half_w, half_d, half_h = obj.width / 2, obj.depth / 2, obj.height / 2
        x, y, z = obj.center

        # Simplified bounding box calculation (assuming object is axis-aligned)
        obj.bbox = np.array(
            [
                [x + half_w, y + half_d, z - half_h],  # 4 points on the bottom surface
                [x - half_w, y + half_d, z - half_h],
                [x - half_w, y - half_d, z - half_h],
                [x + half_w, y - half_d, z - half_h],
                [x + half_w, y + half_d, z + half_h],  # 4 points on the top surface
                [x - half_w, y + half_d, z + half_h],
                [x - half_w, y - half_d, z + half_h],
                [x + half_w, y - half_d, z + half_h],
            ],
            dtype=np.float32,
        )

    def _translate_object_mesh(self, obj: Object, delta: np.ndarray):
        """Synchronous translation of object geometry (rigid body movement)"""
        obj.mesh.apply_translation(delta.astype(np.float32))

    def refresh_scene_graph(self):
        """Rebuild relationship graph with latest objects (lightweight)"""
        self.scene_graph = SceneGraph(self.scene_name, self.cfg, self.objects)

    def refresh_grid_map(self):
        """With latest geometry/Rebuild grid_map using convex hull (relatively time-consuming)"""
        self.scene_mesh = self._get_scene_mesh(self.objects)
        self.grid_map = GridMap(self.scene_name, self.scene_mesh, self.objects, self.cfg)

    def get_object_position(self, object_name: str):
        """
        Get the current position of the object

        Args:
            object_name: Object name, in the format of "label_idx"

        Returns:
            tuple: (center, orientation) Or None if the object does not exist
        """
        for obj in self.objects:
            if f"{obj.label}_{obj.idx}" == object_name:
                return obj.center.copy(), obj.orientation.copy()
        return None

    def list_objects(self):
        """List the names and positions of all objects in the scene"""
        object_info = []
        for obj in self.objects:
            object_info.append(
                {
                    "name": f"{obj.label}_{obj.idx}",
                    "center": obj.center.tolist(),
                    "orientation": obj.orientation.tolist(),
                }
            )
        return object_info
