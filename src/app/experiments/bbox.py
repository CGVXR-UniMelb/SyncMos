import threading
import time

import numpy as np

from src.action_planning.action_planning_module import ActionPlanningModule
from src.motion_synthesis.motion_synthesis_module import MotionSynthesisModule
from src.scene.scene import Scene
from src.type import Character
from src.utils import misc, smpl
from src.utils.polyscope import PolyscopeApp

_DEFAULT_CHARACTER_NAMES = [
    "Amir",
    "Benjamin",
    "Charles",
    "David",
    "Evan",
    "Francis",
]
_DEFAULT_COLOR = [0.7, 0.7, 0.7]
# _DEFAULT_STORY = """
# Amir makes coffee in the kitchen.
# Benjamin works with his laptop sitting on the sofa.
# After making coffee, Amir sits in a near chair to Benjamin and drinks coffee.
# Charles and David are having a conversation in the dining area.
# """

# _DEFAULT_STORY = """
# Amir, Benjamin, Charles, and David sat on the same chair in the restaurant, one after another.
# Each of them sat on the chair alone, then left before the next person sat.
# """

_DEFAULT_STORY = """
Amir sits on the sofa.
"""


class DemoApp(PolyscopeApp):
    def __init__(self, cfg):
        super().__init__(framerate=cfg.framerate)
        self.cfg = cfg
        self.device = cfg.device
        self.scene_name = cfg.scene_name
        self.num_characters = cfg.num_characters
        self.character_names = _DEFAULT_CHARACTER_NAMES[: self.num_characters]
        self.user_instruction = _DEFAULT_STORY.strip()
        self.playing = False
        self.error_occurred = False
        self.is_status_ui_active = True
        self.use_non_blocking_planning = True
        self.logger = misc.get_console_logger("DemoApp", cfg.logging_level)

        # ---------- Export settings ----------
        self.export_out_path = "demo.mp4"
        self.export_fps = 30
        self.export_start_frame = 0  # Export the starting frame
        self.export_end_frame = 0  # Export the ending frame
        self.export_transparent = False
        self.export_reset_first = False
        self.is_exporting = False

        # ---------- Path Visualization settings ----------
        self.show_path_visualization = True  # Path visualization is enabled by default
        self.path_viz_update_interval = 30  # Update every 30 frames
        self.path_viz_last_update = 0
        self.path_visualization_elements = {}  # Store visualization elements
        # Mark whether the first recalculation of the bottle has been annotated back objects.json
        self._annotation_saved = True
        # Object Mesh Handle: label_idx -> polyscope mesh handle
        self.object_mesh_handles = {}

        threading.Thread(
            target=self.action_planning_daemon,
            daemon=True,
        ).start()
        self.reset()

    def action_planning_daemon(self):
        while True:
            time.sleep(0.1)
            if not self.use_non_blocking_planning:
                continue
            if not self.playing or self.error_occurred:
                continue
            if self.fId <= self.max_fId:
                continue

            if self.action_planner.is_new_event_required():
                try:
                    self.action_planner.generate_new_event(self.user_instruction)
                except Exception as e:
                    import traceback

                    error_details = traceback.format_exc()
                    self.logger.error(f"Error occurred while generating new event: {e}")
                    self.logger.error(f"Full traceback:\n{error_details}")
                    self.error_occurred = True
                    self.error(f"Error occurred while generating new event: {e}\nDetails: {error_details}")
            time.sleep(0.3)

    def reset(self):
        super().reset()
        self.scene = Scene(self.scene_name, self.cfg)
        # Do not register the entire scene mesh; register handles for each object individually
        self.object_mesh_handles = {}
        for obj in self.scene.objects:
            name = f"obj_{obj.label}_{obj.idx}"
            handle = self.register_trimesh(name, obj.mesh)
            self.object_mesh_handles[f"{obj.label}_{obj.idx}"] = handle

        self.characters = [Character(name) for name in self.character_names]
        self.character_meshes = []
        self.motion_synthesizer = MotionSynthesisModule(self.scene, self.characters, self.cfg)
        self.action_planner = ActionPlanningModule(
            self.scene, self.characters, self.motion_synthesizer.state_manager, self.cfg
        )

        # Initialize path visualization
        self.init_path_visualization()

        for character in self.characters:
            character.state = self.motion_synthesizer.state_manager.idle_state
            character.body_params = self.motion_synthesizer.create_initial_body_params(
                position=self.scene.grid_map.get_random_position(),
                facing_direction=np.random.rand(2),
                character=character,
            )
            character.current_actions = []
            character.target_position = character.position
            character.target_root_position = character.position
            character.target_facing_direction = character.facing_direction
            character.future_positions = character.position.tile(3, 1)
            character.future_facing_directions = character.facing_direction.tile(3, 1)

            vertices, faces = self.motion_synthesizer.get_current_mesh(character)
            mesh = self.register_mesh(character.name, vertices, faces, color=_DEFAULT_COLOR)
            self.character_meshes.append(mesh)

        self.max_fId = len(self.characters[0].body_params) - 1
        self.fId = len(self.characters[0].body_params)
        self.motion_synthesizer.state_manager.fId = self.fId

        # Immediately update the path visualization after initialization to re-register all elements
        self.update_path_visualization()

        # Recalculate the geometric annotations for bottle_1 first (etc.) to fit the mesh bbox/center/aa_extents Write the results back only during the first run
        try:
            if not self._annotation_saved:
                # Write back results only on the first run objects.json
                self._recompute_object_annotation(
                    label_idx="bottle_1",
                    front_back_axis="shorter",
                    persist=(not self._annotation_saved),
                )
                self._annotation_saved = True
        except Exception as e:
            self.logger.warning(f"Failed to recompute annotation for bottle_1: {e}")

        # Initialize and visualize the bbox of all objects
        self._init_bbox_visualization()

        # Configuration: Attach bottle_1 to the first character’s hand“Hand” (Use wrist joints + Relative offset)
        self.attach_bottle_to_hand = True
        self.attached_label = "bottle_1"
        # Default relative wrist deviation (Right 0.02m Update object properties in memory 0.02m Optional: Persist and write back -0.05m Find the corresponding item and update it
        self.bottle_local_offset = np.array([0.02, 0.02, -0.05], dtype=np.float32)

    # ---------- Utilities: recompute object annotation (bbox etc.) ----------
    def _get_z_aligned_bounding_box(self, mesh):
        import numpy as np
        from scipy.spatial import ConvexHull

        def minimum_bounding_rectangle(points):
            pi2 = np.pi / 2.0
            hull_points = points[ConvexHull(points).vertices]
            edges = hull_points[1:] - hull_points[:-1]
            angles = np.arctan2(edges[:, 1], edges[:, 0])
            angles = np.abs(np.mod(angles, pi2))
            angles = np.unique(angles)
            rotations = np.vstack(
                [np.cos(angles), np.cos(angles - pi2), np.cos(angles + pi2), np.cos(angles)]
            ).T.reshape((-1, 2, 2))
            rot_points = np.dot(rotations, hull_points.T)
            min_x = np.nanmin(rot_points[:, 0], axis=1)
            max_x = np.nanmax(rot_points[:, 0], axis=1)
            min_y = np.nanmin(rot_points[:, 1], axis=1)
            max_y = np.nanmax(rot_points[:, 1], axis=1)
            areas = (max_x - min_x) * (max_y - min_y)
            best_idx = np.argmin(areas)
            x1 = max_x[best_idx]
            x2 = min_x[best_idx]
            y1 = max_y[best_idx]
            y2 = min_y[best_idx]
            r = rotations[best_idx]
            rval = np.zeros((4, 2))
            rval[0] = np.dot([x1, y2], r)
            rval[1] = np.dot([x2, y2], r)
            rval[2] = np.dot([x2, y1], r)
            rval[3] = np.dot([x1, y1], r)
            return rval

        bbox_2D = minimum_bounding_rectangle(np.asarray(mesh.vertices)[:, :2])
        min_z = mesh.bounds[0, 2]
        max_z = mesh.bounds[1, 2]
        bbox = np.vstack(
            [
                np.hstack([bbox_2D, np.full((4, 1), min_z)]),
                np.hstack([bbox_2D, np.full((4, 1), max_z)]),
            ]
        )
        return bbox

    def _cosine_similarity(self, a, b):
        import numpy as np

        return (a * b).sum(axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))

    def _init_obj_annotation(self, obj_mesh, front_back_axis: str = "shorter"):
        import numpy as np

        assert front_back_axis in ["shorter", "longer"], "front_back_axis should be either 'shorter' or 'longer'"
        bbox = self._get_z_aligned_bounding_box(obj_mesh)
        r, s, t = bbox[1] - bbox[0], bbox[3] - bbox[0], bbox[4] - bbox[0]
        shorter_axis = r if np.linalg.norm(r) <= np.linalg.norm(s) else s
        longer_axis = r if shorter_axis is s else s
        orientation = shorter_axis if front_back_axis == "shorter" else longer_axis
        orientation = orientation / np.linalg.norm(orientation)
        width = np.linalg.norm(longer_axis) if front_back_axis == "shorter" else np.linalg.norm(shorter_axis)
        depth = np.linalg.norm(shorter_axis) if front_back_axis == "shorter" else np.linalg.norm(longer_axis)
        height = np.linalg.norm(t)

        obj_normals = np.asarray(obj_mesh.vertex_normals)
        dominant_normal = np.mean(obj_normals, axis=0)
        if self._cosine_similarity(dominant_normal, orientation) < self._cosine_similarity(
            dominant_normal, -orientation
        ):
            orientation = -orientation

        aa_extents = np.max(bbox, axis=0) - np.min(bbox, axis=0)
        center = np.mean(bbox, axis=0)
        return (
            bbox.astype(np.float32),
            orientation.astype(np.float32),
            float(width),
            float(depth),
            float(height),
            aa_extents.astype(np.float32),
            center.astype(np.float32),
        )

    def _recompute_object_annotation(self, label_idx: str, front_back_axis: str = "shorter", persist: bool = False):
        import json
        from pathlib import Path

        # Search for objects
        target = None
        for obj in self.scene.objects:
            if f"{obj.label}_{obj.idx}" == label_idx:
                target = obj
                break
        if target is None:
            raise ValueError(f"Object {label_idx} not found")

        bbox, orientation, width, depth, height, aa_extents, center = self._init_obj_annotation(
            target.mesh, front_back_axis
        )
        # If bbox is registered, it can be refreshed
        target.bbox = bbox
        target.orientation = orientation
        target.width = width
        target.depth = depth
        target.height = height
        target.aa_extents = aa_extents
        target.center = center
        target.has_orientation = True

        # Register bounding boxes for all objects in the scene objects.json
        if persist:
            scene_objects_path = Path(f"./configs/scenes/{self.scene_name}/objects.json")
            with scene_objects_path.open() as f:
                annos = json.load(f)
            # Clear existing bounding box visualizations
            for it in annos:
                if it["label"] == target.label and it["idx"] == target.idx:
                    it["bbox"] = bbox.tolist()
                    it["orientation"] = orientation.tolist()
                    it["width"] = width
                    it["depth"] = depth
                    it["height"] = height
                    it["aa_extents"] = aa_extents.tolist()
                    it["center"] = center.tolist()
                    it["has_orientation"] = True
                    break
            with scene_objects_path.open("w") as f:
                json.dump(annos, f, indent=2)

        # Register bounding boxes for individual objects (yellow wireframes)
        try:
            self._update_bbox_for_object(label_idx)
        except Exception:
            pass

    def _init_bbox_visualization(self):
        """Refresh their wireframe visualizations based on the current bbox of objects"""
        import polyscope as ps

        if not hasattr(self, "_bbox_handles"):
            self._bbox_handles = {}
        # Clean up the existing bbox first
        for name in list(self._bbox_handles.keys()):
            try:
                ps.remove_curve_network(name)
            except Exception:
                pass
        self._bbox_handles.clear()

        # Registration by object bbox (Yellow Line Frame)
        for obj in self.scene.objects:
            name = f"bbox_{obj.label}_{obj.idx}"
            handle = self.register_bounding_box(name, obj.bbox[:, :3], radius=0.01, color=[1.0, 1.0, 0.0])
            self._bbox_handles[name] = handle

    def _update_bbox_for_object(self, label_idx: str):
        """Refresh line frame visualization based on the current bbox of objects"""
        import polyscope as ps

        target = None
        for obj in self.scene.objects:
            if f"{obj.label}_{obj.idx}" == label_idx:
                target = obj
                break
        if target is None:
            return
        name = f"bbox_{label_idx}"
        # Remove the old ones
        try:
            ps.remove_curve_network(name)
        except Exception:
            pass
        # Re-register
        handle = self.register_bounding_box(name, target.bbox[:, :3], radius=0.01, color=[1.0, 1.0, 0.0])
        if not hasattr(self, "_bbox_handles"):
            self._bbox_handles = {}
        self._bbox_handles[name] = handle

    def do_export(self):
        if self.is_exporting:
            return
        self.is_exporting = True
        try:
            from src.utils.export import export_polyscope_animation

            self.logger.info(f"Export started -> {self.export_out_path}")

            # Calculate the export range
            start_frame = self.export_start_frame
            end_frame = self.export_end_frame
            total_frames = end_frame - start_frame + 1

            self.logger.info(f"Exporting frames {start_frame} to {end_frame} (total: {total_frames})")

            export_polyscope_animation(
                self,
                out_path=self.export_out_path,
                fps=self.export_fps,
                start_frame=start_frame,  # Use new parameter names
                end_frame=end_frame,  # Use new parameter names
                transparent=self.export_transparent,
                reset_first=self.export_reset_first,
            )
            self.logger.info(f"Export finished -> {self.export_out_path}")
        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            self.error_occurred = True
            self.error(f"Export failed: {e}\nFull traceback:\n{error_details}")
        finally:
            self.is_exporting = False

    def callback(self):
        if self.fId <= self.max_fId:  # when showing previous frames
            for character, character_mesh in zip(self.characters, self.character_meshes):
                vertices, _ = self.motion_synthesizer.get_mesh_at(character, self.fId)
                character_mesh.update_vertex_positions(vertices)

            # Update the path visualization when reviewing, showing the path status corresponding to the current frame
            # Decide whether to update in real-time based on user settings
            if self.show_path_visualization:  # Update every frame during review
                self.update_path_visualization_for_frame()

            if self.playing:
                self.fId += 1
            return

        if not self.playing:
            return
        if self.error_occurred:
            return

        # if non-blocking planning is disabled, planning is done in the main thread (here)
        # in other case, planning is done in a separate thread (`action_planning_daemon`)
        if not self.use_non_blocking_planning:
            if self.action_planner.is_new_event_required():
                try:
                    self.action_planner.generate_new_event(self.user_instruction)
                except Exception as e:
                    import traceback

                    error_details = traceback.format_exc()
                    self.logger.error(f"Error occurred while generating new event: {e}")
                    self.logger.error(f"Full traceback:\n{error_details}")
                    self.error_occurred = True
                    self.error(f"Error occurred while generating new event: {e}\nDetails: {error_details}")

        # if a new event is created, it's allocated to the involved characters
        if not self.action_planner.event_queue.empty():
            event, parsed_event = self.action_planner.event_queue.get()
            event.created_fId = self.fId
            try:
                self.action_planner.allocate_new_plan(event, parsed_event)
            except Exception as e:
                import traceback

                error_details = traceback.format_exc()
                self.logger.error(f"Error occurred while allocating new plan: {e}\nFull traceback:\n{error_details}")
                self.error_occurred = True
                self.error(f"Error occurred while allocating new plan: {e}\nFull traceback:\n{error_details}")

        # update character state / motion
        for character, character_mesh in zip(self.characters, self.character_meshes):
            self.motion_synthesizer.advance_motion_frame(character)

        # Attach bottle_1 to the character’s right wrist + Local offset)
        if getattr(self, "attach_bottle_to_hand", False) and len(self.characters) > 0:
            try:
                char = self.characters[0]
                # Get the current frame’s SMPL joint and obtain the right wrist’s world coordinates
                joints = self.motion_synthesizer.bm(char.body_params[-1:]).joints[0].detach().cpu().numpy()  # (J,3)
                right_wrist_idx = smpl.get_joint_idx("right_wrist")
                wrist = joints[right_wrist_idx]

                # Construct a local coordinate system (based on the character’s orientation)
                fd = char.facing_direction.cpu().numpy()[0]
                fwd = fd / (np.linalg.norm(fd) + 1e-8)
                fwd = np.array([fwd[0], fwd[1], 0.0], dtype=np.float32)
                up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                right = np.cross(up, fwd)
                right /= np.linalg.norm(right) + 1e-8
                R = np.stack([right, fwd, up], axis=1)
                if not hasattr(self, "bottle_local_offset"):
                    self.bottle_local_offset = np.array([0.02, 0.02, -0.05], dtype=np.float32)

                center = wrist + (R @ self.bottle_local_offset)

                # Determine the orientation using the forearm direction (wrist) - Elbow, and projection onto the horizontal plane
                try:
                    right_elbow_idx = smpl.get_joint_idx("right_elbow")
                    elbow = joints[right_elbow_idx]
                    forearm = wrist - elbow
                    norm = np.linalg.norm(forearm) + 1e-8
                    forearm = forearm / norm
                    forearm_xy = np.array([forearm[0], forearm[1], 0.0], dtype=np.float32)
                    if np.linalg.norm(forearm_xy) < 1e-4:
                        new_orientation = fwd.copy()
                    else:
                        new_orientation = forearm_xy / (np.linalg.norm(forearm_xy) + 1e-8)
                except Exception:
                    new_orientation = fwd.copy()

                self.scene.move_object(
                    "bottle_1",
                    center,
                    new_orientation=new_orientation,
                    move_mesh=True,
                    update_scene_graph=False,
                    update_grid_map=False,
                )
                # Only updating the vertices of the object handle
                try:
                    obj = next(o for o in self.scene.objects if f"{o.label}_{o.idx}" == "bottle_1")
                    h = self.object_mesh_handles.get("bottle_1") or self.object_mesh_handles.get("obj_bottle_1")
                    if h is not None:
                        h.update_vertex_positions(obj.mesh.vertices)
                except Exception as e2:
                    self.logger.debug(f"Failed to update attached bottle mesh vertices: {e2}")
                # Refreshing the bbox visualization
                self._update_bbox_for_object("bottle_1")
            except Exception as e:
                self.logger.warning(f"Failed to attach bottle_1: {e}")

        # Updating the path visualization
        if self.show_path_visualization and self.fId - self.path_viz_last_update >= self.path_viz_update_interval:
            self.update_path_visualization()
            self.path_viz_last_update = self.fId

        # update character mesh
        for character, character_mesh in zip(self.characters, self.character_meshes):
            vertices, _ = self.motion_synthesizer.get_current_mesh(character)
            character_mesh.update_vertex_positions(vertices)

        # update frame idx
        self.fId += 1
        self.max_fId = self.fId - 1
        self.motion_synthesizer.state_manager.fId = self.fId
        assert self.max_fId == len(self.characters[0].body_params) - 1

    def ui_callback(self):
        self.gui.SetNextWindowSize((1000, 200), self.gui.ImGuiCond_FirstUseEver)
        self.gui.Begin("Status", self.is_status_ui_active)
        self.gui.TextUnformatted(f"Characters: {self.character_names}")
        self.gui.Separator()
        if self.gui.TreeNode("Available Action Labels"):
            for action_label in self.motion_synthesizer.motion_generator.get_candidate_actions():
                self.gui.TextUnformatted(f" - {action_label}")
            self.gui.TreePop()
        if self.gui.TreeNode("Events"):
            for event in self.action_planner.narrator.previous_events:
                if self.gui.TreeNode(f"{event} ({event.state})"):
                    if event.parsed_event is not None:
                        for character, parsed in event.parsed_event.items():
                            self.gui.TextUnformatted(f" - {character}:")
                            self.gui.TextUnformatted(f"  - Target Actions: {parsed['target_action']}")
                            self.gui.TextUnformatted(f"  - Relationships: {parsed['relationships']}")
                    self.gui.TreePop()
            self.gui.TreePop()

        if self.gui.TreeNode("Characters"):
            for character in self.characters:
                if self.gui.TreeNode(f"{character.name}"):
                    self.gui.TextUnformatted(
                        f"Event: {character.event} ({character.event.state if character.event is not None else ''})"
                    )
                    self.gui.TextUnformatted(f"State: {character.state}")
                    self.gui.TextUnformatted(f"Current Actions: {character.current_actions}")
                    self.gui.TextUnformatted(f"Target Actions: {character.target_actions}")
                    if self.gui.TreeNode("Relationships: "):
                        for relationship in character.relationships:
                            self.gui.TextUnformatted(f" - {relationship}")
                        self.gui.TreePop()
                    self.gui.TreePop()
            self.gui.TreePop()

        if self.gui.TreeNode("Export Animation (MP4)"):
            # Output file path
            changed, self.export_out_path = self.gui.InputText("Output Path", self.export_out_path)

            # FPS
            _, self.export_fps = self.gui.SliderInt("FPS", self.export_fps, 1, 120)

            self.gui.TextUnformatted(f"Current Frame: {self.fId} / {self.max_fId}")
            self.gui.TextUnformatted(f"Total Available Frames: {self.max_fId + 1}")

            # Frame number selection: starting frame and ending frame
            self.gui.TextUnformatted("Export Frame Range:")
            _, self.export_start_frame = self.gui.SliderInt("Start Frame", self.export_start_frame, 0, self.max_fId)
            _, self.export_end_frame = self.gui.SliderInt("End Frame", self.export_end_frame, 0, self.max_fId)

            # Calculating the actual number of frames exported
            total_export_frames = self.export_end_frame - self.export_start_frame + 1
            self.gui.TextUnformatted(f"Total Export Frames: {total_export_frames}")

            # Preview of the export range
            self.gui.TextUnformatted(f"Export Range: Frame {self.export_start_frame} to {self.export_end_frame}")

            # Other switches
            _, self.export_transparent = self.gui.Checkbox("Transparent Background", self.export_transparent)
            _, self.export_reset_first = self.gui.Checkbox("Reset Before Export", self.export_reset_first)

            # Status display
            self.gui.TextUnformatted(f"Status: {'Exporting...' if self.is_exporting else 'Idle'}")

            # Export button
            if self.gui.Button("Export MP4") and not self.is_exporting:
                self.do_export()

            self.gui.TreePop()

        if self.gui.TreeNode("Path Visualization"):
            # Path visualization switch
            _, self.show_path_visualization = self.gui.Checkbox("Show Path Visualization", self.show_path_visualization)

            if self.gui.Button("Update Path Visualization"):
                self.update_path_visualization()

            self.gui.SameLine()
            if self.gui.Button("Clear Path Visualization"):
                self.clear_path_visualization()

            # Update interval settings
            _, self.path_viz_update_interval = self.gui.SliderInt(
                "Update Interval (frames)", self.path_viz_update_interval, 1, 100
            )

            # Display path information
            if self.gui.TreeNode("Path Information"):

                # Display data save status
                if hasattr(self, "path_visualization_data") and self.path_visualization_data:
                    data_status = f"Data Saved: {len(self.path_visualization_data)} characters"
                    self.gui.TextUnformatted(data_status)

                    # Display frame data statistics for each character
                    for char_name, char_frames in self.path_visualization_data.items():
                        if char_frames:
                            frame_count = len(char_frames)
                            frame_range = (
                                f"{min(char_frames.keys())} - {max(char_frames.keys())}" if frame_count > 0 else "None"
                            )
                            self.gui.TextUnformatted(f"  {char_name}: {frame_count} frames ({frame_range})")
                else:
                    self.gui.TextUnformatted("Data Saved: None")

                self.gui.Separator()

                for character in self.characters:
                    if self.gui.TreeNode(f"{character.name} Path"):
                        if hasattr(character, "grid_path") and character.grid_path:
                            self.gui.TextUnformatted(f"Grid Path Length: {len(character.grid_path)}")
                            self.gui.TextUnformatted(
                                f"Current Grid: {character.grid_path[0] if character.grid_path else 'None'}"
                            )
                            self.gui.TextUnformatted(
                                f"Target Grid: {character.grid_path[-1] if character.grid_path else 'None'}"
                            )
                        else:
                            self.gui.TextUnformatted("No grid path available")

                        if hasattr(character, "target_position") and character.target_position is not None:
                            target = character.target_position.cpu().numpy()
                            self.gui.TextUnformatted(
                                f"Target Position: ({target[0, 0]:.2f}, {target[0, 1]:.2f}, {target[0, 2]:.2f})"
                            )

                        if hasattr(character, "target_root_position") and character.target_root_position is not None:
                            root = character.target_root_position.cpu().numpy()
                            self.gui.TextUnformatted(
                                f"Target Root Position: ({root[0, 0]:.2f}, {root[0, 1]:.2f}, {root[0, 2]:.2f})"
                            )

                        self.gui.TreePop()
                self.gui.TreePop()

            self.gui.TreePop()
        self.gui.End()

        self.gui.Text(f"Error Occurred: {self.error_occurred}")
        self.gui.Text(f"Left Query Count: {self.action_planner.narrator.left_query_count}")
        self.gui.Text(f"Narrator: {self.action_planner.narrator.model} (temperature: {self.action_planner.narrator.temperature})")  # fmt: skip
        self.gui.Text(f"Event Parser: {self.action_planner.event_parser.model} (temperature: {self.action_planner.event_parser.temperature})")  # fmt: skip

        _, self.fId = self.gui.SliderInt("Frame", self.fId, 0, self.max_fId)

        if self.gui.Button(("Play" if not self.playing else "Pause")):
            self.playing = not self.playing

        self.gui.SameLine()
        if self.gui.Button("Reset"):
            self.playing = False
            self.reset()

        self.gui.SameLine()
        if self.gui.Button("Increase Query Budget"):
            self.action_planner.narrator.left_query_count += 1

        _, self.use_non_blocking_planning = self.gui.Checkbox(
            "Use Non-Blocking Planning", self.use_non_blocking_planning
        )
        _, self.user_instruction = self.gui.InputTextMultiline("User Instruction", self.user_instruction)

        # Automatically update path visualization
        if self.show_path_visualization and self.fId - self.path_viz_last_update >= self.path_viz_update_interval:
            self.update_path_visualization()
            self.path_viz_last_update = self.fId

    def init_path_visualization(self):
        """Initialize path visualization"""
        try:
            # Ensure the character list exists
            if not hasattr(self, "characters") or not self.characters:
                self.logger.warning("No characters available for path visualization")
                self.path_visualization_elements = {}
                return

            # Clear previous visualization elements
            self.clear_path_visualization()

            # Initialize the visualization element dictionary
            self.path_visualization_elements = {}

            # Create visualization elements for each character
            for character in self.characters:
                if character is None:
                    continue

                char_name = character.name
                if char_name is None:
                    char_name = f"Character_{len(self.path_visualization_elements)}"

                self.path_visualization_elements[char_name] = {
                    "path_curve": None,
                    "current_pos": None,
                    "target_pos": None,
                    "root_pos": None,
                    "path_points": None,
                    "label": None,
                }

            self.logger.info(f"Initialized path visualization for {len(self.path_visualization_elements)} characters")

        except Exception as e:
            self.logger.error(f"Failed to initialize path visualization: {e}")
            import traceback

            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Ensure there is at least one empty dictionary
            self.path_visualization_elements = {}

    def update_path_visualization(self):
        """Update path visualization and save all necessary data"""
        try:
            import polyscope as ps
            import trimesh
            import numpy as np

            # Ensure the visualization element dictionary is initialized
            if not hasattr(self, "path_visualization_elements") or not self.path_visualization_elements:
                self.init_path_visualization()

            # Initialize the path data storage dictionary (if not existing)
            if not hasattr(self, "path_visualization_data"):
                self.path_visualization_data = {}

            for character in self.characters:
                char_name = character.name

                # Ensure that the visual element dictionary for this character exists
                if char_name not in self.path_visualization_elements:
                    self.path_visualization_elements[char_name] = {
                        "path_curve": None,
                        "current_pos": None,
                        "target_pos": None,
                        "root_pos": None,
                        "path_points": None,
                        "label": None,
                    }

                # Initialize the frame data dictionary for this character (if it doesn’t exist)
                if char_name not in self.path_visualization_data:
                    self.path_visualization_data[char_name] = {}

                # Obtain the current frame ID
                current_frame_id = self.fId if hasattr(self, "fId") else 0

                # Save all the data required for the current frame
                self.path_visualization_data[char_name][current_frame_id] = {
                    "current_position": None,
                    "target_position": None,
                    "target_root_position": None,
                    "path_points": None,
                    "path_edges": None,
                }

                # 1. Save the data of the current position
                if hasattr(character, "position") and character.position is not None:
                    try:
                        current_pos = character.position.cpu().numpy()
                        self.path_visualization_data[char_name][current_frame_id][
                            "current_position"
                        ] = current_pos.copy()

                        # Draw the current position (Blue sphere)
                        current_sphere = trimesh.creation.uv_sphere(radius=0.1)
                        current_sphere.apply_translation(current_pos[0])

                        # Check if it already exists; if it does, remove it first
                        if self.path_visualization_elements[char_name]["current_pos"] is not None:
                            try:
                                ps.remove_surface_mesh(self.path_visualization_elements[char_name]["current_pos"])
                            except:
                                pass

                        self.register_trimesh(f"{char_name}_current", current_sphere, color=[0.0, 0.5, 1.0])
                        self.path_visualization_elements[char_name]["current_pos"] = f"{char_name}_current"
                    except Exception as e:
                        self.logger.warning(f"Failed to visualize current position for {char_name}: {e}")

                # 2. Save the data of the target position
                if hasattr(character, "target_position") and character.target_position is not None:
                    try:
                        target_pos = character.target_position.cpu().numpy()
                        self.path_visualization_data[char_name][current_frame_id]["target_position"] = target_pos.copy()

                        # Draw the target position (Red sphere)
                        target_sphere = trimesh.creation.uv_sphere(radius=0.1)
                        target_sphere.apply_translation(target_pos[0])

                        # Check if it already exists; if it does, remove it first
                        if self.path_visualization_elements[char_name]["target_pos"] is not None:
                            try:
                                ps.remove_surface_mesh(self.path_visualization_elements[char_name]["target_pos"])
                            except:
                                pass

                        self.register_trimesh(f"{char_name}_target", target_sphere, color=[1.0, 0.0, 0.0])
                        self.path_visualization_elements[char_name]["target_pos"] = f"{char_name}_target"
                    except Exception as e:
                        self.logger.warning(f"Failed to visualize target position for {char_name}: {e}")

                # 3. Save the data of the target root position
                if hasattr(character, "target_root_position") and character.target_root_position is not None:
                    try:
                        root_pos = character.target_root_position.cpu().numpy()
                        self.path_visualization_data[char_name][current_frame_id][
                            "target_root_position"
                        ] = root_pos.copy()

                        # Draw the target root position (Orange sphere)
                        root_sphere = trimesh.creation.uv_sphere(radius=0.05)
                        root_sphere.apply_translation(root_pos[0])
                        self.register_trimesh(f"{char_name}_root", root_sphere, color=[1.0, 0.5, 0.0])
                        self.path_visualization_elements[char_name]["root_pos"] = f"{char_name}_root"
                    except Exception as e:
                        self.logger.warning(f"Failed to visualize root position for {char_name}: {e}")

                # 4. Save the grid path data
                if hasattr(character, "grid_path") and character.grid_path:
                    try:
                        path_points = []
                        for grid_point in character.grid_path:
                            if hasattr(self.scene, "grid_map") and hasattr(self.scene.grid_map, "grid_points"):
                                world_pos = self.scene.grid_map.grid_points[grid_point]
                                path_points.append(world_pos)

                        if len(path_points) > 1:
                            # Ensure path_points is a numpy array
                            if not isinstance(path_points[0], np.ndarray):
                                path_points = np.array(path_points)
                            elif isinstance(path_points, list):
                                path_points = np.array(path_points)

                            # Save the path point data
                            self.path_visualization_data[char_name][current_frame_id][
                                "path_points"
                            ] = path_points.copy()

                            # Create path lines
                            edges = []
                            for i in range(len(path_points) - 1):
                                edges.append([i, i + 1])

                            # Ensure edges are a numpy array and save them
                            edges = np.array(edges)
                            self.path_visualization_data[char_name][current_frame_id]["path_edges"] = edges.copy()

                            # Check if it already exists; if it does, remove it first
                            if self.path_visualization_elements[char_name]["path_curve"] is not None:
                                try:
                                    ps.remove_curve_network(self.path_visualization_elements[char_name]["path_curve"])
                                except:
                                    pass

                            # Register as a curve network
                            cn = ps.register_curve_network(
                                f"{char_name}_path", path_points, edges, color=[0.0, 1.0, 0.0], radius=0.02
                            )
                            try:
                                cn.set_radius(0.2, relative=False)
                            except Exception:
                                pass
                            self.path_visualization_elements[char_name]["path_curve"] = f"{char_name}_path"

                            # Add path point markers
                            for i, point in enumerate(path_points):
                                if i == 0:  # Start point
                                    start_sphere = trimesh.creation.uv_sphere(radius=0.05)
                                    start_sphere.apply_translation(point)
                                    self.register_trimesh(
                                        f"{char_name}_path_start", start_sphere, color=[0.0, 0.8, 0.0]
                                    )
                                elif i == len(path_points) - 1:  # End point
                                    end_sphere = trimesh.creation.uv_sphere(radius=0.05)
                                    end_sphere.apply_translation(point)
                                    self.register_trimesh(f"{char_name}_path_end", end_sphere, color=[0.0, 0.8, 0.0])
                    except Exception as e:
                        self.logger.warning(f"Failed to visualize grid path for {char_name}: {e}")
                        import traceback

                        self.logger.warning(f"Grid path error traceback: {traceback.format_exc()}")

                # 5. Save and draw character name labels
                if hasattr(character, "position") and character.position is not None:
                    try:
                        current_pos = character.position.cpu().numpy()
                        # Add text labels above the characters
                        label_pos = current_pos[0] + np.array([0, 0, 0.3])
                        label_sphere = trimesh.creation.uv_sphere(radius=0.02)
                        label_sphere.apply_translation(label_pos)
                        self.register_trimesh(f"{char_name}_label", label_sphere, color=[1.0, 1.0, 1.0])
                        self.path_visualization_elements[char_name]["label"] = f"{char_name}_label"
                    except Exception as e:
                        self.logger.warning(f"Failed to visualize label for {char_name}: {e}")

            # self.logger.info("Path visualization updated successfully")

        except Exception as e:
            self.logger.error(f"Failed to update path visualization: {e}")
            import traceback

            self.logger.error(f"Traceback: {traceback.format_exc()}")

            # Add debugging information
            self.logger.error("Debug info:")
            self.logger.error(f"  - Characters: {[c.name if c else 'None' for c in self.characters]}")
            self.logger.error(f"  - Path visualization elements: {self.path_visualization_elements}")
            if hasattr(self, "scene") and hasattr(self.scene, "grid_map"):
                self.logger.error(f"  - Grid map available: {hasattr(self.scene.grid_map, 'grid_points')}")
            else:
                self.logger.error("  - Grid map not available")

    def update_path_visualization_for_frame(self):
        """Update path visualization for specific frames (used to display historical path states during rewind)"""
        try:
            import polyscope as ps
            import trimesh
            import numpy as np

            # Ensure the visualization element dictionary is initialized
            if not hasattr(self, "path_visualization_elements") or not self.path_visualization_elements:
                self.init_path_visualization()

            # Obtain the current frame ID
            current_frame_id = self.fId

            # Check if there is any saved path visualization data
            if not hasattr(self, "path_visualization_data") or not self.path_visualization_data:
                self.logger.warning("No path visualization data available")
                return

            for character in self.characters:
                char_name = character.name

                # Ensure that the visual element dictionary for this character exists
                if char_name not in self.path_visualization_elements:
                    self.path_visualization_elements[char_name] = {
                        "path_curve": None,
                        "current_pos": None,
                        "target_pos": None,
                        "root_pos": None,
                        "path_points": None,
                        "label": None,
                    }

                # Check if the character has saved data in the current frame, No, because every time self.path_viz_update_interval Draw the saved current position
                if (
                    char_name not in self.path_visualization_data
                    or current_frame_id not in self.path_visualization_data[char_name]
                ):
                    # self.logger.warning(f"No data for character {char_name} at frame {current_frame_id}")
                    continue

                frame_data = self.path_visualization_data[char_name][current_frame_id]

                # 1. Draw the saved target position (Blue sphere)
                if frame_data.get("current_position") is not None:
                    try:
                        current_pos = frame_data["current_position"]
                        current_sphere = trimesh.creation.uv_sphere(radius=0.1)
                        current_sphere.apply_translation(current_pos[0])

                        # Check if it already exists; if it does, remove it first
                        if self.path_visualization_elements[char_name]["current_pos"] is not None:
                            try:
                                ps.remove_surface_mesh(self.path_visualization_elements[char_name]["current_pos"])
                            except:
                                pass

                        self.register_trimesh(f"{char_name}_current", current_sphere, color=[0.0, 0.5, 1.0])
                        self.path_visualization_elements[char_name]["current_pos"] = f"{char_name}_current"
                    except Exception as e:
                        self.logger.warning(f"Failed to visualize saved current position for {char_name}: {e}")

                # 2. Draw the saved target root position (Red sphere)
                if frame_data.get("target_position") is not None:
                    try:
                        target_pos = frame_data["target_position"]
                        target_sphere = trimesh.creation.uv_sphere(radius=0.1)
                        target_sphere.apply_translation(target_pos[0])

                        # Check if it already exists; if it does, remove it first
                        if self.path_visualization_elements[char_name]["target_pos"] is not None:
                            try:
                                ps.remove_surface_mesh(self.path_visualization_elements[char_name]["target_pos"])
                            except:
                                pass

                        self.register_trimesh(f"{char_name}_target", target_sphere, color=[1.0, 0.0, 0.0])
                        self.path_visualization_elements[char_name]["target_pos"] = f"{char_name}_target"
                    except Exception as e:
                        self.logger.warning(f"Failed to visualize saved target position for {char_name}: {e}")

                # 3. Draw the saved grid path (Orange sphere)
                if frame_data.get("target_root_position") is not None:
                    try:
                        root_pos = frame_data["target_root_position"]
                        root_sphere = trimesh.creation.uv_sphere(radius=0.05)
                        root_sphere.apply_translation(root_pos[0])
                        self.register_trimesh(f"{char_name}_root", root_sphere, color=[1.0, 0.5, 0.0])
                        self.path_visualization_elements[char_name]["root_pos"] = f"{char_name}_root"
                    except Exception as e:
                        self.logger.warning(f"Failed to visualize saved target root position for {char_name}: {e}")

                # 4. Green lines (Add character name labels)
                if frame_data.get("path_points") is not None and frame_data.get("path_edges") is not None:
                    try:
                        path_points = frame_data["path_points"]
                        edges = frame_data["path_edges"]

                        if len(path_points) > 1:
                            # Check if it already exists; if it does, remove it first
                            if self.path_visualization_elements[char_name]["path_curve"] is not None:
                                try:
                                    ps.remove_curve_network(self.path_visualization_elements[char_name]["path_curve"])
                                except:
                                    pass

                            # Register as a curve network
                            ps.register_curve_network(
                                f"{char_name}_path", path_points, edges, color=[0.0, 1.0, 0.0], radius=0.02
                            )
                            self.path_visualization_elements[char_name]["path_curve"] = f"{char_name}_path"

                            # Add path point markers
                            for i, point in enumerate(path_points):
                                if i == 0:  # Start point
                                    start_sphere = trimesh.creation.uv_sphere(radius=0.05)
                                    start_sphere.apply_translation(point)
                                    self.register_trimesh(
                                        f"{char_name}_path_start", start_sphere, color=[0.0, 0.8, 0.0]
                                    )
                                elif i == len(path_points) - 1:  # End point
                                    end_sphere = trimesh.creation.uv_sphere(radius=0.05)
                                    end_sphere.apply_translation(point)
                                    self.register_trimesh(f"{char_name}_path_end", end_sphere, color=[0.0, 0.8, 0.0])
                    except Exception as e:
                        self.logger.warning(f"Failed to visualize saved grid path for {char_name}: {e}")
                        import traceback

                        self.logger.warning(f"Grid path error traceback: {traceback.format_exc()}")

                # 5. Add Role Name Tab - Use the saved current position
                if frame_data.get("current_position") is not None:
                    try:
                        current_pos = frame_data["current_position"]
                        # Add text labels above the characters
                        label_pos = current_pos[0] + np.array([0, 0, 0.3])
                        label_sphere = trimesh.creation.uv_sphere(radius=0.02)
                        label_sphere.apply_translation(label_pos)
                        self.register_trimesh(f"{char_name}_label", label_sphere, color=[1.0, 1.0, 1.0])
                        self.path_visualization_elements[char_name]["label"] = f"{char_name}_label"
                    except Exception as e:
                        self.logger.warning(f"Failed to visualize label for {char_name}: {e}")

        except Exception as e:
            self.logger.error(f"Failed to update path visualization for frame: {e}")
            import traceback

            self.logger.error(f"Traceback: {traceback.format_exc()}")

    def clear_path_visualization(self):
        """Clear path visualization"""
        try:
            import polyscope as ps

            # Ensure the visualization element dictionary exists
            if not hasattr(self, "path_visualization_elements"):
                return

            # Remove all grids related to path visualization
            for char_name, elements in self.path_visualization_elements.items():
                if elements is None:
                    continue

                for element_type, element_name in elements.items():
                    if element_name is not None:
                        try:
                            # Try to remove different types of visualization elements
                            try:
                                ps.remove_surface_mesh(element_name)
                            except:
                                pass

                            try:
                                ps.remove_curve_network(element_name)
                            except:
                                pass

                        except Exception as e:
                            self.logger.debug(f"Failed to remove element {element_name}: {e}")

            # Clear the element dictionary
            self.path_visualization_elements = {}

            # Clear the saved path data
            if hasattr(self, "path_visualization_data"):
                self.path_visualization_data = {}

            # Force a refresh of the Polyscope display
            try:
                ps.refresh()
            except:
                pass

        except Exception as e:
            self.logger.error(f"Failed to clear path visualization: {e}")
            import traceback

            self.logger.error(f"Traceback: {traceback.format_exc()}")


def run(cfg) -> None:
    """Run the retained experiment with configuration supplied by the shared CLI."""
    app = DemoApp(cfg)
    app.run()
