import threading
import time

import numpy as np
import torch
import os

from src.action_planning.action_planning_module import ActionPlanningModule
from src.motion_synthesis.motion_synthesis_module import MotionSynthesisModule
from src.scene.scene import Scene
from src.type import Character
from src.utils import misc, smpl
from src.utils.polyscope import PolyscopeApp
from src.utils.camera_control import CameraController
from src.motion_synthesis.states import State
from src.utils.grid_utils import grid_to_world
from src.utils.motion_io import save_motion_data

_DEFAULT_CHARACTER_NAMES = [
    "Amir",
    "Benjamin",
    "Charles",
    "David",
    "Evan",
    "Francis",
    "George",
    "Henry",
    "Isaac",
    "Jacob",
    "Kevin",
    "Leo",
    "Michael",
    "Nathan",
    "Oliver",
    "Peter",
    "Quentin",
    "Ryan",
    "Samuel",
    "Thomas",
    "Victor",
    "William",
    "Xavier",
    "Yusuf",
    "Zachary",
]
_DEFAULT_COLOR = [0.7, 0.7, 0.7]


# LINGO
_DEFAULT_STORY = """
Amir, Benjamin, and Charles pick up the beer. 
They walk to the table together.
They drink the beer together.
"""
_DEFAULT_STORY = """
Amir, Benjamin, and Charles pick up the beer. 
They sit at the table together.
They drink the beer together.
"""
# _DEFAULT_STORY = """
# Amir, Benjamin, and Charles pick up the beer seperately.
# They sit at the table seperately.
# They drink the beer together.
# """
# _DEFAULT_STORY = """
# Charles play the guitar near the table.
# Amir, Benjamin pick up the beer.
# They sit at the table near Charles and listen to the music together.
# They drink the beer together.
# """
# _DEFAULT_STORY = """
# Amir, Benjamin, and Charles sit on the sofa together.
# They drink the beer together.
# """
character_initial_grid_positions = {
    "Amir": [17, 26],
    "Benjamin": [9, 15],
    "Charles": [7, 19],
}
# character_initial_grid_positions = {
# }


class DemoApp(PolyscopeApp):
    def __init__(self, cfg):

        super().__init__(framerate=cfg.framerate)
        self.cfg = cfg
        self.device = cfg.device

        # Initialize logger first - needed for logging during initialization
        self.logger = misc.get_console_logger("DemoApp", cfg.logging_level)

        self.scene_name = cfg.scene_name
        self.num_characters = cfg.num_characters
        self.character_names = _DEFAULT_CHARACTER_NAMES[: self.num_characters]
        self.scene = Scene(self.scene_name, self.cfg)

        self.user_instruction = _DEFAULT_STORY.strip()
        # env_instruction = os.environ.get("USER_INSTRUCTION", "").strip()
        # if env_instruction:
        #     self.user_instruction = env_instruction
        self.character_initial_grid_positions = character_initial_grid_positions

        self.motion_dir = os.environ.get("MOTION_OUT_DIR", "motion/parallel")

        self.playing = True
        self.error_occurred = False
        self.is_status_ui_active = True
        self.use_non_blocking_planning = False

        # ---------- Camera/View settings ----------
        self.camera_controller = CameraController(
            scene_name=self.scene_name, config_dir="./configs/scenes", logger=self.logger
        )

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
        self.visualize_grid_path = False  # Temporarily Disable Path Visualization
        self.path_viz_update_interval = 30  # Update every 30 frames
        self.path_viz_last_update = 0
        self.path_visualization_elements = {}  # Store visualization elements

        # ---------- Motion Visualization settings ----------
        self.show_origin_motion = False  # False=Display Time-warped Action, True=Display Original Action

        threading.Thread(
            target=self.action_planning_daemon,
            daemon=True,
        ).start()
        self.reset()

        self.current_bottle_grip_offset = 0.0
        self.current_wrist_to_palm_offset = 0.0
        self.bottle_grip_offset = (
            -0.1
        )  # Offset distance along the long axis of the bottle towards the body of the bottle (meters)
        self.wrist_to_palm_offset = -0.1  # Distance from the wrist to the center of the palm (meters)

        # Load saved camera configuration if exists
        self.camera_controller.load_camera_config()

        self.saved = False

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
        # self.scene = Scene(self.scene_name, self.cfg)
        # self.register_trimesh("scene", self.scene.scene_mesh)
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
        if self.visualize_grid_path:
            self.init_path_visualization()

        for character in self.characters:
            character.state = self.motion_synthesizer.state_manager.idle_state

            if character.name in self.character_initial_grid_positions:
                grid_xy = self.character_initial_grid_positions[character.name]
                gx, gy = int(grid_xy[0]), int(grid_xy[1])
                x_min, y_min = self.scene.grid_map.bbox[0][:2]
                world = grid_to_world(
                    gx, gy, grid_size=self.scene.grid_map.grid_step, bbox_min=(x_min, y_min), return_center=True, z=0.0
                )

                character.body_params = self.motion_synthesizer.create_initial_body_params(
                    position=world, facing_direction=np.random.rand(2), character=character
                )
            else:
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
        # self.update_path_visualization()

        # Apply the saved or default camera perspective
        if hasattr(self, "camera_controller"):
            preset = self.camera_controller.current_camera_preset
            if preset:
                self.camera_controller.apply_camera_preset(preset)
            else:
                self.camera_controller.apply_custom_camera()

    def save_motion_data(self):
        """Save animation data to .npz Files"""
        try:
            # Get narrator type and parser mode
            narrator_type = getattr(self.cfg.narrator, "type", "baseline")
            parser_mode = getattr(self.cfg.event_parser, "parser_mode", "single")

            save_motion_data(
                characters=self.characters,
                scene_name=self.scene_name,
                num_characters=self.num_characters,
                user_instruction=self.user_instruction,
                character_initial_grid_positions=self.character_initial_grid_positions,
                fId=self.fId,
                max_fId=self.max_fId,
                motion_dir=self.motion_dir,
                narrator_type=narrator_type,
                parser_mode=parser_mode,
                logger=self.logger,
            )
        except Exception as e:
            self.logger.error(f"Failed to save motion data: {e}")

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
                # Ensure that the frame index does not exceed the actual number of frames for the character
                char_max_frame = len(character.body_params) - 1
                frame_idx = min(self.fId, char_max_frame)
                if frame_idx >= 0:
                    vertices, _ = self.motion_synthesizer.get_mesh_at(
                        character, frame_idx, use_origin=self.show_origin_motion
                    )
                    character_mesh.update_vertex_positions(vertices)

            # Update the path visualization when reviewing, showing the path status corresponding to the current frame
            # Decide whether to update in real-time based on user settings
            if self.visualize_grid_path and self.show_path_visualization:  # Update every frame during review
                self.update_path_visualization_for_frame()

            if self.playing:
                self.fId += 1

            if self.fId > 10 and len(self.characters) > 0:
                char = self.characters[0]
                if len(char.body_params) > 0:
                    if (
                        self.current_bottle_grip_offset != self.bottle_grip_offset
                        or self.current_wrist_to_palm_offset != self.wrist_to_palm_offset
                    ):
                        self.current_bottle_grip_offset = self.bottle_grip_offset
                        self.current_wrist_to_palm_offset = self.wrist_to_palm_offset
                        self.attach_bottle_to_hand(char, bottle_name="bottle_1")
            return
        # current plan is finished
        if (self.fId > 3000) or (
            hasattr(self.action_planner.narrator, "plan_generated")
            and hasattr(self.action_planner.narrator, "event_list")
            and self.action_planner.narrator.plan_generated
            and self.action_planner.narrator.event_list == []
            and all([character.state == State.IDLE for character in self.characters])
            and all([character.target_actions == [] for character in self.characters])
        ):
            self.playing = False
            # Saving animation data
            if not self.saved:
                self.save_motion_data()
                self.saved = True
            self.unshow()
            return
        # for baseline method (ONLY when NOT using timeline narrator)
        # Fix: Should not exit prematurely in Timeline mode; wait for all events in event_list to complete
        elif (
            self.fId > 100
            and not (hasattr(self.action_planner.narrator, "event_list") and self.action_planner.narrator.event_list)
            and all([character.state == State.IDLE for character in self.characters])
            and all([character.target_actions == [] for character in self.characters])
        ):
            self.playing = False
            # Saving animation data
            if not self.saved:
                self.save_motion_data()
                self.saved = True
            self.unshow()
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

        # Updating the path visualization
        if (
            self.visualize_grid_path
            and self.show_path_visualization
            and self.fId - self.path_viz_last_update >= self.path_viz_update_interval
        ):
            self.update_path_visualization()
            self.path_viz_last_update = self.fId

        # update character mesh
        for character, character_mesh in zip(self.characters, self.character_meshes):
            vertices, _ = self.motion_synthesizer.get_current_mesh(character, use_origin=self.show_origin_motion)
            character_mesh.update_vertex_positions(vertices)

        # update frame idx
        self.fId += 1
        self.max_fId = self.fId - 1
        self.motion_synthesizer.state_manager.fId = self.fId
        # when interacting with other characters, the body_params is not updated by the first character
        # thus, we cannot fulfill the condition
        # assert self.max_fId == len(self.characters[0].body_params) - 1

        # Attach the bottle to the character’s hand
        if self.fId > 10 and len(self.characters) > 0:
            char = self.characters[0]
            if len(char.body_params) > 0:
                self.attach_bottle_to_hand(char, bottle_name="bottle_1")

    def attach_bottle_to_hand(self, character, bottle_name="bottle_1"):
        return
        """
        Attach the bottle to the character’s hand, allowing it to move and rotate with the hand’s posture
        
        Args:
            character: Character object
            bottle_name: Name of the bottle object (default) "bottle_1")
        """
        # Ensure that the frame index is within the valid range
        if len(character.body_params) == 0:
            self.logger.warning(f"Character {character.name} has no body_params")
            return

        frame_idx = min(self.fId, len(character.body_params) - 1)
        current_body_params = character.body_params[frame_idx : frame_idx + 1]

        # Get the current frame’s SMPL joint and obtain the right wrist’s world coordinates
        joints = self.motion_synthesizer.bm(current_body_params).joints[0].detach().cpu().numpy()  # (J,3)
        right_wrist_idx = smpl.get_joint_idx("right_wrist")
        wrist = joints[right_wrist_idx]

        # 1) Global rotation of the wrist, R_wrist (3x3)
        try:
            with torch.no_grad():
                transl = current_body_params[:, :3]
                global_orient = smpl.quaternion_to_axis_angle(current_body_params[:, 3:7])
                body_pose = smpl.quaternion_to_axis_angle(current_body_params[:, 7:].reshape(-1, 21, 4)).reshape(-1, 63)

            body_params_dict = {
                "transl": transl,
                "global_orient": global_orient,
                "body_pose": body_pose,
            }

            device = transl.device
            num_frames = 1
            for key in self.motion_synthesizer.bm.required_keys:
                if key not in body_params_dict:
                    body_params_dict[key] = torch.zeros(
                        (num_frames, self.motion_synthesizer.bm.body_param_dims[key]),
                        dtype=torch.float32,
                        device=device,
                    )

            out = self.motion_synthesizer.bm.bm(
                return_full_pose=True,
                pose2rot=True,
                **body_params_dict,
            )

            if hasattr(out, "A"):  # smplx BodyModel is common
                R_wrist = out.A[0, right_wrist_idx, :3, :3].detach().cpu().numpy()
            elif hasattr(out, "joint_transforms"):
                R_wrist = out.joint_transforms[0, right_wrist_idx, :3, :3].detach().cpu().numpy()
            elif hasattr(out, "rot_mats"):
                R_wrist = out.rot_mats[0, right_wrist_idx].detach().cpu().numpy()
            else:
                raise RuntimeError("SMPL output missing wrist rotation data")
        except Exception as e:
            self.logger.debug(f"Failed to get wrist rotation: {e}, using fallback")
            # Downgrade option: Use the forearm direction
            try:
                right_elbow_idx = smpl.get_joint_idx("right_elbow")
                elbow = joints[right_elbow_idx]
                forearm = wrist - elbow
                forearm = forearm / (np.linalg.norm(forearm) + 1e-8)
                up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                right = np.cross(forearm, up)
                right = right / (np.linalg.norm(right) + 1e-8)
                up_corrected = np.cross(right, forearm)
                R_wrist = np.stack([right, forearm, up_corrected], axis=1)
            except Exception:
                # Final downgrade option
                fd = character.facing_direction.cpu().numpy()[0]
                fwd = fd / (np.linalg.norm(fd) + 1e-8)
                fwd = np.array([fwd[0], fwd[1], 0.0], dtype=np.float32)
                up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                right = np.cross(up, fwd)
                right /= np.linalg.norm(right) + 1e-8
                R_wrist = np.stack([right, fwd, up], axis=1)

        # 2) Select the one for alignment"Hand axis" (Choose one according to your needs)
        #    Common: R_wrist[:,1] (Forward direction of the hand) or R_wrist[:,2] (Palm facing upward)
        hand_axis_world = R_wrist[:, 2]  # Keep your hands on this. Keep the bottle straight
        hand_axis_world = hand_axis_world / (np.linalg.norm(hand_axis_world) + 1e-8)

        # 3) The bottle’s"Long axis"Direction within the model (assumed to be canonical) +Z)
        z_can = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # 4) Calculate the rotation of z_can to hand_axis_world (Rodrigues formula)
        v = hand_axis_world
        k = np.cross(z_can, v)
        s = np.linalg.norm(k)
        c = float(np.dot(z_can, v))
        if s < 1e-8:
            # Collinear, may require a 180-degree flip
            if c > 0.0:
                R_align = np.eye(3, dtype=np.float32)
            else:
                # 180°: Flip around any axis perpendicular to z_can
                R_align = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
        else:
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=np.float32)
            R_align = np.eye(3, dtype=np.float32) + K + K @ K * ((1.0 - c) / (s**2))

        # 5)  (Optional) Fine-tuning rotation for gripping: e.g., tilting the bottle mouth slightly downward/Outward
        #    Here, a fixed small offset is given (a slight rotation around the bottle’s own X-axis)
        theta = np.deg2rad(10.0)  # 10-degree fine-tuning
        R_grasp = np.array(
            [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]], dtype=np.float32
        )

        R_obj = R_align @ R_grasp  # Final 3D rotation of the bottle (world system)

        # Calculate the placement of the bottle: position the bottle within the palm
        # Strategy: Start from the wrist position, shift in the opposite direction along the palm, to bring the bottle into the palm

        # Calculate the center position of the bottle:
        # 1. Start from the wrist
        # 2. Move forward along the hand (from wrist to fingers) to the center of the palm
        # 3. Then adjust along the long axis of the bottle

        # Obtain the actual direction from wrist to fingers: Use the extension in the forearm direction
        # Or use the direction from wrist to the base of the middle finger (if there is data for the middle finger joint)
        try:
            # Try using the extension direction of the forearm (at the elbow)->The direction of the wrist) as the forward direction of the hand
            right_elbow_idx = smpl.get_joint_idx("right_elbow")
            elbow = joints[right_elbow_idx]
            forearm_dir = wrist - elbow
            palm_forward = forearm_dir / (np.linalg.norm(forearm_dir) + 1e-8)
        except Exception:
            # Downgrade: Use the wrist rotation matrix
            palm_forward = R_wrist[:, 1]
            palm_forward = palm_forward / (np.linalg.norm(palm_forward) + 1e-8)

        # Move from wrist to the center of the palm (about 10cm along the forearm extension direction)
        palm_center = wrist + palm_forward * self.wrist_to_palm_offset

        # Offset from the center of the palm along the long axis of the bottle
        center = palm_center - hand_axis_world * self.bottle_grip_offset

        # Add a lateral offset to the inner side of the palm
        palm_side = R_wrist[:, 0]  # Palm horizontal
        palm_side = palm_side / (np.linalg.norm(palm_side) + 1e-8)
        center += palm_side * 0.02  # Horizontal offset of 2cm

        # 6) Use the forward horizontal projection of the wrist as yaw, only updating pose
        yaw_dir_world = R_wrist[:, 1]
        hand_xy = np.array([yaw_dir_world[0], yaw_dir_world[1], 0.0], dtype=np.float32)
        if np.linalg.norm(hand_xy) < 1e-6:
            # Downgrade: Use character orientation
            fd = character.facing_direction.cpu().numpy()[0]
            new_orientation = np.array([fd[0], fd[1], 0.0], dtype=np.float32)
            new_orientation = new_orientation / (np.linalg.norm(new_orientation) + 1e-8)
        else:
            new_orientation = hand_xy / (np.linalg.norm(hand_xy) + 1e-8)

        self.scene.move_object(
            bottle_name,
            center,
            new_orientation=new_orientation,
            move_mesh=False,
            update_scene_graph=False,
            update_grid_map=False,
        )

        # 7) Use rest vertices to construct absolute pose, avoiding cumulative rotation
        try:
            obj = next(o for o in self.scene.objects if f"{o.label}_{o.idx}" == bottle_name)
            h = self.object_mesh_handles.get(bottle_name) or self.object_mesh_handles.get(f"obj_{bottle_name}")

            # Obtain the original vertices of the bottle during initialization (only done once, never deleted thereafter)
            if not hasattr(self, "_bottle_original_rest_V"):
                # First acquisition: Save original vertices to permanent variables
                self._bottle_original_rest_V = obj.mesh.vertices.copy()
                self._bottle_original_rest_centroid = self._bottle_original_rest_V.mean(0, keepdims=True)
                self.logger.info(
                    f"Permanently cached original bottle vertices, centroid at: {self._bottle_original_rest_centroid}"
                )

            # Use the permanently saved original vertices
            V_rest = self._bottle_original_rest_V
            c_local = self._bottle_original_rest_centroid

            # Construct vertices in the world coordinate system: Rotate and move to the target position
            V_world = (V_rest - c_local) @ R_obj.T + center

            # Directly set vertex positions
            obj.mesh.vertices[:] = V_world
            if h is not None:
                h.update_vertex_positions(obj.mesh.vertices)

            # Debugging: Output position information every 100 frames
            if self.fId % 100 == 11:
                current_centroid = obj.mesh.vertices.mean(0)
                self.logger.info(
                    f"Frame {self.fId}: wrist={wrist}, center={center}, bottle_centroid={current_centroid}"
                )
        except Exception as e2:
            self.logger.debug(f"Failed to apply 3D rotation to bottle mesh: {e2}")

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

        # Motion Visualization Toggle
        if self.gui.TreeNode("Motion Visualization"):
            self.gui.TextUnformatted("Toggle between original and time-warped motion:")

            # Check if any character has origin data
            has_origin_data = any(
                c.body_params_origin is not None and len(c.body_params_origin) > 0 for c in self.characters
            )

            if not has_origin_data:
                self.gui.TextUnformatted("(Original motion data not yet available)")
                self.gui.TextUnformatted("(Will be available after parallel interactions)")

            changed, new_value = self.gui.Checkbox("Show Original Motion", self.show_origin_motion)
            if changed:
                self.show_origin_motion = new_value
                self.logger.info(
                    f"Motion visualization switched to: {'Original' if self.show_origin_motion else 'Time-Warped'}"
                )

            self.gui.Separator()
            if self.show_origin_motion:
                self.gui.TextUnformatted("Currently showing: Original Motion (No Time Warp)")
                if has_origin_data:
                    # Show which characters have origin data
                    for c in self.characters:
                        if c.body_params_origin is not None and len(c.body_params_origin) > 0:
                            self.gui.TextUnformatted(f"  - {c.name}: {len(c.body_params_origin)} origin frames")
            else:
                self.gui.TextUnformatted("Currently showing: Time-Warped Motion")

            self.gui.TreePop()

        if self.gui.TreeNode("Bottle Attachment Settings"):
            # Initialize default values (if not existing)
            if not hasattr(self, "bottle_grip_offset"):
                self.bottle_grip_offset = -0.1
            if not hasattr(self, "wrist_to_palm_offset"):
                self.wrist_to_palm_offset = -0.1

            # Bottle Longitudinal Offset
            self.gui.TextUnformatted("Bottle Grip Offset (along bottle axis):")
            changed1, new_grip_offset = self.gui.SliderFloat("Grip Offset (m)", self.bottle_grip_offset, -0.3, 0.3)
            if changed1:
                self.bottle_grip_offset = new_grip_offset
                self.logger.info(f"Bottle grip offset updated to: {self.bottle_grip_offset}")

            # Offset from Wrist to Palm Center
            self.gui.TextUnformatted("Wrist to Palm Center Offset (along forearm):")
            changed2, new_palm_offset = self.gui.SliderFloat("Palm Offset (m)", self.wrist_to_palm_offset, -0.3, 0.3)
            if changed2:
                self.wrist_to_palm_offset = new_palm_offset
                self.logger.info(f"Wrist to palm offset updated to: {self.wrist_to_palm_offset}")

            # Display Current Value
            self.gui.Separator()
            self.gui.TextUnformatted(f"Current Grip Offset: {self.bottle_grip_offset:.3f} m")
            self.gui.TextUnformatted(f"Current Palm Offset: {self.wrist_to_palm_offset:.3f} m")

            # Reset Button
            if self.gui.Button("Reset to Default"):
                self.bottle_grip_offset = -0.1
                self.wrist_to_palm_offset = -0.1
                self.logger.info("Bottle attachment settings reset to default")

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

        if self.visualize_grid_path and self.gui.TreeNode("Path Visualization"):
            # Path visualization switch
            _, self.show_path_visualization = self.gui.Checkbox("Show Path Visualization", self.show_path_visualization)

            if self.visualize_grid_path and self.gui.Button("Update Path Visualization"):
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

        # Camera control UI
        if self.gui.TreeNode("Camera Control"):
            self.gui.TextUnformatted("Camera Presets:")

            # Preset perspective button
            preset_names = self.camera_controller.get_preset_names()
            for i, preset_name in enumerate(preset_names):
                if self.gui.Button(f"Load: {preset_name}"):
                    self.camera_controller.apply_camera_preset(preset_name)
                    # Update the value in the input field
                    self._cam_pos_x_str = f"{self.camera_controller.custom_camera_position[0]:.2f}"
                    self._cam_pos_y_str = f"{self.camera_controller.custom_camera_position[1]:.2f}"
                    self._cam_pos_z_str = f"{self.camera_controller.custom_camera_position[2]:.2f}"
                    self._cam_target_x_str = f"{self.camera_controller.custom_camera_target[0]:.2f}"
                    self._cam_target_y_str = f"{self.camera_controller.custom_camera_target[1]:.2f}"
                    self._cam_target_z_str = f"{self.camera_controller.custom_camera_target[2]:.2f}"
                if i < len(preset_names) - 1 and (i + 1) % 4 != 0:
                    self.gui.SameLine()

            self.gui.Separator()

            # Customize camera position - Use the input field instead of the slider
            self.gui.TextUnformatted("Custom Camera Position:")

            # Initialize the string buffer (if not existing)
            if not hasattr(self, "_cam_pos_x_str"):
                self._cam_pos_x_str = f"{self.camera_controller.custom_camera_position[0]:.2f}"
            if not hasattr(self, "_cam_pos_y_str"):
                self._cam_pos_y_str = f"{self.camera_controller.custom_camera_position[1]:.2f}"
            if not hasattr(self, "_cam_pos_z_str"):
                self._cam_pos_z_str = f"{self.camera_controller.custom_camera_position[2]:.2f}"

            # X coordinate input
            changed_x, self._cam_pos_x_str = self.gui.InputText("Cam X", self._cam_pos_x_str)
            if changed_x:
                try:
                    new_x = float(self._cam_pos_x_str)
                    self.camera_controller.set_position(
                        new_x,
                        self.camera_controller.custom_camera_position[1],
                        self.camera_controller.custom_camera_position[2],
                    )
                except ValueError:
                    pass  # Ignore invalid inputs

            # Y coordinate input
            changed_y, self._cam_pos_y_str = self.gui.InputText("Cam Y", self._cam_pos_y_str)
            if changed_y:
                try:
                    new_y = float(self._cam_pos_y_str)
                    self.camera_controller.set_position(
                        self.camera_controller.custom_camera_position[0],
                        new_y,
                        self.camera_controller.custom_camera_position[2],
                    )
                except ValueError:
                    pass

            # Z coordinate input
            changed_z, self._cam_pos_z_str = self.gui.InputText("Cam Z", self._cam_pos_z_str)
            if changed_z:
                try:
                    new_z = float(self._cam_pos_z_str)
                    self.camera_controller.set_position(
                        self.camera_controller.custom_camera_position[0],
                        self.camera_controller.custom_camera_position[1],
                        new_z,
                    )
                except ValueError:
                    pass

            # Customize camera target - Use the input field
            self.gui.TextUnformatted("Camera Target:")

            # Initialize the string buffer (if not existing)
            if not hasattr(self, "_cam_target_x_str"):
                self._cam_target_x_str = f"{self.camera_controller.custom_camera_target[0]:.2f}"
            if not hasattr(self, "_cam_target_y_str"):
                self._cam_target_y_str = f"{self.camera_controller.custom_camera_target[1]:.2f}"
            if not hasattr(self, "_cam_target_z_str"):
                self._cam_target_z_str = f"{self.camera_controller.custom_camera_target[2]:.2f}"

            # Target X input
            changed_tx, self._cam_target_x_str = self.gui.InputText("Target X", self._cam_target_x_str)
            if changed_tx:
                try:
                    new_tx = float(self._cam_target_x_str)
                    self.camera_controller.set_target(
                        new_tx,
                        self.camera_controller.custom_camera_target[1],
                        self.camera_controller.custom_camera_target[2],
                    )
                except ValueError:
                    pass

            # Target Y input
            changed_ty, self._cam_target_y_str = self.gui.InputText("Target Y", self._cam_target_y_str)
            if changed_ty:
                try:
                    new_ty = float(self._cam_target_y_str)
                    self.camera_controller.set_target(
                        self.camera_controller.custom_camera_target[0],
                        new_ty,
                        self.camera_controller.custom_camera_target[2],
                    )
                except ValueError:
                    pass

            # Target Z input
            changed_tz, self._cam_target_z_str = self.gui.InputText("Target Z", self._cam_target_z_str)
            if changed_tz:
                try:
                    new_tz = float(self._cam_target_z_str)
                    self.camera_controller.set_target(
                        self.camera_controller.custom_camera_target[0],
                        self.camera_controller.custom_camera_target[1],
                        new_tz,
                    )
                except ValueError:
                    pass

            # Projection mode selection
            self.gui.TextUnformatted("Projection Mode:")
            if self.gui.RadioButton("Perspective", self.camera_controller.current_projection_mode == "perspective"):
                self.camera_controller.set_projection_mode("perspective")
                self.camera_controller.apply_custom_camera()
            self.gui.SameLine()
            if self.gui.RadioButton("Orthographic", self.camera_controller.current_projection_mode == "orthographic"):
                self.camera_controller.set_projection_mode("orthographic")
                self.camera_controller.apply_custom_camera()

            self.gui.Separator()

            # Apply/Save button
            if self.gui.Button("Apply Custom Camera"):
                self.camera_controller.apply_custom_camera()

            self.gui.SameLine()
            if self.gui.Button("Save Camera Config"):
                self.camera_controller.save_camera_config()

            self.gui.SameLine()
            if self.gui.Button("Reload Config"):
                # Reload configuration file
                self.camera_controller.load_camera_config()
                # Apply the custom camera position saved in the configuration file (instead of the preset)
                self.camera_controller.apply_custom_camera()
                # Update the value in the input field
                self._cam_pos_x_str = f"{self.camera_controller.custom_camera_position[0]:.2f}"
                self._cam_pos_y_str = f"{self.camera_controller.custom_camera_position[1]:.2f}"
                self._cam_pos_z_str = f"{self.camera_controller.custom_camera_position[2]:.2f}"
                self._cam_target_x_str = f"{self.camera_controller.custom_camera_target[0]:.2f}"
                self._cam_target_y_str = f"{self.camera_controller.custom_camera_target[1]:.2f}"
                self._cam_target_z_str = f"{self.camera_controller.custom_camera_target[2]:.2f}"
                self.logger.info(
                    f"Reloaded camera config: "
                    f"pos=[{self.camera_controller.custom_camera_position[0]:.2f}, "
                    f"{self.camera_controller.custom_camera_position[1]:.2f}, "
                    f"{self.camera_controller.custom_camera_position[2]:.2f}], "
                    f"target=[{self.camera_controller.custom_camera_target[0]:.2f}, "
                    f"{self.camera_controller.custom_camera_target[1]:.2f}, "
                    f"{self.camera_controller.custom_camera_target[2]:.2f}]"
                )

            # Read current perspective button
            self.gui.Separator()
            self.gui.TextUnformatted("Capture Current View from Polyscope:")
            self.gui.TextUnformatted("(Use after manually adjusting camera with mouse)")
            if self.gui.Button("Read Current Camera View"):
                # Read the actual current camera state from Polyscope (instead of internal variables)
                camera_info = self.camera_controller.get_current_camera_from_polyscope()
                if camera_info:
                    # Update the input field to display the current values
                    self._cam_pos_x_str = f"{camera_info['position'][0]:.2f}"
                    self._cam_pos_y_str = f"{camera_info['position'][1]:.2f}"
                    self._cam_pos_z_str = f"{camera_info['position'][2]:.2f}"
                    self._cam_target_x_str = f"{camera_info['target'][0]:.2f}"
                    self._cam_target_y_str = f"{camera_info['target'][1]:.2f}"
                    self._cam_target_z_str = f"{camera_info['target'][2]:.2f}"
                    self.logger.info(
                        f"Captured camera view from Polyscope: "
                        f"pos=[{camera_info['position'][0]:.2f}, "
                        f"{camera_info['position'][1]:.2f}, "
                        f"{camera_info['position'][2]:.2f}], "
                        f"target=[{camera_info['target'][0]:.2f}, "
                        f"{camera_info['target'][1]:.2f}, "
                        f"{camera_info['target'][2]:.2f}]"
                    )

            self.gui.SameLine()
            if self.gui.Button("Copy to Clipboard"):
                # Copy current camera parameters to the clipboard (as Python code)
                pos = self.camera_controller.custom_camera_position
                target = self.camera_controller.custom_camera_target
                projection = self.camera_controller.current_projection_mode

                clipboard_text = f"""# Camera Configuration
camera.set_position({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})
camera.set_target({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})
camera.set_projection_mode("{projection}")
camera.apply_custom_camera()"""

                try:
                    import pyperclip

                    pyperclip.copy(clipboard_text)
                    self.logger.info("Camera parameters copied to clipboard")
                except ImportError:
                    self.logger.warning("pyperclip not installed, cannot copy to clipboard")
                except Exception as e:
                    self.logger.warning(f"Failed to copy to clipboard: {e}")

            # Display the configuration file path
            self.gui.Separator()
            self.gui.TextUnformatted(f"Config File: {self.camera_controller.camera_config_file}")
            self.gui.TextUnformatted(f"(Scene-specific, saved separately for {self.scene_name})")

            # Save as a new preset
            self.gui.Separator()
            self.gui.TextUnformatted("Save Current View as Preset:")
            if not hasattr(self, "_new_preset_name"):
                self._new_preset_name = "My View"

            changed, self._new_preset_name = self.gui.InputText("Preset Name", self._new_preset_name)
            if self.gui.Button("Save as New Preset"):
                if self._new_preset_name.strip():
                    self.camera_controller.save_current_view_as_preset(self._new_preset_name.strip())
                    self.camera_controller.save_camera_config()

            # Display current camera information
            self.gui.Separator()
            state = self.camera_controller.get_current_state()
            self.gui.TextUnformatted(f"Current Preset: {state['preset']}")
            self.gui.TextUnformatted(
                f"Position: [{state['position'][0]:.2f}, " f"{state['position'][1]:.2f}, {state['position'][2]:.2f}]"
            )
            self.gui.TextUnformatted(
                f"Target: [{state['target'][0]:.2f}, " f"{state['target'][1]:.2f}, {state['target'][2]:.2f}]"
            )
            self.gui.TextUnformatted(f"Projection: {state['projection']}")

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

        self.gui.SameLine()
        if self.gui.Button("Save Motion Data"):
            self.save_motion_data()
            self.logger.info("Manual save triggered")

        _, self.use_non_blocking_planning = self.gui.Checkbox(
            "Use Non-Blocking Planning", self.use_non_blocking_planning
        )
        _, self.user_instruction = self.gui.InputTextMultiline("User Instruction", self.user_instruction)

        # Automatically update path visualization
        if (
            self.visualize_grid_path
            and self.show_path_visualization
            and self.fId - self.path_viz_last_update >= self.path_viz_update_interval
        ):
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
            if self.visualize_grid_path and (
                not hasattr(self, "path_visualization_elements") or not self.path_visualization_elements
            ):
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
                        "hand_pos": None,
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
                    "hand_position": None,
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

                # 6. Save and Draw Hand Target Position
                if hasattr(character, "target_hand_position") and character.target_hand_position is not None:
                    try:
                        hand_pos = character.target_hand_position.cpu().numpy()
                        self.path_visualization_data[char_name][current_frame_id]["hand_position"] = hand_pos.copy()

                        # Draw Hand Target Position (Purple Sphere)
                        hand_sphere = trimesh.creation.uv_sphere(radius=0.05)
                        hand_sphere.apply_translation(hand_pos[0])

                        # Check if it already exists; if it does, remove it first
                        if (
                            char_name in self.path_visualization_elements
                            and "hand_pos" in self.path_visualization_elements[char_name]
                            and self.path_visualization_elements[char_name]["hand_pos"] is not None
                        ):
                            try:
                                ps.remove_surface_mesh(self.path_visualization_elements[char_name]["hand_pos"])
                            except:
                                pass

                        self.register_trimesh(f"{char_name}_hand", hand_sphere, color=[0.8, 0.0, 0.8], enabled=True)
                        self.path_visualization_elements[char_name]["hand_pos"] = f"{char_name}_hand"
                    except Exception as e:
                        import traceback

                        self.logger.warning(f"Failed to visualize hand position for {char_name}: {e}")
                        self.logger.warning(f"Hand position error traceback: {traceback.format_exc()}")
                        self.logger.warning(
                            f"Character attributes: target_hand_position={getattr(character, 'target_hand_position', 'None')}"
                        )
                        self.logger.warning(
                            f"Visualization elements: {self.path_visualization_elements.get(char_name, 'Not found')}"
                        )
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
            if self.visualize_grid_path and (
                not hasattr(self, "path_visualization_elements") or not self.path_visualization_elements
            ):
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
                        "hand_pos": None,
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

                # 6. Draw Saved Hand Target Position
                if frame_data.get("hand_position") is not None:
                    try:
                        hand_pos = frame_data["hand_position"]
                        hand_sphere = trimesh.creation.uv_sphere(radius=0.05)
                        hand_sphere.apply_translation(hand_pos[0])

                        # Check if it already exists; if it does, remove it first
                        if (
                            char_name in self.path_visualization_elements
                            and "hand_pos" in self.path_visualization_elements[char_name]
                            and self.path_visualization_elements[char_name]["hand_pos"] is not None
                        ):
                            try:
                                ps.remove_surface_mesh(self.path_visualization_elements[char_name]["hand_pos"])
                            except:
                                pass

                        self.register_trimesh(f"{char_name}_hand", hand_sphere, color=[0.8, 0.0, 0.8], enabled=True)
                        if char_name not in self.path_visualization_elements:
                            self.path_visualization_elements[char_name] = {}
                        self.path_visualization_elements[char_name]["hand_pos"] = f"{char_name}_hand"
                    except Exception as e:
                        import traceback

                        self.logger.warning(f"Failed to visualize hand position for {char_name} (frame): {e}")
                        self.logger.warning(f"Hand position frame error traceback: {traceback.format_exc()}")
                        self.logger.warning(f"Frame data: {frame_data.get('hand_position', 'Not found')}")
                        self.logger.warning(
                            f"Visualization elements: {self.path_visualization_elements.get(char_name, 'Not found')}"
                        )

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
