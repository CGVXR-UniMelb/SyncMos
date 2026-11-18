import threading
import time

import numpy as np
import torch

from src.action_planning.action_planning_module import ActionPlanningModule
from src.motion_synthesis.motion_synthesis_module import MotionSynthesisModule
from src.scene.scene import Scene
from src.type import Character
from src.utils import misc, smpl
from src.utils.polyscope import PolyscopeApp
from src.motion_synthesis.states import State
from src.utils.grid_utils import grid_to_world

from src.motion_synthesis.motion_generator import experiment_stats

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

# _DEFAULT_STORY = """
# Amir sits on the sofa.
# """

# _DEFAULT_STORY = """
# Amir is the waiter and Benjamin is the customer.
# Amir picks up a bottle near the bar and socializing area.
# Benjamin sits on the chair in the dining area.
# Amir hands over the bottle to Benjamin near where Benjamin is sitting.
# After that, Amir go back to pick up another bottle and Benjamin drinks the bottle.
# """

# _DEFAULT_STORY = """
# Amir walk to the door of the restaurant and leave the restaurant.
# Benjamin came from the outside of the restaurant and sat down on the chair near the table 4.
# Charles grab a bottle and give it to Benjamin where Benjamin is sitting.
# """

# _DEFAULT_STORY = """
# Amir, Benjamin sits near the same table.
# Charles grab a bottle and sits down near the same table as Amir and Benjamin.
# Charles hand over the bottle to Benjamin when both of them are still sitting.
# Benjamin hand over the bottle to Amir when both of them are still sitting.
# Amir hands over the bottle to Charles when both of them are still sitting.
# To achieve this, please put the handover position between two characters.
# """

# _DEFAULT_STORY = """
# Charles grab a bottle.
# Charles hand over the bottle to Benjamin.
# Benjamin hand over the bottle to Amir.
# Amir hands over the bottle to Charles.
# """

# _DEFAULT_STORY = """
# Amir, Benjamin and Charles sitting together near the same table at the same time.
# """


# _DEFAULT_STORY = """
# Amir sit on the chair near the table 4.
# Amir left the restaurant..
# After Amir stood up, Benjamin sit on the same chair as Amir was sitting.
# And Benjamin left the restaurant.
# After Benjamin stood up, Charles sit on the same chair as Benjamin was sitting.S
# """

# _DEFAULT_STORY = """
# Amir bend forward and pick up bottle_1 from table_3.
# Amir straighten up and put down bottle_1 on bar_counter_1.
# Amir pick up another bottle from wine_rack_1.
# Amir read the label on the bottle.
# Amir put down the bottle on bar_counter_1.
# Amir pick up a glass from bar_counter_1 and drink.
# Benjamin wave his hand toward Amir.
# Amir turn right.
# Amir pick up bottle_1 again and hand it to Benjamin.
# Benjamin pick up the bottle from Amir.
# Benjamin drink.
# Amir sit down on chair_6 near table_4.
# Benjamin sit down on chair_3 near table_4.
# Amir eat from plate_5 on table_4.
# Benjamin eat from plate_6 on table_4.
# Amir turn left.
# Amir type on laptop_1 on reception_desk_1.
# Benjamin drink again.
# Amir pick up potted_plant_2 near reception_desk_1.
# Amir put down potted_plant_2 on bar_counter_1.
# Amir turn on the light near bar_logo_1.
# Benjamin take photo of bar_logo_1 using his phone.
# Amir swing slightly with rhythm.
# Benjamin talk on phone.
# Amir listen to music.
# Amir pick up a new bottle from wine_rack_11.
# Amir hand over the bottle to Benjamin.
# Benjamin put down the new bottle on table_4.
# Benjamin eat again.
# Amir wave to the bartender.
# Benjamin wave to the bartender.
# """

character_initial_grid_positions = {}
# _DEFAULT_STORY = """
# Amir picks up a bottle.
# Amir hands over the bottle to Benjamin at grid [21, 27].
# Benjamin hands over the bottle to Charles at grid [21, 23].
# Charles hands over the bottle to David at grid [13, 18].
# David hands over the bottle to Evan at grid [8, 18].
# Evan hands over the bottle to Francis at grid [3, 18].
# """
# character_initial_grid_positions = {
#     "Amir": [17, 27],
#     "Benjamin": [22, 32],
#     "Charles": [20, 24],
#     "David": [12, 19],
#     "Evan": [7, 19],
#     "Francis": [2, 19]
# }


# _DEFAULT_STORY = """
# Amir picks up a bottle near table_2.
# Amir hands over the bottle to Benjamin at grid [13, 18].
# Benjamin hands over the bottle to Charles at grid [8, 18].
# """
# character_initial_grid_positions = {
#     "Amir": [17, 27],
#     "Benjamin": [12, 18],
#     "Charles": [7, 19],
# }

# _DEFAULT_STORY = """
# Amir picks up a bottle near table_2.
# Amir hands over the bottle to Benjamin at grid [10, 18].
# """
# character_initial_grid_positions = {
#     "Amir": [17, 27],
#     "Benjamin": [5, 20]
# }

# Global statistics - Record only one instance per run
# Format: [scene_name, num_characters, best_idx_A, best_idx_B, middle, mapping_after_idx_A, mapping_after_idx_B]


def generate_handover_story(num_characters, character_names, scene_name, scene_grid_map):
    """
    Automatically generate handover stories based on the number of characters, using random positions

    Args:
        num_characters: Number of characters
        character_names: List of character names
        scene_name: Scene name
        scene_grid_map: The grid_map object of the scene

    Returns:
        The generated story string
    """
    if num_characters < 2:
        return "Not enough characters for handover."

    story_lines = []

    # The first character picks up the bottle
    story_lines.append(f"{character_names[0]} picks up a bottle.")

    # Generate a random position for each handover and convert it to grid coordinates
    from src.utils.grid_utils import world_to_grid

    # Generate a handover chain: each character passes to the next character
    for i in range(num_characters - 1):
        current_char = character_names[i]
        next_char = character_names[i + 1]

        # Obtain random world coordinates
        random_world_pos = scene_grid_map.get_random_position()

        # Convert to grid coordinates
        grid_x, grid_y = world_to_grid(
            random_world_pos[0],
            random_world_pos[1],
            grid_size=scene_grid_map.grid_step,
            bbox_min=(scene_grid_map.bbox[0][0], scene_grid_map.bbox[0][1]),
            round_to_int=True,
        )

        if grid_x <= 1:
            grid_x = 3
        if grid_y <= 1:
            grid_y = 3
        location = f"at grid [{grid_x}, {grid_y}]"
        story_lines.append(f"{current_char} hands over the bottle to {next_char} {location}.")

    return "\n".join(story_lines)


# Default story (will be generated dynamically based on num_characters when DemoApp initializes)
_DEFAULT_STORY = """
Amir picks up a bottle.
Amir hands over the bottle to Benjamin near table_4.
Benjamin hands over the bottle to Charles near table_2.
"""


class DemoApp(PolyscopeApp):
    def __init__(self, cfg):
        self.visualize_grid_path = False

        super().__init__(framerate=cfg.framerate)
        self.cfg = cfg
        self.device = cfg.device
        self.scene_name = cfg.scene_name
        self.num_characters = cfg.num_characters
        self.character_names = _DEFAULT_CHARACTER_NAMES[: self.num_characters]

        # Clear previous statistics and initialize current running statistics
        experiment_stats.STATISTICS = [self.scene_name, self.num_characters]

        # Initialize the scene to obtain the grid_map
        self.logger = misc.get_console_logger("DemoApp", cfg.logging_level)
        self.scene = Scene(self.scene_name, self.cfg)

        # Automatically generate handover stories based on the number of characters and the scene’s grid_map
        generated_story = generate_handover_story(
            self.num_characters,
            self.character_names,
            self.scene_name,
            self.scene.grid_map,  # Pass in the grid_map object
        )
        self.user_instruction = generated_story.strip()
        self.logger.info(f"Generated story for {self.num_characters} characters:")
        self.logger.info(f"\n{self.user_instruction}")

        self.playing = True
        self.error_occurred = False
        self.is_status_ui_active = True
        self.use_non_blocking_planning = False

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
        # No need to recreate the scene, as it has been created in __init__
        # self.scene = Scene(self.scene_name, self.cfg)

        # Clean up the previous object mesh handles
        if hasattr(self, "object_mesh_handles"):
            self.object_mesh_handles.clear()

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

            if character.name in character_initial_grid_positions:
                grid_xy = character_initial_grid_positions[character.name]
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
                    vertices, _ = self.motion_synthesizer.get_mesh_at(character, frame_idx)
                    character_mesh.update_vertex_positions(vertices)

            # Update the path visualization when reviewing, showing the path status corresponding to the current frame
            # Decide whether to update in real-time based on user settings
            if self.visualize_grid_path and self.show_path_visualization:  # Update every frame during review
                self.update_path_visualization_for_frame()

            if self.playing:
                self.fId += 1
            return
        # current plan is finished
        if (
            hasattr(self.action_planner.narrator, "plan_generated")
            and hasattr(self.action_planner.narrator, "event_list")
            and self.action_planner.narrator.plan_generated
            and self.action_planner.narrator.event_list == []
            and all([character.state == State.IDLE for character in self.characters])
            and all([character.target_actions == [] for character in self.characters])
        ):
            self.playing = False

            # Save the statistics information to a file
            # Format: scene_name, num_characters, best_idx_A, best_idx_B, middle, mapping_after_idx_A, mapping_after_idx_B
            with open("stat_scale.txt", "a") as f:
                # Convert the list into a string, separated by commas
                f.write(",".join(map(str, experiment_stats.STATISTICS)) + "\n")
            self.logger.info(f"Statistics saved: {experiment_stats.STATISTICS}")

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
            vertices, _ = self.motion_synthesizer.get_current_mesh(character)
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
