from src.scene.scene import Scene
from src.utils import misc
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


_DEFAULT_STORY = """
Amir picks up a bottle.
Amir hands over the bottle to Benjamin in an empty area.
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
        self.playing = True
        self.error_occurred = False
        self.is_status_ui_active = True
        self.use_non_blocking_planning = False
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
        self.reset()

        self.current_bottle_grip_offset = 0.0
        self.current_wrist_to_palm_offset = 0.0
        self.bottle_grip_offset = (
            -0.1
        )  # Offset distance along the long axis of the bottle towards the body of the bottle (meters)
        self.wrist_to_palm_offset = -0.1  # Distance from the wrist to the center of the palm (meters)

    def reset(self):
        super().reset()
        self.scene = Scene(self.scene_name, self.cfg)
        # self.register_trimesh("scene", self.scene.scene_mesh)
        # Do not register the entire scene mesh; register handles for each object individually
        self.object_mesh_handles = {}
        for obj in self.scene.objects:
            name = f"obj_{obj.label}_{obj.idx}"
            handle = self.register_trimesh(name, obj.mesh)
            self.object_mesh_handles[f"{obj.label}_{obj.idx}"] = handle

        self.max_fId = 0
        self.fId = 0

        # Immediately update the path visualization after initialization to re-register all elements
        # self.update_path_visualization()

    def callback(self):
        if self.fId <= self.max_fId:  # when showing previous frames

            if self.playing:
                self.fId += 1

            return

        if not self.playing:
            return
        if self.error_occurred:
            return

        # update frame idx
        self.fId += 1
        self.max_fId = self.fId - 1

    def ui_callback(self):
        self.gui.SetNextWindowSize((1000, 200), self.gui.ImGuiCond_FirstUseEver)
        self.gui.Begin("Status", self.is_status_ui_active)
        self.gui.Separator()

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

        self.gui.End()

        self.gui.Text(f"Error Occurred: {self.error_occurred}")

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

        _, self.user_instruction = self.gui.InputTextMultiline("User Instruction", self.user_instruction)


def run(cfg) -> None:
    """Run the retained experiment with configuration supplied by the shared CLI."""
    app = DemoApp(cfg)
    app.run()
