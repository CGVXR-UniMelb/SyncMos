from __future__ import annotations

import glob
import os
import traceback
from typing import Optional

import numpy as np
import torch

from src.pipeline.motion_source import FileMotionSource, MotionSource
from src.rendering.smpl_sequence_renderer import SMPLSequenceRenderer
from src.scene.scene import Scene
from src.utils import misc
from src.utils.camera_control import CameraController
from src.utils.polyscope import PolyscopeApp

_DEFAULT_COLOR = [0.7, 0.7, 0.7]


class PolyscopeMotionViewer(PolyscopeApp):
    """Polyscope frontend for file replay, realtime generation, and continued generation."""

    def __init__(
        self,
        cfg,
        motion_source: Optional[MotionSource] = None,
        renderer: Optional[SMPLSequenceRenderer] = None,
        motion_dir: str = "motion",
        app_name: str = "Motion Viewer",
        on_user_instruction_change=None,
        on_save_motion_data=None,
        on_load_motion_file=None,
        on_generate_new_motion=None,
        on_continue_from_source=None,
        debug_info_provider=None,
        generation_stats_provider=None,
    ):
        super().__init__(app_name=app_name, framerate=cfg.framerate)
        self.cfg = cfg
        self.device = cfg.device
        self.logger = misc.get_console_logger(self.__class__.__name__, cfg.logging_level)
        self.renderer = renderer or SMPLSequenceRenderer(device=self.device)
        self.motion_source = motion_source
        self.on_user_instruction_change = on_user_instruction_change
        self.on_save_motion_data = on_save_motion_data
        self.on_load_motion_file = on_load_motion_file
        self.on_generate_new_motion = on_generate_new_motion
        self.on_continue_from_source = on_continue_from_source
        self.debug_info_provider = debug_info_provider
        self.generation_stats_provider = generation_stats_provider
        self.motion_dir = motion_dir
        self.available_motion_files = []
        self.selected_motion_file_idx = 0

        self.scene = None
        self.scene_name = None
        self.character_names = []
        self.character_meshes = {}
        self.object_mesh_handles = {}
        self.camera_controller = None

        self.motion_loaded = False
        self.playing = True
        self.loop_playback = True
        self.show_origin_motion = False
        self.user_instruction = ""
        self.fId = 0
        self.max_fId = 0

        self.export_out_path = "replay.mp4"
        self.export_fps = 30
        self.export_start_frame = 0
        self.export_end_frame = 0
        self.export_transparent = False
        self.export_reset_first = False
        self.is_exporting = False

        self.show_all_time_overlay = False
        self.all_time_stride = 5
        self.all_time_min_brightness = 0.25
        self.all_time_min_alpha = 0.2
        self.hide_live_meshes_on_overlay = True
        self.all_time_use_range = False
        self.all_time_start_frame = 0
        self.all_time_end_frame = 0
        self.all_time_traj_radius = 0.03
        self.all_time_keyframe_reset = False
        self.all_time_keyframe_text = ""
        self.all_time_keyframes = []
        self.all_time_traj_segments = {}
        self.all_time_traj_segment_text = {}
        self.all_time_mesh_names = set()
        self.all_time_traj_names = set()

        self.bottle_grip_offset = -0.1
        self.wrist_to_palm_offset = -0.1
        self.use_non_blocking_planning = False

        self.scan_motion_files()
        if self.motion_source is not None:
            self.load_source(self.motion_source)

    def scan_motion_files(self):
        pattern = os.path.join(self.motion_dir, "**", "*.npz")
        self.available_motion_files = sorted(glob.glob(pattern, recursive=True))

    def load_motion_file(self, filepath: str):
        if self.on_load_motion_file is not None:
            source = self.on_load_motion_file(filepath)
        else:
            source = FileMotionSource(filepath, device=self.device, logger=self.logger)
        if source is not None:
            self.load_source(source)

    def load_source(self, source: MotionSource):
        self.clear_all_time_overlay()
        self.motion_source = source
        metadata = source.get_metadata()
        same_scene = self.motion_loaded and self.scene_name == metadata.scene_name
        self.scene_name = metadata.scene_name or getattr(self.cfg, "scene_name", "")
        self.character_names = list(metadata.character_names)
        self.user_instruction = metadata.user_instruction
        self.max_fId = max(0, source.latest_frame_id())
        self.fId = 0
        self.all_time_end_frame = self.max_fId
        self.all_time_traj_segments = {name: [] for name in self.character_names}
        self.all_time_traj_segment_text = {name: "" for name in self.character_names}

        if not same_scene:
            self.reset()
            self.scene = Scene(self.scene_name, self.cfg)
            self.object_mesh_handles = {}
            for obj in self.scene.objects:
                display_name = f"obj_{obj.label}_{obj.idx}"
                self.object_mesh_handles[f"{obj.label}_{obj.idx}"] = self.register_trimesh(display_name, obj.mesh)

            self.camera_controller = CameraController(
                scene_name=self.scene_name,
                config_dir="./configs/scenes",
                logger=self.logger,
            )
            self.camera_controller.load_camera_config()
            self.camera_controller.apply_custom_camera()

        self.character_meshes = {}
        first_frame = self._source_frame(0)
        if first_frame is not None:
            mesh_frame = self.renderer.render_frame(
                self._body_params_for_display(first_frame),
                frame_id=first_frame.frame_id,
            )
            for name, mesh in mesh_frame.character_meshes.items():
                self.character_meshes[name] = self.register_mesh(
                    name,
                    mesh.vertices,
                    mesh.faces,
                    color=_DEFAULT_COLOR,
                )

        self.export_end_frame = self.max_fId
        self.motion_loaded = True

    def callback(self):
        if self.motion_source is None or not self.motion_loaded:
            return

        self.max_fId = max(self.max_fId, self.motion_source.latest_frame_id())
        if self.all_time_end_frame < self.max_fId:
            self.all_time_end_frame = self.max_fId
        frame = self._source_frame(self.fId)
        if frame is None:
            return

        mesh_frame = self.renderer.render_frame(
            self._body_params_for_display(frame),
            frame_id=frame.frame_id,
        )
        for name, char_mesh in mesh_frame.character_meshes.items():
            if name not in self.character_meshes:
                self.character_meshes[name] = self.register_mesh(
                    name,
                    char_mesh.vertices,
                    char_mesh.faces,
                    color=_DEFAULT_COLOR,
                )
            else:
                self.character_meshes[name].update_vertex_positions(char_mesh.vertices)

        if self.playing:
            self.fId += 1
            if self.fId > self.max_fId:
                if self.motion_source.is_finished() and self.loop_playback:
                    self.fId = 0
                else:
                    self.fId = self.max_fId

    def ui_callback(self):
        self.render_logs()
        self.gui.SetNextWindowSize((1000, 720), self.gui.ImGuiCond_FirstUseEver)
        _, opened = self.gui.Begin("Motion Viewer", True)
        if not opened:
            self.gui.End()
            return

        self._render_user_instruction_ui()
        self._render_source_ui()
        if self.motion_source is not None:
            self._render_realtime_debug_ui()
            self._render_animation_info_ui()
            self._render_generation_stats_ui()
            self._render_playback_ui()
            self._render_motion_visualization_ui()
            self._render_bottle_attachment_ui()
            self._render_export_ui()
            self._render_camera_ui()
        else:
            self.selectable_text("empty_view", "Load a saved motion or generate a new realtime motion.")

        self.gui.End()

    def _render_user_instruction_ui(self):
        changed_text, value = self.labeled_control(
            self.gui.InputTextMultiline, "User Instruction", self.user_instruction
        )
        if changed_text:
            self._set_user_instruction(value)

    def _render_source_ui(self):
        if not self.labeled_control(self.gui.TreeNode, "Source / Generation"):
            return
        self.selectable_text(
            "source_status", f"Status: {self.motion_source.status() if self.motion_source else 'no source'}"
        )
        self.selectable_text("directory", f"Motion directory: {self.motion_dir}")
        if self.labeled_control(self.gui.Button, "Refresh File List"):
            self.scan_motion_files()
        self.gui.SameLine()
        if self.labeled_control(self.gui.Button, "Generate New Motion") and self.on_generate_new_motion is not None:
            source = self.on_generate_new_motion(self.user_instruction)
            if source is not None:
                self.load_source(source)

        can_continue = self.motion_source is not None
        if can_continue:
            self.gui.SameLine()
            if (
                self.labeled_control(self.gui.Button, "Continue From Loaded")
                and self.on_continue_from_source is not None
            ):
                source = self.on_continue_from_source(self.motion_source, self.user_instruction)
                if source is not None:
                    self.load_source(source)

        self.gui.Separator()
        if len(self.available_motion_files) == 0:
            self.selectable_text("empty_directory", "No motion files found")
        else:
            self.selectable_text("file_count", f"Available files ({len(self.available_motion_files)}):")
            for i, filepath in enumerate(self.available_motion_files):
                label = os.path.relpath(filepath, self.motion_dir)
                if self.gui.RadioButton(f"##select_{label}", self.selected_motion_file_idx == i):
                    self.selected_motion_file_idx = i
                self.gui.SameLine()
                self.selectable_text(f"file_{label}", label)
                if self.gui.IsItemClicked():
                    self.selected_motion_file_idx = i
            if self.labeled_control(self.gui.Button, "Load Selected File"):
                self.load_motion_file(self.available_motion_files[self.selected_motion_file_idx])
        self.gui.TreePop()

    def _render_realtime_debug_ui(self):
        debug = self.debug_info_provider() if self.debug_info_provider is not None else {}
        if not debug:
            return
        self.selectable_text("error", f"Error Occurred: {debug.get('error_occurred', False)}")
        if "left_query_count" in debug:
            self.selectable_text("queries", f"Left Query Count: {debug['left_query_count']}")
        if "narrator" in debug:
            self.selectable_text("narrator", f"Narrator: {debug['narrator']}")
        if "event_parser" in debug:
            self.selectable_text("parser", f"Event Parser: {debug['event_parser']}")

        if self.labeled_control(self.gui.TreeNode, "Available Action Labels"):
            labels = debug.get("action_labels", [])
            self.selectable_text("actions", "\n".join(f" - {label}" for label in labels))
            self.gui.TreePop()
        if self.labeled_control(self.gui.TreeNode, "Events"):
            self.selectable_text("events", "\n".join(map(str, debug.get("events", []))))
            self.gui.TreePop()
        if self.labeled_control(self.gui.TreeNode, "Characters"):
            for character in debug.get("characters", []):
                if self.labeled_control(self.gui.TreeNode, character.get("name", "Character")):
                    for key in ["event", "state", "current_actions", "target_actions", "relationships"]:
                        self.selectable_text(key, f"{key}: {character.get(key)}")
                    self.gui.TreePop()
            self.gui.TreePop()

    def _render_animation_info_ui(self):
        metadata = self.motion_source.get_metadata()
        if self.labeled_control(self.gui.TreeNode, "Animation Info"):
            self.selectable_text("scene", f"Scene: {metadata.scene_name}")
            self.selectable_text(
                "character_count", f"Characters: {metadata.num_characters or len(metadata.character_names)}"
            )
            self.selectable_text("character_names", f"Character Names: {', '.join(metadata.character_names)}")
            self.selectable_text("frames", f"Total Frames: {self.max_fId + 1}")
            self.gui.Separator()
            self.selectable_text("story_label", "Story:")
            self.selectable_text("story", self.user_instruction)
            self.gui.TreePop()

    def _render_generation_stats_ui(self):
        stats = self.generation_stats_provider() if self.generation_stats_provider is not None else {}
        render_stats = self.renderer.stats_snapshot() if hasattr(self.renderer, "stats_snapshot") else {}
        if not stats and not render_stats:
            return
        if not self.labeled_control(self.gui.TreeNode, "Generation Stats"):
            return

        frame_per_sec = stats.get("frame_per_sec", 0.0)
        joints_per_sec = stats.get("joints_per_sec", 0.0)
        smpl_per_sec = render_stats.get("smpl_per_sec", 0.0)
        self.selectable_text("fps", f"frame/s: {frame_per_sec:.2f}")
        self.selectable_text("joint_rate", f"joints/s: {joints_per_sec:.2f}")
        self.selectable_text("smpl_rate", f"smpl/s: {smpl_per_sec:.2f}")
        self.gui.Separator()

        generated_frames = int(stats.get("total_generated_frames", max(0, self.max_fId + 1)))
        generated_joints = int(stats.get("total_generated_joint_frames", 0))
        joints_per_frame = max(1, int(stats.get("motion_joint_count", 28)))
        generated_frame_equivalent = (
            generated_joints / joints_per_frame if generated_joints else float(generated_frames)
        )
        played_frame = int(self.fId)
        generated_vs_played_ratio = min(1.0, played_frame / max(1.0, generated_frame_equivalent))
        self.selectable_text("generated_frames", f"Generated Frames: {generated_frames}")
        self.selectable_text("generated_joints", f"Generated Joints: {generated_joints}")
        self.selectable_text("progress_label", "Playback vs Generated Joints:")
        self.gui.ProgressBar(generated_vs_played_ratio, (0.0, 0.0))
        self.selectable_text(
            "progress", f"frame {played_frame} / {generated_frame_equivalent:.1f} generated-frame-equivalent"
        )
        self.gui.Separator()

        self.selectable_text("cuts", f"Cut Events: {stats.get('total_cut_events', 0)}")
        self.selectable_text("cut_frames", f"Cut Frames: {stats.get('total_cut_frames', 0)}")
        if self.labeled_control(self.gui.TreeNode, "Cut Frame Distribution"):
            distribution = stats.get("cut_frame_distribution", {})
            if not distribution:
                self.selectable_text("no_cuts", "No cuts recorded")
            for cut_frames, count in distribution.items():
                self.selectable_text(f"distribution_{cut_frames}", f"{cut_frames} frames: {count} events")
            self.gui.TreePop()
        if self.labeled_control(self.gui.TreeNode, "Generated Frame Distribution"):
            distribution = stats.get("generated_frame_distribution", {})
            if not distribution:
                self.selectable_text("no_segments", "No generated segments recorded")
            for generated, count in distribution.items():
                self.selectable_text(f"distribution_{generated}", f"{generated} frames: {count} segments")
            self.gui.TreePop()
        self.gui.TreePop()

    def _render_playback_ui(self):
        self.gui.Separator()
        _, self.fId = self.labeled_control(self.gui.SliderInt, "Frame", int(self.fId), 0, max(0, int(self.max_fId)))
        if self.labeled_control(self.gui.Button, "Play" if not self.playing else "Pause"):
            self.playing = not self.playing
        self.gui.SameLine()
        if self.labeled_control(self.gui.Button, "Reset"):
            self.fId = 0
            self.playing = False
        self.gui.SameLine()
        if self.labeled_control(self.gui.Button, "Save Motion Data") and self.on_save_motion_data is not None:
            self.on_save_motion_data()

        changed, self.loop_playback = self.labeled_control(self.gui.Checkbox, "Loop", self.loop_playback)

    def _render_motion_visualization_ui(self):
        if not self.labeled_control(self.gui.TreeNode, "Motion Visualization"):
            return
        has_origin_data = self._has_origin_data()
        if not has_origin_data:
            self.selectable_text("no_original", "(Original motion data not available)")
        changed, self.show_origin_motion = self.labeled_control(
            self.gui.Checkbox, "Show Original Motion", self.show_origin_motion
        )
        if changed and self.show_all_time_overlay:
            self.build_all_time_overlay()

        self.gui.Separator()
        self.selectable_text(
            "displayed_motion",
            (
                "Currently showing: Original Motion (No Time Warp)"
                if self.show_origin_motion
                else "Currently showing: Time-Warped Motion"
            ),
        )

        self.gui.Separator()
        changed_overlay, self.show_all_time_overlay = self.labeled_control(
            self.gui.Checkbox, "Show 4D Overlay (All Frames)", self.show_all_time_overlay
        )
        if changed_overlay:
            if self.show_all_time_overlay:
                if self.hide_live_meshes_on_overlay:
                    self._set_live_mesh_visibility(False)
                self.build_all_time_overlay()
            else:
                self.clear_all_time_overlay()
                self._set_live_mesh_visibility(True)

        changed_hide, self.hide_live_meshes_on_overlay = self.labeled_control(
            self.gui.Checkbox, "Hide Live Meshes", self.hide_live_meshes_on_overlay
        )
        if changed_hide and self.show_all_time_overlay:
            self._set_live_mesh_visibility(not self.hide_live_meshes_on_overlay)

        changed_stride, self.all_time_stride = self.labeled_control(
            self.gui.SliderInt, "Overlay Frame Stride", self.all_time_stride, 1, 60
        )
        changed_brightness, self.all_time_min_brightness = self.labeled_control(
            self.gui.SliderFloat, "Overlay Min Brightness", self.all_time_min_brightness, 0.05, 1.0
        )
        changed_alpha, self.all_time_min_alpha = self.labeled_control(
            self.gui.SliderFloat, "Overlay Min Alpha", self.all_time_min_alpha, 0.05, 1.0
        )
        changed_kf_reset, self.all_time_keyframe_reset = self.labeled_control(
            self.gui.Checkbox, "Reset Brightness/Alpha at Keyframes (1 -> Min)", self.all_time_keyframe_reset
        )
        changed_kf_text, self.all_time_keyframe_text = self.labeled_control(
            self.gui.InputText, "Keyframes (comma-separated, reset points)", self.all_time_keyframe_text
        )
        if changed_kf_text:
            self.all_time_keyframes = self._parse_keyframes(self.all_time_keyframe_text, self.max_fId)
        changed_range, self.all_time_use_range = self.labeled_control(
            self.gui.Checkbox, "Use Overlay Frame Range", self.all_time_use_range
        )
        changed_start, self.all_time_start_frame = self.labeled_control(
            self.gui.SliderInt, "Overlay Start Frame", self.all_time_start_frame, 0, max(0, self.max_fId)
        )
        changed_end, self.all_time_end_frame = self.labeled_control(
            self.gui.SliderInt, "Overlay End Frame", self.all_time_end_frame, 0, max(0, self.max_fId)
        )
        changed_radius, self.all_time_traj_radius = self.labeled_control(
            self.gui.SliderFloat, "Trajectory Radius", self.all_time_traj_radius, 0.005, 0.2
        )

        if (
            any(
                [
                    changed_stride,
                    changed_brightness,
                    changed_alpha,
                    changed_kf_reset,
                    changed_kf_text,
                    changed_range,
                    changed_start,
                    changed_end,
                    changed_radius,
                ]
            )
            and self.show_all_time_overlay
        ):
            self.build_all_time_overlay()

        self.gui.Separator()
        self.selectable_text("trajectory_help", "Trajectory Segments (use start-end, comma-separated)")
        for char_name in self.character_names:
            changed_seg, text = self.labeled_control(
                self.gui.InputText, f"{char_name} Segments", self.all_time_traj_segment_text.get(char_name, "")
            )
            if changed_seg:
                self.all_time_traj_segment_text[char_name] = text
                self.all_time_traj_segments[char_name] = self._parse_traj_segments(text, self.max_fId)
                if self.show_all_time_overlay:
                    self.build_all_time_overlay()
        self.gui.TreePop()

    def _render_bottle_attachment_ui(self):
        if not self.labeled_control(self.gui.TreeNode, "Bottle Attachment Settings"):
            return
        self.selectable_text("grip_help", "Bottle Grip Offset (along bottle axis):")
        _, self.bottle_grip_offset = self.labeled_control(
            self.gui.SliderFloat, "Grip Offset (m)", self.bottle_grip_offset, -0.3, 0.3
        )
        self.selectable_text("wrist_help", "Wrist to Palm Center Offset (along forearm):")
        _, self.wrist_to_palm_offset = self.labeled_control(
            self.gui.SliderFloat, "Palm Offset (m)", self.wrist_to_palm_offset, -0.3, 0.3
        )
        self.gui.Separator()
        self.selectable_text("grip_offset", f"Current Grip Offset: {self.bottle_grip_offset:.3f} m")
        self.selectable_text("palm_offset", f"Current Palm Offset: {self.wrist_to_palm_offset:.3f} m")
        if self.labeled_control(self.gui.Button, "Reset to Default"):
            self.bottle_grip_offset = -0.1
            self.wrist_to_palm_offset = -0.1
        self.gui.TreePop()

    def _render_export_ui(self):
        if not self.labeled_control(self.gui.TreeNode, "Export Animation (MP4)"):
            return
        _, self.export_out_path = self.labeled_control(self.gui.InputText, "Output Path", self.export_out_path)
        _, self.export_fps = self.labeled_control(self.gui.SliderInt, "FPS", self.export_fps, 1, 120)
        self.selectable_text("frame", f"Current Frame: {self.fId} / {self.max_fId}")
        self.selectable_text("available_frames", f"Total Available Frames: {self.max_fId + 1}")
        self.selectable_text("range_label", "Export Frame Range:")
        _, self.export_start_frame = self.labeled_control(
            self.gui.SliderInt, "Start Frame", self.export_start_frame, 0, max(0, self.max_fId)
        )
        _, self.export_end_frame = self.labeled_control(
            self.gui.SliderInt, "End Frame", self.export_end_frame, 0, max(0, self.max_fId)
        )
        self.selectable_text(
            "export_frames", f"Total Export Frames: {self.export_end_frame - self.export_start_frame + 1}"
        )
        self.selectable_text(
            "export_range", f"Export Range: Frame {self.export_start_frame} to {self.export_end_frame}"
        )
        _, self.export_transparent = self.labeled_control(
            self.gui.Checkbox, "Transparent Background", self.export_transparent
        )
        _, self.export_reset_first = self.labeled_control(
            self.gui.Checkbox, "Reset Before Export", self.export_reset_first
        )
        self.selectable_text("export_status", f"Status: {'Exporting...' if self.is_exporting else 'Idle'}")
        if self.labeled_control(self.gui.Button, "Export MP4") and not self.is_exporting:
            self.do_export()
        self.gui.TreePop()

    def _render_camera_ui(self):
        if self.camera_controller is None or not self.labeled_control(self.gui.TreeNode, "Camera Control"):
            return
        self.selectable_text("presets", "Camera Presets:")
        preset_names = self.camera_controller.get_preset_names()
        for i, preset_name in enumerate(preset_names):
            if self.labeled_control(self.gui.Button, f"Load: {preset_name}"):
                self.camera_controller.apply_camera_preset(preset_name)
                self._sync_camera_text_fields()
            if i < len(preset_names) - 1 and (i + 1) % 4 != 0:
                self.gui.SameLine()

        self.gui.Separator()
        self.selectable_text("camera_position", "Custom Camera Position:")
        self._ensure_camera_text_fields()
        self._camera_float_input("Cam X", "_cam_pos_x_str", self.camera_controller.set_position, 0)
        self._camera_float_input("Cam Y", "_cam_pos_y_str", self.camera_controller.set_position, 1)
        self._camera_float_input("Cam Z", "_cam_pos_z_str", self.camera_controller.set_position, 2)

        self.selectable_text("camera_target", "Camera Target:")
        self._camera_float_input("Target X", "_cam_target_x_str", self.camera_controller.set_target, 0)
        self._camera_float_input("Target Y", "_cam_target_y_str", self.camera_controller.set_target, 1)
        self._camera_float_input("Target Z", "_cam_target_z_str", self.camera_controller.set_target, 2)

        self.selectable_text("projection_label", "Projection Mode:")
        if self.labeled_control(
            self.gui.RadioButton, "Perspective", self.camera_controller.current_projection_mode == "perspective"
        ):
            self.camera_controller.set_projection_mode("perspective")
            self.camera_controller.apply_custom_camera()
        self.gui.SameLine()
        if self.labeled_control(
            self.gui.RadioButton, "Orthographic", self.camera_controller.current_projection_mode == "orthographic"
        ):
            self.camera_controller.set_projection_mode("orthographic")
            self.camera_controller.apply_custom_camera()

        self.gui.Separator()
        if self.labeled_control(self.gui.Button, "Apply Custom Camera"):
            self.camera_controller.apply_custom_camera()
        self.gui.SameLine()
        if self.labeled_control(self.gui.Button, "Save Camera Config"):
            self.camera_controller.save_camera_config()
        self.gui.SameLine()
        if self.labeled_control(self.gui.Button, "Reload Config"):
            self.camera_controller.load_camera_config()
            self.camera_controller.apply_custom_camera()
            self._sync_camera_text_fields()

        self.gui.Separator()
        self.selectable_text("capture_view", "Capture Current View from Polyscope:")
        self.selectable_text("capture_help", "(Use after manually adjusting camera with mouse)")
        if self.labeled_control(self.gui.Button, "Read Current Camera View"):
            camera_info = self.camera_controller.get_current_camera_from_polyscope()
            if camera_info:
                self._cam_pos_x_str = f"{camera_info['position'][0]:.2f}"
                self._cam_pos_y_str = f"{camera_info['position'][1]:.2f}"
                self._cam_pos_z_str = f"{camera_info['position'][2]:.2f}"
                self._cam_target_x_str = f"{camera_info['target'][0]:.2f}"
                self._cam_target_y_str = f"{camera_info['target'][1]:.2f}"
                self._cam_target_z_str = f"{camera_info['target'][2]:.2f}"
        self.gui.Separator()
        self.selectable_text("config_file", f"Config File: {self.camera_controller.camera_config_file}")
        self.selectable_text("config_help", f"(Scene-specific, saved separately for {self.scene_name})")
        self.gui.Separator()
        self.selectable_text("save_preset", "Save Current View as Preset:")
        if not hasattr(self, "_new_preset_name"):
            self._new_preset_name = "My View"
        _, self._new_preset_name = self.labeled_control(self.gui.InputText, "Preset Name", self._new_preset_name)
        if self.labeled_control(self.gui.Button, "Save as New Preset") and self._new_preset_name.strip():
            self.camera_controller.save_current_view_as_preset(self._new_preset_name.strip())
            self.camera_controller.save_camera_config()

        state = self.camera_controller.get_current_state()
        self.gui.Separator()
        self.selectable_text("preset", f"Current Preset: {state['preset']}")
        self.selectable_text(
            "position",
            f"Position: [{state['position'][0]:.2f}, {state['position'][1]:.2f}, {state['position'][2]:.2f}]",
        )
        self.selectable_text(
            "target", f"Target: [{state['target'][0]:.2f}, {state['target'][1]:.2f}, {state['target'][2]:.2f}]"
        )
        self.selectable_text("projection", f"Projection: {state['projection']}")
        self.gui.TreePop()

    def do_export(self):
        if self.is_exporting:
            return
        self.is_exporting = True
        try:
            from src.utils.export import export_polyscope_animation

            export_polyscope_animation(
                self,
                out_path=self.export_out_path,
                fps=self.export_fps,
                start_frame=self.export_start_frame,
                end_frame=self.export_end_frame,
                transparent=self.export_transparent,
                reset_first=self.export_reset_first,
            )
        except Exception as exc:
            self.logger.error(f"Export failed: {exc}\n{traceback.format_exc()}")
        finally:
            self.is_exporting = False

    def clear_all_time_overlay(self):
        try:
            import polyscope as ps

            for mesh_name in list(self.all_time_mesh_names):
                try:
                    ps.remove_surface_mesh(mesh_name)
                except Exception:
                    pass
            for traj_name in list(self.all_time_traj_names):
                try:
                    ps.remove_curve_network(traj_name)
                except Exception:
                    pass
        except Exception:
            pass
        self.all_time_mesh_names = set()
        self.all_time_traj_names = set()

    def build_all_time_overlay(self):
        if self.motion_source is None:
            return
        self.clear_all_time_overlay()
        import polyscope as ps

        stride = max(1, int(self.all_time_stride))
        start_frame = 0
        end_frame = self.max_fId
        if self.all_time_use_range:
            start_frame = min(max(0, self.all_time_start_frame), self.max_fId)
            end_frame = min(max(0, self.all_time_end_frame), self.max_fId)
            if start_frame > end_frame:
                start_frame, end_frame = end_frame, start_frame

        for char_idx, char_name in enumerate(self.character_names):
            base_color = self._color_for_index(char_idx)
            mesh_ranges = self._invert_segments(
                start_frame,
                end_frame,
                self.all_time_traj_segments.get(char_name, []),
            )
            keyframes = [k for k in self.all_time_keyframes if start_frame <= k <= end_frame]
            for range_start, range_end in mesh_ranges:
                sub_ranges = (
                    self._split_ranges_by_keyframes(range_start, range_end, keyframes)
                    if self.all_time_keyframe_reset
                    else [(range_start, range_end)]
                )
                for sub_start, sub_end in sub_ranges:
                    frame_ids = []
                    params_list = []
                    for frame_id in range(sub_start, sub_end + 1, stride):
                        frame = self.motion_source.get_frame(frame_id)
                        if frame is None:
                            continue
                        params = self._body_params_for_display(frame).get(char_name)
                        if params is None:
                            continue
                        frame_ids.append(frame_id)
                        params_list.append(params)
                    if not params_list:
                        continue
                    vertices_batch, faces = self.renderer.render_body_params_batch(torch.cat(params_list, dim=0))
                    for frame_id, vertices in zip(frame_ids, vertices_batch):
                        t = (frame_id - sub_start) / max(1, sub_end - sub_start)
                        if self.all_time_keyframe_reset:
                            t = 1.0 - t
                        brightness = self.all_time_min_brightness + (1.0 - self.all_time_min_brightness) * t
                        color = [min(1.0, max(0.0, c * brightness)) for c in base_color]
                        mesh_name = f"{char_name}_alltime_{frame_id}"
                        mesh = self.register_mesh(mesh_name, vertices, faces, color=color)
                        if hasattr(mesh, "set_transparency"):
                            mesh.set_transparency(self.all_time_min_alpha + (1.0 - self.all_time_min_alpha) * t)
                        self.all_time_mesh_names.add(mesh_name)

            for seg_idx, (seg_start, seg_end) in enumerate(self.all_time_traj_segments.get(char_name, [])):
                points = []
                for frame_id in range(max(start_frame, seg_start), min(end_frame, seg_end) + 1):
                    frame = self.motion_source.get_frame(frame_id)
                    if frame is None:
                        continue
                    params = self._body_params_for_display(frame).get(char_name)
                    if params is None:
                        continue
                    point = params[0, :3].detach().cpu().numpy().astype(np.float32)
                    point[2] = 0.0
                    points.append(point)
                if len(points) < 2:
                    continue
                points = np.asarray(points, dtype=np.float32)
                edges = np.stack([np.arange(len(points) - 1), np.arange(1, len(points))], axis=1)
                traj_name = f"{char_name}_traj_{seg_idx}"
                ps.register_curve_network(traj_name, points, edges, color=base_color, radius=self.all_time_traj_radius)
                self.all_time_traj_names.add(traj_name)

    def _source_frame(self, frame_id: int):
        if self.motion_source is None:
            return None
        return self.motion_source.get_frame(frame_id)

    def _body_params_for_display(self, frame):
        if self.show_origin_motion and frame.character_body_params_origin:
            params = dict(frame.character_body_params)
            params.update(frame.character_body_params_origin)
            return params
        return frame.character_body_params

    def _set_user_instruction(self, value: str):
        self.user_instruction = value
        if self.on_user_instruction_change is not None:
            self.on_user_instruction_change(value)

    def _has_origin_data(self) -> bool:
        if self.motion_source is None:
            return False
        for frame_id in range(min(self.max_fId + 1, 10)):
            frame = self.motion_source.get_frame(frame_id)
            if frame is not None and frame.character_body_params_origin:
                return True
        return False

    def _set_live_mesh_visibility(self, visible: bool):
        for mesh in self.character_meshes.values():
            if hasattr(mesh, "set_enabled"):
                try:
                    mesh.set_enabled(visible)
                    continue
                except Exception:
                    pass
            if hasattr(mesh, "set_transparency"):
                try:
                    mesh.set_transparency(1.0 if visible else 0.0)
                except Exception:
                    pass

    def _ensure_camera_text_fields(self):
        if not hasattr(self, "_cam_pos_x_str"):
            self._sync_camera_text_fields()

    def _sync_camera_text_fields(self):
        pos = self.camera_controller.custom_camera_position
        target = self.camera_controller.custom_camera_target
        self._cam_pos_x_str = f"{pos[0]:.2f}"
        self._cam_pos_y_str = f"{pos[1]:.2f}"
        self._cam_pos_z_str = f"{pos[2]:.2f}"
        self._cam_target_x_str = f"{target[0]:.2f}"
        self._cam_target_y_str = f"{target[1]:.2f}"
        self._cam_target_z_str = f"{target[2]:.2f}"

    def _camera_float_input(self, label, attr, setter, axis):
        changed, value = self.labeled_control(self.gui.InputText, label, getattr(self, attr))
        setattr(self, attr, value)
        if not changed:
            return
        try:
            number = float(value)
        except ValueError:
            return
        values = (
            list(self.camera_controller.custom_camera_position)
            if label.startswith("Cam ")
            else list(self.camera_controller.custom_camera_target)
        )
        values[axis] = number
        setter(*values)

    def _parse_keyframes(self, text, max_frame):
        frames = []
        for part in [p.strip() for p in text.split(",") if p.strip()]:
            try:
                frames.append(max(0, min(max_frame, int(part))))
            except ValueError:
                pass
        return sorted(set(frames))

    def _parse_traj_segments(self, text, max_frame):
        segments = []
        for part in [p.strip() for p in text.split(",") if p.strip()]:
            if "-" not in part:
                continue
            left, right = [x.strip() for x in part.split("-", 1)]
            try:
                start = max(0, min(max_frame, int(left)))
                end = max(0, min(max_frame, int(right)))
            except ValueError:
                continue
            if start > end:
                start, end = end, start
            segments.append((start, end))
        return segments

    def _invert_segments(self, start_frame, end_frame, segments):
        if start_frame > end_frame:
            return []
        if not segments:
            return [(start_frame, end_frame)]
        segments = sorted(segments)
        ranges = []
        cursor = start_frame
        for start, end in segments:
            if start > cursor:
                ranges.append((cursor, start - 1))
            cursor = max(cursor, end + 1)
        if cursor <= end_frame:
            ranges.append((cursor, end_frame))
        return ranges

    def _split_ranges_by_keyframes(self, start_frame, end_frame, keyframes):
        splits = [start_frame] + [k for k in keyframes if start_frame < k <= end_frame] + [end_frame + 1]
        splits = sorted(set(splits))
        return [(splits[i], splits[i + 1] - 1) for i in range(len(splits) - 1) if splits[i] <= splits[i + 1] - 1]

    def _color_for_index(self, index):
        import colorsys

        r, g, b = colorsys.hsv_to_rgb((index / max(1, len(self.character_names))) % 1.0, 0.75, 0.9)
        return [r, g, b]
