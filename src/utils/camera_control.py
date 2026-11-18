"""
Camera control utilities for Polyscope visualization.
Provides camera preset management, configuration save/load, and view manipulation.
"""

import os
import json
import numpy as np
import polyscope as ps
from typing import Dict, List, Optional, Any
import logging


class CameraController:
    """
    Camera controller for managing camera views in Polyscope.

    Features:
    - Predefined camera presets (default, top view, side view, front view)
    - Custom camera position and target control
    - Save/load camera configurations from JSON files
    - Create and manage custom presets
    - Toggle between perspective and orthographic projection
    """

    def __init__(
        self, scene_name: str = "default", config_dir: str = "./configs/scenes", logger: Optional[logging.Logger] = None
    ):
        """
        Initialize camera controller.

        Args:
            scene_name: Name of the scene (used for config file naming)
            config_dir: Directory where camera config files are stored
            logger: Optional logger instance for logging messages
        """
        self.scene_name = scene_name
        self.config_dir = config_dir
        self.logger = logger or logging.getLogger(__name__)

        # Default camera presets
        self.camera_presets: Dict[str, Dict[str, Any]] = {
            "Default": {"position": [0, -5, 3], "target": [0, 0, 1], "projection": "perspective"},
            "Top View": {
                "position": [
                    0,
                    -0.1,
                    10,
                ],  # Slightly deviate from the Z-axis to avoid being collinear with the up direction
                "target": [0, 0, 0],
                "projection": "orthographic",
            },
            "Side View": {"position": [10, 0, 2], "target": [0, 0, 1], "projection": "perspective"},
            "Front View": {"position": [0, -10, 2], "target": [0, 0, 1], "projection": "perspective"},
        }

        # Current camera state
        self.current_camera_preset: str = "Default"
        self.custom_camera_position: List[float] = [0.0, -5.0, 3.0]
        self.custom_camera_target: List[float] = [0.0, 0.0, 1.0]
        self.current_projection_mode: str = "perspective"

        # Config file path
        self.camera_config_file: str = os.path.join(config_dir, scene_name, "camera_config.json")

    def load_camera_config(self) -> bool:
        """
        Load camera configuration from JSON file.

        Returns:
            True if config was loaded successfully, False otherwise
        """
        try:
            if os.path.exists(self.camera_config_file):
                with open(self.camera_config_file, "r") as f:
                    config = json.load(f)

                self.custom_camera_position = config.get("position", self.custom_camera_position)
                self.custom_camera_target = config.get("target", self.custom_camera_target)
                self.current_projection_mode = config.get("projection", self.current_projection_mode)
                self.current_camera_preset = config.get("preset", self.current_camera_preset)

                # Load custom presets if available
                if "custom_presets" in config:
                    self.camera_presets.update(config["custom_presets"])

                self.logger.info(f"Camera config loaded from {self.camera_config_file}")
                return True
            else:
                self.logger.info(f"No camera config found at {self.camera_config_file}, " "using defaults")
                return False
        except Exception as e:
            self.logger.warning(f"Failed to load camera config: {e}")
            return False

    def save_camera_config(self) -> bool:
        """
        Save current camera configuration to JSON file.

        Returns:
            True if config was saved successfully, False otherwise
        """
        try:
            config = {
                "position": self.custom_camera_position,
                "target": self.custom_camera_target,
                "projection": self.current_projection_mode,
                "preset": self.current_camera_preset,
                "custom_presets": {
                    k: v
                    for k, v in self.camera_presets.items()
                    if k not in ["Default", "Top View", "Side View", "Front View"]
                },
            }

            # Ensure directory exists
            os.makedirs(os.path.dirname(self.camera_config_file), exist_ok=True)

            with open(self.camera_config_file, "w") as f:
                json.dump(config, f, indent=2)

            self.logger.info(f"Camera config saved to {self.camera_config_file}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save camera config: {e}")
            return False

    def apply_camera_preset(self, preset_name: str, fly_to: bool = False) -> bool:
        """
        Apply a predefined camera preset.

        Args:
            preset_name: Name of the preset to apply
            fly_to: If True, animate camera movement to new position

        Returns:
            True if preset was applied successfully, False otherwise
        """
        try:
            if preset_name not in self.camera_presets:
                self.logger.warning(f"Camera preset '{preset_name}' not found")
                return False

            preset = self.camera_presets[preset_name]
            position = np.array(preset["position"], dtype=np.float32)
            target = np.array(preset["target"], dtype=np.float32)
            projection = preset.get("projection", "perspective")

            # Set projection mode
            ps.set_view_projection_mode(projection)
            self.current_projection_mode = projection

            # Set camera position and target
            ps.look_at(position, target, fly_to=fly_to)

            # Update current state
            self.custom_camera_position = position.tolist()
            self.custom_camera_target = target.tolist()
            self.current_camera_preset = preset_name

            self.logger.info(f"Applied camera preset: {preset_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to apply camera preset: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            return False

    def apply_custom_camera(self, fly_to: bool = False) -> bool:
        """
        Apply custom camera position and target.

        Args:
            fly_to: If True, animate camera movement to new position

        Returns:
            True if camera was set successfully, False otherwise
        """
        try:
            position = np.array(self.custom_camera_position, dtype=np.float32)
            target = np.array(self.custom_camera_target, dtype=np.float32)

            ps.set_view_projection_mode(self.current_projection_mode)
            ps.look_at(position, target, fly_to=fly_to)

            self.logger.info(f"Applied custom camera: pos={position}, target={target}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to apply custom camera: {e}")
            return False

    def save_current_view_as_preset(self, preset_name: str) -> bool:
        """
        Save current camera view as a new preset.

        Args:
            preset_name: Name for the new preset

        Returns:
            True if preset was saved successfully, False otherwise
        """
        try:
            if not preset_name or not preset_name.strip():
                self.logger.warning("Preset name cannot be empty")
                return False

            self.camera_presets[preset_name] = {
                "position": self.custom_camera_position.copy(),
                "target": self.custom_camera_target.copy(),
                "projection": self.current_projection_mode,
            }

            self.logger.info(f"Saved current view as preset: {preset_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save preset: {e}")
            return False

    def delete_preset(self, preset_name: str) -> bool:
        """
        Delete a custom preset (default presets cannot be deleted).

        Args:
            preset_name: Name of the preset to delete

        Returns:
            True if preset was deleted, False otherwise
        """
        if preset_name in ["Default", "Top View", "Side View", "Front View"]:
            self.logger.warning(f"Cannot delete default preset: {preset_name}")
            return False

        if preset_name in self.camera_presets:
            del self.camera_presets[preset_name]
            self.logger.info(f"Deleted preset: {preset_name}")
            return True
        else:
            self.logger.warning(f"Preset not found: {preset_name}")
            return False

    def set_position(self, x: float, y: float, z: float) -> None:
        """Set camera position."""
        self.custom_camera_position = [x, y, z]

    def set_target(self, x: float, y: float, z: float) -> None:
        """Set camera target."""
        self.custom_camera_target = [x, y, z]

    def set_projection_mode(self, mode: str) -> bool:
        """
        Set projection mode.

        Args:
            mode: Either "perspective" or "orthographic"

        Returns:
            True if mode was set successfully, False otherwise
        """
        if mode not in ["perspective", "orthographic"]:
            self.logger.warning(f"Invalid projection mode: {mode}")
            return False

        self.current_projection_mode = mode
        return True

    def get_preset_names(self) -> List[str]:
        """Get list of all available preset names."""
        return list(self.camera_presets.keys())

    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current camera state as dictionary.

        Returns:
            Dictionary containing position, target, projection, and preset
        """
        return {
            "position": self.custom_camera_position.copy(),
            "target": self.custom_camera_target.copy(),
            "projection": self.current_projection_mode,
            "preset": self.current_camera_preset,
        }

    def reset_to_default(self) -> bool:
        """Reset camera to default preset."""
        return self.apply_camera_preset("Default")

    def get_current_camera_from_polyscope(self) -> Dict[str, Any]:
        """
        Get current camera parameters from Polyscope's actual viewport state.

        Uses ps.get_view_camera_parameters() to read the real camera state,
        which reflects manual adjustments made with the mouse.

        Returns:
            Dictionary containing current camera position, target, and up direction
        """
        try:
            # Use the official Polyscope API to get current camera parameters
            camera_params = ps.get_view_camera_parameters()

            # Extract position from CameraParameters object
            position = np.array(camera_params.get_position())

            # Extract look direction and calculate target point
            # Polyscope uses look_dir (normalized direction vector)
            # We need to estimate a reasonable distance to place the target
            look_dir = np.array(camera_params.get_look_dir())

            # Strategy: Calculate target distance based on the previous setting
            # or use a reasonable default based on scene scale
            prev_position = np.array(self.custom_camera_position)
            prev_target = np.array(self.custom_camera_target)
            prev_dist = np.linalg.norm(prev_target - prev_position)

            # If previous distance is reasonable, use it
            if 0.1 < prev_dist < 100.0:
                target_dist = prev_dist
            else:
                # Use a default distance that works well for most scenes
                target_dist = 5.0

            # Calculate target: position + look_dir * distance
            target = position + look_dir * target_dist

            # Get up direction
            up = np.array(camera_params.get_up_dir())

            # Update internal state to match Polyscope
            self.custom_camera_position = position.tolist()
            self.custom_camera_target = target.tolist()

            result = {
                "position": position.tolist(),
                "target": target.tolist(),
                "up": up.tolist(),
                "look_dir": look_dir.tolist(),
                "fov": camera_params.get_fov_vertical_deg(),
                "aspect": camera_params.get_aspect(),
            }

            self.logger.info(
                f"Retrieved camera from Polyscope viewport: "
                f"pos=[{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}], "
                f"target=[{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}], "
                f"fov={result['fov']:.1f}°"
            )

            return result

        except Exception as e:
            self.logger.error(f"Failed to get camera from Polyscope: {e}")
            import traceback

            traceback.print_exc()

            # Return current internal state as fallback
            return {
                "position": self.custom_camera_position.copy(),
                "target": self.custom_camera_target.copy(),
                "projection": self.current_projection_mode,
            }

    def sync_from_polyscope(self) -> bool:
        """
        Synchronize internal state with current Polyscope camera.
        Useful after user manually adjusted camera view.

        Returns:
            True if successfully synchronized, False otherwise
        """
        try:
            camera_info = self.get_current_camera_from_polyscope()
            if camera_info and "position" in camera_info:
                self.custom_camera_position = camera_info["position"]
                self.custom_camera_target = camera_info["target"]
                self.logger.info("Synchronized camera state from Polyscope")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to sync from Polyscope: {e}")
            return False

    def get_view_as_json(self) -> str:
        """
        Get the current view as a JSON string.
        This uses Polyscope's official API and can be used to save/restore views.

        The JSON includes camera position, orientation, FOV, projection mode,
        and window size. You can restore it with set_view_from_json().

        Returns:
            JSON string representation of the current view
        """
        try:
            view_json = ps.get_view_as_json()
            self.logger.info("Exported current view as JSON")
            return view_json
        except Exception as e:
            self.logger.error(f"Failed to get view as JSON: {e}")
            return ""

    def set_view_from_json(self, view_json: str) -> bool:
        """
        Set the current view from a JSON string.
        This uses Polyscope's official API to restore a saved view.

        Args:
            view_json: JSON string from get_view_as_json()

        Returns:
            True if successful, False otherwise
        """
        try:
            ps.set_view_from_json(view_json)
            # Sync our internal state after setting the view
            self.sync_from_polyscope()
            self.logger.info("Restored view from JSON")
            return True
        except Exception as e:
            self.logger.error(f"Failed to set view from JSON: {e}")
            return False

    def look_at_bbox(
        self, bbox_min: np.ndarray, bbox_max: np.ndarray, view_type: str = "perspective", distance_factor: float = 2.0
    ) -> bool:
        """
        Automatically position camera to view a bounding box.

        Args:
            bbox_min: Minimum corner of bounding box [x, y, z]
            bbox_max: Maximum corner of bounding box [x, y, z]
            view_type: "perspective", "top", "front", or "side"
            distance_factor: Multiplier for camera distance from bbox

        Returns:
            True if camera was positioned successfully
        """
        try:
            center = (bbox_min + bbox_max) / 2.0
            size = bbox_max - bbox_min
            max_dim = np.max(size)

            if view_type == "top":
                position = center + np.array([0, 0, max_dim * distance_factor])
                projection = "orthographic"
            elif view_type == "front":
                position = center + np.array([0, -max_dim * distance_factor, size[2] * 0.5])
                projection = "perspective"
            elif view_type == "side":
                position = center + np.array([max_dim * distance_factor, 0, size[2] * 0.5])
                projection = "perspective"
            else:  # perspective
                offset = np.array([max_dim * 0.5, -max_dim * distance_factor, max_dim * 0.8])
                position = center + offset
                projection = "perspective"

            self.custom_camera_position = position.tolist()
            self.custom_camera_target = center.tolist()
            self.current_projection_mode = projection

            return self.apply_custom_camera()
        except Exception as e:
            self.logger.error(f"Failed to look at bbox: {e}")
            return False
