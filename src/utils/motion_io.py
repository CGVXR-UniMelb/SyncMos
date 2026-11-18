"""
Motion Data I/O Utilities
For saving and loading animation data
"""

import ast
import os
import warnings
from collections.abc import Mapping
import numpy as np
import torch
from typing import List, Dict, Any, Optional


class MotionMetadataWarning(UserWarning):
    """Optional motion metadata could not be read; playback can still continue."""


def read_initial_grid_positions(
    data: Mapping[str, np.ndarray], filepath: str, logger: Optional[Any] = None
) -> Dict[str, Any]:
    """Read the legacy dictionary text without executing expressions or pickle.

    Invalid/missing optional positions retain the historical empty-dict fallback,
    but now report the filename, field, and cause through a logger or warning.
    """
    field = "character_initial_grid_positions"
    try:
        value = data[field]
        if value.ndim != 0 or value.dtype.kind not in {"U", "S"}:
            raise ValueError("expected a scalar string containing a dictionary")
        text = value.item()
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        positions = ast.literal_eval(text)
        if not isinstance(positions, dict):
            raise ValueError("expected a dictionary")
        return positions
    except (KeyError, SyntaxError, ValueError, TypeError) as error:
        message = f"{filepath}: could not read {field}: {error}; using empty initial positions."
        if logger is not None:
            logger.warning(message)
        else:
            warnings.warn(message, MotionMetadataWarning, stacklevel=2)
        return {}


def save_motion_data(
    characters: List,
    scene_name: str,
    num_characters: int,
    user_instruction: str,
    character_initial_grid_positions: Dict,
    fId: int,
    max_fId: int,
    motion_dir: str = "motion",
    narrator_type: Optional[str] = None,
    parser_mode: Optional[str] = None,
    logger: Optional[Any] = None,
) -> str:
    """
    Save animation data to .npz Files

    Args:
        characters: Character list
        scene_name: Scene name
        num_characters: Number of characters
        user_instruction: User story text
        character_initial_grid_positions: Dictionary of initial grid positions for characters
        fId: Current frame ID
        max_fId: Maximum frame ID
        motion_dir: Save directory (default) "motion")
        narrator_type: Narrator Type (e.g "baseline", "timeline")
        parser_mode: Event Parser mode (e.g "single", "batch", "timeline")
        logger: Log Logger (Optional)

    Returns:
        File Path to Save

    Raises:
        Exception: Throw an anomaly when saving failed
    """
    try:
        # Create subdirectories based on narrator type and parser mode
        if narrator_type and parser_mode:
            # Standardized type name
            narrator_name = f"narrator_{narrator_type}" if narrator_type != "baseline" else "narrator"
            parser_name = f"parser_{parser_mode}" if parser_mode != "single" else "parser"

            motion_dir = os.path.join(motion_dir, f"{narrator_name}_{parser_name}")

        # Create motion Folder (If it does not exist)
        os.makedirs(motion_dir, exist_ok=True)

        # Generate filenames, automatically add serial numbers
        base_filename = f"{scene_name}_{num_characters}chars"
        sequence_num = 1
        while True:
            filename = f"{base_filename}_{sequence_num:03d}.npz"
            filepath = os.path.join(motion_dir, filename)
            if not os.path.exists(filepath):
                break
            sequence_num += 1

        # Data to be saved
        character_names = [c.name for c in characters]
        save_data = {
            "scene_name": scene_name,
            "num_characters": num_characters,
            "character_names": character_names,
            "user_instruction": user_instruction,
            "character_initial_grid_positions": str(character_initial_grid_positions),
            "fId": fId,
            "max_fId": max_fId,
            "total_frames": max_fId + 1,
        }

        # Save data for each role
        for character in characters:
            char_name = character.name

            # Save body_params (Actions after time distortion)
            if character.body_params is not None:
                save_data[f"{char_name}_body_params"] = character.body_params.cpu().numpy()

            # Save body_params_origin (Original actions, if any)
            if character.body_params_origin is not None:
                save_data[f"{char_name}_body_params_origin"] = character.body_params_origin.cpu().numpy()

        # Save to file
        np.savez_compressed(filepath, **save_data)

        if logger:
            logger.info(f"Motion data saved to: {filepath}")
            logger.info(f"Total frames: {max_fId + 1}, Final fId: {fId}")

        return filepath

    except Exception as e:
        if logger:
            import traceback

            logger.error(f"Failed to save motion data: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def load_motion_data(filepath: str, device: str = "cpu", logger: Optional[Any] = None) -> Dict[str, Any]:
    """
    From .npz Load animation data from file

    Args:
        filepath: File Paths
        device: Target device("cpu" Or "cuda")
        logger: Log Logger (Optional)

    Returns:
        Dictionary containing all data

    Raises:
        Exception: Throw exception if loading fails
    """
    try:
        if logger:
            logger.info(f"Loading motion file: {filepath}")

        # The existing writer stores strings/numeric arrays; pickle is unnecessary.
        with np.load(filepath, allow_pickle=False) as data:

            # Extract metadata
            result = {
                "scene_name": str(data["scene_name"]),
                "num_characters": int(data["num_characters"]),
                "character_names": list(data["character_names"]),
                "user_instruction": str(data["user_instruction"]),
                "max_fId": int(data["max_fId"]),
                "fId": int(data["fId"]),
                "total_frames": int(data["total_frames"]),
                "character_data": {},
            }

            result["character_initial_grid_positions"] = read_initial_grid_positions(data, filepath, logger)

            # Load data for each character
            for char_name in result["character_names"]:
                char_data = {}

                # Load body_params (After Time Distortion)
                body_params_key = f"{char_name}_body_params"
                if body_params_key in data:
                    char_data["body_params"] = torch.from_numpy(data[body_params_key]).to(device)

                # Load body_params_origin (Original actions, if any)
                body_params_origin_key = f"{char_name}_body_params_origin"
                if body_params_origin_key in data:
                    char_data["body_params_origin"] = torch.from_numpy(data[body_params_origin_key]).to(device)

                result["character_data"][char_name] = char_data

        if logger:
            logger.info(
                f"Loaded: Scene={result['scene_name']}, Characters={result['num_characters']}, Frames={result['total_frames']}"
            )

        return result

    except Exception as e:
        if logger:
            import traceback

            logger.error(f"Failed to load motion file: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def list_motion_files(motion_dir: str = "motion") -> List[str]:
    """
    List all saved animation files

    Args:
        motion_dir: Motion folder path

    Returns:
        List of file paths (sorted)
    """
    import glob

    if not os.path.exists(motion_dir):
        return []

    pattern = os.path.join(motion_dir, "*.npz")
    return sorted(glob.glob(pattern))


def get_motion_file_info(filepath: str) -> Dict[str, Any]:
    """
    Obtaining basic information of the animation file (no full data loading)

    Args:
        filepath: File Paths

    Returns:
        Dictionary containing basic information
    """
    try:
        with np.load(filepath, allow_pickle=False) as data:

            info = {
                "filepath": filepath,
                "filename": os.path.basename(filepath),
                "scene_name": str(data["scene_name"]),
                "num_characters": int(data["num_characters"]),
                "character_names": list(data["character_names"]),
                "total_frames": int(data["total_frames"]),
                "max_fId": int(data["max_fId"]),
                "file_size_kb": os.path.getsize(filepath) / 1024,
            }

            # Checking for origin data
            has_origin = any(f"{char_name}_body_params_origin" in data for char_name in info["character_names"])
            info["has_origin_data"] = has_origin

        return info

    except Exception as e:
        return {"filepath": filepath, "filename": os.path.basename(filepath), "error": str(e)}
