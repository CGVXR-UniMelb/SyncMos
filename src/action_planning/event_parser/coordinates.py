"""Project parsed grid/object targets into world coordinates without changing plan order."""

from __future__ import annotations

from logging import Logger
from typing import TYPE_CHECKING, Callable, Dict, List

import numpy as np

from src.utils.grid_utils import grid_to_world

if TYPE_CHECKING:
    from src.action_planning.contracts import BatchParsedEvents
    from src.type import Object


class EventCoordinateResolver:
    def __init__(self, objects: list[Object], bbox_min: tuple[float, float], grid_step: float, logger: Logger):
        self.objects = objects
        self.bbox_min = bbox_min
        self.grid_step = grid_step
        self.logger = logger

    def convert(
        self, parsed_events: BatchParsedEvents, calculate_hand_position: Callable[[Object, object], np.ndarray]
    ) -> BatchParsedEvents:
        """Mutate and return the same mapping, retaining seating and fallback policies."""
        object_finder = {f"{obj.label}_{obj.idx}": obj for obj in self.objects}
        shared_seating = self._collect_shared_seating(parsed_events, object_finder)
        seating_offsets, seat_by_character = self._build_seating_offsets(shared_seating, object_finder)
        for event_idx, event_data in parsed_events.items():
            for char_name, char_data in event_data.items():
                self._assign_target_position(
                    char_data, event_idx, char_name, object_finder, seating_offsets, seat_by_character
                )
                # Intelligent setting of hand_target_position height (if LLM does not provide it explicitly)
                if "hand_target" in char_data and "hand_target_position" not in char_data:
                    hand_target = char_data["hand_target"]
                    if isinstance(hand_target, str) and hand_target in object_finder:
                        target_obj = object_finder[hand_target]
                        hand_position = calculate_hand_position(target_obj, char_data.get("target_action", ""))
                        char_data["hand_target_position"] = list(hand_position)
                        self.logger.info(f"Set hand position for '{hand_target}': {hand_position}")

        return parsed_events

    @staticmethod
    def _collect_shared_seating(parsed_events, object_finder):
        seat_labels = {"couch", "sofa", "bench", "bed"}
        shared_seating: Dict[str, List[tuple]] = {}
        for event_idx, event_data in parsed_events.items():
            for char_name, char_data in event_data.items():
                target_grid = char_data.get("target_grid")
                if not isinstance(target_grid, str):
                    continue
                obj = object_finder.get(target_grid)
                if obj is None or obj.label not in seat_labels:
                    continue

                # Enabled only when performing sitting actions/Shared seat allocation activated for lying-type actions
                target_action = char_data.get("target_action", [])
                if isinstance(target_action, list):
                    action_text = " ".join([str(a) for a in target_action if a])
                else:
                    action_text = str(target_action or "")
                action_text = action_text.lower()
                if not any(k in action_text for k in ["sit down", "maintain sit", "lie down", "maintain lie"]):
                    continue

                shared_seating.setdefault(target_grid, []).append((event_idx, char_name, char_data))

        return shared_seating

    @staticmethod
    def _build_seating_offsets(shared_seating, object_finder):
        seating_offsets: Dict[tuple, List[float]] = {}
        seat_by_character: Dict[tuple, List[float]] = {}
        for target_grid, entries in shared_seating.items():
            obj = object_finder[target_grid]

            # Calculate the lateral direction of seats (perpendicular to the object’s orientation)
            orientation = obj.orientation[:2]
            norm = np.linalg.norm(orientation)
            if norm < 1e-6:
                orientation = np.array([1.0, 0.0], dtype=np.float32)
            else:
                orientation = orientation / norm
            lateral = np.array([-orientation[1], orientation[0]], dtype=np.float32)

            if len(entries) <= 1:
                event_idx, char_name, _ = entries[0]
                seating_offsets[(event_idx, char_name)] = [0.0, float(lateral[0]), float(lateral[1])]
                seat_by_character[(char_name, target_grid)] = seating_offsets[(event_idx, char_name)]
                continue

            # Allocate seat spacing based on width, within reasonable limits
            seat_span = max(obj.width * 0.8, 0.6)
            step = seat_span / max(len(entries) - 1, 1)
            step = min(0.6, max(0.35, step))
            start = -step * (len(entries) - 1) / 2.0
            offsets = [start + i * step for i in range(len(entries))]

            # Stable sorting to ensure consistent allocation
            entries_sorted = sorted(entries, key=lambda x: (x[0], x[1]))
            for offset, (event_idx, char_name, char_data) in zip(offsets, entries_sorted):
                seating_offsets[(event_idx, char_name)] = [float(offset), float(lateral[0]), float(lateral[1])]
                seat_by_character[(char_name, target_grid)] = seating_offsets[(event_idx, char_name)]

        return seating_offsets, seat_by_character

    def _assign_target_position(
        self, char_data, event_idx, char_name, object_finder, seating_offsets, seat_by_character
    ) -> None:
        bbox_min = self.bbox_min
        if "target_grid" in char_data:
            target_grid = char_data["target_grid"]

            # Check if target_grid is an array or a string
            if isinstance(target_grid, str):
                # String: object name, find the object’s position
                if target_grid in object_finder:
                    obj = object_finder[target_grid]
                    # Use the center position of the object (Object class uses center property) or shared seat offset
                    offset_info = seating_offsets.get((event_idx, char_name))
                    if offset_info is None and obj.label in {"couch", "sofa", "bench", "bed"}:
                        offset_info = seat_by_character.get((char_name, target_grid))

                    seat_height_offset = 0.2
                    seat_obj_labels = {"chair", "armchair", "stool", "bench", "couch", "sofa", "bed"}

                    if offset_info and obj.label in {"couch", "sofa", "bench", "bed"}:
                        offset, lateral_x, lateral_y = offset_info
                        base_xy = obj.center[:2] + obj.orientation[:2] * (obj.depth * obj.sit_offset_horizontal)
                        base_xy = base_xy + np.array([lateral_x, lateral_y], dtype=np.float32) * offset
                        z = (obj.center[2] if len(obj.center) >= 3 else 0.0) + seat_height_offset
                        world_coords = np.array([base_xy[0], base_xy[1], z], dtype=np.float32)
                        char_data["target_position"] = list(world_coords)
                        self.logger.info(f"Using shared seating offset for '{target_grid}': {world_coords}")
                    else:
                        world_coords = obj.center[:3] if len(obj.center) >= 3 else obj.center
                        if obj.label in seat_obj_labels:
                            world_coords = np.array(world_coords, dtype=np.float32)
                            world_coords[2] = world_coords[2] + seat_height_offset
                        char_data["target_position"] = list(world_coords)
                        self.logger.info(f"Using object '{target_grid}' center position: {world_coords}")
                else:
                    self.logger.warning(f"Object '{target_grid}' not found in scene, using grid [0, 0]")
                    world_coords = grid_to_world(
                        0, 0, grid_size=self.grid_step, bbox_min=bbox_min, return_center=True, z=0.0
                    )
                    char_data["target_position"] = list(world_coords)
            elif isinstance(target_grid, list) and len(target_grid) == 2:
                # Array: grid coordinates, convert to world coordinates
                grid_x, grid_y = target_grid
                world_coords = grid_to_world(
                    grid_x, grid_y, grid_size=self.grid_step, bbox_min=bbox_min, return_center=True, z=0.0
                )
                char_data["target_position"] = list(world_coords)
            else:
                self.logger.error(f"Invalid target_grid format: {target_grid}")


def calculate_smart_hand_position(
    target_obj: Object, action: object, find_surface_height: Callable[[Object], float]
) -> np.ndarray:
    """
    Smartly calculate the hand position height
    Args:
        target_obj: Target object
        action: Action Types
        find_surface_height: Keep sub-levels of access by following the supported surface search method of the parser
    Returns:
        np.ndarray: 3D Position [x, y, z]
    """
    # Use the center of the object for the base position
    hand_position = target_obj.center.copy()

    # Set the height intelligently based on the action type and object type
    # Compatible with action formats in string or list format
    if isinstance(action, list):
        action_str = " ".join([str(a) for a in action if a])
    elif action is None:
        action_str = ""
    else:
        action_str = str(action)
    action_lower = action_str.lower()

    if any(keyword in action_lower for keyword in ["pick up", "grab", "take", "hold"]):
        # Taking action: Using the center height of the object
        hand_position[2] = target_obj.center[2]

    elif any(keyword in action_lower for keyword in ["put down", "place", "drop", "set"]):
        # Placing action: Need to find a supporting surface
        supporting_surface_height = find_surface_height(target_obj)
        hand_position[2] = supporting_surface_height

    elif any(keyword in action_lower for keyword in ["drink", "eat", "use"]):
        # Using action: Setting the height based on the object type
        if target_obj.label.lower() in ["table", "desk", "counter"]:
            hand_position[2] = target_obj.center[2] + target_obj.height / 2  # Table surface
        else:
            hand_position[2] = target_obj.center[2]  # Other object centers

    elif any(keyword in action_lower for keyword in ["open", "close", "touch"]):
        # Contact action: Setting the height based on the object type
        if target_obj.label.lower() in ["door", "cabinet", "drawer"]:
            hand_position[2] = target_obj.center[2] + target_obj.height * 0.3  # Handle height
        else:
            hand_position[2] = target_obj.center[2]  # Other object centers

    else:
        # Default: Using the center height of the object
        hand_position[2] = target_obj.center[2]

    return hand_position


def find_supporting_surface_height(objects: list[Object], target_obj: Object) -> float:
    """
    Finding the height of the supporting surface for the target object
    Args:
        objects: Check objects in the original order of the scene; return the first surface that meets the distance condition
        target_obj: Target object
    Returns:
        float: Height of the supporting surface
    """
    # Searching for nearby tables, counters, and other supporting surfaces
    for obj in objects:
        if obj.label.lower() in ["table", "desk", "counter", "shelf"]:
            # Checking if it’s near a supporting surface
            distance = np.linalg.norm(target_obj.center[:2] - obj.center[:2])
            if distance < 1.0:  # Considered the same supporting surface within 1 meter
                return obj.center[2] + obj.height / 2  # Surface height

    # If no supporting surface is found, use the default height
    return 1.0  # The default height is 1 meter
