import queue
import re
from typing import TYPE_CHECKING, List

import numpy as np
import torch

from ..motion_synthesis.state_manager import StateManager
from ..motion_synthesis.states import State
from ..scene.scene import Scene
from ..type import Character, Event, Object
from ..utils import misc
from ..utils.grid_utils import grid_to_world
from .event_cache import event_ids, get_cached_event, remap_batch_results, unique_events
from .event_parser import get_event_parser
from .event_parser.event_parser import _VALID_RELATIONSHIPS
from .event_parser.position_sampler import PositionSampler
from .narrator import get_narrator
from .scene_describer import get_scene_describer
from .seating import table_facing_direction

if TYPE_CHECKING:
    from .contracts import EventIdParsedEvents


class ActionPlanningModule:
    def __init__(self, scene: Scene, characters: List[Character], state_manager: StateManager, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.scene = scene
        self.characters = characters
        self.state_manager = state_manager

        # Initialize logger first
        self.logger = misc.get_console_logger("ActionPlanningModule", cfg.logging_level)

        self.scene_describer = get_scene_describer(cfg)
        self.scene_description = self.scene_describer.get_scene_description(scene.scene_name, scene.scene_graph)
        self.narrator = get_narrator(characters, scene.objects, cfg)

        # Pass scene_name and grid_map to support BatchEventParser
        self.event_parser = get_event_parser(
            characters, scene.objects, cfg, scene_name=scene.scene_name, grid_map=scene.grid_map
        )

        # Check the parser mode and record it
        self.parser_mode = getattr(cfg.event_parser, "parser_mode", "single")
        self.logger.info(f"Using parser mode: {self.parser_mode}")

        self.event_queue = queue.Queue()
        self.position_sampler = PositionSampler(scene.grid_map, cfg)
        self.position_sampler.update_anchors(characters, scene.objects)

        self.finder = {f"{obj.label}_{obj.idx}": obj for obj in scene.objects}
        self.finder.update({character.name: character for character in self.characters})

        # Batch parse the cache (store by event_id, loose matching: use if the cache exists)
        self._batch_parsed_cache: "EventIdParsedEvents" = {}

        # Use the candidate actions and tags provided by motion_generator
        self.tags = self.state_manager.motion_generator.get_candidate_actions()
        self.hhi_tags = getattr(self.state_manager.motion_generator, "hhi_tags", set())
        self.minimum_distance = cfg.misc.minimum_distance
        self.root_height_offset = cfg.misc.root_height_offset
        self.position_relationships_pattern = re.compile(
            r"^(" + "|".join(rel for rel in _VALID_RELATIONSHIPS if rel != "look at") + ")"
        )

    def is_new_event_required(self) -> bool:
        return self.narrator.is_new_event_required()

    def generate_new_event(self, user_instruction: str | None = None) -> None:
        event, parsed_event = self._get_subsequent_event(user_instruction)
        if event is None:
            self.logger.info("Skipping the planning...")
            return
        self.event_queue.put((event, parsed_event))
        self.logger.info("Planning done. New event is created to the queue.")

    def _get_subsequent_event(self, user_instruction=None):
        event = self.narrator.generate_subsequent_plan(
            self.scene_description,
            self.tags,
            user_instruction,
        )
        if event is None:
            return None, None

        if not self.narrator.event_based_planning:
            parsed_event = event  # already parsed
            dummy_event = self.narrator.previous_events[-1]
            return dummy_event, parsed_event

        if hasattr(self.narrator, "event_list") and event is not None and self.parser_mode == "batch":
            self._refresh_batch_cache(event)
            parsed_event = get_cached_event(event, self.narrator, self._batch_parsed_cache, self.logger)
            if not parsed_event:
                self.logger.warning("Event not found in cache, falling back to single event parsing")
                parsed_event = self._parse_single_event(event)
        else:
            parsed_event = self._parse_single_event(event)
        return event, parsed_event

    def _parse_single_event(self, event):
        return self.event_parser.parse_event(
            event, self.scene_description, self.tags, self.narrator.previous_events[:-1]
        )

    def _refresh_batch_cache(self, event):
        """Reparse the full current/pending set if any event ID is absent from the cache."""
        all_event_ids = []
        if event is not None:
            all_event_ids.extend(event_ids(event, self.narrator))
        for item in self.narrator.event_list:
            all_event_ids.extend(event_ids(item, self.narrator))

        # Check which event_id is not in the cache (loose match: use it if it’s in the cache)
        missing_event_ids = [eid for eid in all_event_ids if eid not in self._batch_parsed_cache]

        # Debugging logs
        self.logger.info(f"📋 Current event_ids: {sorted(all_event_ids)}")
        self.logger.info(f"💾 Cached event_ids: {sorted(self._batch_parsed_cache.keys())}")
        self.logger.info(f"❌ Missing event_ids: {sorted(missing_event_ids)}")

        # If there are missing events, clear the cache and re-parse all events
        if missing_event_ids:
            self.logger.info(
                f"🎯 Timeline + Batch: Detected {len(missing_event_ids)} missing events - clearing cache and re-parsing all events"
            )

            # Clear the cache
            self._batch_parsed_cache.clear()
            self.logger.info("🗑️  Cache cleared")

            # Collect all events that need to be parsed
            events_for_batch = []
            if event is not None:
                events_for_batch.append(event)
            events_for_batch.extend(self.narrator.event_list)

            # Remove duplicates and expand ParallelGroup
            deduped = unique_events(events_for_batch, self.narrator)

            self.logger.info(f"🔄 Re-parsing all {len(deduped)} events...")
            self.logger.info(
                f"   Events to parse: {[self.narrator._get_event_id(e) if hasattr(self.narrator, '_get_event_id') else str(e) for e in deduped]}"
            )
            parsed_events = self._parse_events_batch(deduped)
            self.logger.info(f"   Parsed event_ids: {sorted(parsed_events.keys())}")
            # Replace the cache completely with the new results
            self._batch_parsed_cache = parsed_events
            self.logger.info(f"   Cache rebuilt with: {sorted(self._batch_parsed_cache.keys())}")
        else:
            self.logger.info("✅ All required events in cache, reusing")

    def _parse_events_batch(self, events):
        """Batch parsing of multiple events"""
        self.logger.info(f"Batch parsing {len(events)} events from timeline...")

        if hasattr(self.event_parser, "parse_events"):
            # Inject dependency information before calling (if available)
            if hasattr(self.narrator, "timeline_dependencies") and hasattr(self.event_parser, "timeline_dependencies"):
                self.event_parser.timeline_dependencies = getattr(self.narrator, "timeline_dependencies", [])
            # BatchEventParser
            parsed_results = self.event_parser.parse_events(events, self.tags, self.scene_description, with_usage=False)
            # Debugging: Print the original structure returned by LLM
            self.logger.info("🔍 LLM parsed_results structure:")
            for idx_key, data in parsed_results.items():
                self.logger.info(f"   Index {idx_key}: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                if isinstance(data, dict):
                    for char_name, char_data in data.items():
                        event_id = char_data.get("event_id", "NO_ID") if isinstance(char_data, dict) else "NOT_DICT"
                        self.logger.info(f"      - {char_name}: event_id={event_id}")

            # Map the results to event_id -> parsed_result
            # BatchEventParser returns an index structure {0: {char: data}, 1: {char: data}, ...}
            # Need to map it to event_id based on event_id {event_id: {char: data}, ...}
            # Strategy 1 (Priority): Use event_id output by LLM for precise matching
            # Strategy 2 (Fallback): Match based on character names
            parsed_results = remap_batch_results(events, parsed_results, self.narrator, self.logger)
        elif hasattr(self.event_parser, "parse_parallel_events"):
            # EventParserTimeline with parallel support
            parsed_results = self.event_parser.parse_parallel_events(
                events, self.scene_description, self.tags, self.narrator.previous_events
            )
        else:
            # No individual parsing implementation available; return null result with a warning.
            self.logger.warning(
                "Event parser does not support batch or parallel parsing; falling back is not implemented here."
            )
            parsed_results = {}

        return parsed_results

    def allocate_new_plan(self, event: Event, parsed_event):
        """Apply a parsed plan while preserving the existing update order.

        Event membership is assigned first, then per-character actions and hand
        targets, interaction partners, target positions, and facing directions.
        ParallelEventGroup and parser dictionary variants remain supported.
        """
        self._assign_event_characters(event)
        for character in event.involved_characters:
            char_key = self._find_character_plan_key(character, parsed_event)
            if char_key is None:
                self.logger.warning(
                    f"Parsed event has no entry for character '{character.name}'. Keys: {list(parsed_event.keys())}"
                )
                continue
            plan = parsed_event[char_key]
            self._assign_character_actions(character, plan)
            self._assign_hand_object(character, plan)
            self._assign_hand_position(character, plan)
            self._assign_attachment_intent(character, plan)

        if len(event.involved_characters) == 2:
            character_1 = event.involved_characters[0]
            character_2 = event.involved_characters[1]
            character_1.interactee = character_2
            character_2.interactee = character_1
        self._assign_target_positions(event)

    def _assign_event_characters(self, event):
        """Assign event membership before updating any character plan."""
        is_parallel_group = hasattr(event, "events") and hasattr(event, "parallel_group")
        group_members = event.involved_characters if is_parallel_group else []
        group_id = None
        if is_parallel_group:
            try:
                group_id = "|".join(event.parallel_group)
            except Exception:
                group_id = str(id(event))

        # assign the event to the involved characters
        for character in event.involved_characters:
            if character.state != State.IDLE:  # when looping
                self.state_manager.change_state(character, State.IDLE)
                if character.interactee is not None:
                    self.state_manager.change_state(character.interactee, State.IDLE)
            character.event = event
            if is_parallel_group:
                character.parallel_group_id = group_id
                character.parallel_group_members = [c for c in group_members if c != character]
            else:
                character.parallel_group_id = None
                character.parallel_group_members = []

    def _find_character_plan_key(self, character, parsed_event):
        """Match exact name, case-insensitive name, then a sole result."""
        char_key = None
        if isinstance(parsed_event, dict):
            if character.name in parsed_event:
                char_key = character.name
            else:
                # Case-insensitive matching
                lower_map = {k.lower(): k for k in parsed_event.keys()}
                if character.name.lower() in lower_map:
                    char_key = lower_map[character.name.lower()]
                elif len(parsed_event) == 1:
                    # If there is only one character, use the fallback method
                    char_key = next(iter(parsed_event.keys()))
        return char_key

    def _assign_character_actions(self, character, plan):
        """Apply action labels, relationships, and an optional batch target."""
        # Prioritize using the text field; if not available, use target_action
        if "text" in plan and plan["text"]:
            # The Batch Parser provides text, using text as target_actions
            character.target_actions = [plan["text"]]
        else:
            # Use target_action
            character.target_actions = plan["target_action"]

        character.previous_relationships = character.relationships
        character.relationships = plan["relationships"]
        if not character.relationships:
            character.relationships = character.previous_relationships.copy()

        if any("sit down" in action.lower() for action in character.target_actions):
            target_object = plan.get("target_grid")
            character.seat_object = target_object if isinstance(target_object, str) else None

        # Handle target_position from batch parser (if exists)
        if "target_position" in plan:
            # The batch parser already provides world coordinates
            pos = plan["target_position"]
            character._batch_target_position = torch.tensor(pos, dtype=torch.float32, device=self.device).reshape(1, 3)

    def _assign_hand_object(self, character, plan):
        """Preserve string and legacy dictionary hand-object handling."""
        # First handle hand_target (if exists)
        if "hand_target" in plan:
            hand_target = plan["hand_target"]

            # Compatible with different formats of batch parser and traditional parser
            if isinstance(hand_target, dict):
                # Traditional Parser Format: {"object": "bottle_1"}
                character.target_hand_object = hand_target["object"]
            elif isinstance(hand_target, str):
                # Batch Parser Format: Directly as object name strings
                character.target_hand_object = hand_target
            else:
                # Other cases (e.g., list format of coordinates)
                character.target_hand_object = None
        else:
            character.target_hand_object = None

    def _assign_hand_position(self, character, plan):
        """Dispatch explicit hand positions before object-based inference."""
        # Then handle hand_target_position
        # Rule: Use hand_target_position if available; if not available but there is hand_target, automatically infer
        if "hand_target_position" in plan:
            # If there is an explicit hand_target_position, use it directly
            hand_target_position = plan["hand_target_position"]

            if isinstance(hand_target_position, dict):
                self._assign_structured_hand_position(character, hand_target_position, plan)
            elif isinstance(hand_target_position, list):
                # Numerical XYZ absolute coordinates (compatible with numpy scalar types)
                if len(hand_target_position) == 3 and all(
                    isinstance(v, (int, float, np.integer, np.floating)) for v in hand_target_position
                ):
                    character.target_hand_position = torch.tensor(
                        hand_target_position, dtype=torch.float32, device=self.device
                    ).reshape(1, 3)
                # Area conditions (string list)
                elif all(isinstance(v, str) for v in hand_target_position):
                    character.target_hand_position = self._calculate_hand_target_position_from_area(
                        hand_target_position, "center"
                    )
                else:
                    self.logger.warning(f"Unsupported hand_target_position list format: {hand_target_position}")
                    character.target_hand_position = None
            else:
                self.logger.warning(f"Unsupported hand_target_position type: {type(hand_target_position)}")
                character.target_hand_position = None
        elif character.target_hand_object and character.target_hand_object in self.finder:
            # No hand_target_position, but there is a hand_target; position is automatically inferred
            character.target_hand_position = self._calculate_hand_target_position(
                character.target_hand_object, "center"
            )
        else:
            # None for both
            character.target_hand_position = None

    def _assign_attachment_intent(self, character, plan):
        """Apply explicit or action-inferred attachment/placement intent."""
        # Handling attachment control (whether there is a hand_target or hand_target_position)
        # Decision-making based on the event parser’s attachment control (with intelligent fallback)
        if character.target_hand_object or character.target_hand_position is not None:
            attachment_control = plan.get("attachment_control", None)
            if attachment_control is None:
                # Intelligent inference: If it’s an action to pick up an object, attach; if it’s an action to place an object, detach
                actions_text = (
                    " ".join(character.target_actions)
                    if isinstance(character.target_actions, list)
                    else str(character.target_actions)
                )
                a = actions_text.lower()
                if any(k in a for k in ["pick up", "grab", "hold", "take"]):
                    attachment_control = "attach"
                elif any(k in a for k in ["put down", "place", "drop", "set"]):
                    attachment_control = "detach"
                else:
                    attachment_control = "none"

            if attachment_control == "attach":
                # Set the attachment intention, waiting for the hand to approach the object
                grasp_type = "default"
                if "hand_target" in plan:
                    hand_target = plan["hand_target"]
                    if isinstance(hand_target, dict) and "grasp_type" in hand_target:
                        grasp_type = hand_target["grasp_type"] or "default"

                # target_hand_object can be None (e.g., receiving an object from the air)
                # If it’s None but there is hand_target_position, the system will automatically detect an available object as it approaches
                character.attachment_intent = {
                    "object": character.target_hand_object,  # Allowable value: None
                    "grasp_type": grasp_type,
                    "attached": False,
                    "action_type": "attach",
                }
                character.placement_intent = None
            elif attachment_control == "detach":
                character.attachment_intent = None
                # Set the placement intention, waiting for the hand to approach the target position
                character.placement_intent = {
                    "target_position": (
                        character.target_hand_position.clone() if character.target_hand_position is not None else None
                    ),
                    "placed": False,
                    "action_type": "detach",
                }
            else:
                # attachment_control == "none" No attachment is performed at this time/Separation operation
                character.attachment_intent = None
                character.placement_intent = None
        else:
            # Neither hand_target nor hand_target_position are present
            character.attachment_intent = None
            character.placement_intent = None

    def _assign_structured_hand_position(self, character, hand_target_position, plan):
        """Resolve a grid, object, area, or explicit-height hand target."""
        # Supported by hand_target_grid Provide junction points (The mid-air point where the action is carried out)
        if "hand_target_grid" in hand_target_position:
            grid_xy = hand_target_position["hand_target_grid"]
            if isinstance(grid_xy, (list, tuple)) and len(grid_xy) == 2:
                gx, gy = int(grid_xy[0]), int(grid_xy[1])
                x_min, y_min = self.scene.grid_map.bbox[0][:2]
                world = grid_to_world(
                    gx,
                    gy,
                    grid_size=self.scene.grid_map.grid_step,
                    bbox_min=(x_min, y_min),
                    return_center=True,
                    z=0.0,
                )

                # Determine Z-coordinate
                if hand_target_position.get("position_type") == "air":
                    # In the air: Use explicitly provided height
                    z = float(hand_target_position.get("height", 1.25))
                elif hand_target_position.get("position_type") == "surface" and "object" in hand_target_position:
                    # On the surface: When there is an object, use the surface height of the object
                    obj_name = hand_target_position["object"]
                    if obj_name in self.finder:
                        obj = self.finder[obj_name]
                        # Obtain surface height (center of the object) + Half of the height)
                        z = obj.center[2] + obj.height / 2
                        self.logger.info(f"Using surface height from {obj_name}: {z}")
                    else:
                        self.logger.warning(f"Object '{obj_name}' not found, using default height")
                        z = 1.0
                else:
                    # Default: Ground height
                    z = world[2]

                character.target_hand_position = torch.tensor(
                    [world[0], world[1], z], dtype=torch.float32, device=self.device
                ).reshape(1, 3)
            else:
                self.logger.warning(f"Invalid hand_target_grid: {hand_target_position['hand_target_grid']}")
                character.target_hand_position = None
        elif "object" in hand_target_position:
            character.target_hand_position = self._calculate_hand_target_position(
                hand_target_position["object"], hand_target_position.get("position_type", "center")
            )
        elif "area" in hand_target_position:
            character.target_hand_position = self._calculate_hand_target_position_from_area(
                hand_target_position["area"], hand_target_position.get("position_type", "center")
            )
        elif hand_target_position.get("position_type") == "air" and "height" in hand_target_position:
            # In the air: Use existing XY (prefer target_position, otherwise center of the object), set Z=height
            base_xy = None
            if "target_position" in plan:
                base_xy = plan["target_position"][:2]
            elif character.target_position is not None:
                base_xy = character.target_position.squeeze(0).tolist()[:2]
            elif character.target_hand_object and character.target_hand_object in self.finder:
                obj = self.finder[character.target_hand_object]
                base_xy = obj.center[:2].tolist() if hasattr(obj.center, "tolist") else obj.center[:2]
            else:
                base_xy = [0.0, 0.0]
            z = float(hand_target_position["height"])
            character.target_hand_position = torch.tensor(
                [base_xy[0], base_xy[1], z], dtype=torch.float32, device=self.device
            ).reshape(1, 3)
        else:
            self.logger.warning(f"Unsupported hand_target_position dict format: {hand_target_position}")
            character.target_hand_position = None

    def _assign_target_positions(self, event):
        """Resolve targets in character order, then update facing directions."""
        # infer target positions and target root positions
        position_to_nearby = None
        positions_to_avoid = [c.target_position for c in self.characters if c not in event.involved_characters]
        for character in event.involved_characters:
            # Prefer using target_position provided by the batch parser; hand-target should not cover the landing point
            if hasattr(character, "_batch_target_position") and character._batch_target_position is not None:
                character.target_position = character._batch_target_position
                delattr(character, "_batch_target_position")  # Clean up temporary properties
            else:
                character.target_position = self._infer_target_position(
                    character,
                    position_to_nearby,
                    positions_to_avoid,
                )
            position_to_nearby = character.target_position
            positions_to_avoid.append(character.target_position)

            if any("sit down" in action for action in character.target_actions) or any(
                "lie down" in action for action in character.target_actions
            ):
                character.target_root_position = self._calculate_target_root_position(character.target_position)
            else:
                character.target_position = self.scene.grid_map.get_closest_grid_position(
                    character.target_position,
                    exclude_occupied=False,
                    return_tensor=True,
                )

        # infer target facing directions
        for character in event.involved_characters:
            character.target_facing_direction = self._infer_target_direction(character)

        # The update of object attachment positions is now handled in the MotionGenerator

    def _infer_target_position(
        self,
        character: Character,
        position_to_nearby,
        positions_to_avoid,
    ):
        """Before this funciton is called, characters' relationships should be updated beforehand"""

        if self._is_positional_relationships_unchanged(character.previous_relationships, character.relationships):
            return character.position  # keep the current position

        target_position = self.position_sampler.sample_position(
            conditions=character.relationships,
            position_to_nearby=position_to_nearby,
            positions_to_avoid=positions_to_avoid,
        )
        # if sampled position is too close to the character,
        # we prefer not to move the character
        distance = torch.norm(target_position - character.position)
        if distance < self.minimum_distance:
            target_position = character.position
        return target_position

    def _is_positional_relationships_unchanged(self, prev_relationships, new_relationships):
        # check whether the positional relationships are same as the previous ones
        if not prev_relationships:
            return False

        prev_positional_relationships = set(
            relationship
            for relationship in prev_relationships
            if self.position_relationships_pattern.match(relationship)
        )
        new_positional_relationships = set(
            relationship
            for relationship in new_relationships
            if self.position_relationships_pattern.match(relationship)
        )
        return prev_positional_relationships == new_positional_relationships

    def _infer_target_direction(self, character: Character):
        """Before this function is called, all chracaters' target positions should be updated beforehand"""

        seated_direction = table_facing_direction(character, self.finder, self.device)
        if seated_direction is not None:
            return seated_direction

        target_direction = None
        # if 'look at' relationship is given, we set the target direction towards the anchor
        for relationship in character.relationships:
            if not relationship.startswith("look at"):
                continue

            anchor = relationship.split(" ")[-1]
            anchor = self.finder[anchor]
            if isinstance(anchor, Character):
                anchor_position = anchor.target_position
            else:
                anchor_position = torch.tensor(anchor.center, device=self.device).reshape(1, 3)
            anchor_position[:, 2] = 0.0
            target_direction = anchor_position - character.target_position
            target_direction /= torch.norm(target_direction, dim=-1, keepdim=True)
            break

        # in case of human-scene interaction, we set the anchor's orientation as the target direction
        # or adjust the target direction not to be too far from the anchor's orientation
        for relationship in character.relationships:
            if not (relationship.startswith("sit") or relationship.startswith("lie")):
                continue

            anchor = relationship.split(" ")[-1]
            anchor_orientation = torch.tensor(
                self.finder[anchor].orientation,
                dtype=torch.float32,
                device=self.device,
            ).reshape(1, 3)

            if target_direction is None:
                target_direction = anchor_orientation
            else:
                target_direction = misc.calculate_rotated_direction(
                    initial=anchor_orientation,
                    target=target_direction,
                    threshold_angle=torch.pi / 8,
                )
            break

        if target_direction is None:
            target_direction = character.target_facing_direction
        return target_direction

    def _calculate_hand_target_position(self, object_name: str, position_type: str = "center"):
        """
        Calculate hand target positions (new method)
        Args:
            object_name: Position reference object name or character name
            position_type: Position type ("center", "surface", "top", "side", "handle")
        Returns:
            torch.Tensor: Hand target position (1, 3)
        """
        # First, check if it’s a character name
        for character in self.characters:
            if character.name == object_name:
                if character.target_hand_position is not None:
                    self.logger.info(
                        f"Using {character.name}'s hand position as reference: {character.target_hand_position}"
                    )
                    return character.target_hand_position.clone()
                else:
                    # If the character has no hand position, use the character’s body position
                    self.logger.info(f"Using {character.name}'s body position as reference: {character.position}")
                    return character.position.clone()

        # If it’s not a character, then find the object
        target_object = None
        for obj in self.scene.objects:
            if f"{obj.label}_{obj.idx}" == object_name:
                target_object = obj
                break

        if target_object is None:
            self.logger.warning(f"Object {object_name} not found for hand target position")
            return None

        # Calculate hand position based on the position type
        hand_position = target_object.center.copy()

        if position_type == "surface":
            # For surfaces, the hand position is at the top of the object
            hand_position[2] = target_object.height
        elif position_type == "top":
            # For the top, the hand position is at the top of the object
            hand_position[2] = target_object.height
        elif position_type == "side":
            # For the sides, the hand position is on the side of the object (assuming on the right side)
            pass
        elif position_type == "handle":
            pass
        # position_type == "center" Use the center position of the object

        return torch.tensor(hand_position, dtype=torch.float32, device=self.device).reshape(1, 3)

    def _calculate_hand_target_position_from_area(self, area_conditions: list, position_type: str = "center"):
        """
        Calculate the hand target position based on the Area condition
        Args:
            area_conditions: List of conditions for the Area object, such as ["between table_3 and table_2"]
            position_type: Position type ("center", "surface", "top", "side", "handle")
        Returns:
            torch.Tensor: Hand target position (1, 3)
        """
        # Use position_sampler to sample positions, just like infer_target_position
        target_position = self.position_sampler.sample_position(
            conditions=area_conditions,
            position_to_nearby=None,
            positions_to_avoid=[],
        )

        # Adjust the height based on the position type
        if position_type == "surface":
            # For surfaces, use the default height
            target_position[0][2] = 1.1  # Assume the height of the table
        elif position_type == "top":
            # For the top, use a higher position
            target_position[0][2] = 1.1
        elif position_type == "side":
            # For the sides, use a medium height
            target_position[0][2] = 1.1
        elif position_type == "handle":
            # For the handle, use the handle height
            target_position[0][2] = 1.1
        # position_type == "center" Use the default center position

        self.logger.info(f"Calculated hand target position from area conditions: {target_position}")
        if isinstance(target_position, torch.Tensor):
            return target_position.detach().clone().to(dtype=torch.float32, device=self.device).reshape(1, 3)
        else:
            return torch.tensor(target_position, dtype=torch.float32, device=self.device).reshape(1, 3)

    def _calculate_target_root_position(self, target_position):
        ray_origin = target_position.clone().squeeze().cpu().numpy()
        ray_origin[2] = 2.0
        ray_direction = np.array([[0.0, 0.0, -1.0]])

        locations, _, _ = self.scene.scene_mesh.ray.intersects_location(
            ray_origin[None],
            ray_direction,
        )

        scale = 0.01
        while locations.size == 0:
            new_ray_origin = ray_origin + np.random.rand(3) * scale
            new_ray_origin[2] = 2.0
            locations, _, _ = self.scene.scene_mesh.ray.intersects_location(
                new_ray_origin[None],
                ray_direction,
            )
            scale += 0.01
        locations = locations[locations[:, 2].argmax()].reshape(1, 3)

        target_object: Object = None
        for obj in self.scene.objects:
            if obj.mesh.convex_hull.contains(locations):
                target_object = obj
                break
        self.logger.debug(f"Target Object for Sitting: {target_object}")

        height_offset = target_object.sit_offset_vertical if target_object is not None else self.root_height_offset
        if target_object is not None and target_object.label in {
            "chair",
            "armchair",
            "stool",
            "bench",
            "couch",
            "sofa",
            "bed",
        }:
            height_offset += 0.0
        target_root_position = locations + np.array([0, 0, height_offset])
        target_root_position = torch.tensor(target_root_position, device=self.device).reshape(1, 3)
        return target_root_position
