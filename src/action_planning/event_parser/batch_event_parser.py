"""
Batch Event Parser - Parsing multiple events passed in a single request.
Use grid system to locate ,  It’s also possible use_vision Configure additional scene view.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List, Dict, Any
import base64

import numpy as np

from src.action_planning.cached_llm_client import CachedLLMClient
from src.type import Character, Event, Object
from src.utils.llm_logging import write_llm_log
from .coordinates import EventCoordinateResolver, calculate_smart_hand_position, find_supporting_surface_height
from src.utils import misc

if TYPE_CHECKING:
    from src.action_planning.contracts import BatchParsedEvents, ParsedEvent


class BatchEventParser:
    """
    Batch event parser

    Differences compared to the traditional EventParser:
    1. Parsing the list of events passed in a single LLM request.
    2. Based on use_vision Configure whether the selection should be accompanied by a visual view.
    3. Converting grid coordinates or object names to world coordinates, without creating an Area object.
    4. Assigning sitting positions for shared seats; this does not guarantee that all positions are non-conflicting.
    """

    def __init__(
        self,
        characters: List[Character],
        objects: List[Object],
        scene_name: str,
        grid_map,
        cfg,
    ) -> None:
        # Checking whether caching is enabled
        enable_cache = getattr(cfg.event_parser, "enable_cache", False)
        cache_dir = getattr(cfg.event_parser, "cache_dir", "cache/event_parser")

        self.client = CachedLLMClient(
            provider=cfg.event_parser.provider,
            model=cfg.event_parser.model,
            enable_cache=enable_cache,
            cache_dir=cache_dir,
        )
        self.cfg = cfg
        self.characters = characters
        self.character_names = set(character.name for character in characters)
        self.objects = objects
        self.object_labels = set(f"{obj.label}_{obj.idx}" for obj in objects)
        self.scene_name = scene_name
        self.grid_map = grid_map

        self.model = cfg.event_parser.model
        self.temperature = cfg.event_parser.temperature
        self.use_json_mode = True  # Batch parsing with JSON mode enforced
        self.use_vision = getattr(cfg.event_parser, "use_vision", False)  # Whether to use the vision API

        self.topview_image_path = Path(f"./configs/scenes/{scene_name}/top_view_grid.png")
        if self.use_vision and not self.topview_image_path.is_file():
            from src.scene.topview import ensure_topview_grid

            ensure_topview_grid(self.topview_image_path)

        # Log configuration
        self.log_basedir = Path("./logs/event_parser_batch")
        self.log_id_prefix = None
        self.logger = misc.get_console_logger(f"{self.__class__.__name__}", level=cfg.logging_level)

        # Obtaining grid information from grid_map (compatible with GridMap and GridCoordinateConverter)
        if hasattr(grid_map, "grid_step"):
            # GridMap object (during actual runtime)
            self.grid_step = grid_map.grid_step
            self.grid_width = len(grid_map.grid_points)
            self.grid_height = len(grid_map.grid_points[0]) if self.grid_width > 0 else 0
            self.x_min, self.y_min = grid_map.bbox[0][:2]
            self.x_max, self.y_max = grid_map.bbox[1][:2]
        elif hasattr(grid_map, "grid_size"):
            # GridCoordinateConverter object (during testing)
            self.grid_step = grid_map.grid_size
            self.grid_width = grid_map.grid_width
            self.grid_height = grid_map.grid_height
            self.x_min, self.y_min = grid_map.bbox_min
            if hasattr(grid_map, "bbox_max") and grid_map.bbox_max:
                self.x_max, self.y_max = grid_map.bbox_max
            else:
                # Calculating bbox_max based on grid size
                self.x_max = self.x_min + self.grid_width * self.grid_step
                self.y_max = self.y_min + self.grid_height * self.grid_step
        else:
            raise ValueError(f"Unsupported grid_map type: {type(grid_map)}")

        # Optional: timeline dependency information injected by external (ActionPlanningModule)
        self.timeline_dependencies: List[dict] = []

        # Loading small item categories dynamically from objects.json Loading small item categories from
        self.small_object_labels = self._load_small_object_labels(scene_name)

    def _load_small_object_labels(self, scene_name: str) -> set:
        """
        Loading small item categories dynamically from objects.json Judgment criteria

        Object maximum side length:
        1. Items that can be held <= 0.5m (Excluding furniture-related keywords)
        2. Excludes furniture-type keywords (chair, table, desk, cabinet, shelf, etc.)

        Returns:
            Collection of small item categories (Such as {'cup', 'plate', 'bottle', 'laptop', ...})
        """
        objects_json_path = Path(f"./configs/scenes/{scene_name}/objects.json")

        # Basic small item list (fallback, including common interactable items)
        base_labels = {"cup", "plate", "bottle", "book", "glass", "bowl", "utensil", "laptop"}

        if not objects_json_path.exists():
            self.logger.warning(f"objects.json not found: {objects_json_path}, using base labels")
            return base_labels

        try:
            import json

            with open(objects_json_path, "r") as f:
                objects_data = json.load(f)

            # Excluded large furniture items/Equipment keywords
            exclude_keywords = {
                "chair",
                "table",
                "desk",
                "counter",
                "shelf",
                "cabinet",
                "couch",
                "sofa",
                "bed",
                "stool",
                "bench",
                "fridge",
                "dishwasher",
                "tv",
                "monitor",
                "plant",
                "rack",
                "board",
                "dispenser",
                "logo",
                "wall",
                "door",
                "window",
            }

            small_labels = set()

            for obj_data in objects_data:
                label = obj_data.get("label", "").lower()
                if not label:
                    continue

                # Objects excluding furniture keywords
                if any(keyword in label for keyword in exclude_keywords):
                    continue

                # Observers for object sizes
                aa_extents = obj_data.get("aa_extents", [])
                if len(aa_extents) >= 3:
                    max_size = max(aa_extents)
                    # Small items: maximum edge length <= 0.5m
                    if max_size <= 0.5:
                        small_labels.add(label)
                        self.logger.debug(f"Found small object: {label} (max_size={max_size:.3f}m)")

            # Merge basic list with dynamically discovered tags
            all_labels = base_labels | small_labels

            self.logger.info(f"Loaded {len(all_labels)} small object labels from {scene_name}: {sorted(all_labels)}")
            return all_labels

        except Exception as e:
            self.logger.error(f"Failed to load objects.json: {e}, using base labels")
            return base_labels

    def parse_events(
        self,
        events: List[Event],
        available_action_labels: List[str],
        scene_description: str = "",
        with_usage: bool = False,
    ) -> "BatchParsedEvents | tuple[BatchParsedEvents, int, int]":
        """
        Parse multiple events in batches

        Args:
            events: List of events to be parsed
            available_action_labels: List of available action tags
            scene_description: Scene description text
            with_usage: Whether an additional list of non-empty events is returned prompt/completion Number of tokens.

        Returns:
            {
                0: {  # event index
                    "character_name": {
                        "text": "walk to the table",
                        "target_action": ["walk"],
                        "target_grid": [10, 15],
                        "hand_target": "cup_1",
                        "relationships": ["interact with cup_1"],
                        "parser_source": "batch"
                    }
                },
                1: { ... }
            }

            The outer integer keys are indices of input events, not event_id. When the coordinate transformation is successful
            target_position will also be filled in; the returned results may not include all input events.
            with_usage=Returns True if events are non-empty
            (parsed_events, prompt_tokens, completion_tokens)。
            Always returns a result when events are empty {}, Includes with_usage=When True.
        """
        if not events:
            return {}

        self.logger.info(f"Batch parsing {len(events)} events...")

        # Calls LLM for batch parsing
        response, prompt_tokens, completion_tokens = self._query_llm(
            events,
            available_action_labels,
            scene_description,
        )

        # Parsing response
        parsed_events = self._parse_response(response, events)

        # Converts grid coordinates to world coordinates
        parsed_events = self._convert_grid_to_world_coords(parsed_events)

        # Generates relationships (for compatibility with existing systems)
        parsed_events = self._generate_relationships(parsed_events)

        # Marks the source of parsing
        for event_data in parsed_events.values():
            for char_data in event_data.values():
                char_data["parser_source"] = "batch"

        self.logger.info(f"Batch parsing completed for {len(parsed_events)} events")

        if with_usage:
            return parsed_events, prompt_tokens, completion_tokens
        return parsed_events

    def _query_llm(
        self,
        events: List[Event],
        available_action_labels: List[str],
        scene_description: str = "",
    ):
        """Calls LLM for batch parsing"""
        messages = self._create_messages(events, available_action_labels, scene_description)

        # Prepares API call parameters
        api_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        if self.use_json_mode:
            api_kwargs["response_format"] = {"type": "json_object"}

        response, prompt_tokens, completion_tokens = self.client.get_response(with_usage=True, **api_kwargs)

        self.log(messages, response)
        return response, prompt_tokens, completion_tokens

    def _create_messages(
        self,
        events: List[Event],
        available_action_labels: List[str],
        scene_description: str = "",
    ) -> List[Dict[str, Any]]:
        """Create LLM messages (supporting vision and plain text modes)"""
        messages = [
            {"role": "system", "content": self._create_system_message()},
        ]

        # If vision is enabled and the image exists, send the image
        if self.use_vision and self.topview_image_path.exists():
            user_content = [
                {"type": "text", "text": self._create_user_message(events, available_action_labels, scene_description)},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": self._encode_image_to_base64(self.topview_image_path),
                        "detail": "high",  # Use high-resolution mode for better grid recognition
                    },
                },
            ]
            messages.append({"role": "user", "content": user_content})
            self.logger.info("🖼️  Vision mode enabled: Sending top-view image to LLM")
        else:
            # Plain text mode
            messages.append(
                {
                    "role": "user",
                    "content": self._create_user_message(events, available_action_labels, scene_description),
                }
            )
            if self.use_vision:
                self.logger.warning(f"Vision enabled but image not found: {self.topview_image_path}")

        return messages

    def _encode_image_to_base64(self, image_path: Path) -> str:
        """Base64-encoded data URL for images"""
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/png;base64,{image_data}"

    def _create_system_message(self) -> str:
        """Create system messages"""
        if self.use_vision:
            analysis_instruction = "1. Analyze the scene layout using the provided top-view grid image, and select precise, non-overlapping grid coordinates [x, y] based on the image."
        else:
            analysis_instruction = (
                "1. Analyze the scene layout using the provided object coordinates and grid information"
            )

        return f"""You are a Batch Event Parser specialized in processing multiple events simultaneously for human motion generation.

Begin with a concise checklist (3-7 bullets) of the core sub-tasks before parsing; keep items at the conceptual level.

Your primary task is to efficiently parse multiple events at once, assigning target positions and actions to each character using the scene layout and provided event information while ensuring spatial and logical constraints are upheld.

Key Responsibilities:
{analysis_instruction}
2. Parse multiple events simultaneously to enable global position optimization
3. Assign grid coordinates for character target positions (prefer explicit [x, y] coordinates; avoid using object names as target positions except for seating like chairs/sofas/beds)
4. CRITICAL: DO NOT place characters INSIDE large objects - characters must stand in FREE SPACE beside/near furniture
5. Avoid position conflicts between characters
6. Consider event continuity and spatial relationships
7. Consider spatial continuity only when events share the same character or object dependency. Do not reference unrelated characters from previous events in explanations.
8. For interaction using hand, if which hand to use is specified in the activity description, use the specified hand.

Scene Information:
- Scene: {self.scene_name}
- Grid size: {self.grid_width} × {self.grid_height} cells
- Cell size: {self.grid_step}m (30cm per cell)

Input Specification:
Reasoning behind change:Added an explicit 'Input Specification' section (lines 22-32) listing structured input types (events, available_actions, objects, character_states, constraints) to close gaps and clarify required input fields.
The parser receives a structured JSON input containing:
- events: List of event objects, each with:
- event_id (string)
- description (string)
- character_name (string)
- dependencies (list of event_ids; include 'parallel' flag if required)
- available_actions: Array of valid action labels
- objects: Dictionary of object names/IDs mapped to properties including type, grid positions, and, if possible, a bounding box for free space calculation
- character_states: Optional dictionary for each character's prior state (e.g., previous target_grid, current posture, etc.)
- constraints: Any explicit scene or physical constraints
Parsing details (parallelism, order, dependencies) are governed by the 'dependencies' field. Events sharing a 'parallel' dependency are output under the same index (parallel group); otherwise, each event is indexed separately.
Before any significant event grouping or positioning, state the purpose and essential input(s) being referenced in one line.

Output Format:
You must output a JSON object with this structure:
{{
    "0": {{  // event index
        "character_name": {{
            "event_id": "e1",  // REQUIRED: event ID from the input (e.g., "e1", "e2")
            "text": "walk",  // natural language description using action words from available actions
            "target_action": "walk",  // single action from available actions
            "target_grid": [10, 15],  // grid coordinates [x, y] OR object name "table_1"
            "hand_target": "cup_1"  // (optional) object name for hand interaction
            "hand_target_position": {{"hand_target_grid": [x, y], "position_type": object or "air", "height": "surface" or z(height in meters)}} // (optional) precise hand placement
            "explanation": "Explain the spatial relations, action assignments, and dependency logic behind each parser decision to facilitate traceability and analysis."
        }}
    }},
    "1": {{ ... }}
}}

Rules:
1. target_grid can be EITHER:
   - Grid coordinates [x, y] for general areas/positions
   - Object name "object_id" (e.g., "chair_9", "table_1") - system will find the exact position
        CRITICAL: When using grid coordinates [x, y], DO NOT place characters INSIDE large objects (tables, counters, desks, sofas, beds, etc.). Characters must stand BESIDE/NEAR these objects, not ON them. Use the top-view image to identify free space around furniture.
2. hand_target is for objects requiring hand interaction:
    - Use for: picking up, putting down, opening, touching, etc.
    - Specify object name (e.g., "cup_1", "bottle_2")
    - Omit if no hand interaction needed
3. Optional: hand_target_position for precise hand placement. Use ONE of:
    - {{"object": "table_1", "position_type": "surface"}}
    - {{"area": ["between table_1 and table_2"], "position_type": "center"}}
    - Air/Specific height: {{"object": "cup_1", "position_type": "air", "height": 1.25}} 
    Notes: Use "surface" for tops/shelves, "handle" for handles, "air" for mid-air interactions with specified height. Omit or set null if not applicable.
4. For sitting/lying on furniture: use object name in target_grid (e.g., "chair_9")
5. Avoid assigning overlapping positions to different characters
6. Consider the logical flow between consecutive events
7. Use only actions from the provided available action labels
8. If the event's activity describes an action not in available action labels, select the semantically closest available action to simulate it:
    - Apply the conversion to BOTH "text" and "target_action" fields
    - Examples: "hand over" → "put down", "receive" → "pick up", "give" → "put down"
    - The "text" field should use natural language but with the converted action word
9. Only events explicitly marked with 'parallel' in the dependency list should share the same event index. All other events, even if independent, must be output under separate indices.
10. For parallel handover/receive actions (object transfer between characters):
    - IMPORTANT: The object is a dynamic object (in someone's hand), so CANNOT use hand_target
    - Instead, define a FIXED handover point in air using hand_target_position for BOTH characters:
        {{"hand_target_grid": [gx, gy], "position_type": "air", "height": 1.25}}
    - Both characters move their hands to this pre-agreed fixed point to complete the transfer
    - CRITICAL: Position the two characters' target_grid on OPPOSITE SIDES of the handover point to prevent overlap
        Example: If handover point is [24, 14], place one character at [25, 13] and the other at [23, 15]
    Notes:
        - hand_target_grid should be the midpoint or a safe nearby cell between two characters' target_grid
        - Ensure sufficient distance (at least 2-3 grids) between characters' target_grid positions
        - Use a reasonable height in [1.2, 1.4] meters for air handover
        - hand_target can be omitted (object is dynamic, moving with the giver)
11. For every event, always add an "explanation" field immediately, briefly describing why the action, position, and object have been assigned as such (e.g., 'Standing next to the table to ensure unobstructed pickup', 'Assigned pick up to match preceding put down', etc.).

- For missing/unknown events, characters, action labels, or objects, return an error object specifying the missing or invalid field.
- For out-of-bounds grid coordinates or non-free target locations (placing inside forbidden objects), return a specific event error.
- Omitting optional fields is allowed (unless the action requires them).
- If ambiguity or conflict is detected, return a detailed error for that instance and halt further parsing.

Examples:

Example 1 - Picking up objects (hand_target required):
Event: "[Alice] pick up the bottle from the table (event_id=e1)"
Output:
{{
    "0": {{
        "Alice": {{
            "event_id": "e1",
            "text": "pick up the bottle",
            "target_action": "pick up",
            "target_grid": [5, 20],  // use grid coordinates for a place to stand while picking up
            "hand_target": "bottle_1",  // object to interact with
            "explanation": "Alice wants to use bottle_1, so she picks it up from beside the table, avoiding standing directly on top of it."
        }}
    }}
}}

Explanation:
- Use hand_target to indicate the small object to grasp ("bottle_1").
- target_grid is where the character stands to pick up; hand_target_position omitted → system infers center height of the object.
- IMPORTANT: target_grid [5, 20] must be in FREE SPACE (not inside a table/counter). Character stands BESIDE the table, not ON it.

Example 2 - Sitting on furniture (target_grid as object name):
Event: "[Bob] sit down on chair_9 (event_id=e2)"
Output:
{{
    "0": {{
        "Bob": {{
            "event_id": "e2",
            "text": "sit down on chair",  // only use natual language in text field
            "target_action": "sit down",
            "target_grid": "chair_9",  // use object name for exact seat position
            // no hand_target needed for sitting
            "explanation": "Bob needs to sit for his next activity, so he sits on chair_9, using the furniture anchor."
        }}
    }}
}}

Explanation:
- For seating actions, target_grid uses the furniture object name ("chair_9").
- No hand_target needed; the system will compute a sitting anchor near the chair.

Example 3 - Walking to area (target_grid as coordinates):
Event: "[Carol] walk to the kitchen area (event_id=e3)"
Output:
{{
    "0": {{
        "Carol": {{
            "event_id": "e3",
            "text": "walk",
            "target_action": "walk",
            "target_grid": [5, 20],  // use grid coordinates for general areas
            // no hand_target needed for walking
            "explanation": "Carol needs to move to the kitchen to prepare for the next event, so she walks to the free area at [5, 20]."
        }}
    }}
}}

Explanation:
- Pure locomotion: provide [x, y] grid coordinates for areas.
- No hand interaction required.

Example 4 - Put down on surface (LLM specifies surface hand position):
Event: "[Alice] put down the bottle on the table using left hand (event_id=e4)"
Output:
{{
    "0": {{
        "Alice": {{
            "event_id": "e4",
            "text": "put down the bottle using left hand",
            "target_action": "put down",
            "target_grid": [8, 12],
            "hand_target": "bottle_1",
            "hand_target_position": {{"hand_target_grid": [24, 14], "object": "table_1", "position_type": "surface"}},
            "explanation": "Alice needs to release bottle_1 so it can be taken later, so she puts it down on a specific position on thetable_1, specifying the surface."
        }}
    }}
}}

Explanation:
- Placing requires a supporting surface: specify hand_target_position with {{object:"table_1", position_type:"surface"}}.
- The system will use the table surface height for hand position Z.

Example 5 - Air interaction (LLM specifies explicit height):
Event: "[Carol] raise the cup to show it in the air using right hand (event_id=e6)"
Output:
{{
    "0": {{
        "Carol": {{
            "event_id": "e5",
            "text": "raise the cup in the air using right hand",
            "target_action": "raise",
            "target_grid": [9, 10],
            "hand_target": "cup_1",
            "hand_target_position": {{"object": "cup_1", "position_type": "air", "height": 1.25}},
            "explanation": "Carol wants to show cup_1 to others, so she raises it at 1.25 meters above the ground for visibility."
        }}
    }}
}}

- Use position_type:"air" with an explicit height (meters) when interaction happens mid-air.
- The system combines existing XY (target position or object center) with the provided Z height.

Example 6 - Parallel handover (fixed handover point for dynamic object):
Event (ParallelGroup): "[Bob] hand over the bottle near platform_2 (event_id=e3)" and "[Jack] receive the bottle near platform_2 (event_id=e4)"
Output:
{{
    "0": {{
        "Bob": {{
            "event_id": "e3",
            "text": "put down the bottle using left hand",
            "target_action": "put down",
            "target_grid": [25, 13],
            "hand_target_position": {{"hand_target_grid": [24, 14], "position_type": "air", "height": 1.25}},
            "explanation": "Bob passes the bottle by placing it at a shared air position for Jack to pick up; he stands close to the handover spot to coordinate perfectly."
        }},
        "Jack": {{
            "event_id": "e4",
            "text": "pick up the bottle using right hand",
            "target_action": "pick up",
            "target_grid": [23, 15],
            "hand_target_position": {{"hand_target_grid": [24, 14], "position_type": "air", "height": 1.25}},
            "explanation": "Jack receives the bottle from Bob by synchronizing to the same air space and stands opposite to avoid overlapping positions."
        }}
    }}
}}

Explanation:
- IMPORTANT: "hand over" is not in available actions, so use "put down" for BOTH target_action AND text field
- Similarly, "receive" → "pick up" for BOTH fields
- The bottle is a dynamic object (in Bob's hand), so we define a FIXED handover point using hand_target_position
- Both characters move their hands to grid [24, 14] at height 1.25m to complete the transfer
- hand_target is omitted because the object position is dynamic and cannot be referenced directly
- CRITICAL: target_grid positions MUST be SEPARATED to avoid character overlap! Position characters on OPPOSITE SIDES of the handover point (e.g., Bob at [25, 13] and Jack at [23, 15] are on different sides of the platform/handover point). Never place both characters at the same or adjacent grids.
- the target_grid of characters MUST be OUTSIDE of the platform so they can walk to the target_grid to reach the hand_target_grid
- But the target_grid can be overlapped with chair if the character is sitting on the chair.


Example 7 - Pick up or put down objects while sitting on a chair:
Event: "[Bob] pick up the bottle from the table (event_id=e5)"
Output:
{{
    "0": {{
        "Bob": {{
            "event_id": "e5",
            "text": "pick up the bottle",
            "target_action": "pick up",
            "target_grid": "chair_9",
            "hand_target": "bottle_1",
            "hand_target_position": {{
                "hand_target_grid": [24, 14], 
                "object": "table_4", 
                "position_type": "surface"
            }},
            "explanation": "Bob stays seated on chair_9 and reaches to the tabletop position [24, 14] on table_4 to pick up bottle_1 resting there. The hand_target_grid defines a precise point on the table surface."
        }}
    }}
}}
Explanation:
When a character performs a hand interaction (e.g., pick up, put down) while already sitting on a chair, the target_grid remains the current chair object (e.g., "chair_9") instead of generating a new standing position.
The hand reaches toward a precise tabletop coordinate defined by hand_target_grid, which lies on the surface of the corresponding table (position_type: "surface").
This configuration ensures that seated characters maintain realistic reachability without unnecessary repositioning.
The system interprets hand_target_grid as a contact point on the furniture surface, allowing accurate hand placement during seated manipulation.
This rule applies whenever the character's previous event places them in a seated position (sit down or maintain sit) and the next action involves interaction with nearby objects on tables, counters, or desks.

"errors": [
Reasoning behind change:Added canonical error examples in the output schema (lines 114-119), showing how to specify the relevant event_id and messages for unknown actions/invalid coordinates, providing actionable error feedback.
{{"event_id": "e3", "message": "Unknown action label: 'fly' specified but not in available_actions."}},
{{"event_id": "e5", "message": "Grid position [35, -1] is out of bounds (grid size: 30 x 20)."}}
]
- The "errors" array must always be present (empty if no errors).
- For each error, specify the relevant event_id and a clear, actionable message.
- Optional fields ("hand_target", "hand_target_position") can be omitted or set null if not needed.
- Parsing may continue after non-fatal errors, but must halt if ambiguity or critical conflicts arise.

"""

    def _create_user_message(
        self,
        events: List[Event],
        available_action_labels: List[str],
        scene_description: str = "",
    ) -> str:
        """Create user messages"""
        message_parts = []

        # Scene description (text)
        if scene_description:
            message_parts.append(f"[Scene Description]\n{scene_description}\n")

        # Scene overview (only available in vision mode)
        if self.use_vision and self.topview_image_path.exists():
            message_parts.append(
                f"[Scene Top View with Grid]\nPlease analyze the attached top-view image showing the scene layout with a {self.grid_step}m grid overlay.\n"
            )

        # Character list (excluding positions and states that change dynamically, to maintain cache stability)
        message_parts.append("[Characters]")
        # Sort by name to ensure a stable order
        for char in sorted(self.characters, key=lambda c: c.name):
            message_parts.append(f"- {char.name}")
        message_parts.append("")

        # Small objects in the scene (can be directly referenced)
        # Note: Use initial positions and sort to maintain cache stability
        message_parts.append("[Small Objects (use object name directly)]")
        small_objects = []
        for obj in self.objects:
            if obj.label.lower() in self.small_object_labels:
                grid_x = int((obj.center[0] - self.x_min) / self.grid_step)
                grid_y = int((obj.center[1] - self.y_min) / self.grid_step)
                small_objects.append((f"{obj.label}_{obj.idx}", grid_x, grid_y))
        # Sort by name to ensure a stable order
        small_objects.sort(key=lambda x: x[0])
        for name, gx, gy in small_objects:
            message_parts.append(f"- {name} at grid [{gx}, {gy}]")
        message_parts.append("")

        # No longer list large objects to prevent the LLM from relying too heavily on them
        # Large objects/Furniture (reference positions)
        message_parts.append("[Large Objects/Furniture (for reference)]")
        for obj in self.objects:
            if not any(label in obj.label.lower() for label in self.small_object_labels):
                grid_x = int((obj.center[0] - self.x_min) / self.grid_step)
                grid_y = int((obj.center[1] - self.y_min) / self.grid_step)
                message_parts.append(f"- {obj.label}_{obj.idx} around grid [{grid_x}, {grid_y}]")
        message_parts.append("")

        # Events to be parsed (with event_id, in order following the narrator’s dependency topology)
        # Standardized representation: ParallelGroup expands members to ensure stable caching
        message_parts.append("[Events to Parse]")
        event_idx = 0
        for item in events:
            # Check if it’s a ParallelEventGroup
            if hasattr(item, "events") and hasattr(item, "parallel_group"):
                # ParallelGroup: Expand member events and annotate them
                for ev in item.events:
                    eid = None
                    if hasattr(ev, "timeline_data") and ev.timeline_data:
                        eid = ev.timeline_data.get("event_id")
                    eid_str = f" (event_id={eid}, parallel)" if eid else " (parallel)"
                    message_parts.append(f"{event_idx}. {ev}{eid_str}")
                    event_idx += 1
            else:
                # Ordinary Event
                eid = None
                if hasattr(item, "timeline_data") and item.timeline_data:
                    eid = item.timeline_data.get("event_id")
                eid_str = f" (event_id={eid})" if eid else ""
                message_parts.append(f"{event_idx}. {item}{eid_str}")
                event_idx += 1
        message_parts.append("")

        # Dependency information is passed to the LLM
        if getattr(self, "timeline_dependencies", None):
            message_parts.append("[Dependencies]")
            # Presented in JSON format
            try:
                deps_str = json.dumps(self.timeline_dependencies, ensure_ascii=False)
            except Exception:
                deps_str = str(self.timeline_dependencies)
            message_parts.append(deps_str)
            message_parts.append("")

        # Available actions
        message_parts.append(f"[Available Action Labels]\n{available_action_labels}\n")

        if self.use_vision:
            message_parts.append(
                "Please use the attached top-view image to determine precise [x, y] target_grid coordinates."
            )
            message_parts.append(
                "Rules: target_grid must be [x, y] for areas (e.g., bar_counter, table edges, counters). Only use object name for seating (e.g., 'chair_9')."
            )
            message_parts.append(
                "CRITICAL: Identify FREE SPACE in the image - avoid placing characters inside large furniture/objects (dark areas in the image). Characters should be in open/walkable areas (lighter areas)."
            )
        message_parts.append(
            "Please parse all events and return the results in JSON format, respecting the dependency order and grouping parallel events under the same event index."
        )

        return "\n".join(message_parts)

    def _parse_response(self, response: str, events: List[Event]) -> "BatchParsedEvents":
        """Extract JSON, filter event indices and character names, and add default fields.

        Field values are only locally normalized; no full type checking performed; return an exception if an issue occurs {}.
        """
        try:
            # Attempt to extract JSON
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()

            parsed_data = json.loads(json_str)

            # Verify and normalize data
            result = {}
            for event_idx_str, event_data in parsed_data.items():
                # Skip non-digit keys (e.g., ) "errors", "notes" Wait for)
                if not event_idx_str.isdigit():
                    continue

                event_idx = int(event_idx_str)
                if event_idx >= len(events):
                    continue

                result[event_idx] = {}
                for char_name, char_data in event_data.items():
                    if char_name not in self.character_names:
                        continue

                    # Normalize data format
                    normalized = {
                        "text": char_data.get("text", ""),
                        "target_action": [char_data.get("target_action", "walk")],  # Convert to list
                        "target_grid": char_data.get("target_grid", [0, 0]),
                    }

                    # Process event_id (if provided by LLM)
                    if "event_id" in char_data:
                        normalized["event_id"] = char_data["event_id"]

                    # Process hand_target (optional)
                    if "hand_target" in char_data:
                        normalized["hand_target"] = char_data["hand_target"]

                    # Process hand_target_position (optional)
                    if "hand_target_position" in char_data:
                        normalized["hand_target_position"] = char_data["hand_target_position"]

                    # Process attachment_control (optional)
                    if "attachment_control" in char_data:
                        normalized["attachment_control"] = char_data["attachment_control"]

                    # Process relationships (optional)
                    if "relationships" in char_data:
                        normalized["relationships"] = char_data["relationships"]
                    else:
                        normalized["relationships"] = []

                    result[event_idx][char_name] = normalized

            return result

        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            self.logger.error(f"Response: {response}")
            # Return empty result
            return {}

    def _convert_grid_to_world_coords(self, parsed_events: "BatchParsedEvents") -> "BatchParsedEvents":
        """Enrich the same plan mapping with event/world coordinates (meters, Z up)."""
        bbox_min = (self.x_min, self.y_min)
        resolver = EventCoordinateResolver(self.objects, bbox_min, self.grid_step, self.logger)
        return resolver.convert(parsed_events, self._calculate_smart_hand_position)

    def _calculate_smart_hand_position(self, target_obj: Object, action) -> np.ndarray:
        return calculate_smart_hand_position(target_obj, action, self._find_supporting_surface_height)

    def _find_supporting_surface_height(self, target_obj: Object) -> float:
        return find_supporting_surface_height(self.objects, target_obj)

    def _generate_relationships(self, parsed_events: "BatchParsedEvents") -> "BatchParsedEvents":
        """
        Replace relationships field in-place for compatibility with existing systems.
        Write interact relationships if hand_target is a string; otherwise, write an empty list.
        The relationships originally provided by LLM will not be retained.
        """
        for event_data in parsed_events.values():
            for char_data in event_data.values():
                relationships = []

                # If there is a hand_target, add an interact relationship
                if "hand_target" in char_data:
                    target = char_data["hand_target"]
                    if isinstance(target, str):  # Item names
                        relationships.append(f"interact with {target}")

                char_data["relationships"] = relationships

        return parsed_events

    def parse_event(
        self,
        event: Event,
        scene_description,
        available_action_labels,
        previous_events,
        feedback=None,
        with_usage=False,
        **kwargs,
    ) -> "ParsedEvent | tuple[ParsedEvent, int, int]":
        """
        Single event parsing interface (adapter)

        To be compatible with the existing ActionPlanningModule, a single event parsing interface is provided.
        In practice, batch parsing is still used, but only one event is parsed.

        Args:
            event: Events to be parsed
            scene_description: Scene description
            available_action_labels: Available Action Tabs
            previous_events: Previous events (not used)
            feedback: Feedback (not used)
            with_usage: Whether to return token usage information

        Returns:
            Parsing results with the character name as the key. with_usage=Returns true when True
            (result, prompt_tokens, completion_tokens)。
            When the batch results lack index 0, a default walk plan is constructed for the character in the event.
        """
        self.logger.info(f"Parsing single event (via batch adapter): {event}")

        # Call batch parsing
        if with_usage:
            batch_results, prompt_tokens, completion_tokens = self.parse_events(
                [event], available_action_labels, scene_description, with_usage=True
            )
        else:
            batch_results = self.parse_events([event], available_action_labels, scene_description, with_usage=False)

        # Extract results of individual events
        if 0 in batch_results:
            result = batch_results[0]
        else:
            # Return empty result
            result = {}
            for char in event.involved_characters:
                result[char.name] = {
                    "text": "",
                    "target_action": ["walk"],
                    "target_position": [0.0, 0.0, 0.0],
                    "relationships": [],
                    "parser_source": "batch",
                }

        if with_usage:
            return result, prompt_tokens, completion_tokens
        return result

    def log(self, messages, response, log_id=None):
        """Record logs"""
        if log_id is None:
            log_id = datetime.now().strftime("%y%m%d-%H%M%S")
        if self.log_id_prefix:
            log_id = f"{self.log_id_prefix}-{log_id}"
        log_dir = self.log_basedir
        log_file = log_dir / f"{log_id}.txt"
        write_llm_log(log_file, messages, response, layout="batch")
