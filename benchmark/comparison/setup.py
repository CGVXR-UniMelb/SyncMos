from pathlib import Path
from typing import List

from src.action_planning.narrator import get_narrator
from src.action_planning.scene_describer import get_scene_describer
from src.scene.scene import Scene
from src.type import Character, Event


def setup_scene(runner, scene_name: str):
    runner.scene = Scene(scene_name, runner.cfg)
    runner.scene_name = runner.scene.scene_name
    runner.scene_mesh = runner.scene.scene_mesh
    runner.objects = runner.scene.objects
    runner.grid_map = runner.scene.grid_map
    runner.scene_graph = runner.scene.scene_graph
    runner.object_finder = {f"{obj.label}_{obj.idx}": obj for obj in runner.objects}


def get_max_distraction_level(test_case: dict) -> int:
    max_distraction_level = 0
    for plan in test_case.get("planning_history", []):
        if plan.get("type") == "distractor":
            max_distraction_level = max(max_distraction_level, plan.get("distraction_level", 0))
    for character in test_case.get("current_states", []):
        if character.get("type") == "distractor":
            max_distraction_level = max(max_distraction_level, character.get("distraction_level", 0))
    return max_distraction_level


def setup_modules_and_scenario(runner, test_case, distraction_level: int, repeat_idx: int, narrator_type: str | None):
    runner.current_test_case = test_case
    runner.user_instruction = test_case.get("user_instruction")
    runner.available_action_labels = test_case.get("action_labels")

    # Characters from planning_history/current_states/available_characters/characters
    # Collect all character names (Priority: characters) > current_states > planning_history > available_characters)
    all_character_names = set()

    # Prefer to read from the characters field if available)
    if "characters" in test_case:
        for char_name in test_case.get("characters", []):
            all_character_names.add(char_name)

    # Collect from current_states
    for c in test_case.get("current_states", []):
        if c.get("type") == "distractor" and c.get("distraction_level", 0) > distraction_level:
            continue
        all_character_names.add(c.get("name"))

    # Collect from planning_history
    for plan in test_case.get("planning_history", []):
        if plan.get("type") == "distractor" and plan.get("distraction_level", 0) > distraction_level:
            continue
        for char_name in plan.get("characters", []):
            all_character_names.add(char_name)

    # Collect from available_characters
    for char_name in test_case.get("available_characters", []):
        all_character_names.add(char_name)

    # Create a list of Character objects
    runner.characters: List[Character] = []
    char_dict = {}  # name -> Character objects for subsequent lookup

    # Create from current_states first if available)
    for c in test_case.get("current_states", []):
        if c.get("type") == "distractor" and c.get("distraction_level", 0) > distraction_level:
            continue
        char_name = c.get("name")
        if char_name in all_character_names:
            char_obj = Character(
                name=char_name,
                state=c.get("state", "idle"),
                current_actions=[c.get("current_action", "idle")],
                relationships=c.get("position", []) + c.get("orientation", []),
            )
            runner.characters.append(char_obj)
            char_dict[char_name] = char_obj

    # Create default Character objects for other characters (state)="idle", event=None)
    for char_name in all_character_names:
        if char_name not in char_dict:
            char_obj = Character(
                name=char_name,
                state="idle",
                current_actions=["idle"],
                relationships=[],
            )
            runner.characters.append(char_obj)
            char_dict[char_name] = char_obj

    # Scene description
    runner.scene_describer = get_scene_describer(runner.cfg)
    if runner.scene_description_type in ["narrative", "narrative_vlm"]:
        runner.scene_description = runner.scene_describer.get_pregenerated_scene_description(
            runner.scene_name, idx=repeat_idx
        )
    elif runner.scene_description_type == "narrative_timeline":
        # Let SceneDescriberTimeline handle whether to use pre-generated descriptions
        runner.scene_description = runner.scene_describer.get_scene_description(runner.scene_name, runner.scene_graph)
    else:
        runner.scene_description = runner.scene_describer.get_scene_description(runner.scene_name, runner.scene_graph)

    # Narrator selection (timeline or baseline)
    # Use temp cfg merge to avoid mutating global cfg
    from omegaconf import OmegaConf

    # Convert to a regular dictionary, modify it, and then create OmegaConf
    narrator_cfg_dict = OmegaConf.to_container(runner.cfg.narrator, resolve=True)

    # Debugging logs
    import logging

    logger = logging.getLogger(__name__)
    logger.info(
        f"[DEBUG] Before modification - narrator_type param: {narrator_type}, cfg.narrator.type: {narrator_cfg_dict.get('type')}"
    )

    if narrator_type is not None:
        narrator_cfg_dict["type"] = narrator_type
    else:
        narrator_cfg_dict.pop("type", None)

    logger.info(f"[DEBUG] After modification - narrator_cfg_dict['type']: {narrator_cfg_dict.get('type')}")

    # Create a temporary configuration, disable the struct mode to allow dynamic addition of fields
    temp_cfg = OmegaConf.create({"narrator": narrator_cfg_dict})
    OmegaConf.set_struct(temp_cfg, False)  # Disable the struct mode

    # Also disable the struct mode of the original configuration before merging (temporary)
    original_struct = OmegaConf.is_struct(runner.cfg)
    OmegaConf.set_struct(runner.cfg, False)
    temp_cfg = OmegaConf.merge(runner.cfg, temp_cfg)
    OmegaConf.set_struct(runner.cfg, original_struct)  # Restore the struct mode of the original configuration

    logger.info(f"[DEBUG] After merge - temp_cfg.narrator.type: {temp_cfg.narrator.get('type')}")
    logger.info(
        f"[DEBUG] hasattr check: {hasattr(temp_cfg.narrator, 'type')}, value: {getattr(temp_cfg.narrator, 'type', 'NOT_FOUND')}"
    )

    runner.narrator = get_narrator(runner.characters, runner.objects, temp_cfg)

    # Debug log: Confirm the created narrator type
    logger.info(f"[DEBUG] Created narrator class: {runner.narrator.__class__.__name__}")

    # Planning history to previous_events (Set this first, as it will affect… character.event)
    for plan in test_case.get("planning_history", []):
        if plan.get("type") == "distractor" and plan.get("distraction_level", 0) > distraction_level:
            continue
        event_ = Event(
            involved_characters=[c for c in runner.characters if c.name in plan.get("characters", [])],
            activity=plan.get("activity"),
            state=plan.get("state", "completed"),
        )
        for c in runner.characters:
            if c.name not in plan.get("characters", []):
                continue
            if plan.get("state") == "completed":
                c.event = None
            else:
                c.event = event_

        parsed_event_ = {}
        for parsed in plan.get("details", []):
            parsed_event_[parsed.get("character")] = {}
            parsed_event_[parsed.get("character")]["relationships"] = (
                parsed.get("position", []).copy() + parsed.get("orientation", []).copy()
            )
        event_.parsed_event = parsed_event_
        runner.narrator.previous_events.append(event_)

    # Available characters: Override the character state based on available_characters in test_case
    # Note: This needs to be set after planning_history, as we want to override the state of certain characters
    if "available_characters" in test_case:
        available_names = test_case.get("available_characters", [])
        # Ensure the state of available characters is correct (override the settings from planning_history)
        for c in runner.characters:
            if c.name in available_names:
                # If in available_characters, ensure that the event is None and the state is idle
                c.event = None
                if c.state != "idle":
                    c.state = "idle"
        # Convert to a list of Character objects, sorted by name
        runner.narrator.available_characters = sorted(
            [c for c in runner.characters if c.name in available_names], key=lambda x: x.name
        )
    else:
        # Default: Sort all idle characters with event None by name
        runner.narrator.available_characters = sorted(
            [c for c in runner.characters if c.state == "idle" and c.event is None], key=lambda x: x.name
        )

    # Allow null plan guidance
    if test_case.get("allow_null_plan"):
        runner.narrator.additional_guidelines += (
            "- You can skip the event generation by outputting an empty event that involves no characters in it."
        )
        runner.narrator.additional_guidelines += (
            " This is useful when you need to wait for some characters to become idle before the next event."
        )
        if getattr(runner, "planning_mode", "event").startswith("no_event"):
            runner.narrator.additional_guidelines = runner.narrator.additional_guidelines.replace("event", "plan")

    runner.narrator.log_id_prefix = f"repeat-{repeat_idx}"
    narrator_type_str = narrator_type if narrator_type is not None else "baseline"

    # If it’s a derived case, include the derived case identifier in the path
    test_case_id = test_case.get("id", "unknown")
    if test_case.get("_derived_file"):
        # Derived Use Cases: Using file names (without extensions) as subdirectories
        derived_name = Path(test_case["_derived_file"]).stem
        runner.narrator.log_basedir = (
            runner.log_base_dir
            / "narrator"
            / narrator_type_str
            / f"testcase-{test_case_id}"
            / derived_name
            / f"distraction-{distraction_level}"
        )
    else:
        # Basic Use Cases: Normal paths
        runner.narrator.log_basedir = (
            runner.log_base_dir
            / "narrator"
            / narrator_type_str
            / f"testcase-{test_case_id}"
            / f"distraction-{distraction_level}"
        )
