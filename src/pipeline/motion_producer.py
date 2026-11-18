from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from src.action_planning.action_planning_module import ActionPlanningModule
from src.motion_synthesis.motion_synthesis_module import MotionSynthesisModule
from src.motion_synthesis.states import State
from src.pipeline.motion_source import MotionFrame, MotionMetadata
from src.scene.scene import Scene
from src.type import Character
from src.utils import misc
from src.utils.grid_utils import grid_to_world
from src.utils.motion_io import save_motion_data

DEFAULT_CHARACTER_NAMES = [
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

DEFAULT_STORY = """
Amir picks up a bottle near table_2.
Then Amir hands over the bottle to Benjamin at grid [12, 17].
After that, Amir goes to grid [20, 25].
Benjamin hands over the bottle to Charles at grid [4, 16].
"""

DEFAULT_INITIAL_GRID_POSITIONS = {
    "Amir": [17, 27],
    "Benjamin": [9, 15],
    "Charles": [7, 19],
}


@dataclass
class MotionSequence:
    metadata: MotionMetadata
    frames: list[MotionFrame]
    characters: list[Character]


class MotionProducer:
    """Generates motion body parameters without owning any frontend rendering."""

    def __init__(
        self,
        cfg,
        user_instruction: Optional[str] = None,
        character_initial_grid_positions: Optional[dict[str, list[int]]] = None,
        character_names: Optional[list[str]] = None,
        logger=None,
    ):
        self.cfg = cfg
        self.device = cfg.device
        self.logger = logger or misc.get_console_logger("MotionProducer", cfg.logging_level)
        self.scene_name = cfg.scene_name
        self.num_characters = int(cfg.num_characters)
        self.character_names = character_names or DEFAULT_CHARACTER_NAMES[: self.num_characters]
        self.user_instruction = (user_instruction or DEFAULT_STORY).strip()
        self.character_initial_grid_positions = (
            character_initial_grid_positions
            if character_initial_grid_positions is not None
            else DEFAULT_INITIAL_GRID_POSITIONS
        )

        self.scene: Optional[Scene] = None
        self.characters: list[Character] = []
        self.motion_synthesizer: Optional[MotionSynthesisModule] = None
        self.action_planner: Optional[ActionPlanningModule] = None
        self.fId = 0
        self.max_fId = 0
        self.initialized = False
        self.finished = False
        self.error: Optional[str] = None
        self.saved = False

    def initialize(self) -> MotionMetadata:
        self.scene = Scene(self.scene_name, self.cfg)
        self.characters = [Character(name) for name in self.character_names]
        self.motion_synthesizer = MotionSynthesisModule(self.scene, self.characters, self.cfg)
        self.action_planner = ActionPlanningModule(
            self.scene, self.characters, self.motion_synthesizer.state_manager, self.cfg
        )

        for character in self.characters:
            character.state = self.motion_synthesizer.state_manager.idle_state
            character.body_params = self.motion_synthesizer.create_initial_body_params(
                position=self._initial_position(character),
                facing_direction=np.random.rand(2),
                character=character,
            )
            character.current_actions = []
            character.target_position = character.position
            character.target_root_position = character.position
            character.target_facing_direction = character.facing_direction
            character.future_positions = character.position.tile(3, 1)
            character.future_facing_directions = character.facing_direction.tile(3, 1)

        self.max_fId = len(self.characters[0].body_params) - 1
        self.fId = len(self.characters[0].body_params)
        self.motion_synthesizer.state_manager.fId = self.fId
        self.initialized = True
        return self.get_metadata()

    def step(self) -> Optional[MotionFrame]:
        if not self.initialized:
            self.initialize()
        if self.finished or self.error:
            return None

        try:
            if self._is_finished():
                self.finished = True
                return None

            if self.action_planner.is_new_event_required():
                self.action_planner.generate_new_event(self.user_instruction)

            if not self.action_planner.event_queue.empty():
                event, parsed_event = self.action_planner.event_queue.get()
                event.created_fId = self.fId
                self.action_planner.allocate_new_plan(event, parsed_event)

            if hasattr(self.motion_synthesizer, "advance_motion_frames"):
                self.motion_synthesizer.advance_motion_frames(self.characters)
            else:
                for character in self.characters:
                    self.motion_synthesizer.advance_motion_frame(character)

            frame_id = self.fId
            self.fId += 1
            self.max_fId = self.fId - 1
            self.motion_synthesizer.state_manager.fId = self.fId
            return self._motion_frame(frame_id)
        except Exception as exc:
            self.error = str(exc)
            raise

    def run_until_finished(self, max_frames: Optional[int] = None) -> MotionSequence:
        if not self.initialized:
            self.initialize()
        frames = []
        limit = max_frames if max_frames is not None else getattr(self.cfg, "max_frames", None)
        while not self.finished:
            if limit is not None and len(frames) >= int(limit):
                break
            frame = self.step()
            if frame is not None:
                frames.append(frame)
        return MotionSequence(metadata=self.get_metadata(), frames=frames, characters=self.characters)

    def seed_from_motion_source(self, source) -> None:
        """Use an existing motion source as the prefix for continued generation."""
        if not self.initialized:
            self.initialize()

        metadata = source.get_metadata()
        latest = source.latest_frame_id()
        if latest < 0:
            return

        for character in self.characters:
            frames = []
            origin_frames = []
            for frame_id in range(latest + 1):
                frame = source.get_frame(frame_id)
                if frame is None:
                    continue
                params = frame.character_body_params.get(character.name)
                if params is not None:
                    frames.append(params.detach().clone().to(self.device))
                origin = frame.character_body_params_origin.get(character.name)
                if origin is not None:
                    origin_frames.append(origin.detach().clone().to(self.device))
            if frames:
                character.body_params = torch.cat(frames, dim=0)
                if origin_frames:
                    character.body_params_origin = torch.cat(origin_frames, dim=0)
                character.current_actions = []
                character.target_actions = []
                character.target_position = character.position
                character.target_root_position = character.position
                character.target_facing_direction = character.facing_direction
                character.future_positions = character.position.tile(3, 1)
                character.future_facing_directions = character.facing_direction.tile(3, 1)

        self.fId = latest + 1
        self.max_fId = latest
        self.motion_synthesizer.state_manager.fId = self.fId
        self.user_instruction = metadata.user_instruction or self.user_instruction

    def save(self, path_or_dir: str = "motion") -> str:
        if not self.initialized:
            raise RuntimeError("MotionProducer must be initialized before saving.")
        narrator_type = getattr(getattr(self.cfg, "narrator", None), "type", "baseline")
        parser_mode = getattr(getattr(self.cfg, "event_parser", None), "parser_mode", "single")
        filepath = save_motion_data(
            characters=self.characters,
            scene_name=self.scene_name,
            num_characters=self.num_characters,
            user_instruction=self.user_instruction,
            character_initial_grid_positions=self.character_initial_grid_positions,
            fId=self.fId,
            max_fId=self.max_fId,
            motion_dir=path_or_dir,
            narrator_type=narrator_type,
            parser_mode=parser_mode,
            logger=self.logger,
        )
        if self.motion_synthesizer is not None and hasattr(self.motion_synthesizer, "generation_stats"):
            self.motion_synthesizer.generation_stats.save()
        return filepath

    def get_metadata(self) -> MotionMetadata:
        return MotionMetadata(
            scene_name=self.scene_name,
            character_names=list(self.character_names),
            total_frames=None if not self.finished else self.max_fId + 1,
            user_instruction=self.user_instruction,
            num_characters=self.num_characters,
            character_initial_grid_positions=dict(self.character_initial_grid_positions),
            extras={"fId": self.fId, "max_fId": self.max_fId},
        )

    def _initial_position(self, character: Character):
        if character.name in self.character_initial_grid_positions:
            gx, gy = [int(v) for v in self.character_initial_grid_positions[character.name]]
            x_min, y_min = self.scene.grid_map.bbox[0][:2]
            return grid_to_world(
                gx,
                gy,
                grid_size=self.scene.grid_map.grid_step,
                bbox_min=(x_min, y_min),
                return_center=True,
                z=0.0,
            )
        return self.scene.grid_map.get_random_position()

    def _motion_frame(self, frame_id: int) -> MotionFrame:
        character_body_params = {}
        character_body_params_origin = {}
        for character in self.characters:
            character_body_params[character.name] = character.body_params[-1:].detach().clone()
            if character.body_params_origin is not None and len(character.body_params_origin) > 0:
                character_body_params_origin[character.name] = character.body_params_origin[-1:].detach().clone()
        return MotionFrame(
            frame_id=frame_id,
            character_body_params=character_body_params,
            character_body_params_origin=character_body_params_origin,
        )

    def _is_finished(self) -> bool:
        planner = self.action_planner
        if planner is None or not planner.event_queue.empty():
            return False
        timeline_done = (
            hasattr(planner.narrator, "plan_generated")
            and hasattr(planner.narrator, "event_list")
            and planner.narrator.plan_generated
            and planner.narrator.event_list == []
            and all(
                character.state == State.IDLE
                or (character.looping and character.event is not None and character.event.done)
                for character in self.characters
            )
            and all(character.target_actions == [] for character in self.characters)
        )
        baseline_done = (
            self.fId > 100
            and not (hasattr(planner.narrator, "event_list") and planner.narrator.event_list)
            and all(character.state == State.IDLE for character in self.characters)
            and all(character.target_actions == [] for character in self.characters)
        )
        return timeline_done or baseline_done
