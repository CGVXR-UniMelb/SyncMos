from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import torch

from src.type import Character, Event, Role
from .posture import idle_action_for

if TYPE_CHECKING:
    from .state_manager import StateManager


class State(IntEnum):
    IDLE = 0
    APPROACHING = 1
    INTERACTING = 2
    TRANSITION_IN = 3
    TRANSITION_OUT = 4


class StateBase:
    def __init__(self, manager: StateManager) -> None:
        self.manager = manager
        self.motion_generator = manager.motion_generator
        self.grid_map = manager.grid_map
        # Only using the generator’s set of tags; no longer relying on motion_matcher/motion_db
        self.motion_db = None
        self.hhi_tags = getattr(self.motion_generator, "hhi_tags", set())
        self.hsi_tags = getattr(self.motion_generator, "hsi_tags", set())
        self.partwise_tags = getattr(self.motion_generator, "partwise_tags", set())
        self.looping_tags = getattr(self.motion_generator, "looping_tags", set())

        self.hsi_transition_done_delay = manager.hsi_transition_done_delay
        self.logger = manager.logger

    def enter(self, character: Character, prev_state):
        pass

    def update(self, character: Character):
        raise NotImplementedError

    def exit(self, character: Character, next_state):
        pass

    def __eq__(self, value) -> bool:
        if isinstance(value, str):
            class_name = self.__class__.__name__.lower()
            return class_name.startswith(value.lower().replace("_", ""))
        if isinstance(value, State):  # compare with State enum via string
            class_name = self.__class__.__name__
            enum_name = value.name.title().replace("_", "")
            return class_name.startswith(enum_name)
        if isinstance(value, StateBase):
            return self.__class__ == value.__class__
        raise NotImplementedError

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return class_name.replace("State", "").lower()

    @property
    def fId(self):
        return self.manager.fId

    def _is_parallel_interaction_ready(self, character: Character):
        """
        Checking whether the action is ready to enter the interaction state
        """
        if character.parallel_group_members:
            return self._is_parallel_group_ready(character)
        if character.interactee is None:
            return False

        # Checking whether both characters are in the IDLE state
        if character.state != State.IDLE or character.interactee.state != State.IDLE:
            return False

        # Checking whether both characters have target actions
        if not character.target_actions or not character.interactee.target_actions:
            return False

        # Checking whether both characters are close to the target position
        # char_distance = torch.norm(character.position - character.target_position)
        # interactee_distance = torch.norm(character.interactee.position - character.interactee.target_position)

        # # Only ready when both characters are close to the target position
        # return (char_distance <= self.manager.far_threshold and
        #         interactee_distance <= self.manager.far_threshold)

        # If both characters are in the IDLE state, then the approaching state must have been executed, so they can enter the interacting state simultaneously
        return True

    def _is_parallel_group_ready(self, character: Character) -> bool:
        """Parallel group ready: All members have target actions, are in the IDLE state, and belong to the same event."""
        members = character.parallel_group_members
        if not members:
            return False

        if not character.target_actions or character.state != State.IDLE:
            return False

        for member in members:
            if member.event is None or member.event != character.event:
                return False
            if not member.target_actions:
                return False
            if member.state != State.IDLE:
                return False
        return True

    def _is_at_path_end(self, character: Character) -> bool:
        """Have we reached the end of the planned path? (May not be the original target grid)."""
        grid_path = getattr(character, "grid_path", None)
        if not grid_path:
            return False
        try:
            current_grid = self.grid_map.get_closest_grid_position(
                character.position,
                exclude_occupied=False,
            )
        except Exception:
            return False
        return current_grid == grid_path[-1]


class IdleState(StateBase):
    def enter(self, character: Character, prev_state):
        character.idle_action = idle_action_for(character)
        character.current_actions = [character.idle_action]

        # this helps to reduce the jitterative motion when the character is idle
        if character.target_facing_direction is None:
            character.target_facing_direction = character.facing_direction

    def update(self, character: Character):
        # character remains idle until the next action plan is set
        if not character.target_actions:
            return

        target_position = character.target_position.clone()
        target_position[0, 2] = 0.0
        distance_to_target_position = torch.norm(character.position - target_position)

        # when the target position is changed to far from the current position,
        # the character prioritizes approaching the target position
        if not self.manager.wait_for_interaction_state and distance_to_target_position > self.manager.far_threshold:
            if character.is_resting_posture():
                return self.manager.transition_out_state
            return self.manager.approaching_state

        # when a character is in a parallel group, all members enter INTERACTING together
        if character.parallel_group_members:
            if self._is_parallel_group_ready(character):
                group_members = [character] + character.parallel_group_members
                for member in group_members:
                    if member.state == State.IDLE:
                        self.manager.change_state(member, self.manager.interacting_state)
                return
            return

        # when a character doesn't need to wait for someone to execute an action
        if character.interactee is None:
            return self.manager.interacting_state

        # if a character is waiting for the interactee and them is close enough,
        # we try to match interaction motion from the human-human interaction dataset
        if (
            character.interactee is not None
            and self._is_parallel_interaction_ready(character)
            and character.num_played_frames >= self.motion_generator.motion_update_frequency
        ) or character.interactee.state == State.INTERACTING:
            return self.manager.interacting_state

    def exit(self, character: Character, next_state):
        if next_state != State.TRANSITION_OUT:
            character.current_actions = []
        # these are hacks for mouse-interacting demo
        if character.target_actions == ["stand"]:
            character.target_actions = []
        if character.event is None:
            character.event = Event([], "dummy")
        if self.manager.wait_for_interaction_state:
            self.manager.wait_for_interaction_state = False


class ApproachingState(StateBase):
    def enter(self, character: Character, prev_state):
        character.idle_action = "stand still"
        character.current_actions = ["walk"]

    def update(self, character: Character):
        target_position = character.target_position.clone()
        target_position[0, 2] = 0.0
        distance_to_target_position = torch.norm(character.position - target_position)
        # we cannot get any closer to the target position as there is no path to the target position
        if character.grid_path[0] == character.grid_path[-1]:
            distance_to_target_position = 0.0
        # if we have reached the end of the planned path (e.g., target occupied), allow transition
        elif self._is_at_path_end(character):
            distance_to_target_position = 0.0

        if character.is_planned_to_interact_with_scene():
            # when the character is intended to sit or lie down on some object,
            # we need to frequently check whether there is a matched animation in the human-scene interaction database
            if distance_to_target_position <= self.manager.hsi_matching_start_distance:
                # matched = self.try_match_from_scene_interaction_db(character)
                # if matched:
                return self.manager.transition_in_state

        elif character.is_planned_to_interact_with_human():
            # when the character is intended to interact with another character,
            # we need to frequently check whether there is a matched animation in the human-human interaction database

            # the character waits its interactee in the idle state
            if distance_to_target_position <= self.manager.near_threshold:
                self.manager.wait_for_interaction_state = True
                return self.manager.idle_state

        else:
            if distance_to_target_position <= self.manager.near_threshold:
                if character.target_actions:
                    if character.parallel_group_members and not self._is_parallel_group_ready(character):
                        self.manager.wait_for_interaction_state = True
                        return self.manager.idle_state
                    return self.manager.interacting_state

                character.event.update_completed(character, self.fId)
                return self.manager.idle_state

    def exit(self, character: Character, next_state):
        character.current_actions.remove("walk")


class InteractingState(StateBase):
    def enter(self, character: Character, prev_state):
        character.current_actions += character.target_actions
        character.current_actions = list(set(character.current_actions))  # remove duplicates
        character.target_actions = []

        for i, action in enumerate(character.current_actions):
            if action in self.hhi_tags and character.interactee is None:
                character.current_actions[i] = f"{action}_single"

        # if character.is_resting_posture() and len(character.current_actions) > 1:
        #     character._use_partwise_mm = True
        #     character.current_actions = [
        #         (f"{action}_partwise" if action in self.partwise_tags else action)
        #         for action in character.current_actions
        #     ]
        if character.interactee is not None and character.role == Role.REACTOR:
            character.play_synchronized_motion = True

    def update(self, character: Character):
        """
        Note: reactor character doesn't update its state by itself.
            So, actor character must handle the reactor's state too (= interactee's state)
        """
        if character.play_synchronized_motion:
            return
        if character.looping:
            return

        # fId = character._fIds[0] if character._use_partwise_mm else character.fId
        tag = character.upper_body_action if character._use_partwise_mm else character.main_action
        # if not self.motion_db.is_end_of_motion(fId, tag):
        #     return

        if any("walk" in action for action in character.current_actions):
            target_position = character.target_position.clone()
            target_position[0, 2] = 0.0
            distance_to_target_position = torch.norm(character.position - target_position)
            # we cannot get any closer to the target position as there is no path to the target position
            if character.grid_path[0] == character.grid_path[-1]:
                distance_to_target_position = 0
            # if we have reached the end of the planned path (e.g., target occupied), allow transition
            elif self._is_at_path_end(character):
                distance_to_target_position = 0.0
            if distance_to_target_position > self.manager.near_threshold:
                return
        # scheduler control the segment finished
        elif not character.segment_finished:
            return

        # if the playing animation is finished,
        # if it was looping motion, the character remains in the interacting state
        if tag in self.looping_tags:
            if character.interactee is not None:
                character.interactee.looping = True
                character.interactee.event.update_completed(character.interactee, self.fId, is_looping=True)
            character.looping = True
            character.event.update_completed(character, self.fId, is_looping=True)
            return

        # in normal cases, the character goes to the idle state
        else:
            if character.interactee is not None:
                self.manager.change_state(character.interactee, self.manager.idle_state)
            return self.manager.idle_state

    def exit(self, character: Character, next_state):
        if character.interactee is not None:
            character.interactee = None
            character.role = Role.NONE
            character.play_synchronized_motion = False

        character._use_partwise_mm = False
        character.target_position = self.grid_map.get_closest_grid_position(
            character.position,
            exclude_occupied=True,
            return_tensor=True,
        )
        character.current_actions = [a.replace("_partwise", "") for a in character.current_actions]
        character.looping = False
        if character.event is not None:
            character.event.update_completed(character, self.fId)


class TransitionInState(StateBase):
    def enter(self, character: Character, prev_state):
        if "sit down" in character.target_actions:
            character.current_actions += ["sit down"]
            character.target_actions.remove("sit down")
        elif "lie down" in character.target_actions:
            character.current_actions += ["lie down"]
            character.target_actions.remove("lie down")

    def update(self, character: Character):
        # if self.motion_db.is_end_of_motion(character.fId, character.main_action):
        #     self.logger.debug(f"{character.name}: transition done at fId {self.fId} due to the end of motion")
        #     return self._proceed_to_next_state(character)

        root_distance = torch.norm(character.target_root_position - character.root_position)
        if root_distance > self.manager.hsi_matching_done_root_distance:
            return
        if character.transition_done_fId is None:
            character.transition_done_fId = self.fId
            self.logger.debug(f"{character.name}: transition delay started at fId {self.fId}")
            return
        if self.fId - character.transition_done_fId < self.hsi_transition_done_delay:
            return

        self.logger.debug(f"{character.name}: transition done at fId {self.fId}")
        return self._proceed_to_next_state(character)

    def exit(self, character: Character, next_state):
        character.transition_done_fId = None
        character.accumulated_adjustment = None
        if "sit_transition_in" in character.current_actions:
            character.current_actions.remove("sit_transition_in")
            character.current_actions += ["sit"]
        elif "lie_transition_in" in character.current_actions:
            character.current_actions.remove("lie_transition_in")
            character.current_actions += ["lie"]

    def _proceed_to_next_state(self, character: Character):
        if not character.target_actions:
            character.event.update_completed(character, self.fId)
            return self.manager.idle_state

        if character.interactee is None:
            return self.manager.interacting_state

        # state will be eventually changed to the interacting state
        # this is because more detailed conditions are checked in the idle state (e.g. HHI matching)
        self.manager.change_state(character, self.manager.idle_state)
        return self.manager.update_state(character)


class TransitionOutState(StateBase):
    def enter(self, character: Character, prev_state):
        if character.is_sitting():
            character.current_actions = ["sit_transition_out"]
        elif character.is_lying():
            character.current_actions = ["lie_transition_out"]

    def update(self, character: Character):
        height = character.root_position[0, 2]
        if height >= self.manager.height_threshold:
            return self.manager.approaching_state

    def exit(self, character: Character, next_state):
        if "sit_transition_out" in character.current_actions:
            character.current_actions.remove("sit_transition_out")
        if "lie_transition_out" in character.current_actions:
            character.current_actions.remove("lie_transition_out")
