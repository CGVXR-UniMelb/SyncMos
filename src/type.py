from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, ClassVar, List, Optional

import numpy as np
import torch
import trimesh

from .utils import smpl
from .utils.transform import quaternion_relative_rotation, quaternion_to_axis_angle

if TYPE_CHECKING:
    from src.motion_synthesis.states import StateBase


class Gender(IntEnum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, gender):
        if gender.lower() == "male":
            return cls.MALE
        elif gender.lower() == "female":
            return cls.FEMALE
        else:
            raise ValueError("Invalid gender")


class Role(IntEnum):
    NONE = -1
    ACTOR = 0
    REACTOR = 1


@dataclass
class Event:
    involved_characters: list[Character]
    activity: str
    state: str = "ongoing"
    completed: list[bool] = field(default_factory=list)
    parsed_event = None
    reasoning: str = None
    created_fId: int = None
    completed_fId: int = None

    def __post_init__(self):
        self.activity = self.activity.rstrip(".")
        self.completed = [False] * len(self.involved_characters)

    def __repr__(self) -> str:
        if isinstance(self.involved_characters[0], str):
            return f"[{', '.join(c for c in self.involved_characters)}] {self.activity}"
        return f"[{', '.join(c.name for c in self.involved_characters)}] {self.activity}"

    def __eq__(self, value) -> bool:
        if isinstance(value, str):
            return str(self) == value
        if isinstance(value, Event):
            return str(self) == str(value)
        raise NotImplementedError("Only string and Event comparison is supported.")

    def __hash__(self) -> int:
        return hash(self.content)

    @property
    def is_group_event(self):
        return len(self.involved_characters) > 1

    @property
    def done(self):
        return all(self.completed)

    def update_completed(self, character: Character, fId: int, is_looping=False):
        idx = self.involved_characters.index(character)
        self.completed[idx] = True
        if not all(self.completed):
            return

        if is_looping:
            self.state = "looping"
        else:
            self.state = "completed"
            self.completed_fId = fId
            for c in self.involved_characters:
                c.event = None


@dataclass
class ParallelEventGroup:
    """Parallel event group, containing multiple events that need to be processed simultaneously"""

    events: list[Event]
    parallel_group: list[str]
    state: str = "ongoing"
    completed: list[bool] = field(default_factory=list)
    parsed_event = None
    reasoning: str = None
    created_fId: int = None
    completed_fId: int = None

    def __post_init__(self):
        self.completed = [False] * len(self.events)

    def __repr__(self) -> str:
        event_strs = [str(event) for event in self.events]
        return f"ParallelGroup[{', '.join(event_strs)}]"

    @property
    def involved_characters(self):
        """Return all characters involved in the events"""
        all_characters = []
        for event in self.events:
            all_characters.extend(event.involved_characters)
        return all_characters

    @property
    def done(self):
        return all(self.completed)

    def update_completed(self, character: Character, fId: int, is_looping=False):
        """Update the completion status"""
        # Determine which event the character is in
        for i, event in enumerate(self.events):
            if character in event.involved_characters:
                event.update_completed(character, fId, is_looping)
                if event.done:
                    self.completed[i] = True
                break

        if not all(self.completed):
            return

        if is_looping:
            self.state = "looping"
        else:
            self.state = "completed"
            self.completed_fId = fId
            for c in self.involved_characters:
                c.event = None
                c.parallel_group_id = None
                c.parallel_group_members = []


@dataclass
class Character:
    name: str
    gender: Gender = Gender.MALE
    state: StateBase = None
    current_actions: list = field(default_factory=list)
    idle_action: str = "stand still"
    seat_object: Optional[str] = None
    target_actions: list = field(default_factory=list)
    relationships: list = field(default_factory=list)
    previous_relationships: list = field(default_factory=list)
    event: Event = None
    interactee: Character = None
    role: Role = Role.NONE
    dt: float = 1 / 30
    looping: bool = False

    # for path planning
    velocity_mps: float = 1.0
    halflife_position: float = 0.1
    halflife_rotation: float = 0.3
    target_position: torch.Tensor = None
    spline_curve: torch.Tensor = None
    grid_path: list = field(default_factory=list)
    waiting_frames: int = 0  # Wait for the frame counter
    last_position: torch.Tensor = None  # Position from the previous frame, used to detect whether there is movement
    deadlock_avoidance_active: bool = False  # Is deadlock avoidance being performed

    # for motion matching
    fId: int = -1
    motion_db: str = "base"
    num_played_frames: int = 9999
    play_synchronized_motion: bool = False
    enforce_motion_search: bool = False
    invoke_inertialization: bool = True
    future_positions: torch.Tensor = None
    future_facing_directions: torch.Tensor = None
    target_facing_direction: torch.Tensor = None
    target_root_position: torch.Tensor = None
    transition_done_fId: int = None
    accumulated_adjustment: torch.Tensor = None
    matched_root_position: torch.Tensor = None
    matching_feature: torch.Tensor = None
    # Accumulated playback frames: (T, 91), packed as translation(3), root quaternion(4),
    # and 21 joint quaternions(84). Quaternions use (w, x, y, z).
    body_params: torch.Tensor = None
    offset_x: torch.Tensor = None
    offset_v: torch.Tensor = None

    # for parallel motion visualization toggle (original vs time-warped)
    cache_origin_frames: Optional[torch.Tensor] = None  # format same as cache_frames
    body_params_origin: torch.Tensor = None  # original body params before time warp

    # for partwise motion matching
    _use_partwise_mm: bool = False
    _fIds: List[int] = field(default_factory=lambda: [-1, -1])
    _motion_dbs: List[str] = field(default_factory=lambda: ["base", "base"])
    _update_required: List[bool] = field(default_factory=lambda: [False, False])

    _counter: ClassVar[Counter] = Counter()

    # for lingo
    # this used to check whether current action is changed.
    # because for lingo, one action is constructed by several segmentations
    current_generating_action: str = None
    # which segmentation we are for this action (named step to align with lingo)
    step: int = 0
    cache_frames: Optional[torch.Tensor] = None  # pending playback body parameters, shape (T, 91)
    cache_points: Optional[torch.Tensor] = None  # LINGO joints, shape (B, T_lingo, 3 * J); T_lingo may differ from T
    cache_joints_event: Optional[torch.Tensor] = (
        None  # event-coordinate joints aligned with cache_frames, shape (num_frames, nb_joints, 3)
    )
    points_orig: Optional[torch.Tensor] = None  # LINGO joints from the previous segment
    current_motion_joints_event: Optional[torch.Tensor] = (
        None  # current generated joints in event coordinates, shape (1, nb_joints, 3)
    )
    text_emb: Optional[torch.Tensor] = None  # the text embedding for the current action

    # for generator-driven state switching
    segment_finished: bool = False
    segment_finished_in_buffer: bool = False

    # for hand object interaction
    target_hand_position: Optional[torch.Tensor] = None  # Hand target position (1, 3)
    target_hand_object: Optional[str] = None  # Name of the object for hand interaction
    attached_objects: List[str] = field(default_factory=list)  # List of currently attached objects
    attachment_intent: Optional[dict] = (
        None  # Attachment intent, including object, grasping type, distance threshold, etc
    )
    placement_intent: Optional[dict] = None  # Placement intent, including target position, distance threshold, etc

    # for parallel interaction
    parallel_motion_started: bool = False
    parallel_group_id: Optional[str] = None
    parallel_group_members: List[Character] = field(default_factory=list)

    def __post_init__(self):
        if self.name[0].islower():
            self._counter[self.name] += 1
            self.name = f"{self.name}_{self._counter[self.name]}"

    def __repr__(self) -> str:
        return f"{self.name} ({self.state})"

    def __eq__(self, __value: object) -> bool:
        if __value is None:
            return False
        if isinstance(__value, str):
            return self.name == __value
        if isinstance(__value, Character):
            return self.name == __value.name
        raise NotImplementedError("Only string and Character comparison is supported.")

    @property
    def position(self):
        """Returns the position of the character. The shape is (1, 3), but the z value is fixed to 0."""
        joint_position = self._joint_root_positions()
        if joint_position is not None:
            joint_position = joint_position.clone()
            joint_position[..., 2] = 0
            return joint_position
        return smpl.get_character_positions(self.body_params[-1:])

    @property
    def velocity(self):
        """Returns the velocity of the character. The shape is (1, 3), but the z value is fixed to 0."""
        positions = smpl.get_character_positions(self.body_params[-2:])
        return torch.diff(positions, dim=0) / self.dt

    @property
    def acceleration(self):
        """Returns the acceleration of the character. The shape is (1, 3), but the z value is fixed to 0."""
        positions = smpl.get_character_positions(self.body_params[-3:])
        velocities = torch.diff(positions, dim=0) / self.dt
        return torch.diff(velocities, dim=0) / self.dt

    @property
    def rotation(self):
        """Returns the rotation of the character in quaternion. The shape is (1, 4)."""
        facing_direction = smpl.get_facing_directions(self.body_params[-1:])
        return smpl.get_character_rotations(facing_direction)

    @property
    def angular_velocity(self):
        """Returns the angular_velocity of the character. The shape is (1, 3)."""
        facing_directions = smpl.get_facing_directions(self.body_params[-2:])
        rotations = smpl.get_character_rotations(facing_directions)
        return (
            quaternion_to_axis_angle(
                quaternion_relative_rotation(
                    rotations[:1],
                    rotations[1:],
                )
            )
            / self.dt
        )

    @property
    def facing_direction(self):
        """Returns the facing direction of the character. The shape is (1, 3), but the z value is fixed to 0."""
        joint_facing_direction = self._joint_facing_directions()
        if joint_facing_direction is not None:
            return joint_facing_direction
        return smpl.get_facing_directions(self.body_params[-1:])

    @property
    def root_position(self):
        """Returns the root position of the character. The shape is (1, 3)."""
        joint_root_position = self._joint_root_positions()
        if joint_root_position is not None:
            return joint_root_position
        return smpl.get_root_positions(self.body_params[-1:])

    def _joint_root_positions(self) -> Optional[torch.Tensor]:
        if self.current_motion_joints_event is None:
            return None
        joints = self.current_motion_joints_event
        if joints.ndim == 2:
            joints = joints.unsqueeze(0)
        if joints.shape[-2] < 1:
            return None
        return joints[:, 0, :]

    def _joint_facing_directions(self) -> Optional[torch.Tensor]:
        if self.current_motion_joints_event is None:
            return None
        joints = self.current_motion_joints_event
        if joints.ndim == 2:
            joints = joints.unsqueeze(0)
        if joints.shape[-2] <= 17:
            return None

        hip_lateral = joints[:, 2, :] - joints[:, 1, :]
        shoulder_lateral = joints[:, 17, :] - joints[:, 16, :]
        lateral = hip_lateral + shoulder_lateral
        lateral[..., 2] = 0
        lateral_norm = torch.norm(lateral, dim=-1, keepdim=True)
        if torch.any(lateral_norm <= 1e-6):
            return None

        facing = torch.stack([-lateral[:, 1], lateral[:, 0], torch.zeros_like(lateral[:, 0])], dim=-1)
        facing = facing / torch.clamp(torch.norm(facing, dim=-1, keepdim=True), min=1e-6)

        if self.body_params is not None and len(self.body_params) > 0:
            body_facing = smpl.get_facing_directions(self.body_params[-1:])
            facing = torch.where((facing * body_facing).sum(dim=-1, keepdim=True) < 0, -facing, facing)
        return facing

    @property
    def main_action(self):
        if not self.current_actions:
            if self.is_stationary():
                return "idle"
            return "walk"

        if len(self.current_actions) == 1:
            return self.current_actions[0]

        def _is_posture_related(action: str) -> bool:
            if action is None:
                return False
            a = action.lower()
            return (
                a in {"sit", "lie", "sit down", "lie down"}
                or a.startswith("maintain sit")
                or a.startswith("maintain lie")
                or a.startswith("sit_transition")
                or a.startswith("lie_transition")
            )

        for action in self.current_actions:
            if not _is_posture_related(action):
                return action
        # If all are poses/Transition, return to the first
        return self.current_actions[0]

    @property
    def upper_body_action(self):
        def _is_posture_related(action: str) -> bool:
            if action is None:
                return False
            a = action.lower()
            return (
                a in {"sit", "lie", "sit down", "lie down"}
                or a.startswith("maintain sit")
                or a.startswith("maintain lie")
                or a.startswith("sit_transition")
                or a.startswith("lie_transition")
            )

        for action in self.current_actions:
            if not _is_posture_related(action):
                return action
        return self.current_actions[0] if self.current_actions else "idle"

    @property
    def lower_body_action(self):
        def _is_posture_related(action: str) -> bool:
            if action is None:
                return False
            a = action.lower()
            return (
                a in {"sit", "lie", "sit down", "lie down"}
                or a.startswith("maintain sit")
                or a.startswith("maintain lie")
                or a.startswith("sit_transition")
                or a.startswith("lie_transition")
            )

        for action in self.current_actions:
            if _is_posture_related(action):
                return action
        return self.main_action

    def is_stationary(self):
        return all(p == self.grid_path[0] for p in self.grid_path)

    def is_resting_posture(self):
        return self.is_sitting() or self.is_lying()

    def is_sitting(self):
        return any(
            a is not None
            and (
                a.lower() in {"sit", "sit down"}
                or a.lower().startswith("maintain sit")
                or a.lower().startswith("sit_transition")
            )
            for a in self.current_actions
        )

    def is_lying(self):
        return any(
            a is not None
            and (
                a.lower() in {"lie", "lie down"}
                or a.lower().startswith("maintain lie")
                or a.lower().startswith("lie_transition")
            )
            for a in self.current_actions
        )

    def is_planned(self):
        return self.target_actions

    def is_planned_to_interact_with_scene(self):
        # Only plan the ones that are“Sit down/Lie down”Consider as interacting with the scene; pose maintenance does not trigger planning
        return ("sit down" in self.target_actions) or ("lie down" in self.target_actions)

    def is_planned_to_interact_with_human(self):
        return self.interactee is not None


@dataclass
class Object:
    id: int
    label: str
    idx: int
    aa_extents: np.ndarray
    orientation: np.ndarray
    has_orientation: bool
    width: float
    depth: float
    height: float
    center: np.ndarray
    bbox: np.ndarray
    mesh: trimesh.Trimesh
    sit_offset_horizontal: float = 0.0
    sit_offset_vertical: float = 0.0

    # for hand object interaction
    attached_to: Optional[str] = None  # Name of the character to which it is attached
    attachment_offset: Optional[np.ndarray] = None  # Offset relative to the character’s hand (3,)
    is_attached: bool = False  # Whether it is in an attached state

    def __repr__(self) -> str:
        return f"{self.label}_{self.idx}"

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Object):
            return self.id == __value.id
        elif isinstance(__value, str):
            return f"{self.label}_{self.idx}" == __value
        raise NotImplementedError("Only Object comparison is supported.")

    def __hash__(self) -> int:
        return hash(self.id)

    @property
    def min_z(self):
        return self.bbox[:, 2].min()

    @property
    def max_z(self):
        return self.bbox[:, 2].max()

    @property
    def corners(self):
        return self.bbox[:4, :2]
