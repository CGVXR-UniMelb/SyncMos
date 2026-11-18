from typing import List

import torch

from ..scene.scene import Scene
from ..type import Character
from ..utils import smpl
from ..pipeline.generation_stats import GenerationStats
from .motion_generator.motion_generator import MotionGenerator
from .path_planning.path_planner import PathPlanner
from .state_manager import StateManager
from .states import State


class MotionSynthesisModule:
    def __init__(self, scene: Scene, characters: List[Character], cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.scene = scene
        self.characters = characters

        self.bm = smpl.load_body_model(device=cfg.device)
        self.motion_generator = MotionGenerator(cfg, scene, self.device)
        self.path_planner = PathPlanner(scene.grid_map, cfg)
        self.state_manager = StateManager(self.motion_generator, scene.grid_map, cfg)
        self.generation_stats = GenerationStats(
            motion_joint_count=getattr(self.motion_generator.lingo_cfg.dataset, "nb_joints", 28)
        )

    def _should_use_parallel_motion(self, character: Character, interactee: Character) -> bool:
        """Only use parallel motion for handover-like interactions."""
        if character is None or interactee is None:
            return False

        # Require both to have a hand target (e.g., shared handover point)
        if character.target_hand_position is None or interactee.target_hand_position is None:
            return False

        # Require explicit attachment/placement intent to indicate object transfer
        if any(
            [
                character.attachment_intent,
                character.placement_intent,
                interactee.attachment_intent,
                interactee.placement_intent,
            ]
        ):
            return True

        return False

    def get_current_mesh(self, character: Character, use_origin: bool = False):
        """
        Get the current mesh for a character.

        Args:
            character: Character object
            use_origin: If True, use body_params_origin (original motion without time warp)
                       If False, use body_params (potentially time-warped motion)

        Returns:
            vertices: mesh vertices
            faces: mesh faces
        """
        if use_origin and character.body_params_origin is not None and len(character.body_params_origin) > 0:
            vertices = self.bm(character.body_params_origin[-1:]).vertices[0].cpu().numpy()
        else:
            vertices = self.bm(character.body_params[-1:]).vertices[0].cpu().numpy()
        faces = self.bm.faces
        return vertices, faces

    def get_mesh_at(self, character: Character, fId: int, use_origin: bool = False):
        """
        Get the mesh at a specific frame.

        Args:
            character: Character object
            fId: Frame index
            use_origin: If True, use body_params_origin; if False, use body_params

        Returns:
            vertices: mesh vertices
            faces: mesh faces
        """
        # Try to use origin data if requested
        if use_origin and character.body_params_origin is not None and len(character.body_params_origin) > 0:
            # Check if the frame is within bounds of origin data
            if 0 <= fId < len(character.body_params_origin):
                vertices = self.bm(character.body_params_origin[fId : fId + 1]).vertices[0].cpu().numpy()
                faces = self.bm.faces
                return vertices, faces
            # If out of bounds for origin, fall back to regular body_params

        # Use regular body_params (either not using origin, or origin data not available/out of bounds)
        if fId < 0 or fId >= len(character.body_params):
            raise ValueError(f"Frame index {fId} is out of bounds for character {character.name}.")
        vertices = self.bm(character.body_params[fId : fId + 1]).vertices[0].cpu().numpy()
        faces = self.bm.faces
        return vertices, faces

    def advance_motion_frame(self, character: Character):
        self.state_manager.update_state(character)
        if self.motion_generator.is_motion_update_needed(character):
            if character.state in [State.IDLE, State.APPROACHING]:
                self.path_planner.update_character_path(character, self.characters)
                self.path_planner.update_future_prediction(character)
            # for parallel interaction
            if (
                character.interactee is not None
                and character.state == State.INTERACTING
                and self._should_use_parallel_motion(character, character.interactee)
            ):
                # when the interactee already start interaction, we need to generate the parallel motion for both characters
                if character.interactee.parallel_motion_started:
                    # turn off the flag
                    character.interactee.parallel_motion_started = False
                    character.parallel_motion_started = False

                    # generate the parallel motion for both characters
                    self.motion_generator.generate_parallel_motion(character, character.interactee)

                    # remember to advance the motion frame for the interactee
                    self.motion_generator.advance_motion_frame(character.interactee)
                    # TODO: why body_params length is different? len(character.interactee.body_params) 642 len(character.body_params) 643
                    character.body_params = torch.cat(
                        [character.body_params, character.body_params[-1:].clone()], dim=0
                    )

                # when the interactee haven't start interaction, current character need to wait and set flag to True
                else:
                    character.parallel_motion_started = True
                    return
            # for normal motion generation
            else:
                self.motion_generator.update_best_matching_frame(character)
        self.motion_generator.advance_motion_frame(character)

    def advance_motion_frames(self, characters: List[Character]):
        batchable = []
        fallback = []
        already_pending = []
        for character in characters:
            self.state_manager.update_state(character)
            update_needed = self.motion_generator.is_motion_update_needed(character)
            if not update_needed:
                already_pending.append(character)
                continue
            if character.state in [State.IDLE, State.APPROACHING]:
                self.path_planner.update_character_path(character, self.characters)
                self.path_planner.update_future_prediction(character)
            if (
                character.interactee is not None
                and character.state == State.INTERACTING
                and self._should_use_parallel_motion(character, character.interactee)
            ):
                fallback.append(character)
            else:
                batchable.append(character)

        if batchable:
            for character in already_pending:
                if self._can_cut_future_cache_for_batch(character):
                    if character.state in [State.IDLE, State.APPROACHING]:
                        self.path_planner.update_character_path(character, self.characters)
                        self.path_planner.update_future_prediction(character)
                    self._cut_future_cache(character)
                    batchable.append(character)
            self.motion_generator.update_best_matching_frames_batch(batchable)
            batch_size = len(batchable)
            for character in batchable:
                generated_frames = len(character.cache_frames) if character.cache_frames is not None else 0
                action = character.current_actions[0] if character.current_actions else "stand still"
                self.generation_stats.record_generated_segment(
                    self.state_manager.fId,
                    character.name,
                    generated_frames,
                    action,
                    batch_size,
                )
        for character in fallback:
            if character.interactee is not None and character.interactee.parallel_motion_started:
                character.interactee.parallel_motion_started = False
                character.parallel_motion_started = False
                self.motion_generator.generate_parallel_motion(character, character.interactee)
                self.motion_generator.advance_motion_frame(character.interactee)
                character.body_params = torch.cat([character.body_params, character.body_params[-1:].clone()], dim=0)
            else:
                character.parallel_motion_started = True

        for character in characters:
            if character.parallel_motion_started:
                continue
            self.motion_generator.advance_motion_frame(character)
        self.generation_stats.set_played_frame(self.state_manager.fId)

    def _can_cut_future_cache_for_batch(self, character: Character) -> bool:
        if character is None:
            return False
        if character.interactee is not None and character.state == State.INTERACTING:
            return False
        if character.parallel_motion_started:
            return False
        if character.cache_frames is None or len(character.cache_frames) == 0:
            return False
        if character.points_orig is None:
            return False
        return True

    def _cut_future_cache(self, character: Character) -> None:
        cut_frames = len(character.cache_frames) if character.cache_frames is not None else 0
        self.generation_stats.record_cut(
            self.state_manager.fId,
            character.name,
            cut_frames,
            "join_batch_generation",
        )
        character.cache_frames = None
        character.cache_origin_frames = None
        character.cache_joints_event = None
        character.cache_points = None
        character.num_played_frames = 9999
        character.segment_finished = False
        character.segment_finished_in_buffer = False

    def create_initial_body_params(
        self,
        position,
        facing_direction,
        motion_file="ACCAD/Male1General_c3d/General A1 - Stand_poses.npz",
        character: Character | None = None,
    ) -> torch.Tensor:
        """Delegate initialization and return packed (T, 91) body parameters.

        ``position`` is a NumPy-compatible three-component Event (Z-up) position.
        ``facing_direction`` and ``motion_file`` remain accepted for compatibility
        but are not used. Callers must supply a character: the existing generator
        updates its continuation history despite the legacy None default here.
        The generator retains its existing sampling, fitting and random draws.
        """
        return self.motion_generator._create_initial_body_params(position, character)
