import numpy as np
import torch
import hydra
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as R

from src.type import Character
from ..states import State
from ...scene.scene import Scene

from src.motion_synthesis.lingo_joint_fitting import interpolate_joints, joints_to_smpl
from src.motion_synthesis.lingo_model_loading import init_model
from src.motion_synthesis.lingo_math import transform_points

# Preserve dependency import timing; only native helper loading stays lazy.
import clip  # noqa: F401
import tqdm.auto  # noqa: F401
from src.motion_synthesis.lingo_clip import get_clip_features
from src.motion_synthesis.lingo_geometry import get_mat
from src.motion_synthesis.lingo_math import axis_angle_to_quaternion
from src.motion_synthesis.sampling_utils import DTW
from ..time_warp import TimeWarpModule
from ..posture import idle_action_for
from ..facing_control import align_seated_facing
from .parallel_diagnostics import ParallelMotionDiagnostics


class MotionGenerator:
    """
    Drop-in replacement for MotionMatcher: produces the next frame using a diffusion generator.
    """

    def __init__(self, cfg: DictConfig, scene: Scene, device=None):
        self.cfg = cfg
        self.device = device or cfg.device
        self._motion_update_frequency = cfg.motion_matching.motion_update_frequency
        self.scene = scene

        # load lingo config
        with initialize(version_base=None, config_path=cfg.lingo_config_path):
            lingo_cfg = compose(config_name="config_sample_lingo")
        print(OmegaConf.to_yaml(lingo_cfg))
        self.lingo_cfg = lingo_cfg

        # dataset + model init
        self.dataset = hydra.utils.instantiate(lingo_cfg.dataset)
        self.sampler_body = hydra.utils.instantiate(lingo_cfg.sampler.pelvis)
        self.model_body = init_model(lingo_cfg.model.synhsi_body, device=lingo_cfg.device, eval=True)
        self.model_joints_to_smplx = init_model(lingo_cfg.model.model_smplx, device=lingo_cfg.device, eval=True)
        self.sampler_body.set_dataset_and_model(self.dataset, self.model_body)

        # Legacy initialization consumed a scheduler constructor's CPU random draws
        # before scene occupancy was built. Keep those draws at the same point,
        # but do not read weights or transfer this discarded instance to the GPU.
        # The single live scheduler is loaded below, after scene occupancy.
        self.scheduler = None
        if lingo_cfg.get("use_scheduler", False):
            hydra.utils.instantiate(lingo_cfg.model.scheduler)

        self.auto_regre = lingo_cfg.auto_regre_num
        self.nb_joints = lingo_cfg.dataset.nb_joints
        self.max_window = lingo_cfg.max_window_size
        self.use_pi = lingo_cfg.use_pi
        self.speed = lingo_cfg.get("speed", 1.0)
        self.loc_thresh = lingo_cfg.get("locomotion_threshold", 0.6)  # meters, tune

        assert ((self.max_window - self.auto_regre) * self.lingo_cfg.interp_s) == self._motion_update_frequency

        # change the scene_occ, scene_grid_np, scene_grid_torch in lingo dataset
        # scene_occ, scene_grid_np, scene_grid_torch = self.scene.grid_map.build_scene_occ_from_bbox()
        scene_occ, scene_grid_np, scene_grid_torch = self.scene.grid_map.build_scene_occ_from_mesh_contains()
        self.sampler_body.dataset.scene_occ = scene_occ.unsqueeze(0)  # shape (1, x, y, z)
        self.sampler_body.dataset.scene_grid_np = scene_grid_np
        self.sampler_body.dataset.scene_grid_torch = scene_grid_torch

        # load scheduler model
        if lingo_cfg.use_scheduler:
            scheduler_model = hydra.utils.instantiate(lingo_cfg.model.scheduler)
            scheduler_model.load_state_dict(torch.load(lingo_cfg.scheduler_model_path, map_location=self.device))
            scheduler_model.to(self.device)
            scheduler_model.eval()
        else:
            raise ValueError("scheduler model needed for motion generator")
        self.scheduler = scheduler_model

        # -------- Action classification and synonym normalization based on user-provided list --------
        # Standing/Maintain posture categories
        self.maintain_actions = {
            "stand still",
            "maintain lie",
            "maintain sit",
            "maintain bend",
            "maintain kneel",
            "maintain squat",
        }
        # Walking/Displacement Classes
        self.locomotion_actions = {
            "walk",
            "walk forward",
            "walk back",
            "walk front left",
            "walk front right",
        }
        # Turning/Orientation adjustment categories
        self.orientation_actions = {
            "turn left",
            "turn right",
            "rotate",
        }
        # In-place Pose Changes/No scene contact categories
        self.inplace_actions = {
            "bend forward",
            "straighten up",
            "kneel down",
            "squat down",
            "crawl",
            "wave",
            "toss",
            "swing",
            "stand up",
            "get up",
        }
        # Handheld object interaction categories
        self.hand_object_actions_pick = {"pick up"}
        self.hand_object_actions_put = {"put down"}
        self.hand_object_actions_misc = {
            "take photo",
            "turn on",
            "write",
            "type on",
            "read",
            "drink",
            "eat",
            "talk on",
            "listen to music",
            "brush teeth",
            # Separate play category actions (for detailed selection by LLM)
            "play game",
            "play guitar",
            "play drums",
            "play piano",
        }
        # Scene object interaction categories (static scenes)
        self.scene_interactions = {
            "sit down",
            "lie down",
            "punch",
            "kick",
            "wash",
            "take shower",
            "type on",
            "write",
            "play game",
            "play guitar",
            "play drums",
            "play piano",
        }
        # Do not normalize action aliases; let the upstream LLM choose the exact action names

        # Provides a set of tags consistent with the old motion_db interface, facilitating state machine branching decisions
        # HHI: Human-human interaction (provides a set of common examples; can be expanded with datasets in the future)
        self.hhi_tags = {"handshake", "hug", "wave", "highfive", "push", "pull", "punch", "kick"}
        # HSI: Human-scene interaction/Object interaction (combines scene_interactions with common operations)
        self.hsi_tags = set(self.scene_interactions) | {
            "use laptop",
            "use microwave",
            "use toaster",
            "use coffee maker",
            "turn on record player",
            "turn off record player",
            "sit down",
            "lie down",
        }

        # Some poses can be played independently for body parts (decision made by upper layers if necessary)
        self.partwise_tags = {
            "wave",
            "drink",
            "eat",
            "read",
            "write",
            "type on",
            "take photo",
        }
        # Loop-related tags (used to align with old logic; fragments are controlled by scheduler within the generator)
        self.looping_tags = {"stand still", "idle", "wave"}

        self.time_warp_module = TimeWarpModule(
            lingo_cfg,
            dataset=self.dataset,
            sampler_body=self.sampler_body,
            model_body=self.model_body,
            model_joints_to_smplx=self.model_joints_to_smplx,
            scheduler_model=self.scheduler,
        )
        # registry to store event->motions for time warping

    @property
    def motion_update_frequency(self):
        return self._motion_update_frequency

    # ---------------- public API to match MotionMatcher ----------------
    def is_motion_update_needed(self, character: Character) -> bool:
        # if the action changes, we need to generate new motion
        if (
            character.current_actions
            and self._normalize_action_text(character.current_actions[0]) != character.current_generating_action
        ):
            character.cache_frames = None
            character.step = 0

        if character.enforce_motion_search:
            character.enforce_motion_search = False
            return True

        # if we dont have cache frames, generate it
        if (character.cache_frames is None) or character.cache_frames.shape[0] == 0:
            return True

        # in transition states, we just want to play the motion without interruption
        if character.state in (State.TRANSITION_IN, State.TRANSITION_OUT):
            return False

        return False

    def update_best_matching_frame(self, character: Character) -> None:
        # it should be generate motion frame
        # call it just to align with motion matcher Class
        # generate max_window - auto_regre frames (16 - 2 = 14 frames)
        action_now = character.current_actions[0] if character.current_actions else "stand still"
        canonical = self._normalize_action_text(action_now)
        step = character.step
        cond = self._get_guidance(character)
        canonical = cond.get("raw_text", canonical)
        character.current_generating_action = canonical

        points_orig = character.points_orig

        mat = get_mat(self.lingo_cfg, points_orig)
        fixed_points = points_orig[:, -self.lingo_cfg.auto_regre_num :].reshape(
            self.lingo_cfg.batch_size, self.lingo_cfg.auto_regre_num, self.lingo_cfg.dataset.nb_joints * 3
        )
        fixed_points = self.sampler_body.dataset.normalize_torch(transform_points(fixed_points, torch.inverse(mat)))

        phase = 1
        speed_inter = 3
        pi = torch.tensor(
            [int((step + phase) * (self.lingo_cfg.max_window_size - self.lingo_cfg.auto_regre_num) * speed_inter)]
        ).to(device=self.lingo_cfg.device, dtype=torch.long)

        info_dict = self._sample_step(self.lingo_cfg, character, mat, fixed_points, self.sampler_body, cond, pi)
        points = info_dict["points_orig"][:, self.auto_regre :]  # points in global coordinates and is denormalized

        character.segment_finished_in_buffer = False
        # scheduler
        if self.lingo_cfg.use_scheduler and not cond["is_loco"]:
            points_loco = self.sampler_body.dataset.normalize_torch(transform_points(points, torch.inverse(mat)))
            stop_pred = self.scheduler(points_loco, cond["text_emb"], pi).squeeze(1)
            stop_pred = torch.sigmoid(stop_pred)
            # scheduler deside to finish the segment
            if stop_pred > self.lingo_cfg.scheduler_threshold:
                character.segment_finished_in_buffer = True

        points = align_seated_facing(character, points, self.scene.grid_map, self.lingo_cfg.interp_s)
        # store points for next generation
        character.cache_points = points

        points = points[0]

        points = points.reshape(-1, self.lingo_cfg.dataset.nb_joints, 3)
        points = self.scene.grid_map.lingo_to_event(points)
        points = points.reshape(-1, self.lingo_cfg.dataset.nb_joints * 3)

        # 5) Convert joints->SMPL(X) and append
        pose, transl, _, _ = joints_to_smpl(
            self.model_joints_to_smplx, points, self.lingo_cfg.dataset.joints_ind, self.lingo_cfg.interp_s
        )

        next_segment_body_params = self._pack_body_params_91(transl, pose[:, :3], pose[:, 3:])
        character.cache_frames = next_segment_body_params
        character.cache_joints_event = self._make_cache_joints_event(
            character.cache_points, len(character.cache_frames)
        )
        character.num_played_frames = 0
        character.step = step + 1
        # Write the generated joints to the registry (event/motion) in steps->0: Start a new motion==0 New motion
        # Save the original joints (Lingo global coordinates), Points not assigned to event coordinates are used for TW, and the diffusion data is stored in local_cache

        # if the segment is too long, finish the segment
        if step > 50 and character.state != State.APPROACHING and "walk" not in canonical:
            character.segment_finished_in_buffer = True

    def update_best_matching_frames_batch(self, characters: list[Character]) -> None:
        """Generate cache segments for multiple independent characters in one sampler batch."""
        sample_requests = []
        for character in characters:
            action_now = character.current_actions[0] if character.current_actions else "stand still"
            canonical = self._normalize_action_text(action_now)
            step = character.step
            cond = self._get_guidance(character)
            canonical = cond.get("raw_text", canonical)
            character.current_generating_action = canonical
            points_orig = character.points_orig
            mat = get_mat(self.lingo_cfg, points_orig)
            fixed_points = points_orig[:, -self.lingo_cfg.auto_regre_num :].reshape(
                1,
                self.lingo_cfg.auto_regre_num,
                self.lingo_cfg.dataset.nb_joints * 3,
            )
            fixed_points = self.sampler_body.dataset.normalize_torch(transform_points(fixed_points, torch.inverse(mat)))
            phase = 1
            speed_inter = 3
            pi = torch.tensor(
                [int((step + phase) * (self.lingo_cfg.max_window_size - self.lingo_cfg.auto_regre_num) * speed_inter)],
                device=self.lingo_cfg.device,
                dtype=torch.long,
            )
            sample_requests.append(
                {
                    "character": character,
                    "canonical": canonical,
                    "step": step,
                    "cond": cond,
                    "mat": mat,
                    "fixed_points": fixed_points,
                    "pi": pi,
                }
            )

        if not sample_requests:
            return
        if len(sample_requests) == 1:
            self.update_best_matching_frame(sample_requests[0]["character"])
            return

        info_dict = self._sample_step_batch(sample_requests)
        points_all = info_dict["points_orig"][:, self.auto_regre :]
        body_params_inputs = []

        for idx, request in enumerate(sample_requests):
            character = request["character"]
            cond = request["cond"]
            mat = request["mat"]
            points = points_all[idx : idx + 1]

            character.segment_finished_in_buffer = False
            if self.lingo_cfg.use_scheduler and not bool(cond["is_loco"][0].item()):
                points_loco = self.sampler_body.dataset.normalize_torch(transform_points(points, torch.inverse(mat)))
                stop_pred = self.scheduler(points_loco, cond["text_emb"], request["pi"]).squeeze(1)
                stop_pred = torch.sigmoid(stop_pred)
                if stop_pred > self.lingo_cfg.scheduler_threshold:
                    character.segment_finished_in_buffer = True

            points = align_seated_facing(character, points, self.scene.grid_map, self.lingo_cfg.interp_s)
            character.cache_points = points
            points_event = points[0].reshape(-1, self.lingo_cfg.dataset.nb_joints, 3)
            points_event = self.scene.grid_map.lingo_to_event(points_event)
            # Interpolate each actor's timeline before concatenating the fitting batch.
            # Otherwise the end of one actor is blended into the next actor's position.
            body_params_inputs.append(
                interpolate_joints(
                    points_event.reshape(-1, self.lingo_cfg.dataset.nb_joints * 3), self.lingo_cfg.interp_s
                )
            )

        points_for_smpl = torch.cat(body_params_inputs, dim=0)
        pose, transl, _, _ = joints_to_smpl(
            self.model_joints_to_smplx,
            points_for_smpl,
            self.lingo_cfg.dataset.joints_ind,
            1,  # Timelines already have the configured interpolation applied independently.
        )
        body_params_all = self._pack_body_params_91(transl, pose[:, :3], pose[:, 3:])

        cursor = 0
        segment_frame_len = body_params_all.shape[0] // len(sample_requests)
        for request in sample_requests:
            character = request["character"]
            step = request["step"]
            character.cache_frames = body_params_all[cursor : cursor + segment_frame_len]
            cursor += segment_frame_len
            character.cache_joints_event = self._make_cache_joints_event(
                character.cache_points, len(character.cache_frames)
            )
            character.num_played_frames = 0
            character.step = step + 1
            if step > 50 and character.state != State.APPROACHING and "walk" not in request["canonical"]:
                character.segment_finished_in_buffer = True

    def _sample_step_batch(self, requests: list[dict]):
        batch_size = len(requests)
        original_batch_size = self.sampler_body.batch_size
        original_grid = getattr(self.sampler_body, "grid", None)
        try:
            self.sampler_body.batch_size = batch_size
            if self.sampler_body.dataset.load_scene:
                self.sampler_body.grid = self.sampler_body.dataset.create_meshgrid(batch_size=batch_size).to(
                    self.device
                )

            mat = torch.cat([request["mat"] for request in requests], dim=0)
            fixed_points = torch.cat([request["fixed_points"] for request in requests], dim=0)
            text_emb = torch.cat([request["cond"]["text_emb"] for request in requests], dim=0)
            hand_goal = torch.cat(
                [
                    transform_points(request["cond"]["hand_goal"].reshape(1, 1, 3), torch.inverse(request["mat"]))
                    for request in requests
                ],
                dim=0,
            )
            pelvis_goal_items = []
            pi_items = []
            for request in requests:
                character = request["character"]
                cond = request["cond"]
                pi = request["pi"].clone()
                if bool(cond["is_loco"][0].item()):
                    pi = torch.zeros((1,), dtype=torch.long, device=self.lingo_cfg.device)
                    direction_index = int(0.3 / self.scene.grid_map.grid_step) + 1
                    direction_index = min(direction_index, len(character.grid_path) - 1)
                    if character.grid_path and character.grid_path[0] == character.grid_path[direction_index]:
                        for i in range(direction_index, len(character.grid_path)):
                            if character.grid_path[i] != character.grid_path[0]:
                                direction_index = i
                                break
                    pelvis_goal = self.scene.grid_map.grid_points[character.grid_path[direction_index]]
                    pelvis_goal = (
                        torch.from_numpy(self.scene.grid_map.event_to_lingo(pelvis_goal))
                        .to(self.device)
                        .reshape(1, 1, 3)
                    )
                    pelvis_goal = transform_points(pelvis_goal, torch.inverse(request["mat"]))
                    pelvis_goal_norm = torch.norm(pelvis_goal, dim=-1, keepdim=True)[0, 0, 0]
                    if pelvis_goal_norm >= self.lingo_cfg.speed:
                        pelvis_goal = pelvis_goal / pelvis_goal_norm * self.lingo_cfg.speed
                    pelvis_goal_norm = torch.norm(pelvis_goal, dim=-1, keepdim=True)[0, 0, 0]
                    speed_factor = pelvis_goal[0, 0, 2] / pelvis_goal_norm
                    pelvis_goal = pelvis_goal * ((speed_factor + 1.0) / 4 + 0.5)
                else:
                    pelvis_goal = transform_points(cond["pelvis_goal"].reshape(1, 1, 3), torch.inverse(request["mat"]))
                pelvis_goal_items.append(pelvis_goal)
                pi_items.append(pi)

            pelvis_goal = torch.cat(pelvis_goal_items, dim=0)
            pi = torch.cat(pi_items, dim=0)
            is_pick = torch.cat([request["cond"]["is_pick"] for request in requests], dim=0)
            need_scene = torch.cat([request["cond"]["need_scene"] for request in requests], dim=0)
            need_pelvis_dir = torch.cat([request["cond"]["need_pelvis_dir"] for request in requests], dim=0)
            is_loco = torch.cat([request["cond"]["is_loco"] for request in requests], dim=0)
            need_pi = torch.cat([request["cond"]["need_pi"] for request in requests], dim=0)
            if not self.lingo_cfg.use_pi:
                need_pi = torch.zeros((batch_size,), dtype=torch.bool, device=self.lingo_cfg.device)
                pi = torch.zeros((batch_size,), dtype=torch.long, device=self.lingo_cfg.device)

            scene_flag = torch.zeros((batch_size,), dtype=torch.long, device=self.lingo_cfg.device)
            samples, occs, _ = self.sampler_body.p_sample_loop(
                fixed_points,
                mat,
                scene_flag,
                text_emb,
                pelvis_goal,
                hand_goal,
                is_pick,
                need_scene,
                need_pelvis_dir,
                pi,
                need_pi,
                is_loco,
            )
            points_gene = samples[-1]
            points_orig = transform_points(self.sampler_body.dataset.denormalize_torch(points_gene), mat)
            return {
                "points_orig": points_orig.reshape(
                    batch_size, self.lingo_cfg.max_window_size, 3 * self.lingo_cfg.dataset.nb_joints
                ),
                "occ": occs[-1] if occs else None,
            }
        finally:
            self.sampler_body.batch_size = original_batch_size
            if original_grid is not None:
                self.sampler_body.grid = original_grid

    def advance_motion_frame(self, character: Character) -> None:
        """
        place the next frame from the cache of the character
        """

        next_body_params = character.cache_frames[0].unsqueeze(0)
        character.cache_frames = character.cache_frames[1:]
        character.body_params = torch.cat([character.body_params, next_body_params], dim=0)
        advanced_joint_frame = False
        if character.cache_joints_event is not None and len(character.cache_joints_event) > 0:
            character.current_motion_joints_event = character.cache_joints_event[:1]
            character.cache_joints_event = character.cache_joints_event[1:]
            advanced_joint_frame = True

        # Update body_params_origin: use cache_origin_frames if available, otherwise same as body_params
        if character.cache_origin_frames is not None and character.cache_origin_frames.shape[0] > 0:
            next_origin_params = character.cache_origin_frames[0].unsqueeze(0)
            character.cache_origin_frames = character.cache_origin_frames[1:]
            if character.body_params_origin is None:
                character.body_params_origin = next_origin_params
            else:
                character.body_params_origin = torch.cat([character.body_params_origin, next_origin_params], dim=0)
        else:
            # No origin cache available, just mirror body_params
            if character.body_params_origin is None:
                character.body_params_origin = next_body_params.clone()
            else:
                character.body_params_origin = torch.cat(
                    [character.body_params_origin, next_body_params.clone()], dim=0
                )

        if character.points_orig is None or (
            hasattr(character.points_orig, "shape") and character.points_orig.shape[0] == 0
        ):
            # Use cache_points before the current frame self.lingo_cfg.auto_regre_num Use the current frame as the initial value
            if not advanced_joint_frame:
                self._set_current_motion_joints_event(
                    character, character.cache_points[:, self.lingo_cfg.auto_regre_num]
                )
            character.points_orig = character.cache_points[:, self.lingo_cfg.auto_regre_num].unsqueeze(0)
            # character.cache_points = character.cache_points[:,1:]
        else:
            # Since interp_s frames are automatically generated, it is necessary to check if the value is a multiple of interp_s
            if character.num_played_frames % self.lingo_cfg.interp_s == 0:
                if not advanced_joint_frame:
                    self._set_current_motion_joints_event(character, character.cache_points[:, 0])
                character.points_orig = torch.cat(
                    [character.points_orig, character.cache_points[:, 0].unsqueeze(0)], dim=1
                )
                character.cache_points = character.cache_points[:, 1:]

        # 6) (Optional) Early stop (scheduler) in non-loco segments
        # if self.scheduler is not None and not cond["is_loco"][0]:
        #     self._maybe_stop_with_scheduler(info, mat, cond)

        # 7) book-keeping like the matcher
        character.fId += 1
        character.num_played_frames += 1

        character.segment_finished = False
        if character.cache_frames.shape[0] == 0:
            character.segment_finished = character.segment_finished_in_buffer
            character.segment_finished_in_buffer = False

        # When the current segment ends, check the hand attachment
        # if character.segment_finished:
        #     self.check_and_update_attachments(character)
        # # Update the position of the attachment object
        # self._update_attached_objects(character)

    # generate the parallel motion for both characters
    def generate_parallel_motion(self, character: Character, interactee: Character):
        """Generate, align, and cache one pair without changing sampling order."""
        condA, condB, local_cache_A, local_cache_B = self._prepare_parallel_caches(character, interactee)
        points_A, points_B, origin_A, origin_B, middle = self._denoise_parallel_pair(
            local_cache_A, local_cache_B, condA, condB
        )
        points_all_A = self._parallel_points_tensor(points_A, middle)
        points_all_B = self._parallel_points_tensor(points_B, middle)

        character.segment_finished_in_buffer = True
        interactee.segment_finished_in_buffer = True
        character.cache_points = points_all_A
        interactee.cache_points = points_all_B
        frames_A, frames_B = self._pack_parallel_frames(points_all_A, points_all_B)
        character.cache_frames = frames_A
        interactee.cache_frames = frames_B
        character.cache_joints_event = self._make_cache_joints_event(
            character.cache_points, len(character.cache_frames)
        )
        interactee.cache_joints_event = self._make_cache_joints_event(
            interactee.cache_points, len(interactee.cache_frames)
        )

        origin_points_A = self._parallel_points_tensor(origin_A, middle)
        origin_points_B = self._parallel_points_tensor(origin_B, middle)
        origin_frames_A, origin_frames_B = self._pack_parallel_frames(origin_points_A, origin_points_B)
        character.cache_origin_frames = origin_frames_A
        interactee.cache_origin_frames = origin_frames_B
        character.num_played_frames = 0
        interactee.num_played_frames = 0
        character.step = 0
        interactee.step = 0

    def _prepare_parallel_caches(self, character, interactee):
        """Build guidance and equal-length intermediate caches in A/B order."""
        # get the guidance for both characters
        action_now = character.current_actions[0] if character.current_actions else "stand still"
        canonical = self._normalize_action_text(action_now)

        condA = self._get_guidance(character)
        character.current_generating_action = canonical

        action_now = interactee.current_actions[0] if interactee.current_actions else "stand still"
        canonical = self._normalize_action_text(action_now)

        condB = self._get_guidance(interactee)
        interactee.current_generating_action = canonical

        # get the middle points for both characters

        points_orig_A = character.points_orig
        local_cache_A = self.time_warp_module._middle_step_denoising(
            seg_len=50, points_orig=points_orig_A, trajectory=None, cond=condA
        )

        points_orig_B = interactee.points_orig
        local_cache_B = self.time_warp_module._middle_step_denoising(
            seg_len=50, points_orig=points_orig_B, trajectory=None, cond=condB
        )

        # generate motion for both characters to same length
        if local_cache_A.get_len() < local_cache_B.get_len():
            local_cache_A = self.time_warp_module._middle_step_denoising(
                seg_len=local_cache_B.get_len(),
                points_orig=points_orig_A,
                trajectory=None,
                cond=condA,
                full_seg_len=True,
            )
        elif local_cache_A.get_len() > local_cache_B.get_len():
            local_cache_B = self.time_warp_module._middle_step_denoising(
                seg_len=local_cache_A.get_len(),
                points_orig=points_orig_B,
                trajectory=None,
                cond=condB,
                full_seg_len=True,
            )

        return condA, condB, local_cache_A, local_cache_B

    def _denoise_parallel_pair(self, local_cache_A, local_cache_B, condA, condB):
        """Align wrist contact timing, denoise, and report the same metrics."""
        points_all_np_mid_A = self.time_warp_module._local_cache_to_points_mid(local_cache_A)
        points_all_np_mid_B = self.time_warp_module._local_cache_to_points_mid(local_cache_B)

        # find the best time warping for both characters
        # A: evaluate both wrists and choose better
        best_idx_A, best_dist_A = self._find_best_wrist_frame(
            points_all_np_mid_A, condA["hand_goal"], condA["raw_text"]
        )

        # B: evaluate both wrists and choose better
        best_idx_B, best_dist_B = self._find_best_wrist_frame(
            points_all_np_mid_B, condB["hand_goal"], condB["raw_text"]
        )

        middle = min((best_idx_A + best_idx_B) // 2, points_all_np_mid_A.shape[1] - 1, points_all_np_mid_B.shape[1] - 1)

        points_all_np_A, points_all_origin_np_A = self.time_warp_module._final_denoising(
            seg_len=10,
            local_cache=local_cache_A,
            trajectory=None,
            cond=condA,
            control_idx=best_idx_A,
            control_timing=middle - best_idx_A,
        )
        points_all_np_B, points_all_origin_np_B = self.time_warp_module._final_denoising(
            seg_len=10,
            local_cache=local_cache_B,
            trajectory=None,
            cond=condB,
            control_idx=best_idx_B,
            control_timing=middle - best_idx_B,
        )

        best_idx_A_final, best_dist_A_final = self._find_best_wrist_frame(
            points_all_np_A, condA["hand_goal"], condA["raw_text"]
        )
        best_idx_B_final, best_dist_B_final = self._find_best_wrist_frame(
            points_all_np_B, condB["hand_goal"], condB["raw_text"]
        )

        mapping_after_idx_A = self._find_dtw_metric(
            points_all_np_mid_A.squeeze(0).reshape(points_all_np_mid_A.shape[1], -1)[self.auto_regre :, :],
            points_all_np_A.squeeze(0).reshape(points_all_np_A.shape[1], -1),
            best_idx_A - 2,
        )
        mapping_after_idx_B = self._find_dtw_metric(
            points_all_np_mid_B.squeeze(0).reshape(points_all_np_mid_B.shape[1], -1)[self.auto_regre :, :],
            points_all_np_B.squeeze(0).reshape(points_all_np_B.shape[1], -1),
            best_idx_B - 2,
        )

        best_idx_after_warp, best_dist_after_warp = self._find_best_dual_wrist_frame(
            points_all_np_A, points_all_np_B, condA["hand_goal"], condA["raw_text"], condB["raw_text"]
        )
        best_idx_before_warp, best_dist_before_warp = self._find_best_dual_wrist_frame(
            points_all_origin_np_A, points_all_origin_np_B, condA["hand_goal"], condA["raw_text"], condB["raw_text"]
        )

        ParallelMotionDiagnostics(
            points_all_np_mid_A=points_all_np_mid_A,
            points_all_np_mid_B=points_all_np_mid_B,
            points_all_np_A=points_all_np_A,
            points_all_np_B=points_all_np_B,
            best_idx_A=best_idx_A,
            best_idx_B=best_idx_B,
            middle=middle,
            best_dist_A=best_dist_A,
            best_dist_B=best_dist_B,
            best_idx_A_final=best_idx_A_final,
            best_idx_B_final=best_idx_B_final,
            best_dist_A_final=best_dist_A_final,
            best_dist_B_final=best_dist_B_final,
            best_idx_before_warp=best_idx_before_warp,
            best_dist_before_warp=best_dist_before_warp,
            best_idx_after_warp=best_idx_after_warp,
            best_dist_after_warp=best_dist_after_warp,
            mapping_after_idx_A=mapping_after_idx_A,
            mapping_after_idx_B=mapping_after_idx_B,
            auto_regre=self.auto_regre,
        ).write()
        return points_all_np_A, points_all_np_B, points_all_origin_np_A, points_all_origin_np_B, middle

    def _parallel_points_tensor(self, points, frame_stop):
        """Slice a generated joint sequence and keep its existing dtype/device."""
        return torch.from_numpy(
            points[:, :frame_stop].reshape(self.lingo_cfg.batch_size, -1, self.lingo_cfg.dataset.nb_joints * 3)
        ).to(self.lingo_cfg.device)

    def _pack_parallel_frames(self, points_all_A, points_all_B):
        """Convert both joint sequences, then run SMPL conversion and packing in A/B order."""
        points_A = points_all_A[0]
        points_B = points_all_B[0]

        points_A = points_A.reshape(-1, self.lingo_cfg.dataset.nb_joints, 3)
        points_B = points_B.reshape(-1, self.lingo_cfg.dataset.nb_joints, 3)
        points_A = self.scene.grid_map.lingo_to_event(points_A)
        points_B = self.scene.grid_map.lingo_to_event(points_B)
        points_A = points_A.reshape(-1, self.lingo_cfg.dataset.nb_joints * 3)
        points_B = points_B.reshape(-1, self.lingo_cfg.dataset.nb_joints * 3)

        # 5) Convert joints->SMPL(X) and append
        pose_A, transl_A, _, _ = joints_to_smpl(
            self.model_joints_to_smplx, points_A, self.lingo_cfg.dataset.joints_ind, self.lingo_cfg.interp_s
        )
        pose_B, transl_B, _, _ = joints_to_smpl(
            self.model_joints_to_smplx, points_B, self.lingo_cfg.dataset.joints_ind, self.lingo_cfg.interp_s
        )

        next_segment_body_params_A = self._pack_body_params_91(transl_A, pose_A[:, :3], pose_A[:, 3:])
        next_segment_body_params_B = self._pack_body_params_91(transl_B, pose_B[:, :3], pose_B[:, 3:])

        return next_segment_body_params_A, next_segment_body_params_B

    # ---------------- internal API ----------------
    def _get_guidance(self, character: Character):
        cfg = self.lingo_cfg
        cond = {}

        raw = character.current_actions[0] if character.current_actions else "stand still"
        if character.state == State.IDLE:
            raw = idle_action_for(character)
        canonical = self._normalize_action_text(raw)
        # Normalize the text to fit the model vocabulary
        cond["raw_text"] = canonical
        if cond["raw_text"] == character.current_generating_action:
            cond["text_emb"] = character.text_emb.clone()
        else:
            cond["text_emb"] = get_clip_features(cond["raw_text"]).reshape(cfg.batch_size, 1, -1).to(cfg.device)
            character.text_emb = cond["text_emb"].clone()
        if "sit down" in cond["raw_text"] or "lie down" in cond["raw_text"]:
            cond["pelvis_goal"] = (
                self.scene.grid_map.event_to_lingo(character.target_root_position.float())
                .to(cfg.device)
                .reshape(cfg.batch_size, 1, 3)
            )
        else:
            cond["pelvis_goal"] = (
                self.scene.grid_map.event_to_lingo(character.target_position.float())
                .to(cfg.device)
                .reshape(cfg.batch_size, 1, 3)
            )
        cond["pelvis_goal"][..., 1] = 0.0

        # Use the new hand target position
        if character.target_hand_position is not None:
            cond["hand_goal"] = (
                self.scene.grid_map.event_to_lingo(character.target_hand_position.float())
                .to(cfg.device)
                .reshape(cfg.batch_size, 1, 3)
            )
        else:
            # If there is no hand target, use the default pelvic position
            cond["hand_goal"] = (
                self.scene.grid_map.event_to_lingo(character.target_root_position.float())
                .to(cfg.device)
                .reshape(cfg.batch_size, 1, 3)
            )

        # Set the hand interaction flag
        if self._contains_any_action(canonical, self.locomotion_actions):
            cond["is_pick"] = torch.zeros((cfg.batch_size, 1), dtype=torch.bool).to(cfg.device)
        elif self._contains_any_action(canonical, self.hand_object_actions_pick):
            cond["is_pick"] = torch.ones((cfg.batch_size, 1), dtype=torch.bool).to(cfg.device)
        elif self._contains_any_action(canonical, self.hand_object_actions_put):
            cond["is_pick"] = torch.ones((cfg.batch_size, 1), dtype=torch.bool).to(cfg.device)
        else:
            cond["is_pick"] = torch.zeros((cfg.batch_size, 1), dtype=torch.bool).to(cfg.device)

        if (
            self._contains_any_action(canonical, self.locomotion_actions)
            or self._contains_any_action(canonical, self.scene_interactions)
            or self._contains_any_action(canonical, {"type on", "write"})
        ):
            need_scene = torch.ones((cfg.batch_size,), dtype=torch.bool).to(cfg.device)
        else:
            need_scene = torch.zeros((cfg.batch_size,), dtype=torch.bool).to(cfg.device)
        # need_scene = torch.ones((cfg.batch_size, ), dtype=torch.bool).to(cfg.device)

        if (
            self._contains_any_action(canonical, self.locomotion_actions)
            or self._contains_any_action(canonical, {"sit down", "lie down"})
            or self._contains_any_action(canonical, self.orientation_actions)
        ):
            need_pelvis_dir = torch.ones((cfg.batch_size,), dtype=torch.bool).to(cfg.device)
        else:
            need_pelvis_dir = torch.zeros((cfg.batch_size,), dtype=torch.bool).to(cfg.device)

        cond["need_scene"] = need_scene
        cond["need_pelvis_dir"] = need_pelvis_dir
        is_loco_flag = self._contains_any_action(canonical, self.locomotion_actions) or canonical == "walk"
        cond["is_loco"] = (
            torch.ones((cfg.batch_size,), dtype=torch.bool).to(cfg.device)
            if is_loco_flag
            else torch.zeros((cfg.batch_size,), dtype=torch.bool).to(cfg.device)
        )
        cond["need_pi"] = (
            torch.zeros((cfg.batch_size,), dtype=torch.bool).to(cfg.device)
            if is_loco_flag
            else torch.ones((cfg.batch_size,), dtype=torch.bool).to(cfg.device)
        )

        cond["start_location"] = character.root_position[0]
        cond["start_location"][..., 1] = 0.0

        return cond

    # ---------------- helpers ----------------

    # Auxiliary function: Check if canonical contains any action types
    def _contains_any_action(self, canonical_text, action_set):
        return any(action in canonical_text for action in action_set)

    def _normalize_action_text(self, action_text: str) -> str:
        # Only perform case-insensitivity and space normalization; no synonym mapping is performed
        return (action_text or "").strip().lower()

    def _find_dtw_metric(self, ref, gen, ref_frame):  # ref is x_hat_predited, gen is generated dps.

        dist_metric = lambda x, y: np.linalg.norm(x - y, ord=2)
        dtw_ref = ref.copy()  # seq would be (T, 3*joints)
        dtw_gen = gen.copy()  # seq would be (T, 3*joints)
        dtw_obj = DTW(dist_metric)
        dtw_result = dtw_obj.compute_distance(dtw_ref, dtw_gen)
        target_mapped_frames = []
        for i, ref_idx in enumerate(dtw_result.index1):
            if ref_idx == ref_frame:
                target_mapped_frames.append(dtw_result.index2[i])

        if len(target_mapped_frames) > 0:
            target_frame = target_mapped_frames[0]
        else:
            return -1
            assert False, "target frame not found"  # or you can return the dist threshold frames

        return target_frame

    def _find_best_wrist_frame(self, points_all_np_mid: np.ndarray, hand_goal: torch.Tensor, raw_text: str) -> tuple:
        """
        Evaluate the left and right wrists(20:left_wrist, 21:right_wrist),
        If the raw_text contains"left hand",then only evaluate the left wrist,
        If the raw_text contains"right hand",then only evaluate the right wrist,
        Otherwise, evaluate both wrists and select the frame closest to hand_goal.
        Args:
            points_all_np_mid: (B, T, J, 3)
            hand_goal: (B,1,3) or (B,3) or (3,)
        Returns:
            (best_index:int, best_distance:float)
        """
        if isinstance(hand_goal, torch.Tensor):
            target_np = hand_goal.detach().cpu().numpy()
        else:
            target_np = np.asarray(hand_goal)

        if target_np.ndim == 1:
            target_np = target_np.reshape(1, 3)
        elif target_np.ndim == 3:
            target_np = target_np[:, 0, :]

        B, T, J, _ = points_all_np_mid.shape
        if "left hand" in raw_text:
            wrist_indices = [20]
        elif "right hand" in raw_text:
            wrist_indices = [21]
        else:
            wrist_indices = [20, 21]

        best_idx, best_dist = None, float("inf")
        for hj in wrist_indices:
            traj = points_all_np_mid[:, :, hj, :]  # (B,T,3)
            diffs = traj - target_np[:, None, :]
            dists = np.linalg.norm(diffs, axis=-1)  # (B,T)

            idxs = dists.argmin(axis=1)
            dmin = dists[np.arange(B), idxs]

            # Assume B==1 (current implementation batch=1), take the 0th one
            cand_idx = int(idxs[0]) if hasattr(idxs, "__len__") else int(idxs)
            cand_dist = float(dmin[0]) if hasattr(dmin, "__len__") else float(dmin)
            if cand_dist < best_dist:
                best_dist = cand_dist
                best_idx = cand_idx
                if best_dist < 0.05:
                    return best_idx, best_dist

        return best_idx, best_dist

    def _find_best_dual_wrist_frame(
        self,
        points_all_np_A: np.ndarray,
        points_all_np_B: np.ndarray,
        hand_goal: torch.Tensor,
        raw_text_A: str,
        raw_text_B: str,
    ) -> tuple:
        """
        Find the frame with the minimum sum of distances from the hand to the target for both A and B.
        For each frame, calculate the sum of the distances from A’s hand to the target and B’s hand to the target,
        Then find the minimum value among all frames; this is the index where both hands are closest to the hand target simultaneously.

        Args:
            points_all_np_A: (B, T, J, 3) - A’s action sequence
            points_all_np_B: (B, T, J, 3) - B’s action sequence
            hand_goal: (B,1,3) or (B,3) or (3,) - Hand target position
            raw_text_A: A’s action text description
            raw_text_B: B’s action text description

        Returns:
            (best_index:int, best_distance:float) - Optimal frame index and corresponding minimum distance
        """
        # Convert target position to numpy
        if isinstance(hand_goal, torch.Tensor):
            target_np = hand_goal.detach().cpu().numpy()
        else:
            target_np = np.asarray(hand_goal)

        if target_np.ndim == 1:
            target_np = target_np.reshape(1, 3)
        elif target_np.ndim == 3:
            target_np = target_np[:, 0, :]

        B, T_A, J, _ = points_all_np_A.shape
        _, T_B, _, _ = points_all_np_B.shape

        # Ensure both sequences have the same length
        T = min(T_A, T_B)

        # Determine A’s wrist index
        if "left hand" in raw_text_A:
            wrist_indices_A = [20]
        elif "right hand" in raw_text_A:
            wrist_indices_A = [21]
        else:
            wrist_indices_A = [20, 21]

        # Determine B’s wrist index
        if "left hand" in raw_text_B:
            wrist_indices_B = [20]
        elif "right hand" in raw_text_B:
            wrist_indices_B = [21]
        else:
            wrist_indices_B = [20, 21]

        best_idx, best_dist_sum = None, float("inf")

        # Iterate through all possible wrist combinations
        for hj_A in wrist_indices_A:
            for hj_B in wrist_indices_B:
                # Obtain A and B’s wrist trajectories
                traj_A = points_all_np_A[:, :T, hj_A, :]  # (B,T,3)
                traj_B = points_all_np_B[:, :T, hj_B, :]  # (B,T,3)

                # Calculate the distance from A’s wrist to the target
                diffs_A = traj_A - target_np[:, None, :]
                dists_A = np.linalg.norm(diffs_A, axis=-1)  # (B,T)

                # Calculate the distance from B’s wrist to the target
                diffs_B = traj_B - target_np[:, None, :]
                dists_B = np.linalg.norm(diffs_B, axis=-1)  # (B,T)

                # Calculate the sum of distances
                dists_sum = dists_A + dists_B  # (B,T)

                # Find the frame with the smallest distance sum
                idxs = dists_sum.argmin(axis=1)
                dmin_sum = dists_sum[np.arange(B), idxs]

                # Assume B==1 (current implementation batch=1), take the 0th one
                cand_idx = int(idxs[0]) if hasattr(idxs, "__len__") else int(idxs)
                cand_dist_sum = float(dmin_sum[0]) if hasattr(dmin_sum, "__len__") else float(dmin_sum)

                # Update the best result
                if cand_dist_sum < best_dist_sum:
                    best_dist_sum = cand_dist_sum
                    best_idx = cand_idx

        return best_idx, best_dist_sum

    # ---------------- public helper for LLM ----------------
    def get_candidate_actions(self):
        """
        Return all candidate actions (dedicated, sorted) for direct selection by the upstream LLM.
        """
        groups = [
            self.maintain_actions,
            self.locomotion_actions,
            self.orientation_actions,
            self.inplace_actions,
            self.hand_object_actions_pick,
            self.hand_object_actions_put,
            self.hand_object_actions_misc,
            self.scene_interactions,
        ]
        candidates = set()
        for g in groups:
            candidates.update(g)
        # Fault tolerance: Include common basic names
        candidates.update(["walk", "sit down", "lie down", "stand still"])
        return sorted(candidates)

    def _sample_step(self, cfg, character, mat, fixed_points, sampler, cond, pi):
        raw_text = cond["raw_text"]
        text_emb = cond["text_emb"]
        pelvis_goal = cond["pelvis_goal"]
        pelvis_goal = transform_points(pelvis_goal.reshape(1, 1, 3), torch.inverse(mat))  # convert to local coordinates
        hand_goal = cond["hand_goal"]
        hand_goal = transform_points(hand_goal.reshape(1, 1, 3), torch.inverse(mat))
        is_pick = cond["is_pick"]

        need_scene = cond["need_scene"]
        need_pelvis_dir = cond["need_pelvis_dir"]
        is_loco = cond["is_loco"]
        need_pi = cond["need_pi"]

        speed_new = None
        if is_loco:
            pi = torch.zeros((cfg.batch_size,), dtype=torch.long).to(cfg.device)

            # 0.3 m is the target position; calculate the index of the target position based on grid_step
            direction_index = int(0.3 / self.scene.grid_map.grid_step) + 1
            direction_index = min(direction_index, len(character.grid_path) - 1)
            # if the direction index is the same as the first one, set the direction index to the first moving one
            # print(f'character.grid_path: {character.grid_path}', 'direction_index: ', direction_index)
            if character.grid_path[0] == character.grid_path[direction_index]:
                for i in range(direction_index, len(character.grid_path)):
                    if character.grid_path[i] != character.grid_path[0]:
                        direction_index = i
                        break
            pelvis_goal = self.scene.grid_map.grid_points[character.grid_path[direction_index]]
            pelvis_goal = (
                torch.from_numpy(self.scene.grid_map.event_to_lingo(pelvis_goal))
                .to(self.device)
                .reshape(cfg.batch_size, 1, 3)
            )
            pelvis_goal = transform_points(pelvis_goal, torch.inverse(mat))  # convert to local coordinates
            pelvis_goal_norm = torch.norm(pelvis_goal, dim=-1, keepdim=True)[0, 0, 0]
            if pelvis_goal_norm >= cfg.speed:
                pelvis_goal = pelvis_goal / pelvis_goal_norm * cfg.speed

            pelvis_goal_norm = torch.norm(pelvis_goal, dim=-1, keepdim=True)[0, 0, 0]
            speed_factor = pelvis_goal[0, 0, 2] / pelvis_goal_norm
            speed_factor = (speed_factor + 1.0) / 4 + 0.5
            speed_new = speed_factor

            pelvis_goal = pelvis_goal * speed_new

        if not cfg.use_pi:
            need_pi = torch.zeros((cfg.batch_size,), dtype=torch.bool).to(cfg.device)
            pi = torch.zeros((cfg.batch_size,), dtype=torch.long).to(cfg.device)

        print(
            f"pelvis_goal: {pelvis_goal}",
            "pi: ",
            pi,
            "need_pi: ",
            need_pi,
            "need_scene: ",
            need_scene,
            "need_pelvis_dir: ",
            need_pelvis_dir,
            "raw_text: ",
            raw_text,
            "speed: ",
            speed_new,
        )

        # scene_flag = sampler.dataset.scene_dict[cond['scene_name']]
        # scene_flag = torch.tensor([scene_flag]*cfg.batch_size).to(cfg.device)
        scene_flag = torch.tensor([0] * cfg.batch_size).to(
            cfg.device
        )  # because we only have one scene, so scene flag is always the first one

        # for time warping
        # middle_steps = 30
        # samples, occs, x_0_preds = sampler.p_sample_loop_parallel(
        #     fixed_points, mat, scene_flag, text_emb, pelvis_goal, hand_goal, is_pick,
        #     need_scene, need_pelvis_dir, pi, need_pi, is_loco, middle_steps=middle_steps, noise=None
        # )
        # points_gene = samples[-1]
        # x_0_pred = x_0_preds[-1]
        # x_0_gen = transform_points(sampler.dataset.denormalize_torch(x_0_pred), mat)

        data = {}
        # data['phase'] = 1
        # data['speed_inter'] = 3
        # data['pi'] = pi
        # data['noise'] = points_gene
        # data['x_0_gen'] = x_0_gen

        # normal sampling
        samples, occs, _ = sampler.p_sample_loop(
            fixed_points,
            mat,
            scene_flag,
            text_emb,
            pelvis_goal,
            hand_goal,
            is_pick,
            need_scene,
            need_pelvis_dir,
            pi,
            need_pi,
            is_loco,
        )

        points_gene = samples[-1]
        points_orig = transform_points(sampler.dataset.denormalize_torch(points_gene), mat)

        info_dict = {
            "points_orig": points_orig.reshape(cfg.batch_size, cfg.max_window_size, 3 * cfg.dataset.nb_joints),
            "pelvis_goal": transform_points(pelvis_goal, mat).reshape(cfg.batch_size, 3),  # global coordinates
            "pi": pi,
            "need_pi": need_pi,
            "need_scene": need_scene,
            "need_pelvis_dir": need_pelvis_dir,
            "raw_text": raw_text,
            "speed": speed_new,
            "scene_flag": scene_flag,
            "hand_goal": hand_goal,
            "is_pick": is_pick,
            "occ": occs[-1],
            "data": data,
        }

        return info_dict

    def _pack_body_params_91(self, transl, global_orient_aa, body_pose_aa, num_joints_expected=21):
        """
        transl:            (B, 3)
        global_orient_aa:  (B, 3)           # Axis of the root-angle
        body_pose_aa:      (B, J*3)         # Axis of the remaining J joints-angle Flatten
        num_joints_expected: Expected number of joints (default 21),Corresponding 84-dimensional quaternions)

        return:
            body_params: (B, 91) = [transl(3), root_quat(4), joints_quat(84)]
        """
        transl = torch.from_numpy(transl).to(self.device).float()
        global_orient_aa = torch.from_numpy(global_orient_aa).to(self.device).float()
        body_pose_aa = torch.from_numpy(body_pose_aa).to(self.device).float()
        B = transl.shape[0]
        # 1) Root: axis-angle -> quat (B, 4)
        root_quat = axis_angle_to_quaternion(global_orient_aa)  # Default return (w, x, y, z)

        # 2) Remaining joints: reshape into (B, J, 3) Then convert to quat → (B, J, 4)
        J = body_pose_aa.shape[1] // 3
        assert (
            J == num_joints_expected
        ), f"Expected {num_joints_expected} joints, but received {J}. Please check the number of joints/mapping."
        # NumPy pose[:, 3:] keeps its row stride on CPU; reshape copies only when needed.
        body_pose_aa = body_pose_aa.reshape(B, J, 3)
        joints_quat = axis_angle_to_quaternion(body_pose_aa.reshape(-1, 3)).reshape(B, J, 4)  # (B,J,4)

        # 3) Assemble into (B, 91)
        body_params = torch.cat(
            [transl, root_quat, joints_quat.reshape(B, -1)], dim=1  # (B,3)  # (B,4)  # (B, J*4) Expected J=21 → 84
        )
        return body_params

    def _set_current_motion_joints_event(self, character: Character, points_lingo: torch.Tensor) -> None:
        if character is None or points_lingo is None:
            return
        points = points_lingo
        if points.ndim == 3:
            points = points[0, 0] if points.shape[1] == 1 else points[0]
        if points.ndim == 2 and points.shape[-1] == self.lingo_cfg.dataset.nb_joints * 3:
            points = points[0]
        if points.ndim == 1:
            points = points.reshape(self.lingo_cfg.dataset.nb_joints, 3)
        if points.ndim != 2 or points.shape[-2:] != (self.lingo_cfg.dataset.nb_joints, 3):
            return
        character.current_motion_joints_event = self.scene.grid_map.lingo_to_event(points).reshape(
            1,
            self.lingo_cfg.dataset.nb_joints,
            3,
        )

    def _make_cache_joints_event(
        self, points_lingo: torch.Tensor, frame_count: int | None = None
    ) -> torch.Tensor | None:
        if points_lingo is None:
            return None
        points = points_lingo
        if points.ndim == 3:
            points = points[0]
        if points.ndim == 2 and points.shape[-1] == self.lingo_cfg.dataset.nb_joints * 3:
            pass
        elif points.ndim == 3 and points.shape[-2:] == (self.lingo_cfg.dataset.nb_joints, 3):
            points = points.reshape(points.shape[0], self.lingo_cfg.dataset.nb_joints * 3)
        else:
            return None

        if frame_count is not None and points.shape[0] != frame_count:
            if frame_count == points.shape[0] * self.lingo_cfg.interp_s:
                points = interpolate_joints(points, self.lingo_cfg.interp_s)
            else:
                points = interpolate_joints(points, max(1, int(round(frame_count / points.shape[0]))))
                if points.shape[0] < frame_count:
                    points = torch.cat([points, points[-1:].expand(frame_count - points.shape[0], -1)], dim=0)
                points = points[:frame_count]

        points = points.reshape(-1, self.lingo_cfg.dataset.nb_joints, 3)
        return self.scene.grid_map.lingo_to_event(points)

    def _create_initial_body_params(self, position, character):
        position = torch.from_numpy(self.scene.grid_map.event_to_lingo(position)).to(self.device).float()
        stand_start_idx_list = [np.random.randint(5, 201)]
        joints, mat, _, _, _, _, _, _, _, _, _, _ = self.sampler_body.dataset.__getitem__(stand_start_idx_list[0])
        joints = torch.from_numpy(joints).float().reshape(1, -1, self.lingo_cfg.dataset.nb_joints * 3)
        mat = torch.from_numpy(mat).float().reshape(1, 4, 4)

        points, mat = joints.to(self.device), mat.to(self.device)
        points_orig = self.sampler_body.dataset.denormalize_torch(points)  # (batch_size, max_window_size, nb_joints*3)

        theta = np.pi / 2
        rot_matrix = R.from_euler("y", theta).as_matrix()

        assert self.lingo_cfg.batch_size == 1
        mat[0, :3, :3] = torch.from_numpy(rot_matrix).to(self.device).float()
        points_orig = (
            points_orig.reshape(
                self.lingo_cfg.batch_size, self.lingo_cfg.max_window_size, self.lingo_cfg.dataset.nb_joints, 3
            )
            @ mat[0, :3, :3].t()
        )
        points_orig = points_orig.reshape(
            self.lingo_cfg.batch_size, self.lingo_cfg.max_window_size, self.lingo_cfg.dataset.nb_joints * 3
        )

        translation_shift = points_orig[:, [-self.lingo_cfg.auto_regre_num], :3] - position
        translation_shift[0, 0, 1] = 0.0
        points_orig = points_orig.reshape(self.lingo_cfg.batch_size, -1, self.lingo_cfg.dataset.nb_joints, 3)
        points_orig[:, :, :] -= translation_shift

        points_orig = points_orig.reshape(self.lingo_cfg.batch_size, -1, 3 * self.lingo_cfg.dataset.nb_joints)

        character.points_orig = points_orig
        # character.cache_points = points_orig

        # transfer to event coordinate
        points_orig = points_orig[0]
        points_orig = points_orig.reshape(-1, self.lingo_cfg.dataset.nb_joints, 3)
        points_orig = self.scene.grid_map.lingo_to_event(points_orig)
        if character is not None:
            character.current_motion_joints_event = points_orig[-1:]
        points_orig = points_orig.reshape(-1, self.lingo_cfg.dataset.nb_joints * 3)

        pose, transl, _, _ = joints_to_smpl(
            self.model_joints_to_smplx, points_orig, self.lingo_cfg.dataset.joints_ind, self.lingo_cfg.interp_s
        )
        # Display the history endpoint from which sampling continues. Keep the full
        # history and fitting call unchanged, including their existing random draws.
        body_params = self._pack_body_params_91(transl, pose[:, :3], pose[:, 3:])[-3:]
        return body_params

    def _update_attached_objects(self, character: Character):
        """
        Update the position of the attached object
        Args:
            character: Characters
        """
        for object_name in character.attached_objects:
            # Find the target object
            target_object = None
            for obj in self.scene.objects:
                if f"{obj.label}_{obj.idx}" == object_name:
                    target_object = obj
                    break

            if target_object is None or not target_object.is_attached:
                continue

            # Update the object position
            if character.target_hand_position is not None and target_object.attachment_offset is not None:
                hand_pos = character.target_hand_position.squeeze().cpu().numpy()
                new_center = hand_pos - target_object.attachment_offset

                # Use the move_object method to move the object, ensuring consistent scene state
                self.scene.move_object(
                    object_name=object_name,
                    new_center=new_center,
                    new_orientation=target_object.orientation,  # Maintain the original direction
                    move_mesh=True,  # Synchronize the movement of geometry
                    update_scene_graph=True,  # Update the scene diagram
                    update_grid_map=False,  # Do not update the grid map (performance considerations)
                )

    def _attach_object_to_character(self, character: Character, object_name: str):
        """
        Attach objects to characters
        Args:
            character: Characters
            object_name: Object name
        """
        # Find the target object
        target_object = None
        for obj in self.scene.objects:
            if f"{obj.label}_{obj.idx}" == object_name:
                target_object = obj
                break

        if target_object is None:
            return

        # Set the object attachment state
        target_object.attached_to = character.name
        target_object.is_attached = True

        # Calculate the attachment offset (relative to the character’s hand)
        if character.target_hand_position is not None:
            hand_pos = character.target_hand_position.squeeze().cpu().numpy()
            object_center = target_object.center
            target_object.attachment_offset = hand_pos - object_center

        # Add objects to the character’s attachment list
        if object_name not in character.attached_objects:
            character.attached_objects.append(object_name)

    def _detach_object_from_character(self, character: Character, object_name: str):
        """
        Separate objects from the character
        Args:
            character: Characters
            object_name: Object name
        """
        # Find the target object
        target_object = None
        for obj in self.scene.objects:
            if f"{obj.label}_{obj.idx}" == object_name:
                target_object = obj
                break

        if target_object is None:
            return

        # Clear the object attachment state
        target_object.attached_to = None
        target_object.is_attached = False
        target_object.attachment_offset = None

        # Remove objects from the character’s attachment list
        if object_name in character.attached_objects:
            character.attached_objects.remove(object_name)

    def check_and_update_attachments(self, character: Character):
        """
        Check if the hand is close to the object; if so, perform attachment
        Check if the hand is close to the target position; if so, perform placement
        Args:
            character: Characters
        """
        # Process the attachment intention
        if character.attachment_intent is not None and not character.attachment_intent.get("attached", False):
            self._check_attachment_intent(character)

        # Process the placement intention
        if character.placement_intent is not None and not character.placement_intent.get("placed", False):
            self._check_placement_intent(character)

    def _check_attachment_intent(self, character: Character):
        """
        Check the attachment intention
        Args:
            character: Characters
        """
        intent = character.attachment_intent
        object_name = intent["object"]

        # Find the target object
        target_object = None
        for obj in self.scene.objects:
            if f"{obj.label}_{obj.idx}" == object_name:
                target_object = obj
                break

        if target_object is None:
            return

        self._attach_object_to_character(character, object_name)
        # Update the attachment intention status
        character.attachment_intent["attached"] = True

    def _check_placement_intent(self, character: Character):
        """
        Check the placement intention
        Args:
            character: Characters
        """
        intent = character.placement_intent
        target_position = intent["target_position"]

        if target_position is None:
            return

        self._place_attached_objects(character, target_position)
        # Update the placement intention status
        character.placement_intent["placed"] = True

    def _place_attached_objects(self, character: Character, target_position: torch.Tensor):
        """
        Place the attached object at the target position
        Args:
            character: Characters
            target_position: target positions (1, 3)
        """
        for object_name in character.attached_objects:
            # Find the target object
            target_object = None
            for obj in self.scene.objects:
                if f"{obj.label}_{obj.idx}" == object_name:
                    target_object = obj
                    break

            if target_object is None or not target_object.is_attached:
                continue

            # Place the object at the target position
            target_pos = target_position.squeeze().cpu().numpy()
            target_object.center = target_pos.copy()

            # Update the bounding box
            target_object.bbox = np.array(
                [
                    [
                        target_pos[0] - target_object.width / 2,
                        target_pos[1] - target_object.depth / 2,
                        target_pos[2] - target_object.height / 2,
                    ],
                    [
                        target_pos[0] + target_object.width / 2,
                        target_pos[1] + target_object.depth / 2,
                        target_pos[2] + target_object.height / 2,
                    ],
                ]
            )

            # Separate the objects
            self._detach_object_from_character(character, object_name)
