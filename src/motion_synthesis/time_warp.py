import numpy as np
import torch
from omegaconf import DictConfig
import hydra
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm

from src.motion_synthesis.lingo_model_loading import init_model
from src.motion_synthesis.lingo_math import transform_points
from torch.utils.data import Dataset
from src.motion_synthesis.lingo_geometry import get_mat

# this is for cubic spline interpolation
from scipy.interpolate import CubicSpline

# this class if used for parallel sampling for diffusion model
from src.motion_synthesis.sampling_cache import diffusion_sampling_cache


# basic class for time warp module
class TimeWarpModule:

    def __init__(
        self,
        cfg: DictConfig,
        dataset: Dataset = None,
        sampler_body=None,
        model_body=None,
        model_joints_to_smplx=None,
        scheduler_model=None,
    ) -> None:
        self.cfg = cfg
        self.device = cfg.device

        self.model_joints_to_smplx = model_joints_to_smplx or init_model(
            cfg.model.model_smplx, device=self.device, eval=True
        )
        print("model_joints_to_smplx device: ", next(self.model_joints_to_smplx.parameters()).device)
        self.model_body = model_body or init_model(cfg.model.synhsi_body, device=self.device, eval=True)

        self.synhsi_dataset = dataset or hydra.utils.instantiate(cfg.dataset)
        self.sampler_body = sampler_body or hydra.utils.instantiate(cfg.sampler.pelvis)
        self.sampler_body.set_dataset_and_model(self.synhsi_dataset, self.model_body)

        if scheduler_model is not None:
            self.scheduler_model = scheduler_model
        elif cfg.use_scheduler:
            self.scheduler_model = hydra.utils.instantiate(cfg.model.scheduler)
            self.scheduler_model.load_state_dict(torch.load(cfg.scheduler_model_path, map_location=self.device))
            self.scheduler_model.to(self.device)
            self.scheduler_model.eval()
        else:
            self.scheduler_model = None

    get_mat = staticmethod(get_mat)

    @staticmethod
    def sample_step(cfg, step, mat, fixed_points, sampler, cond, trajectory, pi, is_middle, noise=None):
        raw_text = cond["raw_text"]
        text_emb = cond["text_emb"]
        pelvis_goal = cond["pelvis_goal"]
        pelvis_goal = transform_points(pelvis_goal.reshape(1, 1, 3), torch.inverse(mat))
        hand_goal = cond["hand_goal"]
        hand_goal = transform_points(hand_goal.reshape(1, 1, 3), torch.inverse(mat))
        is_pick = cond["is_pick"]

        need_scene = cond["need_scene"]
        need_pelvis_dir = cond["need_pelvis_dir"]
        is_loco = cond["is_loco"]
        need_pi = cond["need_pi"]

        speed_new = None
        if is_loco:
            raise NotImplementedError("Locomotion is not implemented for time warp module")
            pi = torch.zeros((cfg.batch_size,), dtype=torch.long).to(cfg.device)

            curr_loc = mat[0, :3, 3].cpu().numpy()
            curr_loc = np.array([curr_loc[0], curr_loc[2]]).reshape(1, 2)

            dist = np.linalg.norm(curr_loc - trajectory, axis=1)
            min_idx = np.argmin(dist)
            pelvis_goal = (
                torch.tensor(
                    [
                        trajectory[min(min_idx + 50, len(trajectory) - 1)][0],
                        0,
                        trajectory[min(min_idx + 50, len(trajectory) - 1)][1],
                    ]
                )
                .reshape(1, 1, 3)
                .to(cfg.device)
                .float()
            )

            pelvis_goal = transform_points(pelvis_goal, torch.inverse(mat))

            pelvis_goal_norm = torch.norm(pelvis_goal, dim=-1, keepdim=True)[0, 0, 0]
            if pelvis_goal_norm >= cfg.speed:
                pelvis_goal = pelvis_goal / pelvis_goal_norm * cfg.speed

            theta_list = [[np.pi / 18 * i, -np.pi / 18 * i] for i in range(18)]
            theta_list = np.array(theta_list).reshape(-1)

            for theta in theta_list:
                goal_points = pelvis_goal.reshape(1, 3).repeat(9, 1)
                goal_points[:, 1] += torch.arange(0.1, 1, 0.1).to(cfg.device).reshape(-1)
                goal_points = goal_points.squeeze(0).repeat(20, 1, 1)
                goal_points[:, :, [0, 2]] *= torch.arange(0, 1, 0.05).to(cfg.device).reshape(-1, 1, 1)

                goal_points = transform_points(goal_points, mat)
                goal_occ = sampler.dataset.get_occ_for_points(goal_points, [0])

                if goal_occ.sum() < 5:
                    break

                rotation_matrix = R.from_euler("y", theta).as_matrix()
                rotation_matrix = torch.from_numpy(rotation_matrix).to(cfg.device).float()

                pelvis_goal = pelvis_goal.reshape(1, 3) @ rotation_matrix
                pelvis_goal = pelvis_goal.reshape(1, 1, 3)

            pelvis_goal_norm = torch.norm(pelvis_goal, dim=-1, keepdim=True)[0, 0, 0]
            speed_factor = pelvis_goal[0, 0, 2] / pelvis_goal_norm
            speed_factor = (speed_factor + 1.0) / 4 + 0.5
            speed_new = speed_factor

            pelvis_goal = pelvis_goal * speed_new

        if not cfg.use_pi:
            need_pi = torch.zeros((cfg.batch_size,), dtype=torch.bool).to(cfg.device)
            pi = torch.zeros((cfg.batch_size,), dtype=torch.long).to(cfg.device)

        scene_flag = torch.tensor([0] * cfg.batch_size).to(
            cfg.device
        )  # because we only have one scene, so scene flag is always the first one

        middle_steps = 70
        samples, occs, x_0_preds = sampler.p_sample_loop_parallel(
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
            middle_steps=middle_steps,
            noise=noise,
        )

        points_gene = samples[-1]
        points_orig_out = transform_points(sampler.dataset.denormalize_torch(points_gene), mat)

        x_0_pred = x_0_preds[-1]
        x_0_gen = transform_points(sampler.dataset.denormalize_torch(x_0_pred), mat)

        info_dict = {
            "points_orig": points_orig_out.reshape(cfg.batch_size, cfg.max_window_size, 3 * cfg.dataset.nb_joints),
            "pelvis_goal": transform_points(pelvis_goal, mat).reshape(cfg.batch_size, 3),
            "x_0_gen": x_0_gen,
            "pi": pi,
            "noise": points_gene,
            "need_pi": need_pi,
            "need_scene": need_scene,
            "need_pelvis_dir": need_pelvis_dir,
            "raw_text": raw_text,
            "speed": speed_new,
            "scene_flag": scene_flag,
            "hand_goal": hand_goal,
            "is_pick": is_pick,
            "occ": occs[-1],
        }
        return info_dict

    def _single_event_motion_sampling(self, seg_len: int, points_orig: torch.Tensor, trajectory, cond):
        # Entrance: Call 1 in sequence) Intermediate denoising step, and 2) Final denoising (no time warping)
        local_cache = self._middle_step_denoising(seg_len, points_orig, trajectory, cond)
        # Preserve intermediate conversion and its lifetime before final denoising.
        _points_all_np_mid = self._local_cache_to_points_mid(local_cache)
        points_all_np, points_all_origin_np = self._final_denoising(seg_len, local_cache, trajectory, cond)
        return points_all_np, points_all_origin_np

    def _middle_step_denoising(
        self, seg_len: int, points_orig: torch.Tensor, trajectory, cond, full_seg_len: bool = False
    ):
        # 1) Intermediate denoising, caching x_0 prediction
        local_cache = diffusion_sampling_cache()
        x_0_pred_list = []
        for step in tqdm(range(seg_len)):
            data = {}
            if step == 0:
                data["mat"] = TimeWarpModule.get_mat(self.cfg, points_orig)
                fixed_points_denormalized = points_orig[:, -self.cfg.auto_regre_num :].reshape(
                    self.cfg.batch_size, self.cfg.auto_regre_num, self.cfg.dataset.nb_joints * 3
                )
                data["fixed_points"] = self.sampler_body.dataset.normalize_torch(
                    transform_points(fixed_points_denormalized, torch.inverse(data["mat"]))
                )
            else:
                prev_data = local_cache.get_sampling_information(step - 1)
                data["mat"] = TimeWarpModule.get_mat(self.cfg, prev_data["x_0_gen"])
                fixed_points_denormalized = prev_data["x_0_gen"][:, -self.cfg.auto_regre_num :].reshape(
                    self.cfg.batch_size, self.cfg.auto_regre_num, self.cfg.dataset.nb_joints * 3
                )
                data["fixed_points"] = self.sampler_body.dataset.normalize_torch(
                    transform_points(fixed_points_denormalized, torch.inverse(data["mat"]))
                )

            data["phase"] = 1
            data["speed_inter"] = 3
            data["pi"] = torch.tensor(
                [
                    int(
                        (step + data["phase"])
                        * (self.cfg.max_window_size - self.cfg.auto_regre_num)
                        * data["speed_inter"]
                    )
                ]
            ).to(device=self.cfg.device, dtype=torch.long)

            info_dict = TimeWarpModule.sample_step(
                self.cfg,
                step,
                data["mat"],
                data["fixed_points"],
                self.sampler_body,
                cond,
                trajectory,
                data["pi"],
                is_middle=True,
            )
            data["noise"] = info_dict["noise"]
            data["x_0_gen"] = info_dict["x_0_gen"]

            local_cache.store_sampling_information(step, data)
            points_orig = data["x_0_gen"]

            # Recording intermediate x_0 prediction (aligned with LINGO behavior)
            x_0_pred_list.append(data["x_0_gen"].cpu().numpy()[:, self.cfg.auto_regre_num :])

            # Optional: scheduler and maneuver termination logic (consistent with LINGO; skip if scheduler is not enabled)
            if not full_seg_len and self.cfg.use_scheduler and not cond["is_loco"] and self.scheduler_model is not None:
                points_loco = self.sampler_body.dataset.normalize_torch(
                    transform_points(points_orig, torch.inverse(data["mat"]))
                )
                stop_pred = self.scheduler_model(points_loco, cond["text_emb"], data["pi"]).squeeze(1)
                stop_pred = torch.sigmoid(stop_pred)
                if stop_pred > self.cfg.scheduler_threshold:
                    break

            if not full_seg_len and cond["is_loco"]:
                curr_loc = data["x_0_gen"][0, -1, :3].cpu().numpy().copy()
                curr_loc[1] = 0.0
                end_point = cond["pelvis_goal"].cpu().numpy().copy()
                dist2end = np.linalg.norm(curr_loc - end_point)
                if dist2end < self.cfg.locomotion_threshold:
                    break

        return local_cache

    def _final_denoising(
        self,
        seg_len: int,
        local_cache: diffusion_sampling_cache,
        trajectory,
        cond,
        control_idx: int = None,
        control_timing: int = None,
    ):
        # 2) Final denoising (no time warping, aligned with the original _final_denoising without warping)
        points_all = []
        points_all_origin = []
        # Prepare optional time stretching targets
        target_warping_np = None
        if control_idx is not None and control_timing is not None:
            mid_seq = self._local_cache_to_points_mid(local_cache)  # (B,T,J,3)
            mid_flat = mid_seq.reshape(self.cfg.batch_size, -1, self.cfg.dataset.nb_joints * 3)
            target_warping_np = self._time_warping_function(
                mid_flat, control_idx=control_idx, control_timing=control_timing
            )
        for step in tqdm(range(local_cache.get_len())):
            data = local_cache.get_sampling_information(step)

            if step > 0:
                data_prev = local_cache.get_sampling_information(step - 1)

                data["mat"] = TimeWarpModule.get_mat(self.cfg, data_prev["final_points"])
                fixed_points_denormalized = data_prev["final_points"][:, -self.cfg.auto_regre_num :].reshape(
                    self.cfg.batch_size, self.cfg.auto_regre_num, self.cfg.dataset.nb_joints * 3
                )

                data["mat_for_compare"] = TimeWarpModule.get_mat(self.cfg, data_prev["final_points_origin"])
                fixed_points_denormalized_for_compare = data_prev["final_points_origin"][
                    :, -self.cfg.auto_regre_num :
                ].reshape(self.cfg.batch_size, self.cfg.auto_regre_num, self.cfg.dataset.nb_joints * 3)

                data["fixed_points"] = self.sampler_body.dataset.normalize_torch(
                    transform_points(fixed_points_denormalized, torch.inverse(data["mat"]))
                )
                data["fixed_points_for_compare"] = self.sampler_body.dataset.normalize_torch(
                    transform_points(fixed_points_denormalized_for_compare, torch.inverse(data["mat_for_compare"]))
                )
            else:
                # First step follows the intermediate steps generated mat/fixed_points
                if "mat" not in data:
                    data["mat"] = TimeWarpModule.get_mat(self.cfg, data["x_0_gen"])
                if "fixed_points" not in data:
                    fp_denorm = data["x_0_gen"][:, -self.cfg.auto_regre_num :].reshape(
                        self.cfg.batch_size, self.cfg.auto_regre_num, self.cfg.dataset.nb_joints * 3
                    )
                    data["fixed_points"] = self.sampler_body.dataset.normalize_torch(
                        transform_points(fp_denorm, torch.inverse(data["mat"]))
                    )
                data["mat_for_compare"] = data["mat"]
                data["fixed_points_for_compare"] = data["fixed_points"]

            # Origin (for comparison, no warping)
            info_dict_origin = TimeWarpModule.sample_step(
                self.cfg,
                step,
                data["mat_for_compare"],
                data["fixed_points_for_compare"],
                self.sampler_body,
                cond,
                trajectory,
                data["pi"],
                is_middle=False,
                noise=data["noise"],
            )
            torch.cuda.empty_cache()
            # warping or non-warping Final result
            if target_warping_np is not None:
                # 16-window, 14-step slice size (aligned with the sampler window)
                if step == 0:
                    tw_slice = target_warping_np[:, 0:16, :].copy()
                else:
                    start = 14 * step
                    tw_slice = target_warping_np[:, start : start + 16, :].copy()
                tw_tensor = torch.from_numpy(tw_slice).to(self.cfg.device)
                tw_tensor_local = self.sampler_body.dataset.normalize_torch(
                    transform_points(tw_tensor.float(), torch.inverse(data["mat"]))
                )
                info_dict = TimeWarpModule.sample_step_for_time_warping(
                    self.cfg,
                    step,
                    data["mat"],
                    data["fixed_points"],
                    self.sampler_body,
                    cond,
                    trajectory,
                    data["pi"],
                    is_middle=False,
                    noise=data["noise"],
                    target_warping=tw_tensor_local,
                )
            else:
                info_dict = TimeWarpModule.sample_step(
                    self.cfg,
                    step,
                    data["mat"],
                    data["fixed_points"],
                    self.sampler_body,
                    cond,
                    trajectory,
                    data["pi"],
                    is_middle=False,
                    noise=data["noise"],
                )
            torch.cuda.empty_cache()

            points = info_dict["points_orig"]
            points_origin = info_dict_origin["points_orig"]
            data["final_points"] = points
            data["final_points_origin"] = points_origin

            # Assemble output segments (aligned with update_best_matching_frame behavior)
            points_all.append(points.cpu().numpy()[:, self.cfg.auto_regre_num :])
            points_all_origin.append(points_origin.cpu().numpy()[:, self.cfg.auto_regre_num :])

        points_all_origin_np = np.concatenate(points_all_origin, axis=1).reshape(
            self.cfg.batch_size, -1, self.cfg.dataset.nb_joints, 3
        )
        points_all_np = np.concatenate(points_all, axis=1).reshape(
            self.cfg.batch_size, -1, self.cfg.dataset.nb_joints, 3
        )

        return points_all_np, points_all_origin_np

    # this function is used for time warping (cubic spline interpolation)
    def _time_warping_function(self, time_warping_before: np.ndarray, control_idx: int = 0, control_timing: int = 0):
        # time_warping_before: (B, T, D) where D = J*3
        source_idx = control_idx
        target_idx = control_idx + control_timing
        n_frame = time_warping_before.shape[1]
        # clamp target
        target_idx = max(0, min(n_frame - 1, target_idx))

        original_time = np.arange(n_frame)
        source_points = [0, source_idx, n_frame - 1]
        target_points = [0, target_idx, n_frame - 1]
        cs = CubicSpline(target_points, source_points)
        t_prime = np.arange(n_frame)
        t_target_from_source = cs(t_prime)

        M_prime = np.zeros_like(time_warping_before)
        for i in range(time_warping_before.shape[2]):
            f_d = CubicSpline(original_time, time_warping_before[0, :, i])
            M_prime[0, :, i] = f_d(t_target_from_source)
        return M_prime

    @staticmethod
    def sample_step_for_time_warping(
        cfg, step, mat, fixed_points, sampler, cond, trajectory, pi, is_middle, noise=None, target_warping=None
    ):
        raw_text = cond["raw_text"]
        text_emb = cond["text_emb"]
        pelvis_goal = cond["pelvis_goal"]
        pelvis_goal = transform_points(pelvis_goal.reshape(1, 1, 3), torch.inverse(mat))
        hand_goal = cond["hand_goal"]
        hand_goal = transform_points(hand_goal.reshape(1, 1, 3), torch.inverse(mat))
        is_pick = cond["is_pick"]

        need_scene = cond["need_scene"]
        need_pelvis_dir = cond["need_pelvis_dir"]
        is_loco = cond["is_loco"]
        need_pi = cond["need_pi"]

        if not cfg.use_pi:
            need_pi = torch.zeros((cfg.batch_size,), dtype=torch.bool).to(cfg.device)
            pi = torch.zeros((cfg.batch_size,), dtype=torch.long).to(cfg.device)

        scene_flag = torch.tensor([0] * cfg.batch_size).to(cfg.device)

        # parallel sampling (for noising we have to change)
        middle_steps = 20
        if middle_steps == 0:
            noise = None
            middle_steps = 100
        else:
            noise = sampler.add_middle_noise(noise, 30, 100 - middle_steps)

        if target_warping is not None and hasattr(sampler, "p_sample_loop_parallel_for_time_warping"):
            samples, occs, x_0_preds = sampler.p_sample_loop_parallel_for_time_warping(
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
                middle_steps=middle_steps,
                noise=noise,
                target_warping=target_warping,
            )
        else:
            samples, occs, x_0_preds = sampler.p_sample_loop_parallel(
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
                middle_steps=middle_steps,
                noise=noise,
            )

        points_gene = samples[-1]
        points_orig = transform_points(sampler.dataset.denormalize_torch(points_gene), mat)
        x_0_pred = x_0_preds[-1]
        x_0_gen = transform_points(sampler.dataset.denormalize_torch(x_0_pred), mat)

        info_dict = {
            "points_orig": points_orig.reshape(cfg.batch_size, cfg.max_window_size, 3 * cfg.dataset.nb_joints),
            "pelvis_goal": transform_points(pelvis_goal, mat).reshape(cfg.batch_size, 3),
            "x_0_gen": x_0_gen,
            "pi": pi,
            "noise": points_gene,
            "need_pi": need_pi,
            "need_scene": need_scene,
            "need_pelvis_dir": need_pelvis_dir,
            "raw_text": raw_text,
            "scene_flag": scene_flag,
            "hand_goal": hand_goal,
            "is_pick": is_pick,
            "occ": occs[-1],
        }
        return info_dict

    def _local_cache_to_points_mid(self, local_cache: diffusion_sampling_cache):
        # Concatenate the intermediate cached x_0_gen into a sequence of joint points for the intermediate stage, with a specified shape (B, T, J, 3)
        sequence_length = local_cache.get_len()
        middle_x0_list = []
        for i in range(sequence_length):
            if i == 0:
                part = local_cache.get_sampling_information(i)["x_0_gen"].detach().cpu().numpy()
            else:
                x0 = local_cache.get_sampling_information(i)["x_0_gen"].detach().cpu().numpy()
                part = x0[:, self.cfg.auto_regre_num :]
            middle_x0_list.append(part)

        if len(middle_x0_list) == 0:
            return np.zeros((self.cfg.batch_size, 0, self.cfg.dataset.nb_joints, 3), dtype=np.float32)

        mid_concat = np.concatenate(middle_x0_list, axis=1)
        points_all_np_mid = mid_concat.reshape(self.cfg.batch_size, -1, self.cfg.dataset.nb_joints, 3)
        return points_all_np_mid
