"""SyncMOS controls around the native external LINGO sampler.

The ordinary loop runs in the external checkout through a per-call trace adapter.
The other controls retain their existing implementations and mixed provenance.
"""

import torch
from tqdm import tqdm

from src.motion_synthesis.sampling_utils import debug_assert, extract_ver2


class _LoopTraceAdapter:
    """Expose LINGO's two-result step interface while retaining clean predictions.

    The external loop fixes each noisy sample after a step. Apply the corresponding
    clean-prediction constraint immediately afterwards, preserving the original
    call order. All trace state belongs to this invocation, not the sampler.
    """

    def __init__(self, sampler):
        self._sampler = sampler
        self.clean_predictions: list[torch.Tensor] = []
        self._pending_clean: torch.Tensor | None = None

    def __getattr__(self, name):
        return getattr(self._sampler, name)

    def p_sample(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor | None]:
        sample, occupancy, clean = self._sampler.p_sample(*args, **kwargs)
        self.clean_predictions.append(clean)
        self._pending_clean = clean
        return sample, occupancy

    def set_fixed_points(self, points, *args, **kwargs):
        result = self._sampler.set_fixed_points(points, *args, **kwargs)
        if self._pending_clean is not None:
            self._sampler.set_fixed_points(self._pending_clean, *args, **kwargs)
            self._pending_clean = None
        return result


class SamplingControls:
    """Add SyncMOS controls to the sampler created by create_external_sampler."""

    @torch.no_grad()
    def add_middle_noise(self, x_s, s_timesteps, t_timesteps, noise=None):
        """
        Applies the forward process q(x_t | x_s), adding noise to x_s at timestep s
        to obtain x_t at timestep t (where t > s).

        :param x_s: The current latent tensor at timestep s (e.g., x_50)
        :param s_timesteps: Timesteps tensor corresponding to x_s. Shape: [B,] (e.g., [50, 50, ...])
        :param t_timesteps: Target timesteps tensor (should be more noisy than s). Shape: [B,] (e.g., [70, 70, ...])
        :param noise: Noise tensor (epsilon) to add. If None, generates new noise.
        :return: The noisy latent tensor at t (x_t)
        """
        if noise is None:
            noise = torch.randn_like(x_s)

        # Convert scalars to tensors if needed
        # extract function requires tensors with batch dimension
        batch_size = x_s.shape[0]
        device = x_s.device

        if not isinstance(s_timesteps, torch.Tensor):
            s_timesteps = torch.full((batch_size,), s_timesteps, dtype=torch.long, device=device)
        elif s_timesteps.dim() == 0:
            s_timesteps = s_timesteps.expand(batch_size)

        if not isinstance(t_timesteps, torch.Tensor):
            t_timesteps = torch.full((batch_size,), t_timesteps, dtype=torch.long, device=device)
        elif t_timesteps.dim() == 0:
            t_timesteps = t_timesteps.expand(batch_size)

        # Check if t_timesteps > s_timesteps
        if torch.any(t_timesteps <= s_timesteps):
            raise ValueError("t has must be greater than s")

        alpha_bar_s = extract_ver2(self.alphas_cumprod, s_timesteps, x_s.shape)
        alpha_bar_t = extract_ver2(self.alphas_cumprod, t_timesteps, x_s.shape)

        # x_t = sqrt(alpha_bar_t / alpha_bar_s) * x_s + sqrt(1 - (alpha_bar_t / alpha_bar_s)) * epsilon

        alpha_ratio = alpha_bar_t / alpha_bar_s

        coeff_x_s = torch.sqrt(alpha_ratio)
        coeff_noise = torch.sqrt(1.0 - alpha_ratio)

        x_t = coeff_x_s * x_s + coeff_noise * noise

        return x_t

    @torch.no_grad()
    def p_sample_loop(
        self,
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
    ):
        """Return noisy, occupancy and clean traces without owning the DDPM loop.

        The factory binds ``_native_sample_loop`` to the external implementation. Its
        noise draws, timestep order and progress reporting remain under LINGO's control.
        """
        trace = _LoopTraceAdapter(self)
        samples, occupancies = self._native_sample_loop(
            trace,
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
        return samples, occupancies, trace.clean_predictions

    # for parallel sampling (DK)
    @torch.no_grad()
    def p_sample_loop_parallel(
        self,
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
        middle_steps=20,
        noise=None,
    ):
        device = next(self.model.parameters()).device
        shape = (self.batch_size, self.dataset.max_window_size, self.channel)

        debug_assert(middle_steps <= self.timesteps, "middle_steps should be less than or equal to timesteps")
        if noise is None:
            points = torch.randn(shape, device=device)
            start_step = self.timesteps
            end_step = self.timesteps - middle_steps
        else:
            debug_assert(
                noise.shape == (self.batch_size, self.dataset.max_window_size, self.channel),
                "noise shape should be (batch_size, max_window_size, channel)",
            )
            points = noise
            start_step = self.timesteps - middle_steps
            end_step = 0

        if self.auto_regre_num > 0:
            self.set_fixed_points(
                points, None, fixed_points, mat, joint_id=self.mask_ind, fix_mode=True, fix_goal=False
            )

        imgs = []
        occs = []
        x_0_preds = []

        desc_func = (
            (lambda: r"Fisrt step for finding \hat x_0") if noise is None else (lambda: "Final step for finding x_0")
        )
        for i in tqdm(reversed(range(end_step, start_step)), desc=desc_func(), total=self.timesteps):
            model_used = self.model

            # select sampling code
            points, occ, x_0_pred = self.p_sample(
                model_used,
                points,
                fixed_points,
                mat,
                scene_flag,
                torch.full((self.batch_size,), i, device=device, dtype=torch.long),
                i,
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

            # points, occ, x_0_pred = self.p_sample_dps(model_used, points, fixed_points, mat, scene_flag,
            #                            torch.full((self.batch_size,), i, device=device, dtype=torch.long), i,
            #                            text_emb, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir, pi, need_pi, is_loco
            #                            )
            if self.auto_regre_num > 0:
                self.set_fixed_points(
                    points, None, fixed_points, mat, joint_id=self.mask_ind, fix_mode=True, fix_goal=False
                )
                self.set_fixed_points(
                    x_0_pred, None, fixed_points, mat, joint_id=self.mask_ind, fix_mode=True, fix_goal=False
                )

            points_orig = points
            x_0_pred_orig = x_0_pred

            imgs.append(points_orig)
            x_0_preds.append(x_0_pred_orig)

            if occ is not None:
                occs.append(occ.cpu().numpy())

        return imgs, occs, x_0_preds

    # for time warping (DK)
    @torch.no_grad()
    def p_sample_loop_parallel_for_time_warping(
        self,
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
        middle_steps=20,
        noise=None,
        target_warping=None,
    ):
        device = next(self.model.parameters()).device
        shape = (self.batch_size, self.dataset.max_window_size, self.channel)

        debug_assert(target_warping is not None, "target_warping should be provided")
        debug_assert(middle_steps <= self.timesteps, "middle_steps should be less than or equal to timesteps")

        if noise is None:
            points = torch.randn(shape, device=device)
            start_step = self.timesteps
            end_step = self.timesteps - middle_steps
        else:
            debug_assert(
                noise.shape == (self.batch_size, self.dataset.max_window_size, self.channel),
                "noise shape should be (batch_size, max_window_size, channel)",
            )
            points = noise
            start_step = self.timesteps - middle_steps
            end_step = 0

        if self.auto_regre_num > 0:
            self.set_fixed_points(
                points, None, fixed_points, mat, joint_id=self.mask_ind, fix_mode=True, fix_goal=False
            )

        imgs = []
        occs = []
        x_0_preds = []

        desc_func = (
            (lambda: r"Fisrt step for finding \hat x_0") if noise is None else (lambda: "Final step for finding x_0")
        )
        for i in tqdm(reversed(range(end_step, start_step)), desc=desc_func(), total=self.timesteps):
            model_used = self.model

            # select sampling code
            # points, occ, x_0_pred = self.p_sample(model_used, points, fixed_points, mat, scene_flag,
            #                           torch.full((self.batch_size,), i, device=device, dtype=torch.long), i,
            #                            text_emb, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir, pi, need_pi, is_loco
            #                            )

            points, occ, x_0_pred = self.p_sample_dps_version_2(
                model_used,
                points,
                fixed_points,
                mat,
                scene_flag,
                torch.full((self.batch_size,), i, device=device, dtype=torch.long),
                i,
                text_emb,
                pelvis_goal,
                hand_goal,
                is_pick,
                need_scene,
                need_pelvis_dir,
                pi,
                need_pi,
                is_loco,
                target_warping=target_warping,
            )
            if self.auto_regre_num > 0:
                self.set_fixed_points(
                    points, None, fixed_points, mat, joint_id=self.mask_ind, fix_mode=True, fix_goal=False
                )
                self.set_fixed_points(
                    x_0_pred, None, fixed_points, mat, joint_id=self.mask_ind, fix_mode=True, fix_goal=False
                )

            points_orig = points
            x_0_pred_orig = x_0_pred

            imgs.append(points_orig)
            x_0_preds.append(x_0_pred_orig)

            if occ is not None:
                occs.append(occ.cpu().numpy())

        return imgs, occs, x_0_preds

    def calculate_loss_for_time_warping(self, x_0_pred, target_warping):
        # Preserve the prediction graph; the removed detach would disconnect DPS.
        # x_0_pred_new = x_0_pred.detach().clone().requires_grad_(True)
        # x_0_pred = x_0_pred_new

        # Keep the existing diff, mask, M and unused smoothness calculation.
        diff = x_0_pred - target_warping
        mask = (target_warping[...] != 0).float()
        diff = diff * mask

        N = diff.shape[1]  # number of frames (sequence length)
        M = torch.zeros((N, N), dtype=diff.dtype, device=diff.device)
        for i in range(N):
            if i > 0:
                M[i, i - 1] = -1
            M[i, i] = 2
            if i < N - 1:
                M[i, i + 1] = -1
        M[0, 0] = 1
        M[0, 1] = -1
        M[-1, -2] = -1
        M[-1, -1] = 1

        x_flat = x_0_pred.reshape(x_0_pred.shape[0], N, -1)
        # Keep the legacy computation and tensor lifetime; the loss does not use this term.
        _smoothness = torch.matmul(M, x_flat)

        loss = torch.sum(diff**2)  # + 0.03*torch.sum(_smoothness ** 2)

        # Return the loss instead of computing and returning its gradient here.
        # grad = torch.autograd.grad(loss, x_0_pred, retain_graph=True)[0]
        # return grad
        return loss
