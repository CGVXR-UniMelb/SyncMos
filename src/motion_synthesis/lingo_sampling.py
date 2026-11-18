"""SyncMOS extensions over the native external LINGO sampler.

SamplingControls delegates the ordinary loop to the external sampler and retains
the SyncMOS clean-prediction trace. Parallel, renoise and loss controls remain in
that module with their existing implementations and mixed provenance.
"""

import torch

from src.motion_synthesis.lingo_sampling_controls import SamplingControls


class SamplingExtensions(SamplingControls):
    """Keep clean traces/TW/DPS while delegating the base DDPM step upstream."""

    def get_scheduler(self):
        super().get_scheduler()
        # Match the original cumprod exactly; never reconstruct it by squaring sqrt buffers.
        self.alphas_cumprod = torch.cumprod(1.0 - self.betas, dim=0).to(self.device)

    def p_sample(
        self,
        model,
        x,
        fixed_points,
        mat,
        scene_flag,
        t,
        t_index,
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
        return self._sample(
            model,
            x,
            fixed_points,
            mat,
            scene_flag,
            t,
            t_index,
            text_emb,
            pelvis_goal,
            hand_goal,
            is_pick,
            need_scene,
            need_pelvis_dir,
            pi,
            need_pi,
            is_loco,
            dps=False,
            target_warping=None,
        )

    def p_sample_dps_version_2(
        self,
        model,
        x,
        fixed_points,
        mat,
        scene_flag,
        t,
        t_index,
        text_emb,
        pelvis_goal,
        hand_goal,
        is_pick,
        need_scene,
        need_pelvis_dir,
        pi,
        need_pi,
        is_loco,
        target_warping=None,
    ):
        return self._sample(
            model,
            x,
            fixed_points,
            mat,
            scene_flag,
            t,
            t_index,
            text_emb,
            pelvis_goal,
            hand_goal,
            is_pick,
            need_scene,
            need_pelvis_dir,
            pi,
            need_pi,
            is_loco,
            dps=True,
            target_warping=target_warping,
        )

    @torch.no_grad()
    def _sample(
        self,
        model,
        x,
        fixed_points,
        mat,
        scene_flag,
        t,
        t_index,
        text_emb,
        pelvis_goal,
        hand_goal,
        is_pick,
        need_scene,
        need_pelvis_dir,
        pi,
        need_pi,
        is_loco,
        *,
        dps,
        target_warping,
    ):
        clean, gradient = None, None

        def predict(value, occupancy, timestep, *conditions):
            nonlocal clean, gradient
            # Upstream queries occupancy before this callback, outside the DPS graph.
            with torch.set_grad_enabled(dps):
                leaf = value.detach().clone().requires_grad_(True) if dps else value
                noise = model(leaf, occupancy, timestep, *conditions)
                sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, timestep, leaf.shape)
                sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, timestep, leaf.shape)
                clean = (leaf - sqrt_one_minus * noise) / sqrt_alpha
                if dps:
                    assert target_warping is not None, "target_warping"
                    loss = self.calculate_loss_for_time_warping(clean, target_warping)
                    gradient = torch.autograd.grad(loss, leaf, retain_graph=False)[0].detach()
                clean = clean.detach()
            return noise.detach()

        # The unchanged upstream method owns occupancy, DDPM mean/variance and the
        # sole posterior noise draw. The callable is local, not a model hook/wrapper.
        sample, occupancy = super().p_sample(
            predict,
            x,
            fixed_points,
            mat,
            scene_flag,
            t,
            t_index,
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
        if dps and occupancy is not None:
            occupancy = occupancy.detach()
        if dps and t_index != 0:
            sample = sample - 0.05 * gradient
        return sample, occupancy, clean
