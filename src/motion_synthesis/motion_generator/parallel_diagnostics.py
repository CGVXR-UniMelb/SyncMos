"""Diagnostics emitted by parallel motion generation.

Kept separate from sampling so the legacy stdout, stat.txt, and STATISTICS
formats can be maintained without mixing reporting into the generation flow.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass
class ParallelMotionDiagnostics:
    """Intermediate arrays and contact metrics for one generated pair."""

    points_all_np_mid_A: "np.ndarray"
    points_all_np_mid_B: "np.ndarray"
    points_all_np_A: "np.ndarray"
    points_all_np_B: "np.ndarray"
    best_idx_A: int
    best_idx_B: int
    middle: int
    best_dist_A: float
    best_dist_B: float
    best_idx_A_final: int
    best_idx_B_final: int
    best_dist_A_final: float
    best_dist_B_final: float
    best_idx_before_warp: int
    best_dist_before_warp: float
    best_idx_after_warp: int
    best_dist_after_warp: float
    mapping_after_idx_A: int
    mapping_after_idx_B: int
    auto_regre: int

    def write(self) -> None:
        """Preserve the experiment output text and positional metric order."""
        messages = [
            f"points_all_np_mid_A: {self.points_all_np_mid_A.shape}, points_all_np_mid_B: {self.points_all_np_mid_B.shape}",
            f"points_all_np_A: {self.points_all_np_A.shape}, points_all_np_B: {self.points_all_np_B.shape}",
            f"best_idx_A: {self.best_idx_A}, best_idx_B: {self.best_idx_B}, middle: {self.middle}",
            f"best_dist_A: {self.best_dist_A}, best_dist_B: {self.best_dist_B}",
            f"best_idx_A_final: {self.best_idx_A_final + self.auto_regre}, best_idx_B_final: {self.best_idx_B_final + self.auto_regre}",
            f"best_dist_A_final: {self.best_dist_A_final}, best_dist_B_final: {self.best_dist_B_final}",
            f"best_idx_before_warp: {self.best_idx_before_warp}, best_dist_before_warp: {self.best_dist_before_warp}",
            f"best_idx_after_warp: {self.best_idx_after_warp}, best_dist_after_warp: {self.best_dist_after_warp}",
            f"mapping_after_idx_A: {self.mapping_after_idx_A}, mapping_after_idx_B: {self.mapping_after_idx_B}",
        ]
        for message in messages:
            print(message)
        with open("stat.txt", "a") as f:
            for message in messages:
                f.write(message + "\n")
        try:
            from src.motion_synthesis.motion_generator import experiment_stats

            experiment_stats.STATISTICS.extend(
                [
                    self.best_idx_A - 2,
                    self.best_idx_B - 2,
                    self.middle - 2,
                    self.best_dist_A,
                    self.best_dist_B,
                    self.best_idx_A_final,
                    self.best_idx_B_final,
                    self.best_dist_A_final,
                    self.best_dist_B_final,
                    self.best_idx_before_warp,
                    self.best_dist_before_warp,
                    self.best_idx_after_warp,
                    self.best_dist_after_warp,
                    self.mapping_after_idx_A,
                    self.mapping_after_idx_B,
                ]
            )
        except Exception as e:
            print(f"Warning: Could not update _STATISTICS: {e}")
