"""
Ablation evaluator for narrator variants (dependency-focused).
"""

from typing import List, Dict
from tqdm import tqdm

from src.utils import misc

from .narrator_comparison import (
    NarratorComparisonRunner,
    TIMELINE_NARRATOR_TYPES,
)
from .runner import TestMetadata
from .comparison.io_utils import save_test_case_json
from .ablation.result_types import AblationResult
from .ablation.io_utils import save_ablation_result, generate_ablation_summary

ABLATION_NARRATOR_TYPES = [
    "wo_timeline",
    "timeline_single",
    "wo_timeline_single",
    "timeline_multi_char",
]


class NarratorAblationRunner(NarratorComparisonRunner):
    """Only run the ablation narrator combined evaluation runner"""

    def _get_model_label(self) -> str:
        model = getattr(self.cfg.evaluation, "model", None) or getattr(self.cfg.narrator, "model", None)
        if not model:
            return "unknown_model"
        return str(model).replace("/", "_").replace(":", "_").replace(" ", "_")

    def _get_narrator_label(self) -> str:
        raw = getattr(self.cfg.evaluation, "narrator_types", ABLATION_NARRATOR_TYPES)
        if raw and hasattr(raw, "__iter__"):
            types = list(raw)
        else:
            types = ABLATION_NARRATOR_TYPES
        if len(types) == 1:
            return types[0]
        return "mixed"

    def _setup_directories(self) -> None:
        """ablation/model/narrator_type Path structure"""
        model_label = self._get_model_label()
        narrator_label = self._get_narrator_label()

        from pathlib import Path

        self.log_base_dir = Path(
            f"./logs/ablation/{model_label}/{narrator_label}/" f"{self.run_id}-{self.scene_description_type}"
        )
        self.log_base_dir.mkdir(parents=True, exist_ok=True)

        self.test_cases_dir = self.log_base_dir / "test_cases"
        self.test_cases_dir.mkdir(parents=True, exist_ok=True)

        self.official_comparison_dir = Path("./benchmark/test_cases/narrator_comparison")
        self.official_comparison_dir.mkdir(parents=True, exist_ok=True)

    def __init__(self, cfg):
        super().__init__(cfg)

        raw_narrator_types = getattr(cfg.evaluation, "narrator_types", ABLATION_NARRATOR_TYPES)
        if raw_narrator_types and hasattr(raw_narrator_types, "__iter__"):
            narrator_types = list(raw_narrator_types)
        else:
            narrator_types = ABLATION_NARRATOR_TYPES

        filtered = [t for t in narrator_types if t in ABLATION_NARRATOR_TYPES]
        self.narrator_types = filtered if filtered else ABLATION_NARRATOR_TYPES

        self.ablation_results: List[AblationResult] = []

        self.logger = misc.get_console_logger("NarratorAblation", cfg.logging_level)
        self.logger.info(f"Ablation narrator types: {self.narrator_types}")

    def run_comparison(self) -> None:
        test_cases_by_scene = self._prepare_test_cases()

        for scene_name, test_cases in tqdm(test_cases_by_scene.items(), desc="Scenes"):
            if not test_cases:
                continue
            self._run_scene_comparison(scene_name, test_cases)

        generate_ablation_summary(self.log_base_dir, self.ablation_results, self.logger)

    def _run_test_case_with_variations(self, test_case: dict) -> None:
        for distraction_level in tqdm(self.distraction_levels, desc="Distraction levels"):
            max_distraction_level = self._get_max_distraction_level(test_case)
            if distraction_level > max_distraction_level:
                continue

            for repeat_idx in tqdm(range(self.repeat_count), desc="Repeats", leave=False):
                self.run_ablation_test(test_case, distraction_level, repeat_idx)

    def _get_max_distraction_level(self, test_case: dict) -> int:
        from .comparison.setup import get_max_distraction_level as _get_max_distraction_level_mod

        return _get_max_distraction_level_mod(test_case)

    def run_ablation_test(
        self,
        test_case: dict,
        distraction_level: int,
        repeat_idx: int,
    ) -> None:
        metadata = TestMetadata(test_case["id"], distraction_level, repeat_idx)

        save_test_case_json(test_case, self.test_cases_dir, self.official_comparison_dir, self.logger)

        self.available_action_labels = test_case.get("action_labels", [])

        results: Dict[str, dict] = {}
        for narrator_type in self.narrator_types:
            if narrator_type in TIMELINE_NARRATOR_TYPES:
                result = self._run_timeline_narrator(
                    test_case,
                    distraction_level,
                    repeat_idx,
                    metadata,
                    narrator_type=narrator_type,
                )
                result["mode"] = "timeline"
            else:
                self.baseline_narrator_type = narrator_type
                result = self._run_baseline_incremental(
                    test_case,
                    distraction_level,
                    repeat_idx,
                    metadata,
                )
                result["mode"] = "baseline"

            result["narrator_type"] = narrator_type
            results[narrator_type] = result

        ablation = AblationResult(
            test_case_id=metadata.test_case_id,
            distraction_level=metadata.distraction_level,
            repeat_idx=metadata.repeat_idx,
            results=results,
        )

        self.ablation_results.append(ablation)
        save_ablation_result(self.log_base_dir, ablation, test_case)
