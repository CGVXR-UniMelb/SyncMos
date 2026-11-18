import json
from pathlib import Path
from typing import List
import numpy as np

from .result_types import AblationResult


def save_ablation_result(log_base_dir: Path, ablation: AblationResult, test_case: dict) -> None:
    result_dir = (
        log_base_dir / "results" / f"testcase-{ablation.test_case_id}" / f"distraction-{ablation.distraction_level}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = (
        result_dir / f"tc{ablation.test_case_id:02d}-d{ablation.distraction_level}-r{ablation.repeat_idx}.json"
    )

    payload = {
        "test_case_id": ablation.test_case_id,
        "distraction_level": ablation.distraction_level,
        "repeat_idx": ablation.repeat_idx,
        "scene_name": test_case.get("scene_name"),
        "tags": test_case.get("tags", []),
        "results": ablation.results,
    }

    result_file.write_text(json.dumps(payload, indent=2))


def generate_ablation_summary(log_base_dir: Path, ablation_results: List[AblationResult], logger) -> None:
    summary = {
        "total_tests": len(ablation_results),
        "narrator_types": {},
        "per_test_details": [],
    }

    for r in ablation_results:
        for narrator_type, res in r.results.items():
            entry = summary["narrator_types"].setdefault(
                narrator_type,
                {
                    "total_tests": 0,
                    "total_passed": 0,
                    "total_errors": 0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "event_counts": [],
                    "dependencies_counts": [],
                },
            )
            entry["total_tests"] += 1
            entry["total_passed"] += 1 if res.get("passed", False) else 0
            entry["total_errors"] += 1 if res.get("error", False) else 0
            entry["total_prompt_tokens"] += int(res.get("prompt_tokens", 0))
            entry["total_completion_tokens"] += int(res.get("completion_tokens", 0))
            entry["event_counts"].append(res.get("event_count", 0))
            entry["dependencies_counts"].append(res.get("dependencies_count", 0))

        per_test_entry = {
            "test_case_id": r.test_case_id,
            "distraction_level": r.distraction_level,
            "repeat_idx": r.repeat_idx,
            "results": {},
        }
        for narrator_type, res in r.results.items():
            per_test_entry["results"][narrator_type] = {
                "passed": res.get("passed", False),
                "error": res.get("error", False),
                "error_message": res.get("error_message"),
                "failure_reason": res.get("failure_reason"),
                "prompt_tokens": res.get("prompt_tokens", 0),
                "completion_tokens": res.get("completion_tokens", 0),
                "event_count": res.get("event_count", 0),
                "dependencies_count": res.get("dependencies_count", 0),
                "mode": res.get("mode"),
                "diagnose": res.get("diagnostics", {}),
            }
        summary["per_test_details"].append(per_test_entry)

    for entry in summary["narrator_types"].values():
        total = entry["total_tests"]
        entry["pass_rate"] = (entry["total_passed"] / total * 100) if total > 0 else 0
        entry["avg_event_count"] = float(np.mean(entry["event_counts"])) if entry["event_counts"] else 0.0
        entry["avg_dependencies_count"] = (
            float(np.mean(entry["dependencies_counts"])) if entry["dependencies_counts"] else 0.0
        )
        entry["avg_tokens_per_event"] = (
            ((entry["total_prompt_tokens"] + entry["total_completion_tokens"]) / max(entry["avg_event_count"], 1))
            if total > 0
            else 0.0
        )
        entry.pop("event_counts", None)
        entry.pop("dependencies_counts", None)

    (log_base_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if logger:
        logger.info("=" * 60)
        logger.info("Ablation Summary")
        logger.info("=" * 60)
        for narrator_type, entry in summary["narrator_types"].items():
            logger.info(f"{narrator_type}:")
            logger.info(f"  Total tests: {entry['total_tests']}")
            logger.info(f"  Pass rate: {entry['pass_rate']:.2f}%")
            logger.info(f"  Total errors: {entry['total_errors']}")
            logger.info(f"  Total tokens: {entry['total_prompt_tokens'] + entry['total_completion_tokens']}")
            logger.info(f"  Avg events: {entry['avg_event_count']:.2f}")
            logger.info(f"  Avg dependencies: {entry['avg_dependencies_count']:.2f}")
            logger.info("")
        logger.info("=" * 60)
