import json
from pathlib import Path
from typing import List
import numpy as np


def save_test_case_json(test_case: dict, test_cases_dir: Path, official_dir: Path, logger):
    test_cases_dir.mkdir(parents=True, exist_ok=True)
    official_dir.mkdir(parents=True, exist_ok=True)

    file1 = test_cases_dir / f"testcase_{test_case['id']}.json"
    if not file1.exists():
        file1.write_text(json.dumps(test_case, indent=2))
        if logger:
            logger.debug(f"Saved test case JSON: {file1}")

    file2 = official_dir / f"testcase_{test_case['id']}.json"
    if not file2.exists():
        try:
            file2.write_text(json.dumps(test_case, indent=2))
        except Exception as e:
            if logger:
                logger.warning(f"Failed to save official test case: {file2} ({e})")


def save_comparison_result(log_base_dir: Path, comparison, test_case: dict):
    result_dir = (
        log_base_dir / "results" / f"testcase-{comparison.test_case_id}" / f"distraction-{comparison.distraction_level}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    variant_tag = getattr(comparison, "timeline_variant", "timeline")
    result_file = (
        result_dir
        / f"tc{comparison.test_case_id:02d}-d{comparison.distraction_level}-r{comparison.repeat_idx}-t{variant_tag}.json"
    )

    payload = {
        "test_case_id": comparison.test_case_id,
        "distraction_level": comparison.distraction_level,
        "repeat_idx": comparison.repeat_idx,
        "timeline_variant": getattr(comparison, "timeline_variant", "timeline"),
        "scene_name": test_case.get("scene_name"),
        "tags": test_case.get("tags", []),
        "baseline": {
            "passed": comparison.baseline_passed,
            "error": comparison.baseline_error,
            "error_message": comparison.baseline_error_message,
            "failure_reason": comparison.baseline_failure_reason,
            "prompt_tokens": comparison.baseline_prompt_tokens,
            "completion_tokens": comparison.baseline_completion_tokens,
            "event_count": comparison.baseline_event_count,
            "matched_events": comparison.baseline_matched_events,
            "expected_events": comparison.baseline_expected_events,
            "events_generated": comparison.baseline_events_generated,  # 🆕 Statistically analyze the coverage of pure events
            "matched_dependencies": comparison.baseline_matched_dependencies,
            "expected_dependencies": comparison.baseline_expected_dependencies,
            "tokens_per_event": round(comparison.baseline_tokens_per_event, 2),
            "diagnostics": comparison.baseline_diagnostics,
        },
        "timeline": {
            "variant": getattr(comparison, "timeline_variant", "timeline"),
            "passed": comparison.timeline_passed,
            "error": comparison.timeline_error,
            "error_message": comparison.timeline_error_message,
            "failure_reason": comparison.timeline_failure_reason,
            "prompt_tokens": comparison.timeline_prompt_tokens,
            "completion_tokens": comparison.timeline_completion_tokens,
            "event_count": comparison.timeline_event_count,
            "matched_events": comparison.timeline_matched_events,
            "expected_events": comparison.timeline_expected_events,
            "matched_dependencies": comparison.timeline_matched_dependencies,
            "expected_dependencies": comparison.timeline_expected_dependencies,
            "dependencies_count": comparison.timeline_dependencies_count,
            "event_schema_valid": comparison.timeline_event_schema_valid,
            "dependency_valid": comparison.timeline_dependency_valid,
            "actions_ok": (
                bool(comparison.timeline_planning_result.get("actions_by_character"))
                if isinstance(comparison.timeline_planning_result, dict)
                else None
            ),
            "tokens_per_event": round(comparison.timeline_tokens_per_event, 2),
            "diagnostics": comparison.timeline_diagnostics,
        },
        "comparison": {
            "improvement": comparison.improvement,
            "token_savings_percent": round(comparison.token_savings, 2),
            "token_efficiency_ratio": round(comparison.token_efficiency_ratio, 2),
            "fairness_note": comparison.comparison_fairness,
            "task_type_note": comparison.task_type_note,
        },
    }

    result_file.write_text(json.dumps(payload, indent=2))

    # Generate Diagnostic Report (Phase 2)
    from .diagnostic_report import generate_diagnostic_report

    try:
        generate_diagnostic_report(comparison, test_case, log_base_dir)
    except Exception as e:
        # Failure to Generate Diagnostic Report Should Not Affect Main Process
        import logging

        logging.getLogger(__name__).warning(f"Failed to generate diagnostic report: {e}")


def generate_summary_report(log_base_dir: Path, comparison_results: List, logger):
    summary = {
        "total_tests": len(comparison_results),
        "baseline": {
            "total_passed": sum(1 for r in comparison_results if r.baseline_passed),
            "total_errors": sum(1 for r in comparison_results if r.baseline_error),
            "total_prompt_tokens": int(sum(r.baseline_prompt_tokens for r in comparison_results)),
            "total_completion_tokens": int(sum(r.baseline_completion_tokens for r in comparison_results)),
            "avg_event_count": (
                float(np.mean([r.baseline_event_count for r in comparison_results])) if comparison_results else 0.0
            ),
            "avg_tokens_per_event": (
                float(np.mean([r.baseline_tokens_per_event for r in comparison_results])) if comparison_results else 0.0
            ),
        },
        "timeline": {
            "total_passed": sum(1 for r in comparison_results if r.timeline_passed),
            "total_errors": sum(1 for r in comparison_results if r.timeline_error),
            "total_prompt_tokens": int(sum(r.timeline_prompt_tokens for r in comparison_results)),
            "total_completion_tokens": int(sum(r.timeline_completion_tokens for r in comparison_results)),
            "avg_event_count": (
                float(np.mean([r.timeline_event_count for r in comparison_results])) if comparison_results else 0.0
            ),
            "avg_dependencies_count": (
                float(np.mean([r.timeline_dependencies_count for r in comparison_results]))
                if comparison_results
                else 0.0
            ),
            "avg_tokens_per_event": (
                float(np.mean([r.timeline_tokens_per_event for r in comparison_results])) if comparison_results else 0.0
            ),
        },
        "comparison": {
            "improvements": sum(1 for r in comparison_results if r.improvement),
            "avg_token_savings_percent": (
                float(np.mean([r.token_savings for r in comparison_results])) if comparison_results else 0.0
            ),
            "avg_token_efficiency_ratio": (
                float(np.mean([r.token_efficiency_ratio for r in comparison_results])) if comparison_results else 0.0
            ),
        },
        "per_test_details": [],
    }
    summary["baseline"]["pass_rate"] = (
        summary["baseline"]["total_passed"] / summary["total_tests"] * 100 if summary["total_tests"] > 0 else 0
    )
    summary["timeline"]["pass_rate"] = (
        summary["timeline"]["total_passed"] / summary["total_tests"] * 100 if summary["total_tests"] > 0 else 0
    )

    # Add per-test details
    for r in comparison_results:
        test_detail = {
            "test_case_id": r.test_case_id,
            "distraction_level": r.distraction_level,
            "repeat_idx": r.repeat_idx,
            "timeline_variant": getattr(r, "timeline_variant", "timeline"),
            "baseline": {
                "matched_events": r.baseline_matched_events,
                "expected_events": r.baseline_expected_events,
                "events_generated": r.baseline_events_generated,  # 🆕 Statistically analyze the coverage of pure events
                "events_matched_percent": round(
                    (
                        (r.baseline_matched_events / r.baseline_expected_events * 100)
                        if r.baseline_expected_events and r.baseline_expected_events > 0
                        else 0
                    ),
                    1,
                ),
                "events_generated_percent": round(
                    (
                        (r.baseline_events_generated / r.baseline_expected_events * 100)
                        if r.baseline_expected_events and r.baseline_expected_events > 0
                        else 0
                    ),
                    1,
                ),  # 🆕
                "matched_dependencies": r.baseline_matched_dependencies,
                "expected_dependencies": r.baseline_expected_dependencies,
                "dependencies_satisfied_percent": round(
                    (
                        (r.baseline_matched_dependencies / r.baseline_expected_dependencies * 100)
                        if r.baseline_expected_dependencies and r.baseline_expected_dependencies > 0
                        else 0
                    ),
                    1,
                ),
                "prompt_tokens": r.baseline_prompt_tokens,  # 🆕 Add Token Consumption
                "completion_tokens": r.baseline_completion_tokens,  # 🆕
                "total_tokens": r.baseline_prompt_tokens + r.baseline_completion_tokens,  # 🆕
                "passed": r.baseline_passed,
                "error": r.baseline_error,
            },
            "timeline": {
                "matched_events": r.timeline_matched_events,
                "expected_events": r.timeline_expected_events,
                "events_matched_percent": round(
                    (
                        (r.timeline_matched_events / r.timeline_expected_events * 100)
                        if r.timeline_expected_events and r.timeline_expected_events > 0
                        else 0
                    ),
                    1,
                ),
                "matched_dependencies": r.timeline_matched_dependencies,
                "expected_dependencies": r.timeline_expected_dependencies,
                "dependencies_satisfied_percent": round(
                    (
                        (r.timeline_matched_dependencies / r.timeline_expected_dependencies * 100)
                        if r.timeline_expected_dependencies and r.timeline_expected_dependencies > 0
                        else 0
                    ),
                    1,
                ),
                "prompt_tokens": r.timeline_prompt_tokens,  # 🆕 Add Token Consumption
                "completion_tokens": r.timeline_completion_tokens,  # 🆕
                "total_tokens": r.timeline_prompt_tokens + r.timeline_completion_tokens,  # 🆕
                "passed": r.timeline_passed,
                "error": r.timeline_error,
            },
        }
        summary["per_test_details"].append(test_detail)

    (log_base_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if logger:
        logger.info("=" * 60)
        logger.info("Comparison Summary")
        logger.info("=" * 60)
        logger.info(f"Total tests: {summary['total_tests']}")
        logger.info("")
        logger.info("Baseline (Narrator - Incremental Planning):")
        logger.info("  Task: Incremental planning (2×n derived cases)")
        logger.info(f"  Pass rate: {summary['baseline']['pass_rate']:.2f}%")
        logger.info(f"  Total passed: {summary['baseline']['total_passed']}")
        logger.info(f"  Total errors: {summary['baseline']['total_errors']}")
        logger.info(
            f"  Total tokens: {summary['baseline']['total_prompt_tokens'] + summary['baseline']['total_completion_tokens']}"
        )
        logger.info(f"  Avg events: {summary['baseline']['avg_event_count']:.2f}")
        logger.info(f"  Avg tokens/event: {summary['baseline']['avg_tokens_per_event']:.2f}")
        logger.info("")
        logger.info("Timeline (NarratorTimeline - Batch Planning):")
        logger.info("  Task: Batch planning (1 complete sequence + dependencies)")
        logger.info(f"  Pass rate: {summary['timeline']['pass_rate']:.2f}%")
        logger.info(f"  Total passed: {summary['timeline']['total_passed']}")
        logger.info(f"  Total errors: {summary['timeline']['total_errors']}")
        logger.info(
            f"  Total tokens: {summary['timeline']['total_prompt_tokens'] + summary['timeline']['total_completion_tokens']}"
        )
        logger.info(f"  Avg events: {summary['timeline']['avg_event_count']:.2f}")
        logger.info(f"  Avg dependencies: {summary['timeline']['avg_dependencies_count']:.2f}")
        logger.info(f"  Avg tokens/event: {summary['timeline']['avg_tokens_per_event']:.2f}")
        logger.info("")
        logger.info("Comparison (Normalized Metrics):")
        logger.info(f"  Improvements: {summary['comparison']['improvements']}")
        logger.info(f"  Avg token savings: {summary['comparison']['avg_token_savings_percent']:.2f}%")
        logger.info(f"  Token efficiency ratio: {summary['comparison']['avg_token_efficiency_ratio']:.2f}x")
        logger.info(
            f"  (Timeline is {summary['comparison']['avg_token_efficiency_ratio']:.2f}x more efficient per event)"
        )
        logger.info("")
        logger.info("Per-Test Details:")
        logger.info("-" * 60)
        for r in comparison_results:
            logger.info(f"Test Case {r.test_case_id} (Distraction {r.distraction_level}, Repeat {r.repeat_idx}):")

            # Baseline details
            if r.baseline_matched_events is not None and r.baseline_expected_events is not None:
                baseline_event_pct = (
                    (r.baseline_matched_events / r.baseline_expected_events * 100)
                    if r.baseline_expected_events > 0
                    else 0
                )
                logger.info(
                    f"  Baseline: Events Matched: {r.baseline_matched_events}/{r.baseline_expected_events} ({baseline_event_pct:.1f}%)"
                )

                # 🆕 Display Pure Event Coverage Statistics
                if r.baseline_events_generated is not None:
                    baseline_gen_pct = (
                        (r.baseline_events_generated / r.baseline_expected_events * 100)
                        if r.baseline_expected_events > 0
                        else 0
                    )
                    logger.info(
                        f"            Events Generated: {r.baseline_events_generated}/{r.baseline_expected_events} ({baseline_gen_pct:.1f}%)"
                    )
            else:
                logger.info("  Baseline: Events Matched: N/A")

            if r.baseline_matched_dependencies is not None and r.baseline_expected_dependencies is not None:
                baseline_dep_pct = (
                    (r.baseline_matched_dependencies / r.baseline_expected_dependencies * 100)
                    if r.baseline_expected_dependencies > 0
                    else 0
                )
                logger.info(
                    f"            Dependencies Satisfied: {r.baseline_matched_dependencies}/{r.baseline_expected_dependencies} ({baseline_dep_pct:.1f}%)"
                )
            else:
                logger.info("            Dependencies Satisfied: N/A")

            # Timeline details
            if r.timeline_matched_events is not None and r.timeline_expected_events is not None:
                timeline_event_pct = (
                    (r.timeline_matched_events / r.timeline_expected_events * 100)
                    if r.timeline_expected_events > 0
                    else 0
                )
                logger.info(
                    f"  Timeline: Events Matched: {r.timeline_matched_events}/{r.timeline_expected_events} ({timeline_event_pct:.1f}%)"
                )
            else:
                logger.info("  Timeline: Events Matched: N/A")

            if r.timeline_matched_dependencies is not None and r.timeline_expected_dependencies is not None:
                timeline_dep_pct = (
                    (r.timeline_matched_dependencies / r.timeline_expected_dependencies * 100)
                    if r.timeline_expected_dependencies > 0
                    else 0
                )
                logger.info(
                    f"            Dependencies Satisfied: {r.timeline_matched_dependencies}/{r.timeline_expected_dependencies} ({timeline_dep_pct:.1f}%)"
                )
            else:
                logger.info("            Dependencies Satisfied: N/A")

            logger.info("")
        logger.info("=" * 60)
