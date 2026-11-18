"""Summarize existing comparison results with the original aggregation function."""

import argparse
import json
from pathlib import Path
from typing import Dict


def extract_metrics(summary: Dict) -> Dict:
    """Extract relevant metrics from summary."""
    baseline = summary.get("baseline", {})
    timeline = summary.get("timeline", {})

    # Calculate metrics from per_test_details
    per_test = summary.get("per_test_details", [])

    # Baseline metrics
    baseline_events_matched = [
        test["baseline"].get("events_matched_percent", 0) for test in per_test if "baseline" in test
    ]
    baseline_deps_satisfied = [
        test["baseline"].get("dependencies_satisfied_percent", 0) for test in per_test if "baseline" in test
    ]

    # Timeline metrics
    timeline_events_matched = [
        test["timeline"].get("events_matched_percent", 0) for test in per_test if "timeline" in test
    ]
    timeline_deps_satisfied = [
        test["timeline"].get("dependencies_satisfied_percent", 0) for test in per_test if "timeline" in test
    ]

    return {
        "baseline": {
            "events_matched_percent": (
                sum(baseline_events_matched) / len(baseline_events_matched) if baseline_events_matched else 0
            ),
            "dependencies_satisfied_percent": (
                sum(baseline_deps_satisfied) / len(baseline_deps_satisfied) if baseline_deps_satisfied else 0
            ),
            "total_passed": baseline.get("total_passed", 0),
        },
        "timeline": {
            "events_matched_percent": (
                sum(timeline_events_matched) / len(timeline_events_matched) if timeline_events_matched else 0
            ),
            "dependencies_satisfied_percent": (
                sum(timeline_deps_satisfied) / len(timeline_deps_satisfied) if timeline_deps_satisfied else 0
            ),
            "total_passed": timeline.get("total_passed", 0),
        },
        "total_tests": summary.get("total_tests", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path, help="summary.json files or directories containing them")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    files = set()
    for path in args.summaries:
        if path.is_dir():
            files.update(path.rglob("summary.json"))
        elif path.is_file():
            files.add(path)
        else:
            parser.error(f"Summary path does not exist: {path}")
    if not files:
        parser.error("No summary.json files found")
    results = {str(path): extract_metrics(json.loads(path.read_text(encoding="utf-8"))) for path in sorted(files)}
    text = json.dumps(results, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
