#!/usr/bin/env python3
"""
Run NarratorTimeline two digestors (no_reasoning / no_fewshot)Compare Script

The baseline is run simultaneously by default + Two timeline variants.

Usage:
    python -m benchmark ablation-two --help
"""

from omegaconf import OmegaConf
import argparse
import sys

from benchmark.narrator_comparison import NarratorComparisonRunner

DEFAULT_NARRATOR_TYPES = [
    "timeline_no_reasoning",
    "timeline_no_fewshot",
]


def main():
    parser = argparse.ArgumentParser(description="Run two timeline ablation variants")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/benchmark_comparison_4o_mini.yaml",
        help="Path to benchmark comparison config YAML",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Override evaluation.experiment_name",
    )
    parser.add_argument(
        "--test-cases",
        type=str,
        default=None,
        help="Comma-separated test case IDs (e.g., '0,1,2,10,11,20')",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=None,
        help="Number of repeats per test case",
    )
    parser.add_argument(
        "--distraction-levels",
        type=str,
        default=None,
        help="Comma-separated distraction levels (e.g., '0,1,2')",
    )
    parser.add_argument(
        "--test-case-dir",
        type=str,
        default=None,
        help="Path to a folder containing narrator comparison test cases in JSON",
    )
    parser.add_argument(
        "--narrator-types",
        type=str,
        default=None,
        help=("Comma-separated narrator types to run. " "Default: baseline,timeline_no_reasoning,timeline_no_fewshot"),
    )

    args = parser.parse_args()

    # Loading configurations
    cfg = OmegaConf.load(args.config)

    # Override configuration parameters
    if args.experiment_name:
        cfg.evaluation.experiment_name = args.experiment_name
    if args.test_cases:
        cfg.evaluation.test_cases = [int(x.strip()) for x in args.test_cases.split(",")]
    if args.repeat_count:
        cfg.evaluation.repeat_count = args.repeat_count
    if args.distraction_levels:
        cfg.evaluation.distraction_levels = [int(x.strip()) for x in args.distraction_levels.split(",")]
    if args.test_case_dir:
        cfg.evaluation.test_case_dir = args.test_case_dir

    if args.narrator_types:
        cfg.evaluation.narrator_types = [x.strip() for x in args.narrator_types.split(",")]
    else:
        cfg.evaluation.narrator_types = DEFAULT_NARRATOR_TYPES

    # Check required configurations
    if cfg.evaluation.experiment_name is None:
        print("Error: evaluation.experiment_name must be set")
        print("Use --experiment-name or set it in config file")
        sys.exit(1)

    # Check test case configuration: Either test_cases or test_case_dir must be provided
    has_test_cases = cfg.evaluation.test_cases is not None
    has_test_case_dir = hasattr(cfg.evaluation, "test_case_dir") and cfg.evaluation.test_case_dir

    if not has_test_cases and not has_test_case_dir:
        print("Error: Must provide either --test-cases or --test-case-dir")
        print("  --test-cases: Comma-separated test case IDs (e.g., '902,903')")
        print("  --test-case-dir: Path to directory containing test case JSON files")
        sys.exit(1)

    # If test_case_dir is used, set a default test_cases list (an empty list will be ignored)
    if has_test_case_dir and not has_test_cases:
        cfg.evaluation.test_cases = []

    if cfg.evaluation.repeat_count is None:
        print("Error: evaluation.repeat_count must be set")
        print("Use --repeat-count or set it in config file")
        sys.exit(1)

    if cfg.evaluation.distraction_levels is None:
        print("Error: evaluation.distraction_levels must be set")
        print("Use --distraction-levels or set it in config file")
        sys.exit(1)

    # Synchronize model configuration
    if cfg.evaluation.model:
        cfg.narrator.model = cfg.evaluation.model
        cfg.event_parser.model = cfg.evaluation.model
    if cfg.evaluation.provider:
        cfg.narrator.provider = cfg.evaluation.provider
        cfg.event_parser.provider = cfg.evaluation.provider
    if cfg.evaluation.temperature is not None:
        cfg.narrator.temperature = cfg.evaluation.temperature
        cfg.event_parser.temperature = cfg.evaluation.temperature

    # Set the log level
    cfg.logging_level = "INFO"

    # Ensure the use of pre-generated scene descriptions
    cfg.scene_describer.use_pregenerated_description = True

    # Set to read-only
    OmegaConf.set_readonly(cfg, True)
    OmegaConf.set_struct(cfg, True)

    print(f"\n{'='*60}")
    print("Running Narrator Timeline Ablations (two variants)")
    print(f"Experiment: {cfg.evaluation.experiment_name}")
    if hasattr(cfg.evaluation, "test_case_dir") and cfg.evaluation.test_case_dir:
        print(f"Test cases: Loading from directory: {cfg.evaluation.test_case_dir}")
    else:
        print(f"Test cases: {cfg.evaluation.test_cases}")
    print(f"Repeat count: {cfg.evaluation.repeat_count}")
    print(f"Distraction levels: {cfg.evaluation.distraction_levels}")
    print(f"Narrator types: {cfg.evaluation.narrator_types}")
    print(f"{'='*60}\n")

    runner = NarratorComparisonRunner(cfg)
    runner.run_comparison()

    print(f"\n{'='*60}")
    print(f"Ablation comparison complete! Results saved to: {runner.log_base_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
