"""
Compare the effectiveness of NarratorTimeline and Narrator, with a focus on the effects of dependencies
"""

import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from omegaconf import OmegaConf
from tqdm import tqdm

from src.utils import misc

from .runner import load_test_case, TestMetadata
from .comparison.result_types import ComparisonResult
from .comparison.action_eval import (
    evaluate_timeline_actions as evaluate_timeline_actions,  # Public compatibility re-export.
    match_activity_with_or_options,
)
from .comparison.timeline_validation import validate_timeline_plan as validate_timeline_plan  # Compatibility re-export.
from .comparison.derivation import derive_narrator_cases_from_expected_timeline
from .comparison.io_utils import save_test_case_json, save_comparison_result, generate_summary_report
from .comparison.setup import setup_scene as _setup_scene_mod
from .comparison.setup import setup_modules_and_scenario as _setup_modules_and_scenario_mod
from .comparison.setup import get_max_distraction_level as _get_max_distraction_level_mod

# Constant definitions
NARRATOR_TYPE_TIMELINE = "timeline"
NARRATOR_TYPE_BASELINE = None
TIMELINE_NARRATOR_TYPES = {
    "timeline",
    "timeline_single",
    "timeline_multi_char",
    "timeline_no_reasoning",
    "timeline_no_fewshot",
}
SCENE_NAMES = ["House", "Office", "Restaurant"]


def load_derived_test_cases(base_id: int, derived_dir: Path) -> List[dict]:
    """Load all derived test cases from the specified base test cases

    Args:
        base_id: ID of base test cases
        derived_dir: Directory path for derived test cases

    Returns:
        List of derived test cases, sorted by step and state
    """
    if not derived_dir.exists():
        return []

    # Find all matching derived case files: {base_id}_step{*}_{ongoing|completed}.json
    pattern = f"{base_id}_step*_*.json"
    derived_files = sorted(derived_dir.glob(pattern))

    derived_cases = []
    for file_path in derived_files:
        try:
            case = json.loads(file_path.read_text())
            # Add derived case identifiers and required id fields
            case["_derived_from"] = base_id
            case["_derived_file"] = file_path.name
            # Ensure derived cases have id fields (for log paths, etc.)
            if "id" not in case:
                case["id"] = base_id
            derived_cases.append(case)
        except Exception as e:
            print(f"Warning: Failed to load derived case {file_path}: {e}")

    # Sort by file name (ensure step order)
    derived_cases.sort(key=lambda x: x.get("_derived_file", ""))
    return derived_cases


class NarratorComparisonRunner:
    """Compare NarratorTimeline with Narrator’s runner"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.test_case_ids = cfg.evaluation.test_cases
        self.repeat_count = cfg.evaluation.repeat_count
        self.distraction_levels = cfg.evaluation.distraction_levels
        self.scene_description_type = cfg.scene_describer.description_type

        # Set the type of narrator to run (default runs both)
        raw_narrator_types = getattr(cfg.evaluation, "narrator_types", ["baseline", "timeline"])
        # OmegaConf’s list is ListConfig; need to convert or use hasattr for check
        if raw_narrator_types and hasattr(raw_narrator_types, "__iter__"):
            self.narrator_types = list(raw_narrator_types)  # Convert to standard list format
        else:
            self.narrator_types = ["baseline", "timeline"]

        self.run_id = datetime.now().strftime("%y%m%d-%H%M%S")
        self.logger = misc.get_console_logger("NarratorComparison", cfg.logging_level)

        # Print the configured narrator type
        self.logger.info(f"Configured narrator_types from cfg: {raw_narrator_types}")
        self.logger.info(f"Final narrator_types to run: {self.narrator_types}")

        # Set the directory structure
        self._setup_directories()

        # Save the configuration
        OmegaConf.save(self.cfg, self.log_base_dir / "config.yaml", resolve=True)

        # Load custom test cases
        self.custom_test_case_dir: Optional[Path] = None
        self.custom_test_cases: List[dict] = []
        self._load_custom_test_cases()

        # Store all comparison results
        self.comparison_results: List[ComparisonResult] = []

    def _get_baseline_narrator_type(self):
        """Obtain the baseline narrator type (for ablation or replacement of the baseline)"""
        return getattr(self, "baseline_narrator_type", NARRATOR_TYPE_BASELINE)

    def _setup_directories(self) -> None:
        """Initialize all necessary directory structures"""
        self.log_base_dir = Path(
            f"./logs/benchmark_comparison/{self.cfg.evaluation.experiment_name}/"
            f"{self.run_id}-{self.scene_description_type}"
        )
        self.log_base_dir.mkdir(parents=True, exist_ok=True)

        # Create a test case storage folder for this run (copy in the log)
        self.test_cases_dir = self.log_base_dir / "test_cases"
        self.test_cases_dir.mkdir(parents=True, exist_ok=True)

        # Create an official comparison test case folder (fixed location)
        self.official_comparison_dir = Path("./benchmark/test_cases/narrator_comparison")
        self.official_comparison_dir.mkdir(parents=True, exist_ok=True)

    def _load_custom_test_cases(self) -> None:
        """Load test cases from the custom directory (if configured)"""
        if not hasattr(self.cfg.evaluation, "test_case_dir") or not self.cfg.evaluation.test_case_dir:
            return

        self.custom_test_case_dir = Path(self.cfg.evaluation.test_case_dir)
        if not self.custom_test_case_dir.exists():
            return

        for jf in sorted(self.custom_test_case_dir.glob("*.json")):
            try:
                self.custom_test_cases.append(json.loads(jf.read_text()))
            except Exception as e:
                self.logger.warning(f"Failed to load custom test case: {jf} ({e})")

    def run_comparison(self) -> None:
        """Run the complete comparison test"""
        test_cases_by_scene = self._prepare_test_cases()

        for scene_name, test_cases in tqdm(test_cases_by_scene.items(), desc="Scenes"):
            if not test_cases:
                continue
            self._run_scene_comparison(scene_name, test_cases)

        # Generate a summary report
        generate_summary_report(self.log_base_dir, self.comparison_results, self.logger)

    def _prepare_test_cases(self) -> Dict[str, List[dict]]:
        """Prepare test cases, grouped by scene

        Returns:
            Dictionary with scene names as keys and lists of test cases for that scene as values
        """
        if self.custom_test_cases:
            # Test cases loaded from the custom directory
            grouped: Dict[str, List[dict]] = {scene: [] for scene in SCENE_NAMES}

            # Filter if test_cases (list of test case IDs) is specified
            test_case_ids = set(self.cfg.evaluation.test_cases) if self.cfg.evaluation.test_cases else None

            for tc in self.custom_test_cases:
                # If a test case ID is specified, only the matching test case is loaded
                if test_case_ids is not None and tc.get("id") not in test_case_ids:
                    continue

                scene_name = tc.get("scene_name", "House")
                grouped.setdefault(scene_name, []).append(tc)
            return grouped
        else:
            # Load standard test cases by ID
            return self._group_test_cases_by_scene()

    def _run_scene_comparison(self, scene_name: str, test_cases: List[dict]) -> None:
        """Run all test cases for a single scenario

        Args:
            scene_name: Scene name
            test_cases: List of test cases for this scenario
        """
        _setup_scene_mod(self, scene_name)

        for test_case in tqdm(test_cases, desc="Test cases"):
            self._run_test_case_with_variations(test_case)

    def _run_test_case_with_variations(self, test_case: dict) -> None:
        """Run all variants of a single test case (with different interference levels and repetition counts)

        Args:
            test_case: Test case dictionary
        """
        for distraction_level in tqdm(self.distraction_levels, desc="Distraction levels"):
            max_distraction_level = _get_max_distraction_level_mod(test_case)
            if distraction_level > max_distraction_level:
                continue

            for repeat_idx in tqdm(range(self.repeat_count), desc="Repeats", leave=False):
                self.run_comparison_test(test_case, distraction_level, repeat_idx)

    def run_comparison_test(self, test_case: dict, distraction_level: int, repeat_idx: int) -> None:
        """Comparison of running a single test case

        Args:
            test_case: Test case dictionary
            distraction_level: Interference level
            repeat_idx: Repetition count index
        """
        metadata = TestMetadata(test_case["id"], distraction_level, repeat_idx)

        # Save test case JSON (only when the test case is encountered for the first time)
        save_test_case_json(test_case, self.test_cases_dir, self.official_comparison_dir, self.logger)

        # Ensure that available action labels are set before assigning tasks
        self.available_action_labels = test_case.get("action_labels", [])

        # Derived test cases (only required when running the baseline)
        derived_dir = None
        baseline_result = None
        if "baseline" in self.narrator_types:
            derived_dir = self._derive_narrator_cases_if_needed(test_case, distraction_level, repeat_idx)
            baseline_result = self._run_baseline_narrator(
                test_case, distraction_level, repeat_idx, metadata, derived_dir
            )

        # Run timeline variants (including ablation settings)
        timeline_types = [t for t in self.narrator_types if t in TIMELINE_NARRATOR_TYPES]
        if timeline_types:
            for timeline_type in timeline_types:
                timeline_result = self._run_timeline_narrator(
                    test_case, distraction_level, repeat_idx, metadata, narrator_type=timeline_type
                )
                timeline_event_schema_valid = timeline_result.get("event_schema_valid", True)
                timeline_dependency_valid = timeline_result.get("dependency_valid", True)

                # Compare and save results (record each timeline variant separately)
                comparison = self._compare_results(
                    test_case,
                    metadata,
                    baseline_result,
                    timeline_result,
                    timeline_event_schema_valid,
                    timeline_dependency_valid,
                    timeline_variant=timeline_type,
                )
                self.comparison_results.append(comparison)
                save_comparison_result(self.log_base_dir, comparison, test_case)
        else:
            # Save results once, even when there is no timeline or baseline configured
            comparison = self._compare_results(
                test_case, metadata, baseline_result, None, True, True, timeline_variant=NARRATOR_TYPE_TIMELINE
            )
            self.comparison_results.append(comparison)
            save_comparison_result(self.log_base_dir, comparison, test_case)

    def _derive_narrator_cases_if_needed(self, test_case: dict, distraction_level: int, repeat_idx: int) -> Path:
        """Generate derived narrator test cases if needed

        Args:
            test_case: Test case dictionary
            distraction_level: Interference level
            repeat_idx: Repetition count index

        Returns:
            Derived test case storage directory
        """
        derived_dir = self.official_comparison_dir.parent / "narrator" / "derived"

        # Only generated when there is minimal interference to avoid duplication repeat/Generate at the lowest interference level to prevent duplication
        should_derive = repeat_idx == 0 and distraction_level == min(self.distraction_levels or [0])

        if should_derive:
            try:
                derive_narrator_cases_from_expected_timeline(
                    test_case,
                    self.available_action_labels,
                    derived_dir,
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to derive narrator cases from expected_timeline " f"for tc{test_case['id']}: {e}"
                )

        return derived_dir

    def _run_baseline_narrator(
        self, test_case: dict, distraction_level: int, repeat_idx: int, metadata: TestMetadata, derived_dir: Path
    ) -> dict:
        """Run baseline narrator (generate all events incrementally)

        Args:
            test_case: Test case dictionary
            distraction_level: Interference level
            repeat_idx: Repetition count index
            metadata: Test metadata
            derived_dir: Derived test case directory (no longer used, retains parameter compatibility)

        Returns:
            Run result dictionary
        """
        # Use new incremental baseline testing
        return self._run_baseline_incremental(test_case, distraction_level, repeat_idx, metadata)

    def _run_baseline_incremental(
        self, test_case: dict, distraction_level: int, repeat_idx: int, metadata: TestMetadata
    ) -> dict:
        """Incremental running of the baseline narrator: Generating all events starting from an empty history

        Complete simulation of ongoing events → State transitions for completed events; correctly testing dependencies.

        Args:
            test_case: Test case dictionary (including expected_timeline)
            distraction_level: Interference level
            repeat_idx: Repetition count index
            metadata: Test metadata

        Returns:
            Run result dictionary
        """
        from benchmark.comparison.derivation import get_topological_order, resolve_activity_to_specific

        # Obtaining expected_timeline and extracting event sequences
        expected_timeline = test_case.get("expected_timeline", {})
        if not expected_timeline:
            self.logger.warning(
                f"Test case {test_case['id']} has no expected_timeline, skipping baseline incremental test"
            )
            return self._create_error_result("No expected_timeline in test case")

        events = expected_timeline.get("events", [])
        dependencies = expected_timeline.get("dependencies", [])

        if not events:
            self.logger.warning(f"Test case {test_case['id']} has no events in expected_timeline")
            return self._create_error_result("No events in expected_timeline")

        # Retain topology validation even though the returned order is not consumed here.
        get_topological_order(events, dependencies)

        # Building a dependency graph: event_id -> Subsequent event_ids that depend on this event_id
        self._build_dependency_graph(events, dependencies)

        # Initializing statistics
        total_prompt_tokens = 0
        total_completion_tokens = 0
        generated_events = []
        all_passed = True
        failure_reason = None

        # Collecting diagnostic information
        phase_diagnostics = []
        phase_counter = 0

        # Creating initial test cases (with empty history)
        current_test_case = {
            "id": test_case["id"],
            "scene_name": test_case["scene_name"],
            "action_labels": test_case.get("action_labels", []),
            "user_instruction": test_case.get("user_instruction", ""),
            "planning_history": [],
            "available_characters": sorted(test_case.get("characters", [])),
            "current_states": test_case.get("current_states", []),
        }

        # Tracking completed events
        completed_event_ids = set()
        # Tracking all matching events (including those in the ongoing state)- Only includes events that meet dependencies
        matched_event_ids = set()
        # 🆕 Tracks all generated events, including those with dependency violations)- Used for event coverage statistics
        all_generated_event_ids = set()
        # Tracks events with dependency violations (generated when dependencies are not met)
        dependency_violations = set()  # event_id Gather
        # Tracks failed dependency tests (dependencies violated)
        failed_dependency_tests = set()  # Storage (after_id, before_id) Tuples
        remaining_events = list(events)  # More events need to be generated

        # Continuously generates events until all events are generated or no more can be generated
        # Continues running even if it fails, to count all matching events
        max_phases = len(events) * 3  # Sets the maximum number of stages to avoid infinite loops
        consecutive_failures = 0  # Number of consecutive failures
        max_consecutive_failures = 30  # Maximum number of consecutive failures

        while remaining_events and phase_counter < max_phases:
            # ========== Phase: Generate an executable event ==========
            phase_counter += 1

            # Obtain all currently executable events (dependencies satisfied)
            executable_events = []

            # First, infer common pre-dependencies for events in the parallel group
            parallel_groups = [dep.get("parallel", []) for dep in dependencies if "parallel" in dep]
            event_to_parallel_group = {}
            for group in parallel_groups:
                for event_id in group:
                    event_to_parallel_group[event_id] = group

            for evt in remaining_events:
                evt_id = evt.get("event_id")
                # Check if dependencies are satisfied: all pre-events of evt are completed
                dependencies_met = True

                # Collect all pre-dependencies
                required_predecessors = set()

                # 1. Direct order dependencies
                for dep in dependencies:
                    if "parallel" not in dep and dep.get("before") == evt_id:
                        required_predecessors.add(dep.get("after"))

                # 2. Dependencies inferred through the parallel group
                # If evt_id is in a certain parallel group, find pre-dependencies of other members in the group
                if evt_id in event_to_parallel_group:
                    parallel_group = event_to_parallel_group[evt_id]
                    for sibling_id in parallel_group:
                        if sibling_id != evt_id:
                            # Find pre-dependencies of sibling events
                            for dep in dependencies:
                                if "parallel" not in dep and dep.get("before") == sibling_id:
                                    required_predecessors.add(dep.get("after"))

                # Check whether all prerequisites have been completed
                for required in required_predecessors:
                    if required not in completed_event_ids:
                        dependencies_met = False
                        break

                if dependencies_met:
                    executable_events.append(evt)

            if not executable_events:
                all_passed = False
                if not failure_reason:
                    failure_reason = "No executable events but remaining events exist (dependency deadlock)"
                phase_diagnostics.append(
                    {"phase": phase_counter, "type": "error", "passed": False, "issue": failure_reason}
                )
                consecutive_failures += 1
                # If there are no executable events, proceed cannot be done; exit the loop
                break

            # Generate events (no specific expectations; let the narrator choose freely)
            result = self._generate_any_executable_event(
                current_test_case, executable_events, distraction_level, repeat_idx, phase_counter
            )

            if not result["success"]:
                all_passed = False
                consecutive_failures += 1
                if not failure_reason:  # Only record the first cause of failure
                    failure_reason = result["error"]
                phase_diagnostics.append(
                    {
                        "phase": phase_counter,
                        "type": "normal",
                        "passed": False,
                        "executable_events": [
                            f"{e.get('event_id')}: {e.get('character')} - {e.get('activity')}"
                            for e in executable_events
                        ],
                        "generated": result.get("event_info"),
                        "issue": result["error"],
                        "tokens": {"prompt": result["prompt_tokens"], "completion": result["completion_tokens"]},
                    }
                )

                # Continue running: even if generation fails, simulate ongoing progress → Process completed
                # To allow events that depend on this process to continue running
                total_prompt_tokens += result["prompt_tokens"]
                total_completion_tokens += result["completion_tokens"]

                if executable_events:
                    # Add failed events to the history for simulation purposes)
                    failed_event = executable_events[0]
                    failed_event_id = failed_event.get("event_id")
                    failed_char = failed_event.get("character")
                    failed_activity = failed_event.get("activity")

                    from benchmark.comparison.derivation import resolve_activity_to_specific

                    # ========== Phase A: Mark as ongoing state ==========
                    current_test_case["planning_history"].append(
                        {
                            "idx": len(current_test_case["planning_history"]),
                            "type": "actor",
                            "distraction_level": 0,
                            "state": "ongoing",
                            "characters": [failed_char],
                            "activity": resolve_activity_to_specific(failed_activity),
                            "details": [],
                        }
                    )

                    # Update available_characters
                    current_test_case["available_characters"] = sorted(
                        [c for c in current_test_case["available_characters"] if c != failed_char]
                    )

                    # ========== Phase C: Transition to the completed state ==========
                    #  (Skip the dependency tests for Phase B, as the generation has failed)
                    current_test_case["planning_history"][-1]["state"] = "completed"

                    # Restore available_characters
                    current_test_case["available_characters"].append(failed_char)
                    current_test_case["available_characters"].sort()

                    # Mark as completed (allow events that depend on it to execute)
                    completed_event_ids.add(failed_event_id)

                    # Remove from remaining events
                    remaining_events = [e for e in remaining_events if e.get("event_id") != failed_event_id]

                    self.logger.warning(
                        f"Phase {phase_counter}: Event {failed_event_id} generation failed, but marked as completed (ongoing→completed) to allow dependent events to proceed"
                    )

                # Stop the loop if too many consecutive failures occur
                if consecutive_failures >= max_consecutive_failures:
                    self.logger.warning(f"Stopping after {consecutive_failures} consecutive failures")
                    break

                continue  # Proceed to the next round without breaking

            # Record successful generation (support for single events or parallel groups)
            # Reset the consecutive failure count
            consecutive_failures = 0

            is_parallel_group = result.get("is_parallel_group", False)

            if is_parallel_group:
                # Parallel event groups
                matched_event_ids_from_result = result.get("matched_event_ids", [])
                matched_events_list = result.get("matched_events", [])

                # 🆕 Record all events in the parallel groups as generated
                for event_id in matched_event_ids_from_result:
                    all_generated_event_ids.add(event_id)

                # Check if each event in the parallel groups is present in the executable list
                executable_event_ids = {e.get("event_id") for e in executable_events}
                parallel_group_passed = True
                for event_id in matched_event_ids_from_result:
                    if event_id not in executable_event_ids:
                        # Dependency not met, marked as dependency violation
                        dependency_violations.add(event_id)
                        parallel_group_passed = False
                        self.logger.warning(
                            f"Phase {phase_counter}: Parallel event {event_id} generated but dependencies not met (dependency violation)"
                        )
                    elif event_id in dependency_violations:
                        # Even though it’s in the executable list now, it has already been marked as a dependency violation
                        parallel_group_passed = False
                        self.logger.warning(
                            f"Phase {phase_counter}: Parallel event {event_id} generated again after dependency violation, still marked as failed"
                        )

                phase_diagnostics.append(
                    {
                        "phase": phase_counter,
                        "type": "normal",
                        "passed": parallel_group_passed,
                        "executable_events": [
                            f"{e.get('event_id')}: {e.get('character')} - {e.get('activity')}"
                            for e in executable_events
                        ],
                        "generated": result.get("event_info"),
                        "matched_event_ids": matched_event_ids_from_result,
                        "is_parallel_group": True,
                        "tokens": {"prompt": result["prompt_tokens"], "completion": result["completion_tokens"]},
                    }
                )

                if not parallel_group_passed:
                    all_passed = False
                    if not failure_reason:
                        failure_reason = f"Parallel event group contains events with dependency violations: {[eid for eid in matched_event_ids_from_result if eid in dependency_violations]}"

                total_prompt_tokens += result["prompt_tokens"]
                total_completion_tokens += result["completion_tokens"]
                generated_events.append(result["event_info"])

                # Process each event in the parallel group
                from benchmark.comparison.derivation import resolve_activity_to_specific

                for matched_event in matched_events_list:
                    generated_char = matched_event.get("character")
                    generated_activity = matched_event.get("activity")
                    event_id = matched_event.get("event_id")

                    # Only marked as matched when there are no dependency violations
                    if event_id not in dependency_violations:
                        matched_event_ids.add(event_id)

                    # Added as completed status (parallel events are directly completed)
                    current_test_case["planning_history"].append(
                        {
                            "idx": len(current_test_case["planning_history"]),
                            "type": "actor",
                            "distraction_level": 0,
                            "state": "completed",
                            "characters": [generated_char],
                            "activity": resolve_activity_to_specific(generated_activity),
                            "details": [],
                        }
                    )

                    # Marked as completed
                    completed_event_ids.add(event_id)

                # Remove all matching events from the remaining events
                remaining_events = [
                    e for e in remaining_events if e.get("event_id") not in matched_event_ids_from_result
                ]

                # Continue to the next round of loops (skip the following steps) ongoing/completed Process)
                continue

            else:
                # Single event (may match multiple events, if it’s a composite description)
                generated_event_id = result.get("matched_event_id")

                # Check if multiple events were matched (result of strategy 2)
                matched_event_ids_from_result = result.get("matched_event_ids", [])

                # Check if the matching events are in the executable list
                # If not present, it indicates that dependencies were not met, marked as a dependency violation
                executable_event_ids = {e.get("event_id") for e in executable_events}

                if matched_event_ids_from_result and len(matched_event_ids_from_result) > 1:
                    # Multiple events were matched (result of composite description parsing)
                    for event_id in matched_event_ids_from_result:
                        # 🆕 Regardless of whether dependencies were met or not, it is recorded as generated
                        all_generated_event_ids.add(event_id)

                        # Check whether each matched event is in the executable list
                        if event_id not in executable_event_ids:
                            # Dependency not met, marked as dependency violation
                            dependency_violations.add(event_id)
                            self.logger.warning(
                                f"Phase {phase_counter}: Event {event_id} generated but dependencies not met (dependency violation)"
                            )
                        elif event_id not in dependency_violations:
                            # Only those events that are in the executable list and have no previous dependency violations are considered a successful match
                            matched_event_ids.add(event_id)
                        else:
                            # Even if an event is still in the executable list, if it has previously been marked as a violation, it is still marked as a failure
                            self.logger.warning(
                                f"Phase {phase_counter}: Event {event_id} generated again after dependency violation, still marked as failed"
                            )
                    # Use the first event as the main event (for dependency testing)
                    generated_event_id = matched_event_ids_from_result[0]

                    # Remove the other matching events from the remaining_events
                    for additional_id in matched_event_ids_from_result[1:]:
                        remaining_events = [e for e in remaining_events if e.get("event_id") != additional_id]
                else:
                    # Single event match
                    # 🆕 Regardless of whether dependencies were met or not, it is recorded as generated
                    all_generated_event_ids.add(generated_event_id)

                    # Check whether it is in the executable list
                    if generated_event_id not in executable_event_ids:
                        # Dependency not met, marked as dependency violation
                        dependency_violations.add(generated_event_id)
                        self.logger.warning(
                            f"Phase {phase_counter}: Event {generated_event_id} generated but dependencies not met (dependency violation)"
                        )
                    elif generated_event_id not in dependency_violations:
                        # Only those events that are in the executable list and have no previous dependency violations are considered a successful match
                        matched_event_ids.add(generated_event_id)
                    else:
                        # Even if an event is still in the executable list, if it has previously been marked as a violation, it is still marked as a failure
                        self.logger.warning(
                            f"Phase {phase_counter}: Event {generated_event_id} generated again after dependency violation, still marked as failed"
                        )

            matched_event = next((e for e in executable_events if e.get("event_id") == generated_event_id), None)

            # If the matched event is not in the executable list, it indicates that dependencies were not met
            if not matched_event:
                all_passed = False
                if generated_event_id not in dependency_violations:
                    dependency_violations.add(generated_event_id)
                if not failure_reason:
                    failure_reason = f"Event {generated_event_id} generated but dependencies not met"

                # Record diagnostic information
                phase_diagnostics.append(
                    {
                        "phase": phase_counter,
                        "type": "normal",
                        "passed": False,
                        "executable_events": [
                            f"{e.get('event_id')}: {e.get('character')} - {e.get('activity')}"
                            for e in executable_events
                        ],
                        "generated": result.get("event_info"),
                        "matched_event_id": generated_event_id,
                        "issue": f"Event {generated_event_id} generated but dependencies not met (dependency violation)",
                        "tokens": {"prompt": result["prompt_tokens"], "completion": result["completion_tokens"]},
                    }
                )

                total_prompt_tokens += result["prompt_tokens"]
                total_completion_tokens += result["completion_tokens"]

                # Continues running without interruption, but marked as failed
                # Skips subsequent processing and proceeds to the next round
                continue

            # Matched events are in the executable list, processed normally
            phase_info = {
                "phase": phase_counter,
                "type": "normal",
                "passed": True,
                "executable_events": [
                    f"{e.get('event_id')}: {e.get('character')} - {e.get('activity')}" for e in executable_events
                ],
                "generated": result.get("event_info"),
                "matched_event_id": generated_event_id,
                "tokens": {"prompt": result["prompt_tokens"], "completion": result["completion_tokens"]},
            }

            # If multiple events are matched, additional information is added
            if matched_event_ids_from_result and len(matched_event_ids_from_result) > 1:
                phase_info["matched_event_ids"] = matched_event_ids_from_result
                phase_info["is_composite"] = True
                phase_info["note"] = f"Composite description matched {len(matched_event_ids_from_result)} events"

            phase_diagnostics.append(phase_info)

            total_prompt_tokens += result["prompt_tokens"]
            total_completion_tokens += result["completion_tokens"]
            generated_events.append(result["event_info"])

            # Matched events found
            generated_char = matched_event.get("character")
            generated_activity = matched_event.get("activity")

            # ========== Phase A: Mark as ongoing state ==========
            from benchmark.comparison.derivation import resolve_activity_to_specific

            current_test_case["planning_history"].append(
                {
                    "idx": len(current_test_case["planning_history"]),
                    "type": "actor",
                    "distraction_level": 0,
                    "state": "ongoing",
                    "characters": [generated_char],
                    "activity": resolve_activity_to_specific(generated_activity),
                    "details": [],
                }
            )

            # Updates available_characters (removes the currently executing character)
            current_test_case["available_characters"] = sorted(
                [c for c in current_test_case["available_characters"] if c != generated_char]
            )

            # ========== Phase B: Testing dependencies (ongoing)==========
            # During the ongoing process, other events dependent on this event cannot be executed
            # Build a dependency graph to identify which events directly depend on the current event
            dependent_event_ids = []
            for dep in dependencies:
                if "parallel" not in dep and dep.get("after") == generated_event_id:
                    dependent_event_ids.append(dep.get("before"))

            if dependent_event_ids:
                # Identify characters that are dependent on this event
                dependent_chars = set()
                for evt in remaining_events:
                    if evt.get("event_id") in dependent_event_ids:
                        dependent_chars.add(evt.get("character"))

                # Testing: Calculating executable events (excluding ongoing events)
                phase_counter += 1
                test_result = self._test_dependency_during_ongoing_flexible(
                    current_test_case,
                    generated_event_id,
                    dependent_chars,
                    remaining_events,
                    dependencies,
                    completed_event_ids,
                    distraction_level,
                    repeat_idx,
                    phase_counter,
                )

                # Record dependency tests for diagnosis
                dep_phase_info = {
                    "phase": phase_counter,
                    "type": "dependency_test",
                    "passed": test_result["success"],
                    "context": f"{generated_char}'s event '{generated_activity}' is ongoing",
                    "dependent_characters": list(dependent_chars),
                    "generated": test_result.get("generated_event"),
                    "tokens": {"prompt": test_result["prompt_tokens"], "completion": test_result["completion_tokens"]},
                }

                if not test_result["success"]:
                    all_passed = False
                    consecutive_failures += 1
                    if not failure_reason:  # Only record the first cause of failure
                        failure_reason = test_result["error"]
                    dep_phase_info["issue"] = test_result.get("issue", "Dependency test failed")

                    # Record failed dependency tests
                    # If an event is generated, find the corresponding dependencies
                    if test_result.get("generated_event") and "matched_event_id" in test_result:
                        violated_event_id = test_result["matched_event_id"]
                        # Find the dependencies from generated_event_id to violated_event_id (after) For dependencies with violated_event_id (before) Dependency
                        for dep in dependencies:
                            if "parallel" not in dep:
                                if dep.get("after") == generated_event_id and dep.get("before") == violated_event_id:
                                    failed_dependency_tests.add((dep.get("after"), dep.get("before")))
                                    self.logger.warning(
                                        f"Phase {phase_counter}: Dependency violation detected: {dep.get('after')} → {dep.get('before')}"
                                    )
                                    break
                    else:
                        # Mark all dependencies that depend on generated_event_id and are in dependent_chars as failed
                        # For all dependencys generated_event_id And the character is dependent_chars Dependency in place ,  Mark as Failed
                        for dep in dependencies:
                            if "parallel" not in dep and dep.get("after") == generated_event_id:
                                # Check if the relying character is in dependent_chars in
                                for evt in remaining_events:
                                    if evt.get("event_id") == dep.get("before"):
                                        if evt.get("character") in dependent_chars:
                                            failed_dependency_tests.add((dep.get("after"), dep.get("before")))
                                            self.logger.warning(
                                                f"Phase {phase_counter}: Dependency violation detected (no matched event): {dep.get('after')} → {dep.get('before')}"
                                            )
                                            break

                    # Continue running: Ignore events generated from failed steps, continue marking ongoing events as completed
                    total_prompt_tokens += test_result["prompt_tokens"]
                    total_completion_tokens += test_result["completion_tokens"]

                    phase_diagnostics.append(dep_phase_info)

                    # Stop the loop if too many consecutive failures occur
                    if consecutive_failures >= max_consecutive_failures:
                        self.logger.warning(f"Stopping after {consecutive_failures} consecutive failures")
                        break

                    # Do not break, continue with stage C (convert ongoing events to completed)
                    # This allows the test to continue running even if the dependency test fails
                    # Skipping the part that handles parallel events (because the dependency test failed)
                else:
                    # Dependency test passed, processing continues normally
                    consecutive_failures = 0  # Reset the consecutive failure count
                    if "note" in test_result:
                        dep_phase_info["note"] = test_result["note"]

                    phase_diagnostics.append(dep_phase_info)
                    total_prompt_tokens += test_result["prompt_tokens"]
                    total_completion_tokens += test_result["completion_tokens"]

                    # 🔥 If a valid independent event is generated, it is processed as a real event!
                    if test_result.get("generated_event") and "matched_event_id" in test_result:
                        # This is a valid parallel event that should be recorded
                        parallel_event_id = test_result["matched_event_id"]
                        parallel_char = test_result["generated_event"]["character"]
                        parallel_activity = test_result["generated_event"]["activity"]

                        # Added to the completed state (parallel events can be directly completed)
                        current_test_case["planning_history"].append(
                            {
                                "idx": len(current_test_case["planning_history"]),
                                "type": "actor",
                                "distraction_level": 0,
                                "state": "completed",
                                "characters": [parallel_char],
                                "activity": parallel_activity,
                                "details": [],
                            }
                        )

                        # Marked as completed
                        completed_event_ids.add(parallel_event_id)
                        matched_event_ids.add(parallel_event_id)

                        # Remove from remaining events
                        remaining_events = [e for e in remaining_events if e.get("event_id") != parallel_event_id]

                        generated_events.append(test_result["generated_event"])
                        dep_phase_info["parallel_event_processed"] = True

            # ========== Phase C: Transition to the completed state ==========
            current_test_case["planning_history"][-1]["state"] = "completed"

            # Recovering available_characters (Characters are available again after completing tasks)
            current_test_case["available_characters"].append(generated_char)
            current_test_case["available_characters"].sort()

            # Marking events as completed (It’s only truly completed now)!)
            completed_event_ids.add(generated_event_id)

            # Remove from remaining events
            remaining_events = [e for e in remaining_events if e.get("event_id") != generated_event_id]

        # Loop ends, checking if all events have been generated
        if all_passed and remaining_events:
            all_passed = False
            failure_reason = f"Not all events generated. Remaining: {[e.get('event_id') for e in remaining_events]}"

        # Counting the number of matched events and dependencies
        # Using matched_event_ids instead of completed_event_ids, as some events may be in the ongoing state
        # Excluding events with dependency violations (generated incorrectly when dependencies are not met)
        valid_matched_event_ids = matched_event_ids - dependency_violations
        matched_events_count = len(valid_matched_event_ids)
        expected_events_count = len(events)

        # 🆕 New: Pure event coverage statistics (ignoring dependencies, order, etc.)
        # As soon as the events are generated, it is considered a successful match
        events_generated_count = len(all_generated_event_ids)  # Use the new set to contain all generated events

        # Count the number of dependencies that are satisfied (only consider valid matches, excluding failed dependency tests)
        matched_dependencies_count = 0
        expected_dependencies_count = len(dependencies)

        for dep in dependencies:
            if "parallel" in dep:
                # Parallel dependencies: Check whether all events in the group have successfully matched without any dependency violations
                parallel_group = dep.get("parallel", [])
                if all(eid in matched_event_ids and eid not in dependency_violations for eid in parallel_group):
                    matched_dependencies_count += 1
            else:
                # Sequential dependencies: Check whether both after and before have successfully matched in order, without any dependency violations
                # And ensure that the dependency tests have not failed
                after_id = dep.get("after")
                before_id = dep.get("before")
                dep_tuple = (after_id, before_id)

                # If the dependency test fails, even if all events are completed, it should not be considered a successful match
                if dep_tuple in failed_dependency_tests:
                    continue

                # Use matched_event_ids instead of completed_event_ids
                # Only events that have successfully matched are considered as satisfying the dependency
                if (
                    after_id in matched_event_ids
                    and before_id in matched_event_ids
                    and after_id not in dependency_violations
                    and before_id not in dependency_violations
                ):
                    matched_dependencies_count += 1

        # Determine the reasons for the failures
        failure_reason_type = None
        if not all_passed:
            if "dependency" in (failure_reason or "").lower():
                failure_reason_type = "dependency_test_failed"
            elif "activity" in (failure_reason or "").lower() or "mismatch" in (failure_reason or "").lower():
                failure_reason_type = "activity_mismatch"
            elif "character" in (failure_reason or "").lower():
                failure_reason_type = "character_mismatch"
            else:
                failure_reason_type = "unknown"

        # Construct diagnostic information
        diagnostics = {
            "phases": phase_diagnostics,
            "total_phases": phase_counter,
            "passed_phases": sum(1 for p in phase_diagnostics if p.get("passed")),
            "failed_phase": next((p["phase"] for p in phase_diagnostics if not p.get("passed")), None),
            "matched_events": matched_events_count,
            "expected_events": expected_events_count,
            "events_generated": events_generated_count,  # 🆕 Statistically analyze the coverage of pure events
            "matched_dependencies": matched_dependencies_count,
            "expected_dependencies": expected_dependencies_count,
            "dependency_violations": list(dependency_violations),  # Record events related to dependency violations
            "failed_dependency_tests": [
                {"after": after, "before": before} for after, before in failed_dependency_tests
            ],  # Record failed dependency tests
        }

        # Return results
        return {
            "passed": all_passed,
            "error": not all_passed and failure_reason is not None,
            "error_message": failure_reason if failure_reason else None,
            "failure_reason": failure_reason_type,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "event_count": len(generated_events),
            "dependencies_count": len(dependencies),
            "matched_events": matched_events_count,
            "expected_events": expected_events_count,
            "events_generated": events_generated_count,  # 🆕 Statistically analyze the coverage of pure events
            "matched_dependencies": matched_dependencies_count,
            "expected_dependencies": expected_dependencies_count,
            "planning_result": {
                "type": "incremental_with_dependencies",
                "generated_events": generated_events,
                "expected_events": len(events),
                "steps_completed": len(generated_events),
                "dependencies_tested": True,
            },
            "diagnostics": diagnostics,
        }

    def _create_error_result(
        self, error_message: str, failure_reason: str = "llm_error", diagnostics: dict = None
    ) -> dict:
        """Create a dictionary of error results with diagnostic information)"""
        return {
            "passed": False,
            "error": True,
            "error_message": error_message,
            "failure_reason": failure_reason,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "event_count": 0,
            "dependencies_count": 0,
            "planning_result": {},
            "diagnostics": diagnostics or {},
        }

    def _parse_composite_activity(self, activity: str, characters: list) -> dict:
        """Parse compound descriptions involving multiple characters and actions

        Example:"Amir sits on the couch and reads a book while Benjamin sits on chair and types on the laptop"
        Should be parsed as:
        {
            "Amir": "sits on the couch and reads a book",
            "Benjamin": "sits on chair and types on the laptop"
        }

        Args:
            activity: Compound activity description
            characters: List of involved characters

        Returns:
            Dictionary with character names as keys and action descriptions as values
        """
        import re

        parsed = {}

        # Strategy 1: Try to pass through "while" Split (strongest separator)
        while_parts = re.split(r"\s+while\s+", activity, flags=re.IGNORECASE)

        if len(while_parts) > 1:
            # Yes "while" Separate and find characters for each section
            for part in while_parts:
                for char in characters:
                    if char.lower() in part.lower():
                        # Extract the actions of that character
                        char_pos = part.lower().find(char.lower())
                        after_char = part[char_pos + len(char) :].strip()
                        # Remove the leading comma or connecting words
                        after_char = re.sub(r"^[,\s]+", "", after_char)
                        if after_char:
                            parsed[char] = after_char
        else:
            # Strategy 2: None "while", Try to separate by the character name itself
            # Find the position of all characters in the text
            char_positions = []
            for char in characters:
                char_lower = char.lower()
                activity_lower = activity.lower()
                pos = activity_lower.find(char_lower)
                if pos != -1:
                    char_positions.append((pos, char))

            # Sort by position
            char_positions.sort()

            # Extract the action for each character (text between the name of that character and the next character)
            for i, (pos, char) in enumerate(char_positions):
                start = pos + len(char)

                # Determine the end position (the position of the next character, or the end of the string)
                if i + 1 < len(char_positions):
                    end = char_positions[i + 1][0]
                else:
                    end = len(activity)

                action_text = activity[start:end].strip()

                # Clean up the action text: remove the leading comma, "and"Wait for
                action_text = re.sub(r"^[,\s]+", "", action_text)
                action_text = re.sub(r"^and\s+", "", action_text, flags=re.IGNORECASE)

                # Remove any possible connecting words at the end
                action_text = re.sub(r"\s+(while|and)\s*$", "", action_text, flags=re.IGNORECASE)
                action_text = action_text.rstrip(",").strip()

                if action_text:
                    parsed[char] = action_text

        # If some characters aren’t resolved, try assigning them a common action
        missing_chars = [c for c in characters if c not in parsed]
        if missing_chars:
            # Check if all characters are doing the same thing (e.g.,...) "A, B, and C sit down")
            # Determine the action after finding the last character
            last_char_pos = -1
            last_char = None
            for c in characters:
                pos = activity.lower().rfind(c.lower())
                if pos > last_char_pos:
                    last_char_pos = pos
                    last_char = c

            if last_char_pos != -1 and last_char:
                action = activity[last_char_pos + len(last_char) :].strip()
                action = re.sub(r"^[,\s]+", "", action)
                action = re.sub(r"^and\s+", "", action, flags=re.IGNORECASE)

                # If valid actions are extracted, assign them to all missing characters
                if action and len(action) > 3:  # At least have some substance
                    for c in missing_chars:
                        parsed[c] = action

        # Final fallback: If nothing is resolved, return the original activity
        if not parsed:
            for char in characters:
                parsed[char] = activity

        return parsed

    def _build_dependency_graph(self, events: list, dependencies: list) -> dict:
        """Building a dependency graph: event_id -> Subsequent event_ids that depend on this event_id

        Args:
            events: Event list
            dependencies: Dependency list

        Returns:
            Dependency graph dictionary, e.g:{"e1": ["e2", "e3"], "e2": ["e4"]}
        """
        graph = {}

        for dep in dependencies:
            if isinstance(dep, dict) and "after" in dep and "before" in dep:
                after = dep.get("after")
                before = dep.get("before")
                if after and before:
                    if after not in graph:
                        graph[after] = []
                    graph[after].append(before)

        return graph

    def _generate_any_executable_event(
        self, current_test_case: dict, executable_events: list, distraction_level: int, repeat_idx: int, phase_num: int
    ) -> dict:
        """Generate an executable event and verify it’s in the executable list

        Args:
            current_test_case: Current test cases
            executable_events: List of currently executable events
            distraction_level: Interference level
            repeat_idx: Duplicate index
            phase_num: Phase number

        Returns:
            Generate result dictionary
        """
        from benchmark.comparison.action_eval import match_activity_with_or_options

        try:
            # Do not set expected_next_event, allowing narrator to choose freely
            _setup_modules_and_scenario_mod(
                self, current_test_case, distraction_level, repeat_idx, narrator_type=self._get_baseline_narrator_type()
            )
            self.narrator.scene_description_type = "narrative"

            event_or_group, prompt_tokens, completion_tokens = self.narrator.generate_subsequent_plan(
                self.scene_description,
                self.available_action_labels,
                self.user_instruction,
                with_usage=True,
            )

            # Verification: Are the generated events in the executable list
            if event_or_group is None:
                return {
                    "success": False,
                    "error": f"Phase {phase_num}: Narrator returned null but {len(executable_events)} events are executable",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }

            generated_chars = [c.name for c in getattr(event_or_group, "involved_characters", [])]
            generated_activity = getattr(event_or_group, "activity", "")

            # Attempt to match: Supports matching a single event or multiple characters concurrently
            matched_events = []

            # Strategy 1: Attempt to match a single event (original logic)
            for exe_evt in executable_events:
                exe_char = exe_evt.get("character")
                exe_activity = exe_evt.get("activity", "")

                # Matching conditions: Character match, and activity match
                if exe_char in generated_chars and match_activity_with_or_options(generated_activity, exe_activity):
                    matched_events.append(exe_evt)
                    break

            # Strategy 2: If a single event matches successfully, check if it contains multiple actions (composite description)
            # Example: "Amir picks up a bottle and hands it over to Benjamin, who then puts the bottle down"
            if len(matched_events) == 1 and len(generated_chars) == 1:
                # Check if the generated activities contain multiple actions (via keywords like "and", "then", "who" Wait for)
                activity_lower = generated_activity.lower()
                has_multiple_actions = any(
                    keyword in activity_lower
                    for keyword in [" and ", " then ", " who ", " which ", " after ", " before ", ", who", ", which"]
                )

                if has_multiple_actions:
                    # Try to extract multiple actions from the composite description and match them with multiple expected events
                    # Obtain all expected events (not just those that are executable)
                    expected_timeline = current_test_case.get("expected_timeline", {})
                    all_expected_events = expected_timeline.get("events", [])

                    # Parse the composite description to extract multiple action segments
                    # Use regular expressions to split the composite description
                    import re

                    # Split by common connectives
                    action_parts = re.split(
                        r"\s+(and|then|, who|, which|after|before)\s+", activity_lower, flags=re.IGNORECASE
                    )
                    # Filter out connectives and keep only the action parts
                    action_segments = [
                        part.strip()
                        for part in action_parts
                        if part and part.strip() not in ["and", "then", "who", "which", "after", "before"]
                    ]

                    # If multiple action segments are split out, try to match them
                    if len(action_segments) > 1:
                        additional_matched = []
                        for exp_event in all_expected_events:
                            exp_char = exp_event.get("character", "")
                            exp_activity = exp_event.get("activity", "")
                            exp_id = exp_event.get("event_id", "")

                            # Skip the events that have already been matched
                            if any(e.get("event_id") == exp_id for e in matched_events):
                                continue

                            # Check if the expected events are in the executable list
                            if not any(e.get("event_id") == exp_id for e in executable_events):
                                continue

                            # Check if any segment in the generated activity description matches the expected actions
                            if exp_char in generated_chars:
                                # Check if any action segment matches the expected actions
                                for segment in action_segments:
                                    # Use match_activity_with_or_options for matching
                                    if match_activity_with_or_options(segment, exp_activity):
                                        additional_matched.append(exp_event)
                                        break

                        # If additional matching events are found, add them to the matching list
                        if additional_matched:
                            matched_events.extend(additional_matched)

            # Strategy 3: If no matches are found and multiple characters are generated, attempt to parse the composite description and match multiple events
            if not matched_events and len(generated_chars) > 1:
                # Parse the composite description and extract the actions for each character
                parsed_actions = self._parse_composite_activity(generated_activity, generated_chars)

                # Try to find matching expected events for each character’s actions
                potential_parallel_events = []
                matched_chars = set()

                for char_name, char_action in parsed_actions.items():
                    # Look for matching events for that character in the executable events
                    for exe_evt in executable_events:
                        exe_char = exe_evt.get("character")
                        exe_activity = exe_evt.get("activity", "")

                        if exe_char == char_name and exe_char not in matched_chars:
                            # Check if the extracted actions match the expected actions
                            if match_activity_with_or_options(char_action, exe_activity):
                                potential_parallel_events.append(exe_evt)
                                matched_chars.add(char_name)
                                break

                # If at least one event is matched, accept it (partial matches are allowed)
                if potential_parallel_events:
                    matched_events = potential_parallel_events

            # Strategy 4: If Strategy 1 has already matched 1 event, check if there are any other executable events that perform the same action
            # For example: Generate "Amir, Benjamin, Charles, David, Evan - stand up simultaneously"
            # Executable event: e28: Charles - stand up, e29: David - stand up, e30: Evan - stand up
            # Should match all of these events
            if len(matched_events) == 1 and len(generated_chars) > 1:
                first_matched = matched_events[0]
                first_matched_activity = first_matched.get("activity", "")

                # Find events in executable events where other characters perform the same actions
                for exe_evt in executable_events:
                    exe_char = exe_evt.get("character")
                    exe_activity = exe_evt.get("activity", "")

                    # Skip the events that have already been matched
                    if any(e.get("event_id") == exe_evt.get("event_id") for e in matched_events):
                        continue

                    # Check: Characters are in the generated character list, and the action matches the first matching event
                    if exe_char in generated_chars and match_activity_with_or_options(
                        exe_activity, first_matched_activity
                    ):
                        matched_events.append(exe_evt)

            if not matched_events:
                # Generated events are not in the executable list
                exe_events_str = ", ".join([f"{e.get('character')} - {e.get('activity')}" for e in executable_events])
                return {
                    "success": False,
                    "error": (
                        f"Phase {phase_num}: Generated event not in executable list\n"
                        f"  Generated: {generated_chars} - {generated_activity}\n"
                        f"  Executable events: {exe_events_str}"
                    ),
                    "event_info": {
                        "character": generated_chars[0] if generated_chars else "?",
                        "activity": generated_activity,
                        "matched": False,
                    },
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }

            # Successful match (single or parallel groups)
            if len(matched_events) == 1:
                # Single event match
                matched_event = matched_events[0]
                return {
                    "success": True,
                    "matched_event_id": matched_event.get("event_id"),
                    "matched_event_ids": [matched_event.get("event_id")],  # Also provide a list format
                    "matched_events": [matched_event],  # Provide a complete list of events
                    "event_info": {
                        "character": generated_chars[0] if generated_chars else "?",
                        "activity": generated_activity,
                        "matched": True,
                        "matched_to": f"{matched_event.get('event_id')}: {matched_event.get('character')} - {matched_event.get('activity')}",
                    },
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
            else:
                # Parallel event groups match
                matched_ids = [e.get("event_id") for e in matched_events]
                matched_desc = [
                    f"{e.get('event_id')}: {e.get('character')} - {e.get('activity')}" for e in matched_events
                ]
                return {
                    "success": True,
                    "matched_event_ids": matched_ids,  # Note: Plural form
                    "matched_events": matched_events,  # Return all matching events
                    "is_parallel_group": True,
                    "event_info": {
                        "characters": generated_chars,  # Note: Plural form
                        "activity": generated_activity,
                        "matched": True,
                        "matched_to": " + ".join(matched_desc),
                    },
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }

        except Exception as e:
            self.logger.error(f"Error generating event at phase {phase_num}: {e}")
            return {
                "success": False,
                "error": f"Phase {phase_num}: Error generating event: {str(e)}",
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

    def _generate_and_validate_event(
        self,
        current_test_case: dict,
        expected_char: str,
        expected_activity: str,
        distraction_level: int,
        repeat_idx: int,
        step_idx: int,
    ) -> dict:
        """Generate and validate one event

        Returns:
            Result dictionary, including success, event_info, prompt_tokens, completion_tokens, error
        """
        try:
            # Set expected_next_event
            current_test_case["expected_next_event"] = {"character": expected_char, "activity": expected_activity}

            # Generate event
            _setup_modules_and_scenario_mod(
                self, current_test_case, distraction_level, repeat_idx, narrator_type=self._get_baseline_narrator_type()
            )
            self.narrator.scene_description_type = "narrative"

            event_or_group, prompt_tokens, completion_tokens = self.narrator.generate_subsequent_plan(
                self.scene_description,
                self.available_action_labels,
                self.user_instruction,
                with_usage=True,
            )

            # Verify generated event
            if event_or_group is None:
                return {
                    "success": False,
                    "error": f"Step {step_idx}: Expected event for {expected_char}, but got None",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }

            generated_chars = [c.name for c in getattr(event_or_group, "involved_characters", [])]
            generated_activity = getattr(event_or_group, "activity", "")

            # Check if characters and actions match
            char_match = expected_char in generated_chars
            activity_match = match_activity_with_or_options(generated_activity, expected_activity)

            if not (char_match and activity_match):
                return {
                    "success": False,
                    "error": (
                        f"Step {step_idx}: Event mismatch\n"
                        f"  Expected: {expected_char} - {expected_activity}\n"
                        f"  Generated: {generated_chars} - {generated_activity}"
                    ),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }

            return {
                "success": True,
                "event_info": {"character": expected_char, "activity": generated_activity, "matched": True},
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }

        except Exception as e:
            self.logger.error(f"Error generating event at step {step_idx}: {e}")
            return {
                "success": False,
                "error": f"Step {step_idx}: Error generating event: {str(e)}",
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

    def _test_dependency_during_ongoing_flexible(
        self,
        current_test_case: dict,
        ongoing_event_id: str,
        dependent_chars: set,
        remaining_events: list,
        dependencies: list,
        completed_event_ids: set,
        distraction_level: int,
        repeat_idx: int,
        phase_num: int,
    ) -> dict:
        """Test dependencies: During ongoing period, characters dependent on this event cannot generate events (flexible version)

        Args:
            current_test_case: Current test cases
            ongoing_event_id: ID of ongoing event
            dependent_chars: Set of characters directly dependent on this event
            remaining_events: List of remaining events
            dependencies: All dependency relationships
            completed_event_ids: Set of completed event IDs
            distraction_level: Interference level
            repeat_idx: Duplicate index
            phase_num: Phase number

        Returns:
            Test result dictionary
        """
        from benchmark.comparison.action_eval import match_activity_with_or_options

        try:
            # Calculate executable events (ongoing events have not been completed, so events dependent on them cannot be executed)
            executable_events = []

            # First, infer common pre-dependencies for events in the parallel group
            parallel_groups = [dep.get("parallel", []) for dep in dependencies if "parallel" in dep]
            event_to_parallel_group = {}
            for group in parallel_groups:
                for event_id in group:
                    event_to_parallel_group[event_id] = group

            for evt in remaining_events:
                evt_id = evt.get("event_id")
                dependencies_met = True

                # Collect all pre-dependencies
                required_predecessors = set()

                # 1. Direct order dependencies
                for dep in dependencies:
                    if "parallel" not in dep and dep.get("before") == evt_id:
                        required_predecessors.add(dep.get("after"))

                # 2. Dependencies inferred through the parallel group
                if evt_id in event_to_parallel_group:
                    parallel_group = event_to_parallel_group[evt_id]
                    for sibling_id in parallel_group:
                        if sibling_id != evt_id:
                            for dep in dependencies:
                                if "parallel" not in dep and dep.get("before") == sibling_id:
                                    required_predecessors.add(dep.get("after"))

                # Check whether all prerequisites have been completed (ongoing event is not yet completed)
                for required in required_predecessors:
                    if required not in completed_event_ids:
                        dependencies_met = False
                        break

                if dependencies_met:
                    executable_events.append(evt)

            # Do not set expected_next_event; let the narrator choose freely
            _setup_modules_and_scenario_mod(
                self, current_test_case, distraction_level, repeat_idx, narrator_type=self._get_baseline_narrator_type()
            )
            self.narrator.scene_description_type = "narrative"

            event_or_group, prompt_tokens, completion_tokens = self.narrator.generate_subsequent_plan(
                self.scene_description,
                self.available_action_labels,
                self.user_instruction,
                with_usage=True,
            )

            # Verify the generated events
            if event_or_group is None:
                # No events generated → PASS (Wait correctly)
                return {
                    "success": True,
                    "generated_event": None,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }

            generated_chars = [c.name for c in getattr(event_or_group, "involved_characters", [])]
            generated_activity = getattr(event_or_group, "activity", "")
            generated_char = generated_chars[0] if generated_chars else None

            # Check: Are the generated characters listed in the dependencies
            if generated_char in dependent_chars:
                # Dependency violation! Events for characters that depend on an ongoing event were generated
                return {
                    "success": False,
                    "error": (
                        f"Phase {phase_num}: Dependency violation\n"
                        f"  Event '{ongoing_event_id}' is still ongoing\n"
                        f"  But narrator generated event for: {generated_char}\n"
                        f"  {generated_char}'s next event depends on '{ongoing_event_id}' completing first"
                    ),
                    "issue": f"Generated event for {generated_char} who depends on ongoing event",
                    "generated_event": {"character": generated_char, "activity": generated_activity},
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }

            # Events for independent characters were generated → PASS (Allow parallel processing)
            # However, it is necessary to verify that this event is indeed in the executable list
            matched_event = None
            for exe_evt in executable_events:
                if exe_evt.get("character") == generated_char and match_activity_with_or_options(
                    generated_activity, exe_evt.get("activity", "")
                ):
                    matched_event = exe_evt
                    break

            if not matched_event:
                # Incompatable events were generated
                return {
                    "success": False,
                    "error": f"Phase {phase_num}: Generated event not in executable list during ongoing test",
                    "issue": f"Generated {generated_char} - {generated_activity} which is not executable",
                    "generated_event": {"character": generated_char, "activity": generated_activity},
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }

            # Executable independent events were generated → PASS (This event will be processed in reality)
            return {
                "success": True,
                "note": f"Generated event for independent character {generated_char} (parallel execution OK)",
                "matched_event_id": matched_event.get("event_id"),
                "generated_event": {"character": generated_char, "activity": generated_activity},
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }

        except Exception as e:
            self.logger.error(f"Error in dependency test at phase {phase_num}: {e}")
            return {
                "success": False,
                "error": f"Phase {phase_num}: Error during dependency test: {str(e)}",
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

    def _test_dependency_during_ongoing(
        self,
        current_test_case: dict,
        ongoing_event_id: str,
        dependent_event_ids: list,
        distraction_level: int,
        repeat_idx: int,
        step_idx: int,
        ongoing_char: str,
        dependent_chars: set,
    ) -> dict:
        """When the test is in the ongoing state, the subsequent events that depend on this event should not be generated (return null)

        Only check whether the characters that depend on the current ongoing event are correctly prevented.
        Other independent, parallel characters can generate events normally.

        Args:
            current_test_case: Current test cases
            ongoing_event_id: ID of the ongoing event
            dependent_event_ids: IDs of subsequent events that depend on this event
            distraction_level: Interference level
            repeat_idx: Duplicate index
            step_idx: Step index
            ongoing_char: Characters that are executing the ongoing event
            dependent_chars: Set of characters that depend on this event

        Returns:
            Test result dictionary
        """
        try:
            # Do not set expected_next_event, allowing the narrator to generate freely
            # current_test_case["expected_next_event"] = {}  # Remove this restriction

            # Generate
            _setup_modules_and_scenario_mod(
                self, current_test_case, distraction_level, repeat_idx, narrator_type=self._get_baseline_narrator_type()
            )
            self.narrator.scene_description_type = "narrative"

            event_or_group, prompt_tokens, completion_tokens = self.narrator.generate_subsequent_plan(
                self.scene_description,
                self.available_action_labels,
                self.user_instruction,
                with_usage=True,
            )

            # Verification: If an event is generated, check whether it comes from a character that depends on the current ongoing event
            if event_or_group is not None:
                generated_chars = [c.name for c in getattr(event_or_group, "involved_characters", [])]
                generated_activity = getattr(event_or_group, "activity", "")
                generated_char = generated_chars[0] if generated_chars else None

                # Check: If an event is generated that depends on a character, then fail
                if generated_char in dependent_chars:
                    return {
                        "success": False,
                        "error": (
                            f"Step {step_idx}: Dependency violation\n"
                            f"  Event '{ongoing_event_id}' ({ongoing_char}) is still ongoing\n"
                            f"  But narrator generated event for: {generated_char}\n"
                            f"  {generated_char}'s next event depends on {ongoing_char} completing first"
                        ),
                        "issue": f"Generated event for {generated_char} who depends on ongoing event",
                        "generated_event": {"character": generated_char, "activity": generated_activity},
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    }
                else:
                    # Generating events for independent characters is allowed (parallel execution)
                    return {
                        "success": True,
                        "note": f"Generated event for independent character {generated_char} (parallel execution OK)",
                        "generated_event": {"character": generated_char, "activity": generated_activity},
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    }

            # Returning null is also acceptable; it may mean that all available characters are waiting)
            return {
                "success": True,
                "note": "Narrator returned null (all available characters may be waiting)",
                "generated_event": None,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }

        except Exception as e:
            self.logger.error(f"Error testing dependency at step {step_idx}: {e}")
            return {
                "success": False,
                "error": f"Step {step_idx}: Error testing dependency: {str(e)}",
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

    def _run_baseline_with_derived_cases(
        self, derived_cases: List[dict], distraction_level: int, repeat_idx: int, metadata: TestMetadata
    ) -> dict:
        """Run all derived scenarios and summarize the results

        Args:
            derived_cases: List of derived scenarios
            distraction_level: Interference level
            repeat_idx: Repetition count index
            metadata: Test metadata

        Returns:
            Summarized results of the runs
        """
        baseline_derived_results = []
        baseline_all_passed = True

        for derived_case in derived_cases:
            derived_result = self._run_with_narrator(
                derived_case,
                distraction_level,
                repeat_idx,
                narrator_type=self._get_baseline_narrator_type(),
                metadata=metadata,
            )
            baseline_derived_results.append(
                {"case_file": derived_case.get("_derived_file", "unknown"), "result": derived_result}
            )

            if derived_result and not derived_result.get("error", False):
                if not derived_result.get("passed", False):
                    baseline_all_passed = False
            else:
                baseline_all_passed = False

        # Summarize the results
        return {
            "passed": baseline_all_passed,
            "error": False,
            "prompt_tokens": sum(r["result"].get("prompt_tokens", 0) for r in baseline_derived_results if r["result"]),
            "completion_tokens": sum(
                r["result"].get("completion_tokens", 0) for r in baseline_derived_results if r["result"]
            ),
            "event_count": sum(r["result"].get("event_count", 0) for r in baseline_derived_results if r["result"]),
            "dependencies_count": 0,
            "derived_cases": len(derived_cases),
            "derived_results": baseline_derived_results,
            "planning_result": {
                "type": "derived_cases",
                "total_cases": len(derived_cases),
                "passed_cases": sum(
                    1 for r in baseline_derived_results if r["result"] and r["result"].get("passed", False)
                ),
            },
        }

    def _run_timeline_narrator(
        self,
        test_case: dict,
        distraction_level: int,
        repeat_idx: int,
        metadata: TestMetadata,
        narrator_type: Optional[str] = NARRATOR_TYPE_TIMELINE,
    ) -> dict:
        """Run the timeline narrator and verify the results

        Args:
            test_case: Test case dictionary
            distraction_level: Interference level
            repeat_idx: Repetition count index
            metadata: Test metadata

        Returns:
            Run the result dictionary (containing verification information)
        """
        timeline_result = self._run_with_narrator(
            test_case, distraction_level, repeat_idx, narrator_type=narrator_type, metadata=metadata
        )

        # Special for Timeline: Verify events and dependencies (using diagnostic version)
        if not timeline_result.get("error", False):
            from .comparison.timeline_validation import validate_timeline_plan_with_diagnostics

            ev_ok, dep_ok, dep_diagnostics = validate_timeline_plan_with_diagnostics(
                self.narrator,
                self.current_test_case,
                timeline_result.get("event_or_group") if "event_or_group" in timeline_result else None,
            )
            timeline_result["event_schema_valid"] = ev_ok
            timeline_result["dependency_valid"] = dep_ok

            # Combine the diagnostic information for dependencies
            if "diagnostics" not in timeline_result:
                timeline_result["diagnostics"] = {}
            timeline_result["diagnostics"].update(dep_diagnostics)

            # Timeline based on conditions = Action coverage with dependencies fulfilled
            actions_ok = bool(timeline_result.get("actions_ok", False))
            timeline_result["passed"] = actions_ok and dep_ok

            # Determine the cause of failure
            if not timeline_result["passed"]:
                if not actions_ok:
                    timeline_result["failure_reason"] = "action_mismatch"
                elif not dep_ok:
                    timeline_result["failure_reason"] = "dependency_invalid"
                else:
                    timeline_result["failure_reason"] = "unknown"
        else:
            timeline_result["event_schema_valid"] = True
            timeline_result["dependency_valid"] = True

        return timeline_result

    def _run_with_narrator(
        self,
        test_case: dict,
        distraction_level: int,
        repeat_idx: int,
        narrator_type: Optional[str] = None,
        metadata: Optional[TestMetadata] = None,
    ) -> dict:
        """Run tests using the specified type of narrator

        Args:
            test_case: Test case dictionary
            distraction_level: Interference level
            repeat_idx: Repetition count index
            narrator_type: Narrator type("timeline" Or None)
            metadata: Test metadata

        Returns:
            Run result dictionary
        """
        try:
            # Set up the scene and modules
            _setup_modules_and_scenario_mod(self, test_case, distraction_level, repeat_idx, narrator_type)

            self.narrator.scene_description_type = "narrative"

            # Run the narrator / narrator timeline (Do Not Use event_parser)
            event_or_group, prompt_tokens, completion_tokens = self.narrator.generate_subsequent_plan(
                self.scene_description,
                self.available_action_labels,
                self.user_instruction,
                with_usage=True,
            )

            # Process results based on narrator type
            if narrator_type in TIMELINE_NARRATOR_TYPES:
                return self._process_timeline_result(event_or_group, prompt_tokens, completion_tokens)
            else:
                return self._process_baseline_result(event_or_group, prompt_tokens, completion_tokens)

        except Exception as e:
            return self._handle_narrator_error(e, narrator_type, metadata)

    def _process_timeline_result(self, event_or_group, prompt_tokens: int, completion_tokens: int) -> dict:
        """Result of processing the timeline narrator

        Args:
            event_or_group: Generated events or event groups
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens

        Returns:
            Result dictionary
        """
        from .comparison.action_eval import collect_timeline_events

        event_count = len(collect_timeline_events(self.narrator, event_or_group))
        dependencies_count = len(getattr(self.narrator, "timeline_dependencies", []))
        planning_repr = {
            "pending_event_ids": [
                getattr(e, "timeline_data", {}).get("event_id", "") for e in getattr(self.narrator, "event_list", [])
            ],
            "dependencies_count": dependencies_count,
        }

        # Action evaluation: Direct comparison of activity (support or options)- Using diagnostic version
        from .comparison.action_eval import evaluate_timeline_actions_with_diagnostics

        actions_ok, action_diagnostics = evaluate_timeline_actions_with_diagnostics(
            self.narrator,
            self.current_test_case,
            event_or_group,
        )

        return {
            "passed": actions_ok,  # Dependency verification will be merged in _run_timeline_narrator later
            "error": False,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "planning_result": planning_repr,
            "event_count": event_count,
            "dependencies_count": dependencies_count,
            "actions_ok": actions_ok,
            "diagnostics": action_diagnostics,
            "event_or_group": event_or_group,
        }

    def _process_baseline_result(self, event_or_group, prompt_tokens: int, completion_tokens: int) -> dict:
        """Processing results of baseline narrator

        Args:
            event_or_group: Generated events or event groups
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens

        Returns:
            Result dictionary
        """
        # Using is_new_event_required() Checking whether the next event should be expected
        should_require_event = self.narrator.is_new_event_required()

        if event_or_group is None:
            return self._process_null_event_result(should_require_event, prompt_tokens, completion_tokens)
        else:
            return self._process_single_event_result(event_or_group, prompt_tokens, completion_tokens)

    def _process_null_event_result(
        self, should_require_event: bool, prompt_tokens: int, completion_tokens: int
    ) -> dict:
        """Processing empty event results (narrator returns None)

        Args:
            should_require_event: Whether a new event should be required
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens

        Returns:
            Result dictionary
        """
        expected_is_null = self._is_expected_null(self.current_test_case)
        passed = expected_is_null and not should_require_event

        return {
            "passed": passed,
            "error": False,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "planning_result": None,
            "event_count": 0,
            "dependencies_count": 0,
            "actions_ok": passed,
        }

    def _process_single_event_result(self, event_or_group, prompt_tokens: int, completion_tokens: int) -> dict:
        """Processing results of a single event

        Args:
            event_or_group: Generated events or event groups
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens

        Returns:
            Result dictionary
        """
        chars = [c.name for c in getattr(event_or_group, "involved_characters", [])]
        activity = getattr(event_or_group, "activity", "")
        planning_repr = {
            "characters": chars,
            "activity": activity,
        }

        # Evaluation: Using expected_next_event for direct comparison with activity_by_character, which is no longer needed)
        actions_ok = self._evaluate_next_event_action(
            self.current_test_case,
            {},  # actions_by_character I don’t need any more
            generated_characters=chars,
            generated_event=event_or_group,
        )

        return {
            "passed": actions_ok,
            "error": False,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "planning_result": planning_repr,
            "event_count": 1,
            "dependencies_count": 0,
            "actions_ok": actions_ok,
        }

    def _handle_narrator_error(
        self, error: Exception, narrator_type: Optional[str], metadata: Optional[TestMetadata]
    ) -> dict:
        """Exceptional objects

        Args:
            error: Error result dictionary
            narrator_type: Narrator type
            metadata: Test metadata

        Returns:
            Error Result Dictionary
        """
        err_tb = traceback.format_exc()
        scene = getattr(self, "scene_name", "unknown")
        self.logger.error(
            f"Error occurred during {narrator_type} narrator | "
            f"scene={scene} | metadata={metadata}: {error}\n{err_tb}"
        )
        return {
            "passed": False,
            "error": True,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "planning_result": {},
            "event_count": 0,
            "dependencies_count": 0,
            "error_message": str(error),
            "error_trace": err_tb,
        }

    def _compare_results(
        self,
        test_case: dict,
        metadata: TestMetadata,
        baseline_result: Optional[dict],
        timeline_result: Optional[dict],
        timeline_event_schema_valid: bool,
        timeline_dependency_valid: bool,
        timeline_variant: str = NARRATOR_TYPE_TIMELINE,
    ) -> ComparisonResult:
        """Result of baseline narrator (may be None)

        Args:
            test_case: Test case dictionary
            metadata: Test metadata
            baseline_result: Result of timeline narrator (may be None)
            timeline_result: Whether the timeline event pattern is valid)
            timeline_event_schema_valid: Whether the timeline dependencies are valid
            timeline_dependency_valid: Comparison result object

        Returns:
            Handling the case where a certain narrator is not running
        """
        # Deal with a situation that does not run an narrator
        if baseline_result is None:
            # Create a default baseline result
            baseline_result = {
                "passed": False,
                "error": False,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "planning_result": None,
                "event_count": 0,
                "dependencies_count": 0,
            }

        if timeline_result is None:
            # Create a default timeline result
            timeline_result = {
                "passed": False,
                "error": False,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "planning_result": None,
                "event_count": 0,
                "dependencies_count": 0,
            }

        # Calculate token savings
        baseline_total = baseline_result["prompt_tokens"] + baseline_result["completion_tokens"]
        timeline_total = timeline_result["prompt_tokens"] + timeline_result["completion_tokens"]

        token_savings = (
            self._calculate_token_savings(baseline_total, timeline_total)
            if baseline_total > 0 and timeline_total > 0
            else 0.0
        )
        improvement = (
            self._determine_improvement(baseline_result, timeline_result, baseline_total, timeline_total)
            if baseline_total > 0 and timeline_total > 0
            else "N/A"
        )

        # Calculate normalized metrics (for fair comparison)
        baseline_tokens_per_event = (
            baseline_total / max(baseline_result["event_count"], 1) if baseline_result["event_count"] > 0 else 0.0
        )
        timeline_tokens_per_event = (
            timeline_total / max(timeline_result["event_count"], 1) if timeline_result["event_count"] > 0 else 0.0
        )
        token_efficiency_ratio = (
            baseline_tokens_per_event / timeline_tokens_per_event if timeline_tokens_per_event > 0 else 0.0
        )

        return ComparisonResult(
            test_case_id=metadata.test_case_id,
            distraction_level=metadata.distraction_level,
            repeat_idx=metadata.repeat_idx,
            timeline_variant=timeline_variant,
            baseline_passed=baseline_result["passed"],
            baseline_error=baseline_result["error"],
            baseline_prompt_tokens=baseline_result["prompt_tokens"],
            baseline_completion_tokens=baseline_result["completion_tokens"],
            baseline_planning_result=baseline_result["planning_result"],
            baseline_event_count=baseline_result["event_count"],
            baseline_matched_events=baseline_result.get("matched_events", 0),
            baseline_expected_events=baseline_result.get("expected_events", 0),
            baseline_events_generated=baseline_result.get(
                "events_generated", 0
            ),  # 🆕 Statistically analyze the coverage of pure events
            baseline_matched_dependencies=baseline_result.get("matched_dependencies", 0),
            baseline_expected_dependencies=baseline_result.get("expected_dependencies", 0),
            timeline_passed=timeline_result["passed"],
            timeline_error=timeline_result["error"],
            timeline_prompt_tokens=timeline_result["prompt_tokens"],
            timeline_completion_tokens=timeline_result["completion_tokens"],
            timeline_planning_result=timeline_result["planning_result"],
            timeline_event_count=timeline_result["event_count"],
            timeline_dependencies_count=timeline_result["dependencies_count"],
            timeline_event_schema_valid=timeline_event_schema_valid,
            timeline_dependency_valid=timeline_dependency_valid,
            timeline_matched_events=timeline_result.get("diagnostics", {}).get("matched_events", 0),
            timeline_expected_events=timeline_result.get("diagnostics", {}).get("expected_events", 0),
            timeline_matched_dependencies=timeline_result.get("diagnostics", {}).get("matched_dependencies", 0),
            timeline_expected_dependencies=timeline_result.get("diagnostics", {}).get("expected_dependencies", 0),
            improvement=improvement,
            token_savings=token_savings,
            # Normalized metrics
            baseline_tokens_per_event=baseline_tokens_per_event,
            timeline_tokens_per_event=timeline_tokens_per_event,
            token_efficiency_ratio=token_efficiency_ratio,
            # Diagnostic information (Phase 1)
            baseline_error_message=baseline_result.get("error_message"),
            baseline_failure_reason=baseline_result.get("failure_reason"),
            baseline_diagnostics=baseline_result.get("diagnostics", {}),
            timeline_error_message=timeline_result.get("error_message"),
            timeline_failure_reason=timeline_result.get("failure_reason"),
            timeline_diagnostics=timeline_result.get("diagnostics", {}),
        )

    def _calculate_token_savings(self, baseline_total: int, timeline_total: int) -> float:
        """Calculate percentage of token savings

        Args:
            baseline_total: Total number of tokens for baseline
            timeline_total: Total number of tokens for timeline

        Returns:
            Percentage of savings
        """
        if baseline_total > 0:
            return (baseline_total - timeline_total) / baseline_total * 100
        return 0.0

    def _determine_improvement(
        self, baseline_result: dict, timeline_result: dict, baseline_total: int, timeline_total: int
    ) -> bool:
        """Determine if timeline improves compared to baseline

        Args:
            baseline_result: Baseline result
            timeline_result: Timeline Results
            baseline_total: Total number of tokens for baseline
            timeline_total: Total number of tokens for timeline

        Returns:
            Any improvements
        """
        # Timeline passed, but baseline failed
        if timeline_result["passed"] and not baseline_result["passed"]:
            return True

        # Both results are the same, comparing token consumption
        if timeline_result["passed"] == baseline_result["passed"]:
            if timeline_total < baseline_total:
                return True

        return False

    def _group_test_cases_by_scene(self) -> Dict[str, List[dict]]:
        """Test cases grouped by scene

        Returns:
            Dictionary with scene names as keys and lists of test cases as values
        """
        test_cases_by_scene: Dict[str, List[dict]] = {scene: [] for scene in SCENE_NAMES}

        for id_ in self.test_case_ids:
            test_case = load_test_case(id_)
            scene_name = test_case["scene_name"]
            test_cases_by_scene[scene_name].append(test_case)

        return test_cases_by_scene

    def _is_expected_null(self, test_case: dict) -> bool:
        """Checking for expected null events

        Args:
            test_case: Test case dictionary

        Returns:
            Whether null is expected
        """
        expected = test_case.get("expected_next_event")
        return not expected or expected == {}

    def _evaluate_next_event_action(
        self,
        test_case: dict,
        actions_by_character: Dict[str, List[str]],
        generated_characters: Optional[List[str]] = None,
        generated_event=None,
    ) -> bool:
        """Verifying Baseline based on expected_next_event

        Verification rules:
        - Empty dictionary or None indicates expected null (dependency not met)
        - If expected_next_event contains a list of characters and no activity:
          Requires one event to cover all characters (parallel groups), ignoring actions
        - Otherwise, it checks by character + Activity-based checking is performed
           (The match_activity_with_or_options method is used to support or options)

        Args:
            test_case: Test case dictionary
            actions_by_character: Character action mapping (deprecated, retained for compatibility)
            generated_characters: Generated character list
            generated_event: Generated event objects

        Returns:
            Whether it passes the check
        """
        if not test_case:
            return True

        exp = test_case.get("expected_next_event")

        # An empty dictionary or None indicates a null expectation (no events)
        if not exp or exp == {}:
            return self._validate_null_event(generated_characters, generated_event)

        # Parallel groups: Only checks character sets
        if isinstance(exp.get("characters"), list) and not exp.get("activity"):
            return self._validate_parallel_group(exp, generated_characters)

        # Single event: Checks character and action matching
        return self._validate_single_event(exp, generated_characters, generated_event)

    def _validate_null_event(self, generated_characters: Optional[List[str]], generated_event) -> bool:
        """Verifies empty events (expected null)

        Args:
            generated_characters: Generated character list
            generated_event: Generated event objects

        Returns:
            Verification passed
        """
        return generated_event is None and (generated_characters is None or len(generated_characters) == 0)

    def _validate_parallel_group(self, expected: dict, generated_characters: Optional[List[str]]) -> bool:
        """Verification of parallel group events (multiple characters, no specific actions)

        Args:
            expected: Expected event dictionary
            generated_characters: Generated character list

        Returns:
            Verification passed
        """
        if generated_characters is None:
            return False

        expected_chars = set(map(str, expected["characters"]))
        generated_chars = set(map(str, generated_characters))
        return generated_chars == expected_chars

    def _validate_single_event(
        self, expected: dict, generated_characters: Optional[List[str]], generated_event
    ) -> bool:
        """Verification of individual events (character) + Action)

        Args:
            expected: Expected event dictionary
            generated_characters: Generated character list
            generated_event: Generated event objects

        Returns:
            Verification passed
        """
        # Check character matching
        expected_char = expected.get("character")
        if not expected_char:
            return False

        if generated_characters is None or expected_char not in generated_characters:
            return False

        # If there is an activity field, verify the action
        if "activity" in expected:
            return self._validate_activity(expected, generated_event)

        # If there is no activity field, pass if character matching is sufficient
        return True

    def _validate_activity(self, expected: dict, generated_event) -> bool:
        """Verification of event actions

        Args:
            expected: Expected event dictionary
            generated_event: Generated event objects

        Returns:
            Verification passed
        """
        if generated_event is None:
            return False

        generated_activity = getattr(generated_event, "activity", "")
        expected_activity = expected.get("activity", "")

        if not expected_activity:
            return True  # If there is no expected activity, pass if character matching is sufficient

        # Use match_activity_with_or_options for comparison (supported) (option1 | option2 | option3) Format)
        return match_activity_with_or_options(generated_activity, expected_activity)
