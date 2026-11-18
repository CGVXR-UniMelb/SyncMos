"""
Diagnostic report generator

Generate a detailed diagnostic report for each test case (in Markdown format)
"""

from pathlib import Path
from .result_types import ComparisonResult


class DiagnosticReportGenerator:
    """Produce the test diagnostic report"""

    def __init__(self):
        self.emoji_status = {True: "✅", False: "❌", None: "⚠️"}

    def generate_report(self, comparison: ComparisonResult, test_case: dict, output_path: Path) -> None:
        """Generate the complete diagnostic report

        Args:
            comparison: Compare the results
            test_case: Test case data
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report_lines = []

        # Header
        variant = getattr(comparison, "timeline_variant", "timeline")
        report_lines.append(f"# Testcase {comparison.test_case_id} - Diagnostic Report ({variant})\n")

        # Test Information
        report_lines.extend(self._generate_test_info(test_case, comparison))

        # Baseline Result
        report_lines.extend(self._generate_baseline_section(comparison, test_case))

        # Timeline Result
        report_lines.extend(self._generate_timeline_section(comparison, test_case))

        # Comparison Summary
        report_lines.extend(self._generate_comparison_summary(comparison))

        # Write to file
        output_path.write_text("\n".join(report_lines))

    def _generate_test_info(self, test_case: dict, comparison: ComparisonResult) -> list:
        """Generate the test information section"""
        lines = [
            "## Test Information\n",
            f"- **Scene**: {test_case.get('scene_name', 'N/A')}",
            f"- **Tags**: {', '.join(test_case.get('tags', []))}",
            f"- **Distraction Level**: {comparison.distraction_level}",
            f"- **Repeat Index**: {comparison.repeat_idx}",
            f"- **Instruction**: {test_case.get('user_instruction', 'N/A')}\n",
        ]

        # Expected timeline
        if "expected_timeline" in test_case:
            timeline = test_case["expected_timeline"]
            events = timeline.get("events", [])
            lines.append(f"- **Expected Events**: {len(events)}")
            for event in events:
                char = event.get("character", "?")
                activity = event.get("activity", "?")
                event_id = event.get("event_id", "?")
                lines.append(f"  - `{event_id}`: {char} - {activity}")
            lines.append("")

        return lines

    def _generate_baseline_section(self, comparison: ComparisonResult, test_case: dict) -> list:
        """Generating the Baseline results section"""
        status = self.emoji_status[comparison.baseline_passed if not comparison.baseline_error else None]

        lines = [
            f"## Baseline Result: {status} {'PASSED' if comparison.baseline_passed else 'FAILED' if not comparison.baseline_error else 'ERROR'}\n",
        ]

        # Error or failure reason
        if comparison.baseline_error:
            lines.append(f"**Error**: {comparison.baseline_error_message or 'Unknown error'}\n")
        elif not comparison.baseline_passed:
            lines.append(f"**Failure Reason**: {comparison.baseline_failure_reason or 'Unknown'}\n")

        # Diagnostics
        if comparison.baseline_diagnostics:
            lines.append("### Diagnostic Details\n")
            diag = comparison.baseline_diagnostics

            # Phase results
            if "phases" in diag:
                lines.append("#### Phase-by-Phase Results\n")
                for phase_info in diag["phases"]:
                    phase_num = phase_info.get("phase", "?")
                    passed = phase_info.get("passed", False)
                    phase_status = self.emoji_status[passed]
                    phase_type = phase_info.get("type", "normal")

                    lines.append(
                        f"**Phase {phase_num}** {phase_status} {'PASSED' if passed else 'FAILED'} ({phase_type})\n"
                    )

                    # Displaying different information based on phase_type
                    if phase_type == "dependency_test":
                        # Dependency testing phase
                        if "context" in phase_info:
                            lines.append(f"- **Context**: {phase_info['context']}")
                        if "dependent_characters" in phase_info:
                            dep_chars = phase_info["dependent_characters"]
                            if dep_chars:
                                lines.append(f"- **Dependent Characters**: {', '.join(dep_chars)} (should wait)")
                            else:
                                lines.append("- **Dependent Characters**: None")

                        # Displaying generated events
                        if "generated" in phase_info:
                            gen = phase_info["generated"]
                            if gen:
                                lines.append(
                                    f"- **Generated**: {gen.get('character', '?')} - {gen.get('activity', '?')}"
                                )
                            else:
                                lines.append("- **Generated**: null (no event)")

                        # Note (e.g., parallel execution allowed)
                        if "note" in phase_info:
                            lines.append(f"- **Note**: {phase_info['note']}")

                        # Issue
                        if not passed and "issue" in phase_info:
                            lines.append(f"- **Issue**: {phase_info['issue']}")

                    else:
                        # Normal phase (generating events)
                        # Displaying list of executable events
                        if "executable_events" in phase_info:
                            exe_events = phase_info["executable_events"]
                            if exe_events:
                                lines.append(f"- **Executable Events** ({len(exe_events)} options):")
                                for exe_evt in exe_events:
                                    lines.append(f"  - {exe_evt}")
                            else:
                                lines.append("- **Executable Events**: None (dependency deadlock)")

                        # Displaying generated events
                        if "generated" in phase_info:
                            gen = phase_info["generated"]
                            if gen:
                                matched = gen.get("matched", False)
                                matched_mark = "✅" if matched else "❌"
                                lines.append(
                                    f"- **Generated**: {matched_mark} {gen.get('character', '?')} - {gen.get('activity', '?')}"
                                )
                                if "matched_to" in gen:
                                    lines.append(f"  - **Matched To**: {gen['matched_to']}")
                            else:
                                lines.append("- **Generated**: null (no event)")

                        # Issue
                        if not passed and "issue" in phase_info:
                            lines.append(f"- **Issue**: {phase_info['issue']}")

                    # Tokens (both types)
                    if "tokens" in phase_info:
                        tokens = phase_info["tokens"]
                        lines.append(
                            f"- **Tokens**: {tokens.get('prompt', 0)} (prompt) + {tokens.get('completion', 0)} (completion)"
                        )

                    lines.append("")

            # Summary stats
            if "total_phases" in diag:
                lines.append(f"**Total Phases**: {diag['total_phases']}")
                lines.append(f"**Passed Phases**: {diag.get('passed_phases', 0)}")
                lines.append(f"**Failed Phase**: {diag.get('failed_phase', 'N/A')}\n")

        # Event and dependency matching statistics
        if comparison.baseline_expected_events > 0:
            event_match_rate = (comparison.baseline_matched_events / comparison.baseline_expected_events) * 100
            lines.append(
                f"**Events Matched**: {comparison.baseline_matched_events}/{comparison.baseline_expected_events} ({event_match_rate:.1f}%)"
            )

        if comparison.baseline_expected_dependencies > 0:
            dep_match_rate = (
                comparison.baseline_matched_dependencies / comparison.baseline_expected_dependencies
            ) * 100
            lines.append(
                f"**Dependencies Satisfied**: {comparison.baseline_matched_dependencies}/{comparison.baseline_expected_dependencies} ({dep_match_rate:.1f}%)\n"
            )

        # Token usage
        lines.append(f"**Total Tokens**: {comparison.baseline_prompt_tokens + comparison.baseline_completion_tokens}")
        lines.append(f"- Prompt: {comparison.baseline_prompt_tokens}")
        lines.append(f"- Completion: {comparison.baseline_completion_tokens}")
        lines.append(f"- Tokens per event: {comparison.baseline_tokens_per_event:.2f}\n")

        return lines

    def _generate_timeline_section(self, comparison: ComparisonResult, test_case: dict) -> list:
        """Generating the Timeline results section"""
        status = self.emoji_status[comparison.timeline_passed if not comparison.timeline_error else None]

        variant = getattr(comparison, "timeline_variant", "timeline")
        lines = [
            f"## Timeline Result ({variant}): {status} {'PASSED' if comparison.timeline_passed else 'FAILED' if not comparison.timeline_error else 'ERROR'}\n",
        ]

        # Error or failure reason
        if comparison.timeline_error:
            lines.append(f"**Error**: {comparison.timeline_error_message or 'Unknown error'}\n")
        elif not comparison.timeline_passed:
            lines.append(f"**Failure Reason**: {comparison.timeline_failure_reason or 'Unknown'}\n")

        # Diagnostics
        if comparison.timeline_diagnostics:
            diag = comparison.timeline_diagnostics

            # Event generation
            expected_count = len(test_case.get("expected_timeline", {}).get("events", []))
            generated_count = comparison.timeline_event_count

            # Checking for harmless termination actions
            harmless_cleanup = diag.get("harmless_cleanup_events", [])
            harmless_count = len(harmless_cleanup)

            # Calculating the number of valid events (excluding harmless termination actions)
            effective_generated_count = generated_count - harmless_count
            event_status = self.emoji_status[expected_count == effective_generated_count]

            if harmless_count > 0:
                lines.append(f"### Events Generated: {event_status} {generated_count}/{expected_count}\n")
                lines.append(
                    f"*(Excluding {harmless_count} harmless cleanup event(s): effective count = {effective_generated_count})*\n"
                )
            else:
                lines.append(f"### Events Generated: {event_status} {generated_count}/{expected_count}\n")

            if "generated_events" in diag:
                for evt in diag["generated_events"]:
                    event_id = evt.get("event_id", "?")
                    char = evt.get("character", "?")
                    activity = evt.get("activity", "?")
                    lines.append(f"- `{event_id}`: {char} - {activity}")
                lines.append("")

            # Displaying the filtered harmless termination actions
            if harmless_count > 0:
                lines.append("**Harmless Cleanup Events (Excluded from validation)**:\n")
                for evt in harmless_cleanup:
                    event_id = evt.get("event_id", "?")
                    char = evt.get("character", "?")
                    activity = evt.get("activity", "?")
                    lines.append(f"- `{event_id}`: {char} - {activity}")
                lines.append("")

            # Dependencies
            expected_deps_list = test_case.get("expected_timeline", {}).get("dependencies", [])
            expected_deps = len(expected_deps_list)
            generated_deps = comparison.timeline_dependencies_count
            dep_status = self.emoji_status[comparison.timeline_dependency_valid]

            lines.append(f"### Dependencies: {dep_status} Generated {generated_deps}/{expected_deps}\n")

            # Show expected dependencies
            if expected_deps_list:
                lines.append("**Expected Dependencies**:\n")
                for dep in expected_deps_list:
                    if "parallel" in dep:
                        lines.append(f"- parallel: {dep.get('parallel', [])}")
                    else:
                        lines.append(f"- {dep.get('after', '?')} → {dep.get('before', '?')}")
                lines.append("")

            # Show generated dependencies
            if "generated_dependencies" in diag:
                lines.append("**Generated Dependencies**:\n")
                for dep in diag["generated_dependencies"]:
                    if "parallel" in dep:
                        lines.append(f"- parallel: {dep.get('parallel', [])}")
                    else:
                        lines.append(f"- {dep.get('after', '?')} → {dep.get('before', '?')}")
                lines.append("")

            # Show event ID mapping if semantic matching was used
            if diag.get("semantic_matching_used") and "event_id_mapping" in diag:
                lines.append("**Event ID Mapping** (Generated → Expected):\n")
                mapping = diag["event_id_mapping"]
                for gen_id, exp_id in sorted(mapping.items()):
                    if gen_id != exp_id:
                        lines.append(f"- {gen_id} → {exp_id}")
                if any(gen_id != exp_id for gen_id, exp_id in mapping.items()):
                    lines.append(
                        "\n*Note: LLM generated events in a different order. Dependencies are validated using semantic matching.*\n"
                    )
                else:
                    lines.append("\n*All event IDs match expected values.*\n")
                lines.append("")

            if not comparison.timeline_dependency_valid and "dependency_issues" in diag:
                lines.append("**Dependency Issues**:\n")
                for issue in diag["dependency_issues"]:
                    lines.append(f"- {issue}")
                lines.append("")

            # Action matching
            actions_ok = diag.get("actions_ok", False)
            action_status = self.emoji_status[actions_ok]

            lines.append(f"### Action Matching: {action_status}\n")

            if "activity_mismatches" in diag:
                for mismatch in diag["activity_mismatches"]:
                    char = mismatch.get("character", "?")
                    expected = mismatch.get("expected", [])
                    generated = mismatch.get("generated", [])
                    score = mismatch.get("fuzzy_score", 0.0)

                    lines.append(f"**{char}**: ❌")
                    lines.append(f"- Expected: {', '.join(expected) if expected else 'N/A'}")
                    lines.append(f"- Generated: {', '.join(generated) if generated else 'N/A'}")
                    if score > 0:
                        lines.append(f"- Fuzzy match score: {score:.2f}")
                    if "issue" in mismatch:
                        lines.append(f"- Issue: {mismatch['issue']}")
                    lines.append("")

            if "matched_characters" in diag:
                for char in diag["matched_characters"]:
                    lines.append(f"- ✅ {char}: Activities matched")

        # Event and dependency matching statistics
        lines.append("")
        if comparison.timeline_expected_events > 0:
            event_match_rate = (comparison.timeline_matched_events / comparison.timeline_expected_events) * 100
            lines.append(
                f"**Events Matched**: {comparison.timeline_matched_events}/{comparison.timeline_expected_events} ({event_match_rate:.1f}%)"
            )

        if comparison.timeline_expected_dependencies > 0:
            dep_match_rate = (
                comparison.timeline_matched_dependencies / comparison.timeline_expected_dependencies
            ) * 100
            lines.append(
                f"**Dependencies Satisfied**: {comparison.timeline_matched_dependencies}/{comparison.timeline_expected_dependencies} ({dep_match_rate:.1f}%)\n"
            )

        # Token usage
        lines.append(f"**Total Tokens**: {comparison.timeline_prompt_tokens + comparison.timeline_completion_tokens}")
        lines.append(f"- Prompt: {comparison.timeline_prompt_tokens}")
        lines.append(f"- Completion: {comparison.timeline_completion_tokens}")
        lines.append(f"- Tokens per event: {comparison.timeline_tokens_per_event:.2f}\n")

        return lines

    def _generate_comparison_summary(self, comparison: ComparisonResult) -> list:
        """Generating a comparison summary section"""
        lines = [
            "## Comparison Summary\n",
            f"- **Improvement**: {'Yes ✅' if comparison.improvement else 'No ❌'}",
            f"- **Token Savings**: {comparison.token_savings:.2f}%",
            f"- **Token Efficiency Ratio**: {comparison.token_efficiency_ratio:.2f}x",
            f"  - (Baseline uses {comparison.token_efficiency_ratio:.2f}x more tokens per event)\n",
        ]

        # Recommendations
        lines.append("## Recommendations\n")

        if not comparison.baseline_passed and comparison.baseline_failure_reason:
            lines.append("### For Baseline:")
            if "dependency" in comparison.baseline_failure_reason:
                lines.append("- Review dependency handling logic")
                lines.append("- Check if 'ongoing' state is properly preventing dependent events")
            elif "activity_mismatch" in comparison.baseline_failure_reason:
                lines.append("- Review activity matching criteria")
                lines.append("- Consider fuzzy matching or synonym support")
            lines.append("")

        if not comparison.timeline_passed and comparison.timeline_failure_reason:
            lines.append("### For Timeline:")
            if "action_mismatch" in comparison.timeline_failure_reason:
                lines.append("- Consider adjusting fuzzy match threshold")
                lines.append("- Add synonym mappings for common variations")
                lines.append("- Review LLM prompt for activity formatting guidelines")
            elif "dependency" in comparison.timeline_failure_reason:
                lines.append("- Review dependency generation logic")
                lines.append("- Check if LLM prompt clearly specifies dependency format")
            lines.append("")

        return lines


def generate_diagnostic_report(comparison: ComparisonResult, test_case: dict, log_base_dir: Path) -> None:
    """Generating and saving the diagnostic report

    Args:
        comparison: Compare the results
        test_case: Test case data
        log_base_dir: Log Base Directory
    """
    generator = DiagnosticReportGenerator()

    # Determine Output Path
    output_dir = (
        log_base_dir
        / "diagnostics"
        / f"testcase-{comparison.test_case_id}"
        / f"distraction-{comparison.distraction_level}"
    )
    variant = getattr(comparison, "timeline_variant", "timeline")
    output_file = (
        output_dir
        / f"tc{comparison.test_case_id:03d}-d{comparison.distraction_level}-r{comparison.repeat_idx}-t{variant}.md"
    )

    # Generate Report
    generator.generate_report(comparison, test_case, output_file)
