from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class ComparisonResult:
    test_case_id: int
    distraction_level: int
    repeat_idx: int

    # Baseline
    baseline_passed: bool
    baseline_error: bool
    baseline_prompt_tokens: int
    baseline_completion_tokens: int
    baseline_planning_result: Dict[str, Any]
    baseline_event_count: int

    # Timeline
    timeline_passed: bool
    timeline_error: bool
    timeline_prompt_tokens: int
    timeline_completion_tokens: int
    timeline_planning_result: Dict[str, Any]
    timeline_event_count: int
    timeline_dependencies_count: int
    timeline_event_schema_valid: bool
    timeline_dependency_valid: bool

    # Comparison
    improvement: bool
    token_savings: float

    # Normalized metrics (for fair comparison)
    baseline_tokens_per_event: float = 0.0
    timeline_tokens_per_event: float = 0.0
    token_efficiency_ratio: float = 0.0  # baseline / timeline

    # Event and dependency matching statistics
    baseline_matched_events: int = (
        0  # Number of Expected Events Matched Successfully (Considering Dependency Order, etc.)
    )
    baseline_expected_events: int = 0  # Total Number of Expected Events
    baseline_events_generated: int = (
        0  # 🆕 Pure Event Coverage Statistics (Without Considering Dependency Violations, etc.)
    )
    baseline_matched_dependencies: int = 0  # Number of Dependencies Met
    baseline_expected_dependencies: int = 0  # Total Number of Expected Dependencies
    timeline_matched_events: int = 0  # Number of Expected Events Matched Successfully
    timeline_expected_events: int = 0  # Total Number of Expected Events
    timeline_matched_dependencies: int = 0  # Number of Dependencies Met
    timeline_expected_dependencies: int = 0  # Total Number of Expected Dependencies

    # Fairness notes
    comparison_fairness: str = "not_directly_comparable"
    task_type_note: str = "baseline=incremental (2×n cases), timeline=batch (1 case)"

    # Diagnostic information (Phase 1 enhancement)
    baseline_error_message: Optional[str] = None
    baseline_failure_reason: Optional[str] = None
    baseline_diagnostics: Dict[str, Any] = field(default_factory=dict)

    timeline_error_message: Optional[str] = None
    timeline_failure_reason: Optional[str] = None
    timeline_diagnostics: Dict[str, Any] = field(default_factory=dict)

    # Timeline variant label (for ablation runs)
    timeline_variant: str = "timeline"
