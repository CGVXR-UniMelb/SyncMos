from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class AblationResult:
    test_case_id: int
    distraction_level: int
    repeat_idx: int
    results: Dict[str, Dict[str, Any]]
