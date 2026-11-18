"""Static descriptions of the batch parser's expected dictionary payloads.

These types do not validate or transform LLM output. Fields are optional because
parsing, coordinate conversion, and fallback paths populate different subsets.
Runtime character, event and object definitions live in src.type.
"""

from typing import TypeAlias, TypedDict


class HandTargetSpec(TypedDict, total=False):
    """Structured hand target accepted by the action planner.

    Grid pairs and world-coordinate triples are documented by length; TypedDict
    does not enforce those lengths or the supported combinations of fields.
    """

    object: str
    grasp_type: str | None
    hand_target_grid: list[int] | tuple[int, int]
    area: list[str]
    position_type: str
    height: float | str


class ParsedCharacterPlan(TypedDict, total=False):
    """Expected per-character fields after batch response normalization.

    Raw responses can still contain invalid values: the current parser only
    performs partial checks. This schema describes supported inputs/results,
    not a guarantee that arbitrary model output conforms to the annotations.
    """

    text: str
    target_action: list[str]
    target_grid: list[int] | str
    target_position: list[float]
    relationships: list[str]
    event_id: str
    hand_target: str | HandTargetSpec | list[float] | None
    hand_target_position: HandTargetSpec | list[float] | list[str] | None
    attachment_control: str | None
    parser_source: str


ParsedEvent: TypeAlias = dict[str, ParsedCharacterPlan]
BatchParsedEvents: TypeAlias = dict[int, ParsedEvent]
EventIdParsedEvents: TypeAlias = dict[str, ParsedEvent]
