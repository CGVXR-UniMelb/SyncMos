"""Resolve the table faced by a seated character, including batch-parser plans."""

import re
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..type import Character, Object


def table_facing_direction(
    character: "Character", finder: "Mapping[str, Object | Character]", device: torch.device | str
) -> torch.Tensor | None:
    """Resolve a seated actor's horizontal heading in Event (Z-up) coordinates.

    ``finder`` maps scene names to objects and characters. Object centers and
    ``character.target_position`` must share Event coordinates; the latter is a
    (1, 3) tensor on ``device``. Explicit table references take precedence over
    the table nearest to the actor's named chair in the horizontal plane.

    Return a normalized (1, 3) tensor with a zero Z component, or None when the
    actor is not seated, no table can be resolved, or the direction is nonfinite
    or too short. Character state and input tensors are not modified; no random
    stream is consumed.
    """
    actions = " ".join(character.target_actions).lower()
    if (
        "sit down" not in actions
        and "maintain sit" not in actions
        and getattr(character, "idle_action", None) != "maintain sit"
    ):
        return None

    tables = {name: obj for name, obj in finder.items() if "table" in getattr(obj, "label", "")}
    if not tables:
        return None
    # Only inspect this actor's event, not a neighbour's parallel event.
    event = getattr(character, "event", None)
    events = getattr(event, "events", [event])
    activities = [e.activity for e in events if e is not None and character in e.involved_characters]
    # Empty parser relationships can inherit an earlier event's table. The
    # current event's explicit table therefore takes precedence.
    references = " ".join(activities + character.target_actions + character.relationships)
    table_names = [name for name in re.findall(r"\b[\w]*table_\d+\b", references) if name in tables]
    if table_names:
        table = tables[table_names[0]]
    else:
        # A batch plan may only contain a chair name and world position.
        seat_name = getattr(character, "seat_object", None)
        seat = finder.get(seat_name)
        if seat is None or "chair" not in getattr(seat, "label", ""):
            return None
        table = min(
            tables.values(), key=lambda obj: sum((float(obj.center[i]) - float(seat.center[i])) ** 2 for i in (0, 1))
        )
    direction = (
        torch.as_tensor(table.center, dtype=torch.float32, device=device).reshape(1, 3) - character.target_position
    )
    direction = direction.clone()
    direction[..., 2] = 0
    norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    if not torch.isfinite(direction).all() or torch.any(norm <= 1e-6):
        return None
    return direction / norm
