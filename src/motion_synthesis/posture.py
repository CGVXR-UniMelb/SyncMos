"""Resting posture carried across actions such as drinking while seated."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..type import Character


def idle_action_for(character: "Character") -> str:
    """Choose the resting action label without updating the character.

    Sitting takes precedence over lying, then walking or standing up restores
    ``stand still``. Other actions retain ``idle_action`` when present, with
    ``stand still`` as the fallback. This lookup consumes no random values.
    """
    actions = [action.lower() for action in character.current_actions]
    if any("sit down" in action or "maintain sit" in action for action in actions):
        return "maintain sit"
    if any("lie down" in action or "maintain lie" in action for action in actions):
        return "maintain lie"
    if any(any(word in action for word in ("walk", "stand up", "get up")) for action in actions):
        return "stand still"
    return getattr(character, "idle_action", "stand still")
