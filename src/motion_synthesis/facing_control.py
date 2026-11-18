"""Apply a continuous seated yaw target to sampled joints before body fitting."""

import math
from typing import TYPE_CHECKING

import torch

from .states import State

if TYPE_CHECKING:
    from ..scene.grid_map import GridMap
    from ..type import Character


def align_seated_facing(
    character: "Character", points: torch.Tensor, grid_map: "GridMap", interp_s: int
) -> torch.Tensor:
    """Preserve root paths and joint geometry while easing heading toward the table.

    ``points`` is one nonempty sequence of sampled joints, shaped (1, T, J * 3),
    in LINGO (Y-up) world coordinates. ``grid_map`` converts these to Event
    (Z-up) coordinates, where the character's current and target headings lie.
    ``interp_s`` is the number of playback frames between sampled joint frames;
    the character's ``dt`` and rotation half-life are measured in seconds.

    Return the corrected tensor with the original shape, dtype and device.
    Skipped actions or unusable headings/joint layouts return ``points`` itself.
    Inputs and character state are not modified; no random stream is consumed.

    LINGO's pelvis goal controls position, not an independent facing direction.
    The caller feeds the corrected joints into fitting and continuation history.
    """
    actions = " ".join(character.current_actions).lower()
    seated = (
        "sit down" in actions or "maintain sit" in actions or getattr(character, "idle_action", None) == "maintain sit"
    )
    target = getattr(character, "target_facing_direction", None)
    if (
        not seated
        or character.state == State.APPROACHING
        or any(a in actions for a in ("stand up", "get up", "transition_out"))
        or target is None
    ):
        return points
    joints = grid_map.lingo_to_event(points[0].reshape(len(points[0]), -1, 3))
    if joints.shape[1] <= 17:
        return points
    target = target.to(joints).reshape(-1, 3)[0, :2]
    previous = character.facing_direction.to(joints).reshape(-1, 3)[0, :2]
    if (
        not torch.isfinite(target).all()
        or torch.linalg.vector_norm(target) <= 1e-6
        or torch.linalg.vector_norm(previous) <= 1e-6
    ):
        return points
    lateral = joints[:, 2] - joints[:, 1] + joints[:, 17] - joints[:, 16]
    forward = torch.stack([-lateral[:, 1], lateral[:, 0]], dim=-1)
    if torch.any(torch.linalg.vector_norm(forward, dim=-1) <= 1e-6):
        return points
    target_angle = torch.atan2(target[1], target[0])
    angle = torch.atan2(previous[1], previous[0])
    half_life = max(float(character.halflife_rotation), 1e-6)
    angles = []
    for index in range(len(joints)):
        # The first displayed frame follows history by one playback frame; later
        # raw joints are separated by the configured interpolation interval.
        elapsed = float(character.dt) * (1 if index == 0 else interp_s)
        gain = 1 - math.exp(-math.log(2) * elapsed / half_life)
        delta = target_angle - angle
        angle = angle + gain * torch.atan2(torch.sin(delta), torch.cos(delta))
        angles.append(angle)
    correction = torch.stack(angles) - torch.atan2(forward[:, 1], forward[:, 0])
    cosine, sine = torch.cos(correction)[:, None], torch.sin(correction)[:, None]
    root = joints[:, :1].clone()
    relative = joints - root
    rotated = relative.clone()
    rotated[..., 0] = cosine * relative[..., 0] - sine * relative[..., 1]
    rotated[..., 1] = sine * relative[..., 0] + cosine * relative[..., 1]
    return grid_map.event_to_lingo(rotated + root).reshape_as(points)
