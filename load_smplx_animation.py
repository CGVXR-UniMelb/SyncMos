"""Write SyncMOS SMPL-X motion arrays to a Blender SMPL-X rig.

Run inside Blender with the SMPL-X add-on enabled. The mapping input contains
``transl``, ``global_orient`` and ``body_pose`` in the existing exported layout.
Frames start at zero; hand joints use ``./smplx_handposes.npz``'s relaxed pose.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import bpy
import numpy as np
from mathutils import Quaternion, Vector

SMPLX_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
]
NUM_SMPLX_JOINTS = len(SMPLX_JOINT_NAMES)
NUM_SMPLX_BODYJOINTS = 21
NUM_SMPLX_HANDJOINTS = 15


def animation_data_clear(obj: bpy.types.Object) -> None:
    """Clear the mesh object and its data, leaving the armature action alone."""
    for owner in (obj, obj.data):
        owner.animation_data_clear()


def get_quat_from_rodrigues(
    rodrigues: Sequence[float], rodrigues_reference: Sequence[float] | None = None
) -> Quaternion:
    """Convert an axis-angle vector, adding any SMPL-X pose mean first.

    A reference is added in axis-angle space, not composed as a quaternion.
    Blender's Vector arithmetic preserves the rig's existing float precision.
    """
    rotation = Vector(rodrigues[:3])
    if rodrigues_reference is not None:
        rotation += Vector(rodrigues_reference[:3])
    return Quaternion(rotation.normalized(), rotation.length)


def set_pose_from_rodrigues(
    armature: bpy.types.Object,
    bone_name: str,
    rodrigues: Sequence[float],
    rodrigues_reference: Sequence[float] | None = None,
) -> None:
    """Set one pose bone's quaternion without inserting a keyframe."""
    rotation = get_quat_from_rodrigues(rodrigues, rodrigues_reference)
    bone = armature.pose.bones[bone_name]
    if bone.rotation_mode != "QUATERNION":
        bone.rotation_mode = "QUATERNION"
    bone.rotation_quaternion = rotation


def _rotation_tracks(body: np.ndarray, root: np.ndarray, relaxed_hands: np.ndarray) -> dict[str, np.ndarray]:
    """Build body, constant hand and pelvis quaternion channels in rig order."""
    tracks = {}
    for joint in range(NUM_SMPLX_BODYJOINTS):
        tracks[SMPLX_JOINT_NAMES[joint + 1]] = np.asarray([get_quat_from_rodrigues(frame[joint]) for frame in body])

    hand_start = 1 + NUM_SMPLX_BODYJOINTS + 3
    for name, reference in zip(SMPLX_JOINT_NAMES[hand_start:], relaxed_hands):
        rest_rotation = get_quat_from_rodrigues((0.0, 0.0, 0.0), reference)
        tracks[name] = np.repeat(np.asarray([rest_rotation]), len(body), axis=0)

    tracks["pelvis"] = np.asarray([get_quat_from_rodrigues(frame) for frame in root])
    return tracks


def _write_channels(action: bpy.types.Action, data_path: str, values: np.ndarray, frame_count: int) -> None:
    """Write scalar F-curves, preserving Blender's default interpolation."""
    frames = np.arange(frame_count)
    for component in range(values.shape[1]):
        coordinates = np.column_stack((frames, values[:frame_count, component]))
        curve = action.fcurves.new(data_path, index=component)
        curve.keyframe_points.add(count=frame_count)
        curve.keyframe_points.foreach_set("co", coordinates.ravel())
        curve.update()


def load_smplx_animation(file: Mapping[str, np.ndarray], obj: bpy.types.Object) -> set[str]:
    """Replace the parent rig's action with zero-based SMPL-X motion channels.

    Reset the first ten shape coefficients, refresh joint locations, write root
    translations and joint quaternions, then enable corrective pose shapes at
    frame zero. Face channels and supplied hand poses are intentionally unused,
    matching the existing SyncMOS export/replay convention.
    """
    animation_data_clear(obj)
    bpy.context.view_layer.objects.active = obj

    with np.load("./smplx_handposes.npz", allow_pickle=True) as archive:
        relaxed = np.concatenate(archive["hand_poses"].item()["relaxed"]).reshape(2 * NUM_SMPLX_HANDJOINTS, 3)

    translations = file["transl"].reshape(-1, 3)
    root_rotations = file["global_orient"].reshape(-1, 3)
    body_rotations = np.array(file["body_pose"]).reshape(-1, NUM_SMPLX_BODYJOINTS, 3)

    bpy.ops.object.mode_set(mode="OBJECT")
    shape_keys = obj.data.shape_keys.key_blocks
    for name in (f"Shape{index:03}" for index in range(10)):
        if name in shape_keys:
            shape_keys[name].value = 0.0
        else:
            print(f"ERROR: No key block for: {name}")
    bpy.ops.object.smplx_update_joint_locations("EXEC_DEFAULT")

    rotations = _rotation_tracks(body_rotations, root_rotations, relaxed)
    rig = obj.parent
    action = bpy.data.actions.new(f"{rig.name}Action")
    rig.animation_data_create().action = action
    frame_count = len(body_rotations)
    _write_channels(action, 'pose.bones["root"].location', translations, frame_count)
    for name, quaternions in rotations.items():
        _write_channels(action, f'pose.bones["{name}"].rotation_quaternion', quaternions, frame_count)

    bpy.context.scene.frame_set(0)
    bpy.ops.object.smplx_set_poseshapes("EXEC_DEFAULT")
    return {"FINISHED"}
