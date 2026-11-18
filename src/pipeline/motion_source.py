from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from src.utils.motion_io import load_motion_data


@dataclass
class MotionMetadata:
    scene_name: str = ""
    character_names: list[str] = field(default_factory=list)
    total_frames: Optional[int] = None
    user_instruction: str = ""
    num_characters: int = 0
    character_initial_grid_positions: dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class MotionFrame:
    """One playback frame, with character names mapped to (1, 91) tensors.

    Each tensor packs translation(3), root quaternion(4), and 21 joint
    quaternions(84), in (w, x, y, z) order. The optional origin mapping carries
    corresponding frames before time warping, when available.
    """

    frame_id: int
    character_body_params: dict[str, torch.Tensor]
    character_body_params_origin: dict[str, torch.Tensor] = field(default_factory=dict)


class MotionSource(ABC):
    @abstractmethod
    def get_metadata(self) -> MotionMetadata:
        raise NotImplementedError

    @abstractmethod
    def get_frame(self, frame_id: int) -> Optional[MotionFrame]:
        raise NotImplementedError

    @abstractmethod
    def latest_frame_id(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def is_finished(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def status(self) -> str:
        raise NotImplementedError


class FileMotionSource(MotionSource):
    """MotionSource backed by the existing compressed .npz format."""

    def __init__(self, filepath: str, device: str = "cpu", logger: Optional[Any] = None):
        self.filepath = filepath
        self.device = device
        self.logger = logger
        self.data = load_motion_data(filepath, device=device, logger=logger)
        self.metadata = MotionMetadata(
            scene_name=self.data["scene_name"],
            character_names=list(self.data["character_names"]),
            total_frames=int(self.data["total_frames"]),
            user_instruction=self.data.get("user_instruction", ""),
            num_characters=int(self.data["num_characters"]),
            character_initial_grid_positions=self.data.get("character_initial_grid_positions", {}),
            source_path=filepath,
            extras={
                "fId": self.data.get("fId"),
                "max_fId": self.data.get("max_fId"),
            },
        )

    def get_metadata(self) -> MotionMetadata:
        return self.metadata

    def get_frame(self, frame_id: int) -> Optional[MotionFrame]:
        if frame_id < 0:
            return None
        body_params = {}
        body_params_origin = {}
        found = False
        for char_name in self.metadata.character_names:
            char_data = self.data["character_data"].get(char_name, {})
            params = char_data.get("body_params")
            if params is not None and frame_id < len(params):
                body_params[char_name] = params[frame_id : frame_id + 1]
                found = True
            origin = char_data.get("body_params_origin")
            if origin is not None and frame_id < len(origin):
                body_params_origin[char_name] = origin[frame_id : frame_id + 1]
        if not found:
            return None
        return MotionFrame(
            frame_id=frame_id,
            character_body_params=body_params,
            character_body_params_origin=body_params_origin,
        )

    def latest_frame_id(self) -> int:
        return max(0, int(self.metadata.total_frames or 0) - 1)

    def is_finished(self) -> bool:
        return True

    def status(self) -> str:
        return "loaded"


class RealtimeMotionSource(MotionSource):
    def __init__(self, frame_buffer):
        self.frame_buffer = frame_buffer

    def get_metadata(self) -> MotionMetadata:
        return self.frame_buffer.get_metadata()

    def get_frame(self, frame_id: int) -> Optional[MotionFrame]:
        return self.frame_buffer.get_frame(frame_id)

    def latest_frame_id(self) -> int:
        return self.frame_buffer.latest_frame_id()

    def is_finished(self) -> bool:
        return self.frame_buffer.is_finished()

    def status(self) -> str:
        return self.frame_buffer.status()
