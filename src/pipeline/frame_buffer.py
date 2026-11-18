from __future__ import annotations

import copy
import threading
from typing import Optional

import torch

from .motion_source import MotionFrame, MotionMetadata


class MotionFrameBuffer:
    """Thread-safe buffer for realtime body-parameter frames."""

    def __init__(self, metadata: Optional[MotionMetadata] = None, max_frames: Optional[int] = None):
        self._lock = threading.RLock()
        self._frames: dict[int, MotionFrame] = {}
        self._metadata = metadata
        self._latest_frame_id = -1
        self._finished = False
        self._error: Optional[str] = None
        self._max_frames = max_frames

    def set_metadata(self, metadata: MotionMetadata) -> None:
        with self._lock:
            self._metadata = metadata

    def get_metadata(self) -> MotionMetadata:
        with self._lock:
            if self._metadata is None:
                return MotionMetadata()
            return copy.deepcopy(self._metadata)

    def append_frame(
        self,
        frame_id: int,
        character_body_params: dict[str, torch.Tensor],
        metadata: Optional[MotionMetadata] = None,
        character_body_params_origin: Optional[dict[str, torch.Tensor]] = None,
    ) -> MotionFrame:
        frame = MotionFrame(
            frame_id=frame_id,
            character_body_params=_clone_params(character_body_params),
            character_body_params_origin=_clone_params(character_body_params_origin or {}),
        )
        with self._lock:
            if metadata is not None:
                self._metadata = metadata
            self._frames[frame_id] = frame
            self._latest_frame_id = max(self._latest_frame_id, frame_id)
            if self._max_frames is not None and len(self._frames) > self._max_frames:
                for old_id in sorted(self._frames)[: len(self._frames) - self._max_frames]:
                    del self._frames[old_id]
            return copy.deepcopy(frame)

    def get_frame(self, frame_id: int) -> Optional[MotionFrame]:
        with self._lock:
            frame = self._frames.get(frame_id)
            return copy.deepcopy(frame) if frame is not None else None

    def latest_frame_id(self) -> int:
        with self._lock:
            return self._latest_frame_id

    def is_finished(self) -> bool:
        with self._lock:
            return self._finished

    def mark_finished(self) -> None:
        with self._lock:
            self._finished = True

    def mark_error(self, message: str) -> None:
        with self._lock:
            self._error = message
            self._finished = True

    def status(self) -> str:
        with self._lock:
            if self._error:
                return f"error: {self._error}"
            if self._finished:
                return "finished"
            return "running"


def _clone_params(character_body_params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: params.detach().clone() for name, params in character_body_params.items() if params is not None}
