from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any


class GenerationStats:
    def __init__(self, motion_joint_count: int = 28):
        self.motion_joint_count = motion_joint_count
        self.started_at = time.time()
        self._lock = threading.RLock()
        self.cut_events: list[dict[str, Any]] = []
        self.generated_events: list[dict[str, Any]] = []
        self.played_frame_id = 0

    def record_cut(self, frame_id: int, character_name: str, cut_frames: int, reason: str) -> None:
        if cut_frames <= 0:
            return
        with self._lock:
            self.cut_events.append(
                {
                    "frame_id": int(frame_id),
                    "character": character_name,
                    "cut_frames": int(cut_frames),
                    "reason": reason,
                    "time_sec": time.time() - self.started_at,
                }
            )

    def record_generated_segment(
        self,
        frame_id: int,
        character_name: str,
        generated_frames: int,
        action: str,
        batch_size: int,
    ) -> None:
        if generated_frames <= 0:
            return
        with self._lock:
            self.generated_events.append(
                {
                    "frame_id": int(frame_id),
                    "character": character_name,
                    "generated_frames": int(generated_frames),
                    "generated_joint_frames": int(generated_frames) * self.motion_joint_count,
                    "action": action,
                    "batch_size": int(batch_size),
                    "time_sec": time.time() - self.started_at,
                }
            )

    def set_played_frame(self, frame_id: int) -> None:
        with self._lock:
            self.played_frame_id = max(self.played_frame_id, int(frame_id))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            cut_events = list(self.cut_events)
            generated_events = list(self.generated_events)
            cut_by_character = defaultdict(int)
            generated_by_character = defaultdict(int)
            generated_joints_by_character = defaultdict(int)
            for event in cut_events:
                cut_by_character[event["character"]] += event["cut_frames"]
            for event in generated_events:
                generated_by_character[event["character"]] += event["generated_frames"]
                generated_joints_by_character[event["character"]] += event["generated_joint_frames"]

            cut_distribution = Counter(event["cut_frames"] for event in cut_events)
            generated_distribution = Counter(event["generated_frames"] for event in generated_events)
            total_cut_frames = sum(event["cut_frames"] for event in cut_events)
            total_generated_frames = sum(event["generated_frames"] for event in generated_events)
            total_generated_joint_frames = sum(event["generated_joint_frames"] for event in generated_events)
            elapsed = max(1e-6, time.time() - self.started_at)
            return {
                "elapsed_sec": elapsed,
                "played_frame_id": self.played_frame_id,
                "motion_joint_count": self.motion_joint_count,
                "total_cut_events": len(cut_events),
                "total_cut_frames": total_cut_frames,
                "total_generated_events": len(generated_events),
                "total_generated_frames": total_generated_frames,
                "total_generated_joint_frames": total_generated_joint_frames,
                "frame_per_sec": total_generated_frames / elapsed,
                "joints_per_sec": total_generated_joint_frames / elapsed,
                "cut_by_character": dict(cut_by_character),
                "generated_by_character": dict(generated_by_character),
                "generated_joint_frames_by_character": dict(generated_joints_by_character),
                "cut_frame_distribution": dict(sorted(cut_distribution.items())),
                "generated_frame_distribution": dict(sorted(generated_distribution.items())),
                "cut_events": cut_events,
                "generated_events": generated_events,
            }

    def save(self, out_dir: str = "logs/generation_stats", prefix: str = "generation_stats") -> str:
        os.makedirs(out_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(out_dir, f"{prefix}-{timestamp}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.snapshot(), f, indent=2)
        return path
