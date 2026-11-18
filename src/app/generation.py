"""Offline generation using the same producer as the interactive application."""

from __future__ import annotations

from src.pipeline.motion_producer import MotionProducer


def run(cfg, *, motion_dir: str = "motion", max_frames: int | None = None) -> None:
    producer = MotionProducer(cfg)
    producer.run_until_finished(max_frames=max_frames)
    filepath = producer.save(motion_dir)
    print(filepath)
