from __future__ import annotations

import threading
import time
import traceback

from omegaconf import OmegaConf, read_write

from src.frontend.polyscope_motion_viewer import PolyscopeMotionViewer
from src.pipeline.frame_buffer import MotionFrameBuffer
from src.pipeline.motion_producer import MotionProducer
from src.pipeline.motion_source import FileMotionSource, RealtimeMotionSource


def run_producer_to_buffer(producer: MotionProducer, frame_buffer: MotionFrameBuffer, stop_event: threading.Event):
    try:
        if not producer.initialized:
            initialize_buffer(producer, frame_buffer)
        while not producer.finished and not stop_event.is_set():
            frame = producer.step()
            if frame is not None:
                frame_buffer.append_frame(
                    frame.frame_id,
                    frame.character_body_params,
                    producer.get_metadata(),
                    frame.character_body_params_origin,
                )
            else:
                time.sleep(0.01)
        frame_buffer.set_metadata(producer.get_metadata())
        frame_buffer.mark_finished()
        if not stop_event.is_set() and not producer.saved:
            producer.save("motion")
            producer.saved = True
        if producer.motion_synthesizer is not None:
            producer.motion_synthesizer.generation_stats.save()
    except Exception as exc:
        frame_buffer.mark_error(f"{exc}\n{traceback.format_exc()}")


def initialize_buffer(producer: MotionProducer, frame_buffer: MotionFrameBuffer):
    metadata = producer.initialize()
    frame_buffer.set_metadata(metadata)
    append_producer_prefix(producer, frame_buffer)


def append_producer_prefix(producer: MotionProducer, frame_buffer: MotionFrameBuffer):
    initial_frame_count = max(len(c.body_params) for c in producer.characters if c.body_params is not None)
    for frame_id in range(initial_frame_count):
        frame_buffer.append_frame(
            frame_id,
            {
                c.name: c.body_params[min(frame_id, len(c.body_params) - 1) : min(frame_id, len(c.body_params) - 1) + 1]
                for c in producer.characters
                if c.body_params is not None
            },
            producer.get_metadata(),
            {
                c.name: c.body_params_origin[
                    min(frame_id, len(c.body_params_origin) - 1) : min(frame_id, len(c.body_params_origin) - 1) + 1
                ]
                for c in producer.characters
                if c.body_params_origin is not None and len(c.body_params_origin) > 0
            },
        )


class UnifiedMotionController:
    def __init__(self, cfg):
        self.base_cfg = cfg
        self.producer = None
        self.frame_buffer = None
        self.source = None
        self.stop_event = None
        self.producer_thread = None

    def load_motion_file(self, filepath: str):
        self.stop_generation()
        self.source = FileMotionSource(filepath, device=self.base_cfg.device)
        return self.source

    def generate_new_motion(self, user_instruction: str = ""):
        return self._start_generation(seed_source=None, user_instruction=user_instruction)

    def continue_from_source(self, source, user_instruction: str = ""):
        return self._start_generation(seed_source=source, user_instruction=user_instruction)

    def save_motion_data(self):
        if self.producer is not None:
            self.producer.save("motion")
            if self.producer.motion_synthesizer is not None:
                self.producer.motion_synthesizer.generation_stats.save()

    def update_user_instruction(self, text: str):
        if self.producer is not None:
            self.producer.user_instruction = text

    def debug_info(self):
        producer = self.producer
        if producer is None or producer.action_planner is None or producer.motion_synthesizer is None:
            return {}
        planner = producer.action_planner
        info = {
            "error_occurred": producer.error is not None,
            "characters": [],
        }
        narrator = getattr(planner, "narrator", None)
        if narrator is not None:
            info["left_query_count"] = getattr(narrator, "left_query_count", None)
            model = getattr(narrator, "model", "")
            temperature = getattr(narrator, "temperature", "")
            info["narrator"] = f"{model} (temperature: {temperature})"
            info["events"] = getattr(narrator, "previous_events", [])
        event_parser = getattr(planner, "event_parser", None)
        if event_parser is not None:
            model = getattr(event_parser, "model", "")
            temperature = getattr(event_parser, "temperature", "")
            info["event_parser"] = f"{model} (temperature: {temperature})"
        try:
            info["action_labels"] = producer.motion_synthesizer.motion_generator.get_candidate_actions()
        except Exception:
            info["action_labels"] = []
        for character in producer.characters:
            info["characters"].append(
                {
                    "name": character.name,
                    "event": character.event,
                    "state": character.state,
                    "current_actions": character.current_actions,
                    "target_actions": character.target_actions,
                    "relationships": character.relationships,
                }
            )
        return info

    def generation_stats(self):
        if self.producer is None or self.producer.motion_synthesizer is None:
            return {}
        return self.producer.motion_synthesizer.generation_stats.snapshot()

    def stop_generation(self):
        if self.stop_event is not None:
            self.stop_event.set()
        if self.producer is not None:
            self.producer.finished = True
        if self.frame_buffer is not None:
            self.frame_buffer.mark_finished()
        if self.producer_thread is not None and self.producer_thread.is_alive():
            self.producer_thread.join()
        if self.producer is not None and self.producer.motion_synthesizer is not None:
            self.producer.motion_synthesizer.generation_stats.save()
        self.stop_event = None
        self.producer_thread = None

    def _start_generation(self, seed_source=None, user_instruction: str = ""):
        self.stop_generation()
        cfg = self._cfg_for_source(seed_source)
        character_names = None
        initial_positions = None
        if seed_source is not None:
            metadata = seed_source.get_metadata()
            character_names = list(metadata.character_names)
            initial_positions = metadata.character_initial_grid_positions
            if not user_instruction:
                user_instruction = metadata.user_instruction

        self.producer = MotionProducer(
            cfg,
            user_instruction=user_instruction or None,
            character_initial_grid_positions=initial_positions,
            character_names=character_names,
        )
        self.frame_buffer = MotionFrameBuffer()
        if seed_source is not None:
            self.producer.initialize()
            self.producer.seed_from_motion_source(seed_source)
            self.frame_buffer.set_metadata(self.producer.get_metadata())
            append_producer_prefix(self.producer, self.frame_buffer)
        else:
            initialize_buffer(self.producer, self.frame_buffer)

        self.stop_event = threading.Event()
        self.producer_thread = threading.Thread(
            target=run_producer_to_buffer,
            args=(self.producer, self.frame_buffer, self.stop_event),
            name="MotionProducer",
        )
        self.producer_thread.start()
        self.source = RealtimeMotionSource(self.frame_buffer)
        return self.source

    def _cfg_for_source(self, source):
        cfg = OmegaConf.create(OmegaConf.to_container(self.base_cfg, resolve=False))
        OmegaConf.set_struct(cfg, False)
        OmegaConf.set_readonly(cfg, False)
        if source is not None:
            metadata = source.get_metadata()
            with read_write(cfg):
                cfg.scene_name = metadata.scene_name
                cfg.num_characters = metadata.num_characters or len(metadata.character_names)
                cfg.device = self.base_cfg.device
        return cfg


def run(cfg) -> None:
    """Open the unified viewer; generation starts only through its controls."""
    controller = UnifiedMotionController(cfg)
    app = PolyscopeMotionViewer(
        cfg,
        motion_source=None,
        motion_dir="motion",
        app_name="Unified Motion",
        on_user_instruction_change=controller.update_user_instruction,
        on_save_motion_data=controller.save_motion_data,
        on_load_motion_file=controller.load_motion_file,
        on_generate_new_motion=controller.generate_new_motion,
        on_continue_from_source=controller.continue_from_source,
        debug_info_provider=controller.debug_info,
        generation_stats_provider=controller.generation_stats,
    )
    try:
        app.run()
    finally:
        controller.stop_generation()
