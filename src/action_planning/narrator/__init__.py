from .narrator import Narrator
from .narrator_vlm import NarratorVLM
from .narrator_wo_event import NarratorWithoutEvent
from .narrator_timeline import NarratorTimeline
from .narrator_wo_timeline import NarratorWithoutTimeline
from .narrator_timeline_single import NarratorTimelineSingle
from .narrator_wo_timeline_single import NarratorWithoutTimelineSingle
from .narrator_timeline_multi_char import NarratorTimelineMultiChar
from .narrator_timeline_no_reasoning import NarratorTimelineNoReasoning
from .narrator_timeline_no_fewshot import NarratorTimelineNoFewShot


def get_narrator(characters, objects, cfg):
    if cfg.narrator.disable_event_based_planning:
        return NarratorWithoutEvent(characters, objects, cfg)
    if cfg.scene_describer.description_type == "scene_image":
        return NarratorVLM(characters, objects, cfg)
    if hasattr(cfg.narrator, "type"):
        if cfg.narrator.type == "timeline":
            return NarratorTimeline(characters, objects, cfg)
        if cfg.narrator.type == "timeline_no_reasoning":
            return NarratorTimelineNoReasoning(characters, objects, cfg)
        if cfg.narrator.type == "timeline_no_fewshot":
            return NarratorTimelineNoFewShot(characters, objects, cfg)
        if cfg.narrator.type == "wo_timeline":
            return NarratorWithoutTimeline(characters, objects, cfg)
        if cfg.narrator.type == "timeline_single":
            return NarratorTimelineSingle(characters, objects, cfg)
        if cfg.narrator.type == "wo_timeline_single":
            return NarratorWithoutTimelineSingle(characters, objects, cfg)
        if cfg.narrator.type == "timeline_multi_char":
            return NarratorTimelineMultiChar(characters, objects, cfg)
    return Narrator(characters, objects, cfg)
