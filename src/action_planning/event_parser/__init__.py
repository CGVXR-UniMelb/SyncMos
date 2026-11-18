from .event_parser import EventParser
from .event_parser_vlm import EventParserVLM
from .event_parser_wo_lcps import EventParserWithoutLcps
from .event_parser_timeline import EventParserTimeline
from .batch_event_parser import BatchEventParser


def get_event_parser(characters, objects, cfg, scene_name=None, grid_map=None):
    """Select the configured parser. Batch parsing requires a scene name and grid.

    Selection order: batch, no LCPS, image-based, timeline, then the default parser.
    The legacy timeline parser requires use_examples=false; its example-message
    hook is unimplemented. The default sample selects the separate batch parser.
    """
    parser_mode = getattr(cfg.event_parser, "parser_mode", "single")

    if parser_mode == "batch":
        if scene_name is None or grid_map is None:
            raise ValueError("BatchEventParser requires scene_name and grid_map")
        return BatchEventParser(characters, objects, scene_name, grid_map, cfg)

    if cfg.event_parser.disable_lcps:
        return EventParserWithoutLcps(characters, objects, cfg)
    if cfg.scene_describer.description_type == "scene_image":
        return EventParserVLM(characters, objects, cfg)
    if hasattr(cfg.event_parser, "type") and cfg.event_parser.type == "timeline":
        return EventParserTimeline(characters, objects, cfg)
    return EventParser(characters, objects, cfg)
