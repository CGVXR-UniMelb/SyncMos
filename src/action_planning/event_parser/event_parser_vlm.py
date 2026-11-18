from datetime import datetime
from typing import List
from src.type import Character, Event
from src.utils.llm_logging import write_llm_log
from src.utils.misc import encode_image

from .event_parser import EventParser


class EventParserVLM(EventParser):
    def __init__(self, characters, objects, cfg):
        super().__init__(characters, objects, cfg)
        assert (
            self.scene_description_type == "scene_image"
        ), "EventParserVLM requires scene_description_type to be 'scene_image'"
        self.system_message_file = self.system_messages_dir / "event_parser_vlm.txt"

    def _create_query_message(
        self,
        event: Event,
        scene_description: str,
        available_action_labels: List[str],
        previous_events: List[Event],
        characters: List[Character] = None,
        feedback=None,
    ):
        def _create_user_content(type: str, data: str):
            if type == "text":
                return {"type": type, type: data}
            elif type == "image_url":
                return {"type": type, type: {"url": f"data:image/jpeg;base64,{data}"}}
            else:
                return ValueError(f"There is no type '{type}', type should be 'text' or 'image_url'")

        # text part
        text_part = super()._create_query_message(
            event,
            scene_description[0],
            available_action_labels,
            previous_events,
            characters,
            feedback,
        )
        text_part = text_part.replace("[Scene Description]", "[Object List]")
        text_part = _create_user_content("text", text_part)

        # image part
        scene_image_path = scene_description[1]
        image_data_str = encode_image(scene_image_path)
        image_part = _create_user_content("image_url", image_data_str)

        query_message = [text_part, image_part]
        return query_message

    def log(self, messages, response, log_id=None):
        if log_id is None:
            log_id = datetime.now().strftime("%y%m%d-%H%M%S")
        if self.log_id_prefix:
            log_id = f"{self.log_id_prefix}-{log_id}"
        log_dir = self.log_basedir
        log_file = log_dir / f"{log_id}.txt"
        write_llm_log(log_file, messages, response, layout="vision")
