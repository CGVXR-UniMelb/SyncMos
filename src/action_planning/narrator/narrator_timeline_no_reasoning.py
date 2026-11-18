from typing import List

from src.action_planning.narrator.narrator_timeline import NarratorTimeline
from src.type import Character, Object


class NarratorTimelineNoReasoning(NarratorTimeline):
    def __init__(self, characters: List[Character], objects: List[Object], cfg) -> None:
        super().__init__(characters, objects, cfg)
        self.system_message_file = self.system_messages_dir / "narrator_timeline_no_reasoning.txt"
