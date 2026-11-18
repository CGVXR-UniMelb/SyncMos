from typing import List
from src.action_planning.narrator.narrator_wo_timeline import NarratorWithoutTimeline
from src.type import Character, Object


class NarratorWithoutTimelineSingle(NarratorWithoutTimeline):
    def __init__(self, characters: List[Character], objects: List[Object], cfg) -> None:
        super().__init__(characters, objects, cfg)
        self.system_message_file = self.system_messages_dir / "narrator_wo_timeline_single.txt"

    def is_new_event_required(self):
        # Logic similar to NarratorTimelineSingle
        is_required = super().is_new_event_required()
        if is_required == False:
            return False

        if self.event_list == []:
            return True

        if self._get_next_available_event_idx() is not None:
            return True
        return False

    def generate_subsequent_plan(
        self,
        scene_description,
        available_activities,
        user_instruction=None,
        feedback=None,
        with_usage=False,
        accumulated_prompt_tokens=0,
        accumulated_completion_tokens=0,
    ):
        self.plan_generated = False  # Force flag reset
        return super().generate_subsequent_plan(
            scene_description,
            available_activities,
            user_instruction,
            feedback,
            with_usage,
            accumulated_prompt_tokens,
            accumulated_completion_tokens,
        )
