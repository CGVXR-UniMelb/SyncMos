from typing import List
from src.action_planning.narrator.narrator_timeline import NarratorTimeline
from src.type import Character, Object


class NarratorTimelineSingle(NarratorTimeline):
    def __init__(self, characters: List[Character], objects: List[Object], cfg) -> None:
        super().__init__(characters, objects, cfg)
        self.system_message_file = self.system_messages_dir / "narrator_timeline_single.txt"

        # We always want to generate new plan because we only generate one event at a time
        self.plan_generated = False

    def is_new_event_required(self):
        # Always check if characters are available, but don't rely on cached plan
        # We want to re-query every time the queue is empty
        is_required = super().is_new_event_required()
        if is_required == False:
            return False

        if self.event_list == []:
            # Always return True because 'single' mode generates events one by one
            # We assume if queue is empty, we need to generate the next step
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
        # Override to ensure we don't think we have a "full plan"
        # If event_list is empty, we query LLM for ONE event
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
