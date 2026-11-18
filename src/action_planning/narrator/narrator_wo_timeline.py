from typing import List
from src.action_planning.narrator.narrator_timeline import NarratorTimeline
from src.type import Character, Object


class NarratorWithoutTimeline(NarratorTimeline):
    def __init__(self, characters: List[Character], objects: List[Object], cfg) -> None:
        super().__init__(characters, objects, cfg)
        self.system_messages_dir = self.base_dir / "system_messages"
        self.system_message_file = self.system_messages_dir / "narrator_wo_timeline.txt"

        # Disable timeline related features
        self.use_examples = False

    def _parse_timeline_response(self, response):
        """Parse the simplified response without timeline info"""
        try:
            if not response.get("plan", None):
                self.logger.warning("No plan found in response")
                return

            events = response["plan"].get("events", [])

            # Since there are no dependencies, we just add events in the order they appear
            # We treat them as a sequence where order matters for the same character,
            # but assume independent execution for different characters unless logic dictates otherwise.
            # However, the base NarratorTimeline logic relies on dependencies to sort.
            # Here we just trust the LLM's output order.

            # Clear timeline specific attributes used by the parent class
            self.timeline_dependencies = []
            self.parallel_groups = []

            # Add events directly
            for event_data in events:
                event = self._create_event_from_timeline_data(event_data)
                if event:
                    # In wo_timeline mode, we don't have explicit duration or dependency
                    # We might need to fake some data if the downstream pipeline expects it
                    # But based on the request, we just stripped them from system message.
                    self.event_list.append(event)

            self.logger.info(f"Parsed {len(self.event_list)} events from response")

        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            self.logger.error(f"Error parsing response: {e}\nFull traceback:\n{error_details}")
            self.event_list = []

    def _create_event_from_timeline_data(self, event_data):
        """Override to handle missing timeline fields gracefully"""
        try:
            character_name = event_data["character"]
            activity = event_data["activity"]

            character = None
            for char in self.characters:
                if char.name == character_name:
                    character = char
                    break

            if not character:
                self.logger.warning(f"Character {character_name} not found")
                return None

            from src.type import Event

            event = Event(
                involved_characters=[character],
                activity=activity,
            )

            # Populate dummy timeline data to prevent crashes in checking logic
            event.timeline_data = {
                "event_id": event_data["event_id"],
                "duration": 0,  # Default duration since we don't have it
                "tags": [],
                "location_scope": "",
                "can_time_wrap": True,
                "checked_completed": False,
            }

            return event

        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            self.logger.error(f"Error creating event: {e}\nFull traceback:\n{error_details}")
            return None

    def _check_event_dependencies(self, event):
        """Override to always return True since we don't have dependencies"""
        return True

    def _get_next_available_event_idx(self):
        """Override to simply return the next event index if character is idle"""
        if not self.event_list:
            return None

        # Simplistic strategy: find the first event whose character is idle
        for i, event in enumerate(self.event_list):
            if self._are_event_characters_idle(event):
                self.logger.info(f"Found available event index: {i}")
                return i

        return None
