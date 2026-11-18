from typing import List
from src.action_planning.narrator.narrator_timeline import NarratorTimeline
from src.type import Character, Object, Event


class NarratorTimelineMultiChar(NarratorTimeline):
    def __init__(self, characters: List[Character], objects: List[Object], cfg) -> None:
        super().__init__(characters, objects, cfg)
        self.system_message_file = self.system_messages_dir / "narrator_timeline_multi_char.txt"

    def _create_event_from_timeline_data(self, event_data):
        """Override to handle multi-character events"""
        try:
            character_names = event_data["character"]
            activity = event_data["activity"]

            # Ensure character_names is a list
            if isinstance(character_names, str):
                character_names = [character_names]

            involved_characters = []
            for character_name in character_names:
                # Find corresponding character object
                character = None
                for char in self.characters:
                    if char.name == character_name:
                        character = char
                        break

                if not character:
                    self.logger.warning(f"Character {character_name} not found")
                    # If any character is missing, we might want to skip or fail.
                    # For now, let's skip this character but continue if others exist.
                    continue
                involved_characters.append(character)

            if not involved_characters:
                self.logger.warning(f"No valid characters found for event {event_data}")
                return None

            # Create Event object - Note: Event now supports multiple characters
            event = Event(
                involved_characters=involved_characters,
                activity=activity,
            )

            # Add timeline-specific metadata
            event.timeline_data = {
                "event_id": event_data["event_id"],
                "duration": event_data.get("duration", 0),
                "tags": event_data.get("tags", []),
                "location_scope": event_data.get("location_scope", ""),
                "can_time_wrap": event_data.get("can_time_wrap", True),
                "checked_completed": False,
            }

            return event

        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            self.logger.error(
                f"Error creating multi-char event from timeline data: {e}\nFull traceback:\n{error_details}"
            )
            return None

    def _parse_timeline_response(self, response):
        """Parses timeline response ensuring no parallel groups are processed since we handle them within single events."""
        super()._parse_timeline_response(response)
        # Ensure parallel_groups is empty as we don't support "paralell" keyword in dependencies for this type
        self.parallel_groups = []
