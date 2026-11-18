import re
from copy import deepcopy
from pathlib import Path
from typing import List

import yaml

from src.action_planning.event_parser.event_parser import EventParser
from src.type import Character, Event, Object

_VALID_RELATIONSHIPS = [
    "interact with",
    "adjacent to",
    "close to",
    "to the left of",
    "to the right of",
    "behind",
    "in front of",
    "look at",
    "sit on",
    "lie on",
]


class EventParserTimeline(EventParser):
    """Legacy timeline parser for individual events and parallel event groups.

    This is separate from BatchEventParser, which the default sample selects.
    Example-message construction is not implemented: use_examples must remain
    false when selecting this parser. Existing failure behavior is retained.
    """

    def __init__(
        self,
        characters: List[Character],
        objects: List[Object],
        cfg,
    ) -> None:
        super().__init__(characters, objects, cfg)
        self.system_message_file = self.system_messages_dir / "event_parser_timeline.txt"

    def parse_parallel_events(
        self,
        events: List[Event],
        scene_description,
        available_action_labels,
        previous_events,
        feedback=None,
        with_usage=False,
        accumulated_prompt_tokens=0,
        accumulated_completion_tokens=0,
    ):
        """Parse multiple parallel events at once"""
        self.logger.info(f"Parsing parallel event group with {len(events)} events...")
        self.code_executor.set_previous_events(previous_events)

        response, prompt_tokens, completion_tokens = self._query_llm_parallel(
            events, scene_description, available_action_labels, previous_events, feedback
        )
        response = self._post_process_response(events, response)
        feedback = self._check_validity(events, response, available_action_labels)
        if feedback and self.self_feedback_count < self.max_self_feedback_count:
            self.logger.warning("Invalid parallel response detected, retrying with feedback...")
            self.logger.warning(f"feedback: {feedback}")
            self.self_feedback_count += 1
            return self.parse_parallel_events(
                events,
                scene_description,
                available_action_labels,
                previous_events,
                feedback,
                with_usage,
                accumulated_prompt_tokens + prompt_tokens,
                accumulated_completion_tokens + completion_tokens,
            )

        self.self_feedback_count = 0
        if with_usage:
            return (
                response,
                prompt_tokens + accumulated_prompt_tokens,
                completion_tokens + accumulated_completion_tokens,
            )
        return response

    def _query_llm_parallel(
        self,
        events: List[Event],
        scene_description,
        available_action_labels,
        previous_events,
        feedback=None,
    ):
        args, messages = self.create_api_request_parallel(
            events,
            scene_description,
            available_action_labels,
            previous_events,
            feedback,
            return_with_messages=True,
        )
        response, prompt_tokens, completion_tokens = self.client.get_response(with_usage=True, **args)
        self.log(messages, response)
        return response, prompt_tokens, completion_tokens

    def create_api_request_parallel(
        self,
        events: List[Event],
        scene_description,
        available_action_labels,
        previous_events,
        feedback=None,
        return_with_messages=False,
    ):
        messages = []
        messages.append({"role": "system", "content": self._create_system_message()})
        if self.use_examples:
            for example_file in self.example_files:
                user_part, assistant_part = self._create_example_message(example_file)
                messages.append({"role": "user", "content": user_part})
                messages.append({"role": "assistant", "content": assistant_part})
        messages.append(
            {
                "role": "user",
                "content": self._create_query_message_parallel(
                    events, scene_description, available_action_labels, previous_events, feedback=feedback
                ),
            }
        )

        args = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.use_json_mode:
            args["response_format"] = {"type": "json_object"}

        if return_with_messages:
            return args, messages
        return args

    def _create_example_message(self, example_file: Path):
        """Reject example-message construction for this legacy parser.

        Calls always raise NotImplementedError. The legacy conversion statements
        below remain unreachable; they do not provide example-message support.
        """
        raise NotImplementedError("Not implemented")
        assert example_file.suffix == ".yaml"
        example_data = yaml.safe_load(example_file.read_text())

        # create user message part
        event_to_parse = Event(example_data["event"]["characters"], example_data["event"]["activity"])
        scene_description = (
            (self.scene_descriptions_dir / f"{example_data['scene_name']}_{self.scene_description_type}.txt")
            .read_text()
            .strip()
        )
        if self.scene_description_type == "scene_image":
            img_path = f"./configs/scenes/{example_data['scene_name']}/top_view.png"
            scene_description = (scene_description, img_path)
        available_action_labels = example_data["action_labels"]

        characters = []
        for character in example_data["characters"]:
            characters.append(
                Character(
                    name=character["name"],
                    state=character["state"],
                    current_actions=[character["current_action"]],
                    relationships=character["position"] + character["orientation"],
                )
            )

        previous_events = []
        for event in example_data["previous_events"]:
            previous_events.append(
                Event(
                    event["characters"],
                    event["activity"],
                    event["state"],
                )
            )
            parsed_event = {}
            for parsed in event["parsed"]:
                character = parsed["character"]
                parsed_event[character] = {}
                parsed_event[character]["target_action"] = [parsed["target_action"]]
                parsed_event[character]["relationships"] = parsed["position"] + parsed.get("orientation", [])
            previous_events[-1].parsed_event = parsed_event

        user_part = self._create_query_message(
            event_to_parse,
            scene_description,
            available_action_labels,
            previous_events,
            characters,
        )

        # create assistant message part
        example_data = yaml.safe_load(example_file.read_text())
        assistant_part = example_data["response"].strip()

        return user_part, assistant_part

    def _create_query_message_parallel(
        self,
        events: List[Event],
        scene_description: str,
        available_action_labels: List[str],
        previous_events: List[Event],
        characters: List[Character] = None,
        feedback=None,
    ):
        if characters is None:
            characters = self.characters

        scene_description_str = "[Scene Description]\n" + scene_description
        available_action_labels_str = "[Available Action Labels]\n" + str(available_action_labels)
        current_states_str = "[Current States of Characters]\n"
        for character in characters:
            current_states_str += f"- {character.name}:\n"
            current_states_str += f"  - state: '{character.state}'\n"
            if str(character.state) in ["idle", "interacting"]:
                current_states_str += (
                    f"  - current_action: '{self._get_main_action_label(character.current_actions)}'\n"
                )
                current_states_str += f"  - relationships: {character.relationships}\n"
            else:
                current_states_str += (
                    f"  - current_action: '{self._get_main_action_label(character.current_actions)}'\n"
                )
                current_states_str += f"  - target_action: '{self._get_main_action_label(character.target_actions)}'\n"
                current_states_str += f"  - target_relationships: {character.relationships}\n"

        # Parallel Events Section
        events_str = "[Parallel Events to Parse]\n"
        for i, event in enumerate(events):
            events_str += f"Event {i+1}: {event}\n"

        feedback_str = f"[Feedback]\n{feedback if feedback else 'None'}"

        query_message = [
            scene_description_str,
            available_action_labels_str,
            current_states_str,
            events_str,
            feedback_str,
        ]
        query_message = map(str.strip, query_message)
        query_message = "\n\n".join(query_message).strip()
        return query_message

    def _post_process_response(self, event_or_events, response):
        # Support single events or event lists
        if isinstance(event_or_events, list):
            # Parallel Event Handling
            parsed_events = self.code_executor.execute_response(response)

            # If a single dictionary is returned (result of parsing all characters), it needs to be grouped by event
            if isinstance(parsed_events, dict):
                # Create a parsing result for each event
                event_parsed_results = []
                for event in event_or_events:
                    # Create a parsing result that includes only the relevant characters for each event
                    event_parsed = {}
                    for character in event.involved_characters:
                        if character.name in parsed_events:
                            event_parsed[character.name] = parsed_events[character.name]

                    # Parse the results of handling this event
                    self._process_single_event_parsed_result(event, event_parsed)
                    event_parsed_results.append(event_parsed)

                return event_parsed_results
            else:
                # If a list is returned, process it according to the original logic
                event_parsed_results = []
                for i, event in enumerate(event_or_events):
                    if i < len(parsed_events):
                        parsed_event = parsed_events[i]
                        self._process_single_event_parsed_result(event, parsed_event)
                        event_parsed_results.append(parsed_event)
                    else:
                        # If there is no corresponding parse result, create an empty one
                        event_parsed_results.append({})

                return event_parsed_results
        else:
            # Single event processing (original logic)
            event = event_or_events
            parsed_event = self.code_executor.execute_response(response)

            # convert target_action to list
            for items in parsed_event.values():
                items["target_action"] = [items["target_action"]]

            # integrate position and orientation relationships
            for items in parsed_event.values():
                items["relationships"] = []
                for relationship in items["position"]:
                    if relationship.startswith("interact with"):
                        anchor = relationship.split(" ")[-1]
                        relationship = f"adjacent to {anchor}"
                    items["relationships"].append(relationship)
                for relationship in items.get("orientation", []):
                    items["relationships"].append(relationship)

                del items["position"]
                if "orientation" in items:
                    del items["orientation"]

            # make sure that the target actions include "sit" or "lie"
            # if the relationship includes "sit on" or "lie on"
            # for items in parsed_event.values():
            #     for relationship in items["relationships"]:
            #         if relationship.startswith("sit on"):
            #             items["target_action"].append("sit down")
            #             break
            #         elif relationship.startswith("lie on"):
            #             items["target_action"].append("lie down")
            #             break

            # remove duplicate target actions
            for items in parsed_event.values():
                items["target_action"] = list(set(items["target_action"]))

            # Keep hand_target unchanged for ActionPlanningModule.

            # store parsed event in the event object
            event.parsed_event = deepcopy(parsed_event)

            return parsed_event

    def _process_single_event_parsed_result(self, event: Event, parsed_event: dict):
        """Parse the results of processing a single event"""
        # convert target_action to list
        for items in parsed_event.values():
            items["target_action"] = [items["target_action"]]

        # integrate position and orientation relationships
        for items in parsed_event.values():
            items["relationships"] = []
            for relationship in items["position"]:
                if relationship.startswith("interact with"):
                    anchor = relationship.split(" ")[-1]
                    relationship = f"adjacent to {anchor}"
                items["relationships"].append(relationship)
            for relationship in items.get("orientation", []):
                items["relationships"].append(relationship)

            del items["position"]
            if "orientation" in items:
                del items["orientation"]

        # make sure that the target actions include "sit" or "lie"
        # if the relationship includes "sit on" or "lie on"
        for items in parsed_event.values():
            for relationship in items["relationships"]:
                if relationship.startswith("sit on"):
                    items["target_action"].append("sit down")
                    break
                elif relationship.startswith("lie on"):
                    items["target_action"].append("lie down")
                    break

        # remove duplicate target actions
        for items in parsed_event.values():
            items["target_action"] = list(set(items["target_action"]))

        # Keep hand_target unchanged for ActionPlanningModule.

        # store parsed event in the event object
        event.parsed_event = deepcopy(parsed_event)

    def _check_validity(self, event_or_events, response, available_action_labels):
        if not self.use_self_feedback:
            return ""

        pattern = r"^(" + "|".join(_VALID_RELATIONSHIPS) + ")"

        # Support single events or event lists
        if isinstance(event_or_events, list):
            # Parallel event validation
            events = event_or_events

            # If the response is a dictionary, check all characters
            if isinstance(response, dict):
                for event in events:
                    for character in event.involved_characters:
                        if character.name not in response:
                            feedback = f'You did not generated output for character "{character.name}"'
                            return feedback

                    for character in response:
                        if character in [c.name for c in event.involved_characters]:
                            relationships = response[character]["relationships"]
                            for relationship in relationships:
                                if not re.match(pattern, relationship):
                                    feedback = f'You have generated a relationship that is not valid: "{relationship}"'
                                    feedback += (
                                        "\nPlease regenerate a valid relationship specified in the system message."
                                    )
                                    return feedback

                                anchor = relationship.split(" ")[-1]
                                if anchor not in self.character_names and anchor not in self.object_labels:
                                    feedback = f'You have generated a relationship with non-existent entity as an anchor: "{relationship}" (anchor: "{anchor}")'
                                    feedback += "\nPlease regenerate a valid relationship using an existing entity in the scene."
                                    return feedback
            else:
                # If the response is a list, process it according to the original logic
                for i, event in enumerate(events):
                    if i >= len(response):
                        return f"Missing response for event {i+1}"

                    event_response = response[i]

                    for character in event.involved_characters:
                        if character.name not in event_response:
                            feedback = f'You did not generated output for character "{character.name}" in event {i+1}'
                            return feedback

                    for character in event_response:
                        relationships = event_response[character]["relationships"]
                        for relationship in relationships:
                            if not re.match(pattern, relationship):
                                feedback = f'You have generated a relationship that is not valid: "{relationship}" in event {i+1}'
                                feedback += "\nPlease regenerate a valid relationship specified in the system message."
                                return feedback

                            anchor = relationship.split(" ")[-1]
                            if anchor not in self.character_names and anchor not in self.object_labels:
                                feedback = f'You have generated a relationship with non-existent entity as an anchor: "{relationship}" (anchor: "{anchor}") in event {i+1}'
                                feedback += (
                                    "\nPlease regenerate a valid relationship using an existing entity in the scene."
                                )
                                return feedback
        else:
            # Single event verification (original logic)
            event = event_or_events

            for character in event.involved_characters:
                if character.name not in response:
                    feedback = f'You did not generated output for this character: "{character.name}"'
                    return feedback

            for character in response:
                relationships = response[character]["relationships"]
                for relationship in relationships:
                    if not re.match(pattern, relationship):
                        feedback = f'You have generated a relationship that is not valid: "{relationship}"'
                        feedback += "\nPlease regenerate a valid relationship specified in the system message."
                        return feedback

                    anchor = relationship.split(" ")[-1]
                    if anchor not in self.character_names and anchor not in self.object_labels:
                        feedback = f'You have generated a relationship with non-existent entity as an anchor: "{relationship}" (anchor: "{anchor}")'
                        feedback += "\nPlease regenerate a valid relationship using an existing entity in the scene."
                        return feedback

        return ""

    def _get_main_action_label(self, action_labels):
        if not action_labels:
            return ""
        if len(action_labels) == 1:
            return action_labels[0]
        if all(action.startswith("sit") or action.startswith("lie") for action in action_labels):
            return action_labels[0]
        return next(
            (action for action in action_labels if not action.startswith("sit") and not action.startswith("lie")),
            action_labels[0],
        )
