import json
import re
from pathlib import Path
from typing import List


from src.action_planning.narrator.narrator import Narrator
from src.motion_synthesis.states import State
from src.type import Character, Event, Object, ParallelEventGroup


class NarratorTimeline(Narrator):
    def __init__(self, characters: List[Character], objects: List[Object], cfg) -> None:
        super().__init__(characters, objects, cfg)
        self.system_messages_dir = self.base_dir / "system_messages"
        self.system_message_file = self.system_messages_dir / "narrator_timeline.txt"
        self.event_list = []
        self.timeline_dependencies = []
        # Associated action groups (from dependencies) {"parallel": [..]})
        self.parallel_groups = []  # List[List[str]]

        # currently we only generate one plan
        self.plan_generated = False

        # Timeline narrator does not use examples for now, as the timeline format differs from the regular event format
        # If examples are required, a special timeline example format needs to be created
        self.use_examples = False

    def is_new_event_required(self):
        is_required = super().is_new_event_required()
        if is_required == False:
            return False

        # If event_list is empty, a new event needs to be generated
        if self.event_list == []:
            if self.plan_generated:
                return False
            self.plan_generated = True
            return True

        # Iterate through event_list to find the first event that meets the dependency requirements
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
        self.logger.info("Generating subsequent plan...")

        # If event_list is empty, generate the complete timeline
        if not self.event_list:
            self.logger.info("Event list is empty, generating full timeline...")
            response, prompt_tokens, completion_tokens = self._query_llm(
                scene_description,
                available_activities,
                user_instruction,
                feedback,
            )
            accumulated_prompt_tokens += prompt_tokens
            accumulated_completion_tokens += completion_tokens
            self.left_query_count -= 1
            self.left_query_count = max(0, self.left_query_count)

            response = self._postprocess_before_validation(response)
            feedback = self._check_validity(response)
            if feedback and self.self_feedback_count < self.max_self_feedback_count:
                self.logger.warning("Invalid response detected, retrying with feedback...")
                self.logger.warning(f"feedback: {feedback}")
                self.self_feedback_count += 1
                return self.generate_subsequent_plan(
                    scene_description,
                    available_activities,
                    user_instruction,
                    feedback,
                    with_usage,
                    accumulated_prompt_tokens,
                    accumulated_completion_tokens,
                )

            # Parse the timeline and populate event_list
            self._parse_timeline_response(response)

        # Extract the next event that meets the dependency requirements from event_list
        if self.event_list:
            # Add debugging information
            self.logger.info(f"📋 Timeline status: {len(self.event_list)} events remaining in queue")
            if self.logger.level <= 10:  # DEBUG level
                self.debug_timeline_status()

            next_event = self._get_next_available_event()
            if next_event:
                self.logger.info(f"Extracting next event from timeline: {next_event}")

                if with_usage:
                    return (
                        next_event,
                        accumulated_prompt_tokens,
                        accumulated_completion_tokens,
                    )  # Extract from the timeline items that do not require token consumption

                self.previous_events.append(next_event)
                self.skipped_characters = []
                self.self_feedback_count = 0
                return next_event
            else:
                self.logger.info("No events available due to unmet dependencies")
                if with_usage:
                    return None, accumulated_prompt_tokens, accumulated_completion_tokens
                return None
        else:
            self.logger.info("No more events in timeline")
            if with_usage:
                return None, 0, 0
            return None

    def _create_example_message(self, example_file: Path):
        # Timeline narrator does not use examples for now, as the timeline format differs from the regular event format
        # If examples are required, a special timeline example format needs to be created
        # Temporarily disable examples to avoid errors
        return "", ""

    def _postprocess_before_validation(self, response):
        # Remove comments from the JSON (// The initial line)
        # LLMs sometimes add comments in JSON, which needs to be cleaned up
        if isinstance(response, str):
            # Process line by line, remove unnecessary elements // Remove comments
            lines = response.split("\n")
            cleaned_lines = []
            for line in lines:
                # Check if it contains certain elements // Remove comments
                if "//" in line:
                    # Found // Position
                    comment_pos = line.find("//")
                    # Conducts checks // Check if the previous element is within the string
                    # Simple processing: if so // If there are even number of quotes before the element, it’s not within the string
                    before_comment = line[:comment_pos]
                    quote_count = before_comment.count('"') - before_comment.count('\\"')
                    if quote_count % 2 == 0:
                        # // If not within the string, can be removed
                        line = before_comment.rstrip()
                        # If comments are removed and only whitespace or commas remain, keep this line (maintain JSON structure)
                        if line.strip() and not line.strip().endswith(","):
                            line = line
                        elif line.strip().endswith(","):
                            line = line  # Keep the commas
                        else:
                            continue  # Skip empty lines
                cleaned_lines.append(line)
            response = "\n".join(cleaned_lines)

        response = json.loads(response)
        if not response.get("event", None):
            return response

        # post process activity part
        activity = response["event"]["activity"]
        activity = activity.replace("[", "").replace("]", "")
        activity = re.sub(r"\s\(.*?\)$", "", activity)  # remove event state
        response["event"]["activity"] = activity

        return response

    def _parse_timeline_response(self, response):
        """Parse the timeline response and fill the event_list with events in chronological order"""
        try:
            if not response.get("plan", None):
                self.logger.warning("No plan found in response")
                return

            events = response["plan"].get("events", [])
            dependencies = response["plan"].get("dependencies", [])

            # Save the dependency relationships to instance variables
            self.timeline_dependencies = dependencies
            # Parse parallel groups
            self.parallel_groups = []
            for dep in dependencies:
                if isinstance(dep, dict) and "parallel" in dep:
                    group = dep.get("parallel", [])
                    if isinstance(group, list) and group:
                        ids = [eid for eid in group if isinstance(eid, str)]
                        if ids:
                            self.parallel_groups.append(ids)

            # Sort events based on dependencies
            sorted_events = self._sort_events_by_dependencies(events, dependencies)

            # Convert sorted events into Event objects and add them to event_list
            for event_data in sorted_events:
                event = self._create_event_from_timeline_data(event_data)
                if event:
                    self.event_list.append(event)

            self.logger.info(
                f"Parsed {len(self.event_list)} events from timeline with {len(dependencies)} dependencies"
            )

        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            self.logger.error(f"Error parsing timeline response: {e}\nFull traceback:\n{error_details}")
            self.event_list = []
            self.timeline_dependencies = []
            self.parallel_groups = []
            self.active_parallel_group = []

    def _sort_events_by_dependencies(self, events, dependencies):
        """Sort events based on dependency relationships"""
        # Create a mapping from event_id to events
        event_map = {event["event_id"]: event for event in events}

        # Create a dependency graph: event_id -> [dependencies]
        # For example: event_002 depends on event_001 -> [event_001] Indicate that event_002 depends on event_001
        dependency_graph = {}
        for dep in dependencies:
            # Skip parallel declarations and exception structures
            if not isinstance(dep, dict):
                raise ValueError(f"Dependency {dep} is not a dictionary")
            if "parallel" in dep:
                continue
            after_id = dep.get("after")
            before_id = dep.get("before")
            if not after_id or not before_id:
                continue
            if before_id not in dependency_graph:
                dependency_graph[before_id] = []
            dependency_graph[before_id].append(after_id)

        self.logger.debug(f"Dependency graph: {dependency_graph}")

        # Topological sorting
        visited = set()
        temp_visited = set()
        result = []

        def visit(event_id):
            if event_id in temp_visited:
                raise ValueError("Circular dependency detected")
            if event_id in visited:
                return
            temp_visited.add(event_id)

            # Visit dependent events first
            for dep_id in dependency_graph.get(event_id, []):
                visit(dep_id)

            temp_visited.remove(event_id)
            visited.add(event_id)
            result.append(event_map[event_id])

        # Visit all events
        for event in events:
            if event["event_id"] not in visited:
                visit(event["event_id"])

        self.logger.debug(f"Sorted events: {[e['event_id'] for e in result]}")
        return result

    def _get_next_available_event_idx(self):
        """Get the index of the next event or parallel group that satisfies dependencies"""
        if not self.event_list:
            return None

        # Iterate through pending events, considering parallel groups and character idleness
        for i, event in enumerate(self.event_list):
            eid = self._get_event_id(event)
            group = self._get_parallel_group(eid)
            if group:
                # Parallel group: All events within the group must satisfy dependencies, and all involved characters must be idle
                if self._is_parallel_group_ready(group):
                    # Return the index range of the parallel group
                    parallel_indices = []
                    for j, candidate in enumerate(self.event_list):
                        if self._get_event_id(candidate) in group:
                            parallel_indices.append(j)

                    self.logger.info(
                        f"Found parallel group indices: {parallel_indices} for events: {[self._get_event_id(self.event_list[idx]) for idx in parallel_indices]}"
                    )
                    return parallel_indices
                continue
            else:
                # Non-parallel: Satisfies dependencies and characters are idle
                if self._check_event_dependencies(event) and self._are_event_characters_idle(event):
                    self.logger.info(f"Found single event index: {i} for event: {eid}")
                    return i

        # No event found that satisfies dependencies
        return None

    def _get_next_available_event(self):
        """Use the index to get the next event or parallel group that satisfies dependencies"""
        idx_or_indices = self._get_next_available_event_idx()
        if idx_or_indices is None:
            return None

        if isinstance(idx_or_indices, list):
            # Parallel group: Remove from the end to avoid index issues
            parallel_events = []
            group_ids = []
            for idx in sorted(idx_or_indices, reverse=True):
                event = self.event_list.pop(idx)
                parallel_events.insert(0, event)  # Maintain order
                group_ids.append(self._get_event_id(event))

            # Get information about the parallel group
            group = self._get_parallel_group(group_ids[0])
            parallel_event_group = ParallelEventGroup(parallel_events, group.copy())

            self.logger.info(f"Returning parallel group with {len(parallel_events)} events: {group_ids}")
            return parallel_event_group
        else:
            # Single event
            event = self.event_list.pop(idx_or_indices)
            self.logger.info(f"Returning single event: {self._get_event_id(event)}")
            return event

    def _check_event_dependencies(self, event):
        """Check if the event satisfies all dependencies"""
        if not hasattr(event, "timeline_data") or not event.timeline_data:
            return True  # No timeline data available; considered to satisfy dependencies

        event_id = event.timeline_data.get("event_id")
        if not event_id:
            raise ValueError(f"Event {event} has no event_id")

        # Check if all dependencies of this event have been completed
        dependencies = getattr(self, "timeline_dependencies", [])
        self.logger.debug(f"Checking dependencies for event {event_id}")
        self.logger.debug(f"Total dependencies: {len(dependencies)}")

        # Check for any dependencies (skip parallel declarations)
        for dep in dependencies:
            if not isinstance(dep, dict):
                raise ValueError(f"Dependency {dep} is not a dictionary")
            if "parallel" in dep:
                continue
            self.logger.debug(f"  Dependency: {dep}")
            before_id = dep.get("before")
            after_event_id = dep.get("after")
            if not before_id or not after_event_id:
                continue
            if before_id == event_id:
                self.logger.debug(f"  Event {event_id} depends on {after_event_id}")
                if not self._is_event_completed(after_event_id):
                    self.logger.debug(f"Event {event_id} waiting for dependency {after_event_id}")
                    return False

        return True

    def _are_event_characters_idle(self, event):
        """Are the relevant characters idle (IDLE and not occupying events?)."""
        for character in event.involved_characters:
            state = character.state
            if isinstance(state, str):
                if state.lower() != "idle":
                    return False
            else:
                if state != State.IDLE:
                    return False
            if getattr(character, "event", None) is not None:
                return False
        return True

    def _get_event_id(self, event: Event) -> str:
        if hasattr(event, "timeline_data") and event.timeline_data:
            return event.timeline_data.get("event_id", "")
        return ""

    def _get_parallel_group(self, event_id: str):
        for group in self.parallel_groups:
            if event_id in group:
                return group
        return []

    def _is_parallel_group_ready(self, group_ids: list[str]) -> bool:
        """Are the parallel groups ready: All events within the group have satisfied dependencies, and all involved characters are idle."""
        # Collect event objects within the group (must still be in the queue)
        group_events = []
        for eid in group_ids:
            ev = next((e for e in self.event_list if self._get_event_id(e) == eid), None)
            if ev is None:
                return False
            group_events.append(ev)

        # Dependency check
        for ev in group_events:
            if not self._check_event_dependencies(ev):
                return False

        # Character idle check
        for ev in group_events:
            if not self._are_event_characters_idle(ev):
                return False

        return True

    def _is_event_completed(self, event_id):
        """Check if the event with specified event_id has been completed"""
        # Check if there is an event with that event_id in previous_events and its status is completed
        for prev_event in self.previous_events:
            # Case 1: Ordinary Event
            if isinstance(prev_event, Event):
                if (
                    hasattr(prev_event, "timeline_data")
                    and prev_event.timeline_data
                    and prev_event.timeline_data.get("event_id") == event_id
                ):
                    # Check event status; completed indicates completion
                    return prev_event.state == "completed"

            # Scenario 2: ParallelEventGroup
            elif isinstance(prev_event, ParallelEventGroup):
                # Check the member events in the parallel group
                for sub_event in prev_event.events:
                    if (
                        hasattr(sub_event, "timeline_data")
                        and sub_event.timeline_data
                        and sub_event.timeline_data.get("event_id") == event_id
                    ):
                        # For members of the parallel group, it is necessary to check whether the entire parallel group has completed
                        # Since the events in the parallel group need to be completed simultaneously
                        is_completed = prev_event.state == "completed"
                        self.logger.debug(
                            f"Event {event_id} is part of parallel group, group completed: {is_completed}"
                        )
                        return is_completed

        # If it is not found in previous_events, it means that the event has not started yet
        # In this case, False should be returned, indicating that the dependency is not met
        self.logger.debug(f"Event {event_id} not found in previous events, considering as not completed")
        return False

    def debug_timeline_status(self):
        """Debugging method: Display the current timeline status"""
        self.logger.info("=== Timeline Debug Status ===")
        self.logger.info(f"Event list length: {len(self.event_list)}")
        self.logger.info(f"Dependencies count: {len(self.timeline_dependencies)}")

        # Show the remaining events and their dependency status
        for i, event in enumerate(self.event_list):
            event_id = event.timeline_data.get("event_id", "unknown") if hasattr(event, "timeline_data") else "unknown"
            deps_satisfied = self._check_event_dependencies(event)
            self.logger.info(f"  Pending Event {i}: {event_id} - Dependencies satisfied: {deps_satisfied}")

            # Show the dependencies of the event
            if hasattr(event, "timeline_data") and event.timeline_data:
                event_id = event.timeline_data.get("event_id")
                for dep in self.timeline_dependencies:
                    if dep.get("before") == event_id:
                        after_event_id = dep.get("after")
                        is_completed = self._is_event_completed(after_event_id)
                        self.logger.info(f"    -> Depends on {after_event_id}: {'✓' if is_completed else '✗'}")

        # Show the completion status in the event history
        completed_events = []
        ongoing_events = []
        for prev_event in self.previous_events:
            if (
                hasattr(prev_event, "timeline_data")
                and prev_event.timeline_data
                and prev_event.timeline_data.get("event_id")
            ):
                event_id = prev_event.timeline_data.get("event_id")
                if prev_event.state == "completed":
                    completed_events.append(event_id)
                else:
                    ongoing_events.append(f"{event_id}({prev_event.state})")

        if completed_events:
            self.logger.info(f"Completed events in history: {', '.join(completed_events)}")
        if ongoing_events:
            self.logger.info(f"Ongoing events in history: {', '.join(ongoing_events)}")
        if not completed_events and not ongoing_events:
            self.logger.info("No events in history yet")

        self.logger.info("=== End Timeline Debug ===")

    def _create_event_from_timeline_data(self, event_data):
        """Create an Event object from the timeline data"""
        try:
            character_name = event_data["character"]
            activity = event_data["activity"]

            # Find the corresponding character
            character = None
            for char in self.characters:
                if char.name == character_name:
                    character = char
                    break

            if not character:
                self.logger.warning(f"Character {character_name} not found")
                return None

            # Create an Event object
            event = Event(
                involved_characters=[character],
                activity=activity,
            )

            # Add timeline-related metadata
            event.timeline_data = {
                "event_id": event_data["event_id"],
                "duration": event_data.get("duration", 0),
                "tags": event_data.get("tags", []),
                "location_scope": event_data.get("location_scope", ""),
                "can_time_wrap": event_data.get("can_time_wrap", True),
                "checked_completed": False,  # Check if the completion status has been verified, Upon completion of each event, check if there are new events that satisfy dependencies
            }

            return event

        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            self.logger.error(f"Error creating event from timeline data: {e}\nFull traceback:\n{error_details}")
            return None
