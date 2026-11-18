import json
import re
from collections import deque
from pathlib import Path
from typing import List, Dict, Set


def expand_parallel_dependencies(deps: List[dict], id2event: Dict[str, dict]) -> List[dict]:
    """Infer shared dependencies for events in parallel groups

    Events within a parallel group should have the same predecessor dependencies. For example:
    - If e3 depends on e1 and e2
    - and e3 and e4 are parallel
    - then e4 should also depend on e1 and e2

    Args:
        deps: Original list of dependencies
        id2event: Mapping from event ID to event object

    Returns:
        Extended list of dependencies
    """
    # Step 1: Identify all parallel groups
    parallel_groups: List[Set[str]] = []
    for d in deps:
        if isinstance(d, dict) and "parallel" in d:
            grp = d.get("parallel", []) or []
            if grp:
                parallel_groups.append(set(grp))

    # Step 2: Collect predecessors for each event (only non-parallel dependencies)
    predecessors: Dict[str, Set[str]] = {eid: set() for eid in id2event.keys()}
    for d in deps:
        if isinstance(d, dict) and "parallel" not in d:
            a = d.get("after")
            b = d.get("before")
            if a in id2event and b in id2event:
                predecessors[b].add(a)

    # Step 3: Add shared predecessors for all events in parallel groups
    expanded_deps = list(deps)
    for grp in parallel_groups:
        # Collect predecessors for all events in the parallel group (take the union)
        all_predecessors: Set[str] = set()
        for eid in grp:
            if eid in predecessors:
                all_predecessors |= predecessors[eid]

        # Add all shared predecessors to each event in the parallel group
        for eid in grp:
            for pred in all_predecessors:
                # Check if this dependency already exists
                dep_exists = any(
                    isinstance(d, dict) and d.get("after") == pred and d.get("before") == eid for d in expanded_deps
                )
                if not dep_exists:
                    expanded_deps.append({"after": pred, "before": eid})

    return expanded_deps


def resolve_activity_to_specific(activity: str) -> str:
    """Select the first specific value from the activity that contains OR options

    Example:
        "sit down on (chair_2 | chair_4 | chair_5 | chair_6) in the dining area"
        -> "sit down on chair_2 in the dining area"

    Args:
        activity: Activity string that may contain OR options

    Returns:
        Parsed specific activity strings
    """
    if not activity:
        return activity

    # Searching for all (option1 | option2 | ...) Format segment
    or_pattern = r"\(([^)]+(?:\s*\|\s*[^)]+)+)\)"
    matches = list(re.finditer(or_pattern, activity))

    if not matches:
        # No or option, returns directly
        return activity

    # Replace from back to front to avoid index changes
    result = activity
    for match in reversed(matches):
        start, end = match.span()
        content = match.group(1)
        # Split option, select the first (remove spaces)
        options = [opt.strip() for opt in content.split("|")]
        if options:
            # Replace the entire content with the first option (option1 | option2 | ...) Part
            result = result[:start] + options[0] + result[end:]

    return result


def get_topological_order(events: List[dict], dependencies: List[dict]) -> List[List[dict]]:
    """Obtain the topological sorting sequence of events

    Args:
        events: Event list, each event should contain event_id, character, Fields like activity
        dependencies: Dependency list, including after/before Dependencies of the parallel type

    Returns:
        List of events arranged in topological order. Each internal list contains events that can be executed concurrently.
        Example: [[event1, event2], [event3], [event4, event5]]
    """
    # Establish a mapping from id to event
    id2event = {}
    for ev in events:
        eid = ev.get("event_id")
        if eid:
            id2event[eid] = ev

    if not id2event:
        return []

    # Expand parallel group dependencies
    deps = expand_parallel_dependencies(dependencies, id2event)

    # Extract parallel groups
    parallel_groups = []
    for d in deps:
        if isinstance(d, dict) and "parallel" in d:
            grp = set(d.get("parallel", []) or [])
            if grp:
                parallel_groups.append(grp)

    # Construct a dependency graph (adjacency list) and in-degree
    indeg = {eid: 0 for eid in id2event.keys()}
    graph = {eid: [] for eid in id2event.keys()}
    for d in deps:
        if isinstance(d, dict) and "parallel" not in d:
            a = d.get("after")
            b = d.get("before")
            if a in id2event and b in id2event:
                graph[a].append(b)
                indeg[b] += 1

    # Topological sorting (Kahn’s algorithm), prioritize identifying parallel groups
    q = deque([eid for eid, deg in indeg.items() if deg == 0])
    steps = []

    while q:
        zero_set = set(q)
        used = set()
        matched = False

        # Try to match parallel groups
        for grp in parallel_groups:
            if grp.issubset(zero_set):
                # Find a complete parallel group, as a step
                steps.append([id2event[eid] for eid in sorted(list(grp))])
                used |= grp
                matched = True
                break

        if not matched:
            # No matching parallel group found; treat all zero-degree events as a step
            steps.append([id2event[eid] for eid in sorted(list(zero_set))])
            used |= zero_set

        # Update queue: remove used events
        new_q = deque()
        for eid in list(q):
            if eid not in used:
                new_q.append(eid)

        # Process successors of used events
        q = deque()
        for eid in used:
            for v in graph.get(eid, []):
                indeg[v] -= 1
                if indeg[v] == 0:
                    new_q.append(v)
        q = new_q

        # Remove matched parallel groups
        if matched:
            parallel_groups = [g for g in parallel_groups if not g.issubset(used)]

    return steps


def derive_narrator_cases_from_expected_timeline(test_case: dict, available_action_labels: List[str], output_dir: Path):
    """Derive narrator test cases from expected_timeline

    Args:
        test_case: Test cases containing expected_timeline
        available_action_labels: List of available action tags
        output_dir: Output directory
    """
    et = (test_case or {}).get("expected_timeline", {}) or {}
    events = et.get("events", [])
    deps = et.get("dependencies", [])
    if not events:
        return

    id2event = {}
    all_chars_set = set()
    for ev in events:
        eid = ev.get("event_id")
        if not eid:
            continue
        id2event[eid] = ev
        ch = ev.get("character")
        if ch:
            all_chars_set.add(ch)

    # Sort all characters in alphabetical order
    all_chars = sorted(list(all_chars_set))

    # Expand parallel group dependencies (fix issues with parallel group predecessor inference)
    deps = expand_parallel_dependencies(deps, id2event)

    parallel_groups = []
    for d in deps:
        if isinstance(d, dict) and "parallel" in d:
            grp = set(d.get("parallel", []) or [])
            if grp:
                parallel_groups.append(grp)

    indeg = {eid: 0 for eid in id2event.keys()}
    graph = {eid: [] for eid in id2event.keys()}
    for d in deps:
        if isinstance(d, dict) and "parallel" not in d:
            a = d.get("after")
            b = d.get("before")
            if a in id2event and b in id2event:
                graph[a].append(b)
                indeg[b] += 1

    q = deque([eid for eid, deg in indeg.items() if deg == 0])
    steps = []
    while q:
        zero_set = set(q)
        used = set()
        matched = False
        for grp in parallel_groups:
            if grp.issubset(zero_set):
                steps.append(sorted(list(grp)))
                used |= grp
                matched = True
                break
        if not matched:
            steps.append(sorted(list(zero_set)))
            used |= zero_set

        new_q = deque()
        for eid in list(q):
            if eid not in used:
                new_q.append(eid)
        q = deque()
        for eid in used:
            for v in graph.get(eid, []):
                indeg[v] -= 1
                if indeg[v] == 0:
                    new_q.append(v)
        q = new_q

        if matched:
            parallel_groups = [g for g in parallel_groups if not g.issubset(used)]

    output_dir.mkdir(parents=True, exist_ok=True)
    base_id = test_case.get("id", 0)

    def extract_labels(activity: str) -> List[str]:
        labels = []
        if not activity:
            return labels
        act = activity.lower().replace("_", " ")
        for label in available_action_labels:
            l = label.lower()
            if l in act or l.replace("_", " ") in act:
                labels.append(label)
        return labels

    common = {
        "scene_name": test_case.get("scene_name"),
        "action_labels": test_case.get("action_labels", []),
        "user_instruction": test_case.get("user_instruction", ""),
    }

    linear = [step for step in steps if step]

    # Identify all events in dependencies (used to determine if test cases need to be generated)
    events_in_dependency = set()
    for d in deps:
        if isinstance(d, dict) and "parallel" not in d:
            a = d.get("after")
            b = d.get("before")
            if a in id2event and b in id2event:
                events_in_dependency.add(a)
                events_in_dependency.add(b)
        elif isinstance(d, dict) and "parallel" in d:
            # All events in the parallel group are in dependencies
            for eid in d.get("parallel", []):
                if eid in id2event:
                    events_in_dependency.add(eid)

    # Build dependency mapping: after -> Before completion, after should be generated)
    dependency_map = {}  # after_id -> [before_ids]
    for d in deps:
        if isinstance(d, dict) and "parallel" not in d:
            a = d.get("after")
            b = d.get("before")
            if a in id2event and b in id2event:
                if a not in dependency_map:
                    dependency_map[a] = []
                dependency_map[a].append(b)

    def has_next_dependent_event(step_ids: List[str]) -> bool:
        """Check if there will be a next dependent event after the current step is completed"""
        for eid in step_ids:
            if eid in dependency_map:
                return True
        return False

    def build_history_upto(step_idx: int, ongoing_ids: List[str] | None = None):
        """Build planning_history. For events that have already occurred, use specific activity values (without or option)"""
        hist = []
        for i in range(step_idx):
            for eid in linear[i]:
                ev = id2event[eid]
                # For events that have already occurred, use specific activity (first option)
                activity = resolve_activity_to_specific(ev.get("activity", ""))
                hist.append(
                    {
                        "idx": len(hist),
                        "type": "actor",
                        "distraction_level": 0,
                        "state": "completed",
                        "characters": [ev.get("character")],
                        "activity": activity,
                        "details": [],
                    }
                )
        ongoing_ids = ongoing_ids or []
        for eid in ongoing_ids:
            ev = id2event[eid]
            # For ongoing events, also use specific activity (first option)
            activity = resolve_activity_to_specific(ev.get("activity", ""))
            hist.append(
                {
                    "idx": len(hist),
                    "type": "actor",
                    "distraction_level": 0,
                    "state": "ongoing",
                    "characters": [ev.get("character")],
                    "activity": activity,
                    "details": [],
                }
            )
        return hist

    def build_available(ongoing_ids: List[str] | None = None):
        """Build a list of available characters, sorted alphabetically"""
        ongoing_chars = set()
        for eid in ongoing_ids or []:
            ev = id2event[eid]
            if ev.get("character"):
                ongoing_chars.add(ev["character"])
        # Return in alphabetical order
        return sorted([c for c in all_chars if c not in ongoing_chars])

    for k, step_ids in enumerate(linear):
        # Generate use cases only for steps in the dependency relationship
        step_in_dependency = any(eid in events_in_dependency for eid in step_ids)
        if not step_in_dependency:
            continue

        # Check if there will be a next dependent event after the current step is completed
        has_next = has_next_dependent_event(step_ids)

        # ongoing: The current step is in progress, dependency not met, expected next event to be null
        # Only generate ongoing use cases when there will be a next dependent event after completion
        if has_next:
            ongoing_hist = build_history_upto(k, ongoing_ids=step_ids)
            available_chars = build_available(step_ids)
            expected_next = {}  # Dependency not met, expected null
            ongoing_case = {
                **common,
                "planning_history": ongoing_hist,
                "available_characters": available_chars,
                "expected_next_event": expected_next,
            }
            (output_dir / f"{base_id}_step{k+1}_ongoing.json").write_text(json.dumps(ongoing_case, indent=2))

        # completed: After the current step is completed, expected next dependent event
        # Only generate completed use cases when there will be a next dependent event after completion
        if has_next:
            completed_hist = build_history_upto(k + 1, ongoing_ids=[])
            available_next = build_available([])
            # Find the next dependent event (via dependency_map)
            next_dependent_ids = []
            for eid in step_ids:
                if eid in dependency_map:
                    next_dependent_ids.extend(dependency_map[eid])

            # If the next dependent event is found, use it; otherwise, proceed to the next step in the topological sort
            if next_dependent_ids:
                # Remove duplicates and maintain order
                seen = set()
                unique_next_ids = []
                for eid in next_dependent_ids:
                    if eid not in seen:
                        seen.add(eid)
                        unique_next_ids.append(eid)

                if len(unique_next_ids) > 1:
                    # Parallel groups
                    expected_next2 = {
                        "characters": [id2event[eid]["character"] for eid in unique_next_ids if eid in id2event]
                    }
                elif len(unique_next_ids) == 1:
                    ev2 = id2event[unique_next_ids[0]]
                    # Use the activity directly, instead of inferring the target_action
                    expected_next2 = {"character": ev2.get("character"), "activity": ev2.get("activity", "")}
                else:
                    expected_next2 = {}
            elif k + 1 < len(linear):
                # Back to the next step in the topological sort
                next_ids = linear[k + 1]
                if len(next_ids) > 1:
                    expected_next2 = {"characters": [id2event[eid]["character"] for eid in next_ids]}
                else:
                    ev2 = id2event[next_ids[0]]
                    # Use the activity directly, instead of inferring the target_action
                    expected_next2 = {"character": ev2.get("character"), "activity": ev2.get("activity", "")}
            else:
                expected_next2 = {}

            completed_case = {
                **common,
                "planning_history": completed_hist,
                "available_characters": available_next,
                "expected_next_event": expected_next2,
            }
            (output_dir / f"{base_id}_step{k+1}_completed.json").write_text(json.dumps(completed_case, indent=2))
