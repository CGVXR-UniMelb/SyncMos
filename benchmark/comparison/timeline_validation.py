from typing import Tuple, Dict, Any, Optional
import re


def _are_activities_similar(activity1: str, activity2: str) -> bool:
    """Check if two actions are similar (even if they don’t match exactly)

    Example:
    - "turn off the light" and "turn off lights" They should be considered similar
    - "put down the bottle" and "put down bottle" They should be considered similar

    Args:
        activity1: First action
        activity2: Second action

    Returns:
        Return True if similar, otherwise return False
    """
    if not activity1 or not activity2:
        return False

    # Convert to lowercase and remove extra spaces
    a1 = re.sub(r"\s+", " ", activity1.lower().strip())
    a2 = re.sub(r"\s+", " ", activity2.lower().strip())

    # If they are exactly the same, return True directly
    if a1 == a2:
        return True

    # Extract core action words (remove location information, articles, etc.)
    def extract_core_action(act: str) -> str:
        # Remove common location phrases
        act = re.sub(r"\s+(near|at|on|in|from|to)\s+[^,]+", "", act)
        # Remove articles
        act = re.sub(r"\b(the|a|an)\s+", "", act)
        # Remove extra spaces
        act = re.sub(r"\s+", " ", act).strip()
        return act

    core1 = extract_core_action(a1)
    core2 = extract_core_action(a2)

    # Check if core actions are the same (ignore singular and plural forms)
    # Example "turn off light" and "turn off lights"
    if core1 == core2:
        return True

    # Check if there are only differences in singular and plural forms (at the word level)
    # Example "turn off light" and "turn off lights"
    def normalize_plural_words(text: str) -> str:
        words = text.split()
        normalized_words = []
        for word in words:
            # Remove the trailing part 's' Or 'es' (But keep verbs like "turns")
            if len(word) > 3:  # Avoid processing too short words
                if word.endswith("es") and not word.endswith("ses"):
                    normalized_words.append(word[:-2])
                elif word.endswith("s") and not word.endswith("ss"):
                    normalized_words.append(word[:-1])
                else:
                    normalized_words.append(word)
            else:
                normalized_words.append(word)
        return " ".join(normalized_words)

    core1_normalized = normalize_plural_words(core1)
    core2_normalized = normalize_plural_words(core2)

    if core1_normalized == core2_normalized:
        return True

    # If core actions are different (e.g "turn on" vs "turn off"), Directly return False
    # Extract the core verb part of the action for comparison
    def get_action_verb(text: str) -> str:
        # Extract the first two words as action verbs (e.g "turn off", "pick up")
        words = text.split()
        if len(words) >= 2:
            return f"{words[0]} {words[1]}"
        elif len(words) == 1:
            return words[0]
        return text

    verb1 = get_action_verb(core1_normalized)
    verb2 = get_action_verb(core2_normalized)

    # If core verbs are different (e.g "turn on" vs "turn off"), Not similar
    if verb1 != verb2:
        # However, some common synonyms are allowed (e.g "pick up" vs "grab")
        # Only obvious differences in actions are checked here
        if verb1.split()[0] != verb2.split()[0]:  # The first word is different
            return False

    # Use match_activity_with_or_options for matching (if available)
    # But only when the core verbs are the same
    if verb1 == verb2:
        try:
            from .action_eval import match_activity_with_or_options

            if match_activity_with_or_options(activity1, activity2):
                return True
        except:
            pass

    return False


def validate_timeline_plan(narrator, current_test_case, extra_events: Optional[object] = None) -> Tuple[bool, bool]:
    try:
        from .action_eval import collect_timeline_events

        events = collect_timeline_events(narrator, extra_events)
        dependencies = (
            getattr(narrator, "timeline_dependencies", []) if hasattr(narrator, "timeline_dependencies") else []
        )

        # Schema check
        schema_ok = True
        seen_ids = set()
        valid_ids = set()
        for e in events:
            if not getattr(e, "involved_characters", None):
                schema_ok = False
                break
            if not getattr(e, "activity", ""):
                schema_ok = False
                break
            tl = getattr(e, "timeline_data", {}) or {}
            eid = tl.get("event_id", "")
            if not eid or eid in seen_ids:
                schema_ok = False
                break
            seen_ids.add(eid)
            valid_ids.add(eid)

        # Dependency check against expected
        dep_ok = True
        expected_deps = ((current_test_case or {}).get("expected_timeline", {}) or {}).get("dependencies", [])

        def normalize(dep):
            if not isinstance(dep, dict):
                return None
            if "parallel" in dep:
                grp = dep.get("parallel", [])
                if not isinstance(grp, list) or not grp:
                    return None
                return ("parallel", frozenset(grp))
            else:
                b = dep.get("before")
                a = dep.get("after")
                if not b or not a:
                    return None
                return ("order", a, b)

        expected_set = set()
        for d in expected_deps:
            n = normalize(d)
            if n is None:
                dep_ok = False
                break
            expected_set.add(n)

        produced_set = set()
        for d in dependencies:
            n = normalize(d)
            if n is None:
                dep_ok = False
                break
            # ID existence
            if n[0] == "parallel":
                if not all(gid in valid_ids for gid in n[1]):
                    dep_ok = False
                    break
            elif n[0] == "order":
                _, a, b = n
                if a not in valid_ids or b not in valid_ids:
                    dep_ok = False
                    break
            produced_set.add(n)

        if dep_ok and not expected_set.issubset(produced_set):
            dep_ok = False

        return schema_ok, dep_ok
    except Exception:
        return False, False


def validate_timeline_plan_with_diagnostics(
    narrator, current_test_case, extra_events: Optional[object] = None
) -> Tuple[bool, bool, Dict[str, Any]]:
    """Verify the timeline plan and return detailed diagnostic information

    Returns:
        (schema_ok, dep_ok, diagnostics): Tuple containing schema verification, dependency verification results, and detailed diagnostic information
    """
    diagnostics = {
        "dependency_issues": [],
        "missing_dependencies": [],
        "extra_dependencies": [],
        "invalid_event_ids": [],
    }

    try:
        from .action_eval import collect_timeline_events

        events = collect_timeline_events(narrator, extra_events)
        dependencies = (
            getattr(narrator, "timeline_dependencies", []) if hasattr(narrator, "timeline_dependencies") else []
        )

        # Schema check
        schema_ok = True
        seen_ids = set()
        valid_ids = set()
        for e in events:
            if not getattr(e, "involved_characters", None):
                schema_ok = False
                diagnostics["dependency_issues"].append("Event missing involved_characters")
                break
            if not getattr(e, "activity", ""):
                schema_ok = False
                diagnostics["dependency_issues"].append("Event missing activity")
                break
            tl = getattr(e, "timeline_data", {}) or {}
            eid = tl.get("event_id", "")
            if not eid:
                schema_ok = False
                diagnostics["dependency_issues"].append("Event missing event_id")
                break
            if eid in seen_ids:
                schema_ok = False
                diagnostics["dependency_issues"].append(f"Duplicate event_id: {eid}")
                break
            seen_ids.add(eid)
            valid_ids.add(eid)

        # Dependency check against expected - Use semantic matching
        dep_ok = True
        expected_events_list = ((current_test_case or {}).get("expected_timeline", {}) or {}).get("events", [])
        expected_deps = ((current_test_case or {}).get("expected_timeline", {}) or {}).get("dependencies", [])

        # Step 1: Pass through activity + Create generated_id for character → Mapping of expected_id
        from .action_eval import match_activity_with_or_options, is_harmless_cleanup_action
        import re

        event_id_mapping = {}  # generated_id -> expected_id
        matched_expected_ids = set()
        unmatched_generated_events = []  # Record unmatched generated events

        # Auxiliary function: checks whether a merge action can match multiple expected actions
        def check_merged_action_match(gen_activity: str, exp_activity: str, gen_char: str, exp_char: str) -> bool:
            """Checks the generated merge actions (e.g 'put down X and Y')whether they can match the expected actions (e.g 'put down X')

            Example:'put down the bottle and food' should be able to match 'put down bottle' and 'put down food'
            """
            if gen_char != exp_char:
                return False

            # checks if it is a merge action pattern (X and Y)
            merged_pattern = r"^(.+?)\s+(the\s+)?(\w+)\s+and\s+(the\s+)?(\w+)"
            merged_match = re.match(merged_pattern, gen_activity.lower())

            if merged_match:
                action_prefix = merged_match.group(1)  # e.g., "put down"
                item1 = merged_match.group(3)  # e.g., "bottle"
                item2 = merged_match.group(5)  # e.g., "food"

                # checks whether the expected action matches one of the items
                exp_pattern = rf"^{re.escape(action_prefix)}\s+(the\s+)?({item1}|{item2})"
                if re.match(exp_pattern, exp_activity.lower()):
                    return True

            return False

        for gen_event in events:
            gen_chars = [c.name for c in getattr(gen_event, "involved_characters", [])]
            gen_activity = getattr(gen_event, "activity", "")
            gen_id = (getattr(gen_event, "timeline_data", {}) or {}).get("event_id", "")

            if not gen_id or not gen_chars:
                continue

            # attempts to match to expected event (may match multiple times, if it is a merge action)
            matched_count = 0
            for exp_event in expected_events_list:
                exp_id = exp_event.get("event_id", "")
                exp_char = exp_event.get("character", "")
                exp_activity = exp_event.get("activity", "")

                if exp_id in matched_expected_ids:
                    continue

                # Match condition 1: characters are the same, and activity matches directly
                if exp_char in gen_chars and match_activity_with_or_options(gen_activity, exp_activity):
                    if matched_count == 0:  # establishes an ID mapping for the first match
                        event_id_mapping[gen_id] = exp_id
                    matched_expected_ids.add(exp_id)
                    matched_count += 1
                    continue

                # Match condition 2: merge action matches (e.g "put down X and Y" Match "put down X")
                # Note: pass gen_chars[0] as gen_char and exp_char as exp_char
                if check_merged_action_match(gen_activity, exp_activity, gen_chars[0] if gen_chars else "", exp_char):
                    if matched_count == 0:
                        event_id_mapping[gen_id] = exp_id
                    matched_expected_ids.add(exp_id)
                    matched_count += 1
                    continue

                # Match condition 3: semantic similarity matching (e.g "turn off the light near the dining area" Match "turn off light")
                # This can handle cases with location information
                if exp_char in gen_chars and _are_activities_similar(gen_activity, exp_activity):
                    if matched_count == 0:
                        event_id_mapping[gen_id] = exp_id
                    matched_expected_ids.add(exp_id)
                    matched_count += 1
                    continue

            # If no expected events are matched, record it
            if matched_count == 0:
                unmatched_generated_events.append(
                    {"event_id": gen_id, "character": gen_chars[0] if gen_chars else "?", "activity": gen_activity}
                )

        # Steps 1.5: Identify harmless cleanup actions from unmatched generated events
        # However, note: If this action is similar to the expected action, it should not be considered a harmless cleanup
        # Important: If an event has already been matched in event_id_mapping, it should not be identified as harmless cleanup
        harmless_cleanup_events = set()
        for evt in unmatched_generated_events:
            evt_id = evt["event_id"]
            evt_activity = evt["activity"]
            evt_char = evt["character"]

            # If an event has already been matched in event_id_mapping, it indicates that the expected event has been matched, and it should not be considered a harmless cleanup
            if evt_id in event_id_mapping:
                continue

            # Check whether this action is similar to any expected actions (even if not exactly the same)
            # If similar, it indicates a possible failure in matching, and it should not be considered a harmless cleanup
            is_similar_to_expected = False
            for exp_event in expected_events_list:
                exp_char = exp_event.get("character", "")
                exp_activity = exp_event.get("activity", "")

                # If the character is the same and the actions are similar (using fuzzy matching)
                if exp_char == evt_char:
                    # Check whether the actions are similar (simple keyword matching)
                    # Example "turn off the light" and "turn off lights" They should be considered similar
                    if _are_activities_similar(evt_activity, exp_activity):
                        is_similar_to_expected = True
                        break

            # Only when the action is in the harmless cleanup mode and not similar to the expected action, should it be considered a harmless cleanup
            if is_harmless_cleanup_action(evt_activity) and not is_similar_to_expected:
                harmless_cleanup_events.add(evt_id)
                diagnostics["harmless_cleanup_events"] = diagnostics.get("harmless_cleanup_events", [])
                diagnostics["harmless_cleanup_events"].append(evt)

        # Step 2: Convert generated dependencies using expected event IDs
        def normalize(dep, use_mapping=False):
            if not isinstance(dep, dict):
                return None
            if "parallel" in dep:
                grp = dep.get("parallel", [])
                if not isinstance(grp, list) or not grp:
                    return None
                if use_mapping:
                    # Convert event IDs
                    mapped_grp = [event_id_mapping.get(gid, gid) for gid in grp]
                    return ("parallel", frozenset(mapped_grp))
                return ("parallel", frozenset(grp))
            else:
                b = dep.get("before")
                a = dep.get("after")
                if not b or not a:
                    return None
                if use_mapping:
                    # Convert event IDs
                    mapped_a = event_id_mapping.get(a, a)
                    mapped_b = event_id_mapping.get(b, b)
                    return ("order", mapped_a, mapped_b)
                return ("order", a, b)

        expected_set = set()
        for d in expected_deps:
            n = normalize(d, use_mapping=False)
            if n is None:
                dep_ok = False
                diagnostics["dependency_issues"].append(f"Invalid expected dependency format: {d}")
                break
            expected_set.add(n)

        produced_set = set()
        for d in dependencies:
            n_original = normalize(d, use_mapping=False)
            n_mapped = normalize(d, use_mapping=True)

            if n_original is None:
                dep_ok = False
                diagnostics["dependency_issues"].append(f"Invalid produced dependency format: {d}")
                break

            # Check if dependencies involve harmless cleanup actions; if so, skip them
            skip_dep = False
            if n_original[0] == "parallel":
                # If any event in the parallel group is a cleanup action, skip the entire parallel group
                if any(gid in harmless_cleanup_events for gid in n_original[1]):
                    skip_dep = True
            elif n_original[0] == "order":
                _, a, b = n_original
                # If either end of the dependency is a cleanup action, skip it
                if a in harmless_cleanup_events or b in harmless_cleanup_events:
                    skip_dep = True

            if skip_dep:
                continue  # Do not count in dependency verification

            # Check if the ID exists (use the original ID)
            if n_original[0] == "parallel":
                invalid_ids = [gid for gid in n_original[1] if gid not in valid_ids]
                if invalid_ids:
                    dep_ok = False
                    diagnostics["invalid_event_ids"].extend(invalid_ids)
                    diagnostics["dependency_issues"].append(
                        f"Parallel dependency references invalid event IDs: {invalid_ids}"
                    )
                    break
            elif n_original[0] == "order":
                _, a, b = n_original
                if a not in valid_ids:
                    dep_ok = False
                    diagnostics["invalid_event_ids"].append(a)
                    diagnostics["dependency_issues"].append(
                        f"Order dependency 'after' references invalid event ID: {a}"
                    )
                    break
                if b not in valid_ids:
                    dep_ok = False
                    diagnostics["invalid_event_ids"].append(b)
                    diagnostics["dependency_issues"].append(
                        f"Order dependency 'before' references invalid event ID: {b}"
                    )
                    break

            # Compare using the mapped ID
            produced_set.add(n_mapped)

        # Counting the number of matched events and dependencies
        matched_events_count = len(matched_expected_ids)
        expected_events_count = len(expected_events_list)
        matched_dependencies_count = len(produced_set & expected_set)
        expected_dependencies_count = len(expected_set)

        # Add statistics to diagnostics
        diagnostics["matched_events"] = matched_events_count
        diagnostics["expected_events"] = expected_events_count
        diagnostics["matched_dependencies"] = matched_dependencies_count
        diagnostics["expected_dependencies"] = expected_dependencies_count
        diagnostics["semantic_matching_used"] = True
        diagnostics["event_id_mapping"] = event_id_mapping

        if dep_ok:
            # Check for missing expected dependencies
            missing = expected_set - produced_set
            if missing:
                dep_ok = False
                for m in missing:
                    if m[0] == "parallel":
                        diagnostics["missing_dependencies"].append(f"parallel: {list(m[1])}")
                    elif m[0] == "order":
                        diagnostics["missing_dependencies"].append(f"{m[1]} → {m[2]}")

            # Check for extra dependencies (informational)
            extra = produced_set - expected_set
            for e in extra:
                if e[0] == "parallel":
                    diagnostics["extra_dependencies"].append(f"parallel: {list(e[1])}")
                elif e[0] == "order":
                    diagnostics["extra_dependencies"].append(f"{e[1]} → {e[2]}")

        # Add event ID mapping information to diagnostics
        if event_id_mapping:
            diagnostics["event_id_mapping"] = event_id_mapping
            diagnostics["semantic_matching_used"] = True

        return schema_ok, dep_ok, diagnostics
    except Exception as ex:
        diagnostics["dependency_issues"].append(f"Exception during validation: {str(ex)}")
        return False, False, diagnostics
