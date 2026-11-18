"""Event ID, cache lookup, and batch-result mapping policies.

These helpers preserve payload identity and input iteration order. The planner
owns the cache dictionary and decides when to clear or replace it.
"""


def event_ids(item, narrator) -> list[str]:
    """Extract all event_ids from events or ParallelGroup"""
    ids = []
    if hasattr(item, "events") and hasattr(item, "parallel_group"):
        # ParallelGroup
        for ev in item.events:
            eid = narrator._get_event_id(ev) if hasattr(narrator, "_get_event_id") else str(id(ev))
            ids.append(eid)
    else:
        # Normal Event
        eid = narrator._get_event_id(item) if hasattr(narrator, "_get_event_id") else str(id(item))
        ids.append(eid)
    return ids


def unique_events(events_for_batch, narrator) -> list:
    """Expand parallel groups and keep the first event for each ID, in input order."""
    seen = set()
    deduped = []
    for ev in events_for_batch:
        # Process ParallelGroup
        if hasattr(ev, "events") and hasattr(ev, "parallel_group"):
            for sub_ev in ev.events:
                eid = narrator._get_event_id(sub_ev) if hasattr(narrator, "_get_event_id") else str(id(sub_ev))
                if eid not in seen:
                    seen.add(eid)
                    deduped.append(sub_ev)
        else:
            eid = narrator._get_event_id(ev) if hasattr(narrator, "_get_event_id") else str(id(ev))
            if eid not in seen:
                seen.add(eid)
                deduped.append(ev)

    return deduped


def get_cached_event(event, narrator, cache, logger) -> dict:
    """Preserve group merge order and the single-entry fallback, without copying plans."""
    parsed_event = {}

    # Debugging: Record current event information
    if hasattr(event, "events") and hasattr(event, "parallel_group"):
        logger.info(f"🔄 Retrieving ParallelGroup with {len(event.events)} events from cache")
        event_ids = [
            narrator._get_event_id(sub_ev) if hasattr(narrator, "_get_event_id") else "" for sub_ev in event.events
        ]
        logger.info(f"   ParallelGroup event_ids: {event_ids}")
    else:
        event_id = narrator._get_event_id(event) if hasattr(narrator, "_get_event_id") else ""
        logger.info(f"🔄 Retrieving single event {event_id} from cache")

    # Process ParallelGroup: Combine the parsing results of all member events
    if hasattr(event, "events") and hasattr(event, "parallel_group"):
        for sub_ev in event.events:
            eid = narrator._get_event_id(sub_ev) if hasattr(narrator, "_get_event_id") else ""
            if eid and eid in cache:
                # Combine data from all characters involved in the event
                cached_data = cache[eid]
                logger.info(
                    f"   ✅ Found {eid} in cache with {len(cached_data) if isinstance(cached_data, dict) else 0} character(s)"
                )
                parsed_event.update(cached_data)
            else:
                logger.warning(f"   ❌ ParallelGroup member {eid} not found in cache")
                logger.warning(f"      Available in cache: {list(cache.keys())}")
    else:
        # Normal Event
        event_id = ""
        if hasattr(narrator, "_get_event_id"):
            try:
                event_id = narrator._get_event_id(event)
            except Exception:
                event_id = ""
        if event_id:
            parsed_event = cache.get(event_id, {})

        # If no match is found and there is only one entry in the batch cache, use that entry as a fallback (compatible with scenarios without event_id)
        if not parsed_event and len(cache) == 1:
            parsed_event = next(iter(cache.values()))

    return parsed_event


def remap_batch_results(events, parsed_results, narrator, logger) -> dict:
    """Match event IDs first; only unmatched events use the legacy character fallback."""
    remapped = {}
    for ev in events:
        eid = narrator._get_event_id(ev) if hasattr(narrator, "_get_event_id") else ""
        if not eid:
            continue

        # Obtain the character names for the event
        if hasattr(ev, "involved_characters"):
            char_names = [c.name if hasattr(c, "name") else str(c) for c in ev.involved_characters]
        else:
            char_names = []

        logger.info(f"🎯 Mapping event {eid} with characters {char_names}")

        # Strategy 1: Prioritize precise matching using event_id
        found, matched_chars = _match_event_id(parsed_results, char_names, eid, logger)

        if found:
            remapped[eid] = matched_chars
            logger.info(f"✅ Event {eid} mapped with {len(matched_chars)} character(s) via event_id")

        # Strategy 2 (fallback): If no match is found using event_id, use character names for matching
        if not found:
            found, matched_chars_fallback = _match_character_names(parsed_results, char_names, eid, logger)

            if found:
                remapped[eid] = matched_chars_fallback
                logger.info(
                    f"✅ Event {eid} mapped with {len(matched_chars_fallback)} character(s) via character name fallback"
                )

        if not found:
            logger.warning(f"Event {eid} with characters {char_names} not found in LLM output")
            remapped[eid] = {}

    # Debugging: Print the final remapped results
    logger.info(f"📊 Final remapped results: {len(remapped)} events mapped")
    for eid, char_data in remapped.items():
        logger.info(f"   Event {eid}: {list(char_data.keys()) if char_data else 'EMPTY'}")

    return remapped


def _match_event_id(parsed_results, char_names, eid, logger):
    found = False
    matched_chars = {}  # Collect all matching character data
    for llm_idx_key in list(parsed_results.keys()):
        llm_data = parsed_results[llm_idx_key]
        if not isinstance(llm_data, dict):
            continue

        for char_name in char_names:
            if char_name in llm_data:
                char_data = llm_data[char_name]
                # Check if there is an event_id match
                if isinstance(char_data, dict) and char_data.get("event_id") == eid:
                    matched_chars[char_name] = char_data
                    found = True
                    logger.debug(f"Mapped event {eid} character {char_name} via event_id match")

    return found, matched_chars


def _match_character_names(parsed_results, char_names, eid, logger):
    found = False
    matched_chars_fallback = {}
    for llm_idx_key in list(parsed_results.keys()):
        llm_data = parsed_results[llm_idx_key]
        if not isinstance(llm_data, dict):
            continue

        for char_name in char_names:
            if char_name in llm_data:
                # Check if the character data is already occupied by another event (via event_id)
                char_data = llm_data[char_name]
                if isinstance(char_data, dict) and "event_id" in char_data:
                    # If there is an existing event_id, skip this step (wait for precise matching)
                    continue

                # If a match is found, collect the data for that character
                matched_chars_fallback[char_name] = llm_data[char_name]
                found = True
                logger.debug(f"Mapped event {eid} character {char_name} via character name fallback")

    return found, matched_chars_fallback
