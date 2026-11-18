"""UTF-8 conversation logs with the existing text, vision, and batch layouts."""

import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, TextIO


def write_llm_log(
    path: Path,
    messages: Sequence[Mapping[str, Any]],
    response: str,
    *,
    layout: Literal["text", "vision", "batch"] = "text",
) -> None:
    """Keep native newlines and overwrite semantics; only the encoding is fixed.

    Callers retain their timestamp, prefix, and directory naming conventions.
    I/O failures propagate, as they did before the shared writer was introduced.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        if layout == "batch":
            _write_batch_messages(stream, messages)
        else:
            _write_example_messages(stream, messages, vision=layout == "vision")
        stream.write(f"{'='*30} Response {'='*30}\n")
        try:
            response_json = json.loads(response)
            stream.write(json.dumps(response_json, indent=2, ensure_ascii=False))
        except (json.JSONDecodeError, TypeError):
            stream.write(response)


def _write_example_messages(stream: TextIO, messages: Sequence[Mapping[str, Any]], *, vision: bool) -> None:
    for i, message in enumerate(messages):
        if i == 0:
            title = "System Message"
        elif i == len(messages) - 1:
            title = "Query Messages"
        elif i % 2 == 1:
            title = f"Example {(i-1)//2} Query"
        else:
            title = f"Example {(i-1)//2} Response"
        stream.write(f"{'='*30} {title} {'='*30}\n")
        content = message["content"]
        if vision and i > 0 and (i == len(messages) - 1 or i % 2 == 1):
            content = content[0]["text"]
        stream.write(content)
        stream.write("\n\n")


def _write_batch_messages(stream: TextIO, messages: Sequence[Mapping[str, Any]]) -> None:
    for i, message in enumerate(messages):
        title = "System Message" if i == 0 else "User Message"
        stream.write(f"{'='*30} {title} {'='*30}\n")
        content = message["content"]
        if isinstance(content, str):
            stream.write(content)
        elif isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    stream.write(item["text"])
                elif item.get("type") == "image_url":
                    stream.write(f"\n[Image: {item['image_url']['url'][:100]}...]\n")
        stream.write("\n\n")
