"""
json_utils.py
Robust JSON extraction from noisy or truncated LLM responses.

Handles:
- Clean JSON                  → direct parse
- Markdown fences             → strip then parse
- Truncated JSON              → heal by closing open brackets/braces
- Preamble/postamble text     → regex extraction of outermost { }
"""

import json
import re
from typing import Any


def _heal_truncated_json(text: str) -> str:
    """
    Attempt to close a JSON string that was cut off mid-stream.
    Works by counting unclosed braces and brackets, then appending
    the necessary closing characters.
    """
    # Find the start of the JSON object
    start = text.find("{")
    if start == -1:
        return text
    text = text[start:]

    # Walk the string tracking open structures
    stack = []
    in_string = False
    escape_next = False

    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in ("]", "}"):
            if stack and stack[-1] == ch:
                stack.pop()

    # Close any still-open string
    if in_string:
        text += '"'

    # Append missing closers in reverse order
    text += "".join(reversed(stack))
    return text


def extract_json(text: str) -> dict[str, Any]:
    """
    Extract a JSON object from a raw LLM response string.

    Strategy (in order):
    1. Direct parse.
    2. Strip markdown fences, retry.
    3. Extract outermost { } block, retry.
    4. Heal truncated JSON, retry.

    Raises:
        ValueError: if all strategies fail.
    """

    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown fences
    stripped = re.sub(
        r"```(?:json)?\s*(.*?)\s*```",
        r"\1",
        text,
        flags=re.DOTALL,
    ).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 3. Grab outermost { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            candidate = match.group()
        else:
            candidate = text
    else:
        candidate = text

    # 4. Heal truncated JSON
    healed = _heal_truncated_json(candidate)
    try:
        return json.loads(healed)
    except json.JSONDecodeError:
        pass

    raise ValueError(
        f"No valid JSON found in LLM response. "
        f"First 200 chars: {text[:200]!r}"
    )