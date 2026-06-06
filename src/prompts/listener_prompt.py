LISTENER_PROMPT = """\
You are a literary analyst. Identify who is the intended recipient of the dialogue below.

Known characters in this novel:
{character_list}

Speaker: {speaker}
Dialogue: "{quote}"

Surrounding context:
{context}

Rules:
1. Return valid JSON only. No explanation, no markdown.
2. Choose only from the known characters list above.
3. If the speaker is addressing a group or nobody specific, set listener to "UNKNOWN".
4. confidence is a float between 0.0 and 1.0.

JSON Schema:
{{
  "listener": "<character name or UNKNOWN>",
  "confidence": 0.0
}}
"""