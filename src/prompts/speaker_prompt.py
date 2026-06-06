SPEAKER_PROMPT = """\
You are a literary analyst. Identify who speaks the dialogue below.

Known characters in this novel:
{character_list}

Dialogue to attribute:
"{quote}"

Surrounding context:
{context}

Rules:
1. Return valid JSON only. No explanation, no markdown.
2. Choose only from the known characters list above.
3. If genuinely ambiguous, set speaker to "UNKNOWN" and confidence to 0.0.
4. confidence is a float between 0.0 and 1.0.

JSON Schema:
{{
  "speaker": "<character name or UNKNOWN>",
  "confidence": 0.0
}}
"""