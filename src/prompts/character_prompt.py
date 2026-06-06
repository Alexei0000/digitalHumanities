CHARACTER_PROMPT = """\
You are a literary analyst. Extract every named character that appears in the text below.

Rules:
1. Return valid JSON only. No explanation, no markdown, no preamble.
2. Include common nicknames and title variants as aliases.
3. Do NOT include locations, organizations, or mythical creatures.
4. If no characters are found, return {{"characters": []}}.

JSON Schema:
{{
  "characters": [
    {{
      "name": "<full canonical name>",
      "aliases": ["<nickname>", "<title variant>"]
    }}
  ]
}}

TEXT:
{text}
"""