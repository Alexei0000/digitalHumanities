DIALOGUE_PROMPT = """\
You are a literary analyst. Extract every spoken utterance from the text below.

Dialogue appears in many forms — extract ALL of them:
- Standard double quotes:  "Hello," she said.
- Single quotes:           'Get out,' he growled.
- Em-dash dialogue:        —I won't go, she insisted.
- Said/asked/replied tags: He told her he would return. (indirect)
- Thought as speech:       She wondered, why is he here?
- Any line where a character clearly speaks, shouts, whispers, asks, replies, cries, mutters, or thinks aloud.

Rules:
1. Return valid JSON only. No explanation, no markdown, no preamble.
2. Include both direct speech (quoted text) and indirect speech (reported speech).
3. For direct quotes, preserve the exact wording.
4. For indirect speech, paraphrase the content naturally.
5. Do NOT include pure narration that has no speech act.
6. If genuinely no dialogue is present, return {{"dialogues": []}}.

quote_type must be exactly: "direct" or "indirect"

JSON Schema:
{{
  "dialogues": [
    {{
      "quote": "<exact words or natural paraphrase>",
      "quote_type": "direct"
    }}
  ]
}}

TEXT:
{text}
"""