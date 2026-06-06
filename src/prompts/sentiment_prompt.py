SENTIMENT_PROMPT = """\
You are a literary analyst specialising in sentiment and subtext.

Evaluate the emotional tone of the following dialogue from the speaker's perspective toward the listener.
Consider: irony, sarcasm, subtext, narrative context, power dynamics.

Speaker:  {speaker}
Listener: {listener}
Dialogue: "{quote}"

Context:
{context}

Score the sentiment on this scale:
  -5  very hostile / hateful
  -4  hostile
  -3  strongly negative
  -2  negative / cold
  -1  mildly negative
   0  neutral
  +1  mildly positive / polite
  +2  warm / friendly
  +3  strongly positive
  +4  affectionate / loving
  +5  deeply affectionate / euphoric

Rules:
1. Return valid JSON only. No explanation, no markdown.
2. score must be an integer between -5 and 5.
3. emotion is a short label, e.g. "hostility", "affection", "sarcasm", "anxiety", "admiration".

JSON Schema:
{{
  "score": 0,
  "emotion": "<short label>"
}}
"""