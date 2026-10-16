import os
import time

from google import genai
from google.genai import types

GEN_MODEL = os.getenv("GEN_MODEL", "gemini-3.8-flash")
MAX_TRIES = 5

SYSTEM = (
    "You answer questions about the user's documents. "
    "Use only the excerpts provided. If they do not contain the answer, say so plainly "
    "instead of guessing. Cite sources as (file, p.N) after the claim they support."
)


def ask(question, hits, history, api_key):
    excerpts = "\n\n".join(f"[{c.doc}, p.{c.page}]\n{c.text}" for c, _ in hits)
    earlier = "\n".join(f"{role}: {text}" for role, text in history[-6:])
    prompt = (
        f"Conversation so far:\n{earlier or '(none)'}\n\n"
        f"Excerpts:\n{excerpts}\n\n"
        f"Question: {question}"
    )
    client = genai.Client(api_key=api_key)
    for attempt in range(MAX_TRIES):
        try:
            res = client.models.generate_content(
                model=GEN_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM, temperature=0.2
                ),
            )
            return res.text
        except Exception as err:
            busy = "503" in str(err) or "UNAVAILABLE" in str(err)
            if not busy or attempt == MAX_TRIES - 1:
                raise
            time.sleep(4 * (attempt + 1))
