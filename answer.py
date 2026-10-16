import os

from google import genai
from google.genai import types

GEN_MODEL = os.getenv("GEN_MODEL", "gemini-2.5-flash")

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
    res = client.models.generate_content(
        model=GEN_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM, temperature=0.2),
    )
    return res.text
