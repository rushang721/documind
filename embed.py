import os

import numpy as np
from google import genai
from google.genai import types

EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
BATCH_SIZE = 80


def embed_texts(texts, api_key, task="RETRIEVAL_DOCUMENT"):
    """Embed texts and L2-normalise, so a dot product equals cosine similarity."""
    client = genai.Client(api_key=api_key)
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        res = client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task),
        )
        vectors.extend(e.values for e in res.embeddings)
    m = np.array(vectors, dtype="float32")
    return m / np.linalg.norm(m, axis=1, keepdims=True)
