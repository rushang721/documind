import numpy as np

from embed import embed_texts


def top_chunks(query, chunks, matrix, api_key, k=5):
    q = embed_texts([query], api_key, task="RETRIEVAL_QUERY")[0]
    scores = matrix @ q
    best = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in best]
