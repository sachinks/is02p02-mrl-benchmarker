"""Embedding + MRL truncation utilities.

Talks to a local Ollama server, returns L2-normalized numpy vectors,
and truncates them Matryoshka-style (slice first k dims, re-normalize).
"""

import numpy as np
import requests

from config import settings

# nomic-embed-text expects task prefixes on the input text.
_PREFIX = {
    "document": "search_document: ",
    "query": "search_query: ",
}


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Scale a vector to unit length so cosine similarity == dot product."""
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm


def embed(text: str, kind: str = "document") -> np.ndarray:
    """Embed `text` via Ollama and return a unit-length float32 vector.

    kind: "document" for corpus items, "query" for search queries.
    """
    if kind not in _PREFIX:
        raise ValueError(f"kind must be 'document' or 'query', got {kind!r}")

    resp = requests.post(
        f"{settings.ollama_url}/api/embeddings",
        json={"model": settings.embed_model, "prompt": _PREFIX[kind] + text},
        timeout=60,
    )
    resp.raise_for_status()
    vec = np.asarray(resp.json()["embedding"], dtype=np.float32)
    return _l2_normalize(vec)


def truncate(vec: np.ndarray, dim: int) -> np.ndarray:
    """MRL truncation: keep the first `dim` dims, then re-normalize.

    Re-normalizing is essential — slicing shortens the vector, so we put
    it back on the unit sphere before any cosine comparison.
    """
    if dim > vec.shape[0]:
        raise ValueError(f"dim {dim} exceeds vector size {vec.shape[0]}")
    return _l2_normalize(vec[:dim])


if __name__ == "__main__":
    # Self-test: every truncated vector must have L2 norm ~= 1.0
    v = embed("The quick brown fox jumps over the lazy dog", kind="document")
    print(f"full embedding dims: {v.shape[0]}")
    for d in (64, 128, 256, 512, 768):
        t = truncate(v, d)
        print(f"  dim={d:>3}  ->  shape={t.shape[0]:>3}  L2norm={np.linalg.norm(t):.6f}")
