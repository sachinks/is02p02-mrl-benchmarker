"""bench/embed.py — embedding and MRL truncation utilities.

Provides two public functions used throughout the benchmark:

  embed(text, kind)    -> 768-d unit-length float32 ndarray
  truncate(vec, dim)   -> dim-d unit-length float32 ndarray

Both functions enforce L2 normalisation so downstream code can use plain
dot products for cosine similarity.  The self-test (``python -m bench.embed``)
runs 25 hard assertions across 5 texts and 5 dims to confirm correctness
before any benchmark runs.
"""

import numpy as np
import requests

from config import settings

# nomic-embed-text requires task prefixes to apply the correct
# internal representation path (document vs query encoder heads).
_PREFIX = {
    "document": "search_document: ",
    "query": "search_query: ",
}


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Scale *vec* to unit length (L2 norm == 1.0).

    This is the operation that makes cosine similarity collapse to a plain
    dot product: ``cos(a, b) = (a·b) / (|a||b|) = a·b`` when |a|==|b|==1.

    Handles the zero-vector edge case by returning the vector unchanged
    (a zero vector has no meaningful direction to normalise to).

    Args:
        vec: a 1-D numpy array of any dtype.

    Returns:
        A new float32 array with L2 norm == 1.0, or *vec* unchanged if
        ``np.linalg.norm(vec) == 0``.
    """
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm


def embed(text: str, kind: str = "document") -> np.ndarray:
    """Embed *text* via the local Ollama server and return a unit-length vector.

    Prepends the appropriate task prefix (``search_document:`` or
    ``search_query:``) before sending the text to Ollama.  Skipping the
    prefix is a common silent bug — the model degrades retrieval quality
    without error.

    The raw Ollama response is a float32 array of size 768 (for
    nomic-embed-text).  This function L2-normalises it before returning,
    so all subsequent comparisons use dot products.

    Args:
        text: the raw string to embed.  The task prefix is prepended
            automatically; do not add it yourself.
        kind: ``"document"`` for corpus items, ``"query"`` for search
            queries.  These map to ``search_document:`` and
            ``search_query:`` prefixes respectively.

    Returns:
        A float32 numpy array of shape ``(768,)`` with L2 norm == 1.0.

    Raises:
        ValueError: if *kind* is not ``"document"`` or ``"query"``.
        requests.HTTPError: if the Ollama server returns a non-2xx status.
        requests.ConnectionError: if the Ollama server is not reachable.
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
    """Truncate *vec* to *dim* dimensions and re-normalise — the MRL move.

    Slicing a unit vector to its first *dim* components shortens it: the
    slice no longer lies on the unit sphere.  Re-normalising after slicing
    is **mandatory** before any cosine comparison; without it, truncated
    vectors of different dims are on different-radius spheres and their dot
    products are silently wrong.

    This is the single most common MRL implementation bug.  The self-test
    asserts every truncated vector has norm == 1.0 to catch regressions.

    Args:
        vec: a unit-length 1-D float32 numpy array.  Expected to have been
            produced by ``embed()`` or already normalised.
        dim: number of leading dimensions to keep.  Must be <= ``vec.shape[0]``.

    Returns:
        A float32 numpy array of shape ``(dim,)`` with L2 norm == 1.0.

    Raises:
        ValueError: if *dim* exceeds the vector's size.
    """
    if dim > vec.shape[0]:
        raise ValueError(f"dim {dim} exceeds vector size {vec.shape[0]}")
    return _l2_normalize(vec[:dim])


if __name__ == "__main__":
    DIMS = [64, 128, 256, 512, 768]
    TEST_TEXTS = [
        ("The quick brown fox jumps over the lazy dog", "document"),
        ("Matryoshka embeddings store meaning in nested prefixes", "document"),
        ("How do I reduce memory usage in a vector database?", "query"),
        ("A REST API exposes resources over HTTP using GET and POST", "document"),
        ("what is the best truncation dimension for retrieval?", "query"),
    ]
    NORM_TOL = 1e-5

    print(f"Self-test: {len(TEST_TEXTS)} texts x {len(DIMS)} dims = "
          f"{len(TEST_TEXTS) * len(DIMS)} norm checks\n")

    for text, kind in TEST_TEXTS:
        v = embed(text, kind=kind)
        full_norm = np.linalg.norm(v)
        assert abs(full_norm - 1.0) < NORM_TOL, (
            f"FAIL full-dim norm={full_norm:.6f} for: {text!r}"
        )
        for d in DIMS:
            t = truncate(v, d)
            t_norm = np.linalg.norm(t)
            assert abs(t_norm - 1.0) < NORM_TOL, (
                f"FAIL dim={d} norm={t_norm:.6f} for: {text!r}"
            )
        print(f"  OK  [{kind:8s}] {text[:55]!r}  full={full_norm:.6f}  "
              + "  ".join(f"dim{d}={np.linalg.norm(truncate(v, d)):.6f}" for d in DIMS))

    print("\nAll norm checks passed. Truncation is sound.")
