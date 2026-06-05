"""bench/benchmark.py — MRL benchmark engine.

Measures, across a sweep of truncation dimensions, how much retrieval
quality degrades and how much memory is saved when MRL embeddings are
truncated.

Methodology
-----------
1. Embed all 35 corpus docs + 9 queries once at full 768-d (``embed_all``).
2. Confirm the full-dim reference is sane (``sanity_check``).
3. For each dim in [64, 128, 256, 512, 768]:
   a. Truncate all doc and query vectors to *dim*-d and re-normalise.
   b. Compute recall@K vs the full-dim top-K as ground truth.
   c. Measure average search latency and corpus memory footprint.
4. Print a formatted results table.

Run:  python -m bench.benchmark
"""

import time

import numpy as np

from bench.corpus import CORPUS, QUERIES, SANITY
from bench.embed import embed, truncate

DIMS = [64, 128, 256, 512, 768]
K = 5
LATENCY_REPEATS = 50


def rank(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """Return doc indices sorted best-first by cosine similarity.

    Because all vectors are unit-length, cosine similarity equals the dot
    product.  The matrix multiply scores all docs in one operation.

    Args:
        query_vec: a unit-length 1-D float32 array (the query embedding).
        doc_matrix: a 2-D float32 array of shape ``(n_docs, dim)`` where
            every row is a unit-length doc embedding.

    Returns:
        A 1-D int64 array of doc indices sorted by descending score
        (best match at index 0).
    """
    scores = doc_matrix @ query_vec
    return np.argsort(-scores)


def embed_all():
    """Embed every corpus doc and query once at full 768-d.

    Embedding all items upfront and deriving truncated versions from the
    full-dim vectors is cheaper than re-embedding at each dim and ensures
    the truncation is the only variable between dim sweeps.

    Returns:
        A tuple ``(doc_ids, doc_vecs, query_vecs)`` where:
          - ``doc_ids``    : list of string ids in CORPUS order
          - ``doc_vecs``   : float32 ndarray of shape ``(35, 768)``
          - ``query_vecs`` : list of 9 float32 ndarrays, each shape ``(768,)``
    """
    doc_ids = [d[0] for d in CORPUS]
    doc_vecs = np.vstack([embed(d[1], kind="document") for d in CORPUS])
    query_vecs = [embed(q[0], kind="query") for q in QUERIES]
    return doc_ids, doc_vecs, query_vecs


def gold_topk(doc_vecs: np.ndarray, query_vecs: list, k: int) -> list[set]:
    """Compute the full-dim top-k doc index sets used as ground truth.

    At dim=768 (full dimension) these rankings define "correct."  Recall@k
    at any truncated dim measures how many of these top-k indices survive.

    Args:
        doc_vecs: full-dim doc matrix, shape ``(n_docs, 768)``.
        query_vecs: list of full-dim query vectors.
        k: number of top results to keep per query.

    Returns:
        A list of sets, one per query.  Each set contains the integer doc
        indices of the top-k results at full dimension.
    """
    return [set(rank(q, doc_vecs)[:k].tolist()) for q in query_vecs]


def recall_at_k(
    truncated_doc_vecs: np.ndarray,
    truncated_query_vecs: list,
    gold: list[set],
    k: int,
) -> float:
    """Compute mean recall@k of truncated rankings vs the full-dim gold sets.

    For each query, recall@k = |truncated top-k ∩ gold top-k| / k.
    The mean is taken across all 9 queries.

    At dim=768 (no truncation), recall is 1.000 by construction — this
    doubles as a correctness check.

    Args:
        truncated_doc_vecs: doc matrix truncated to the current dim.
        truncated_query_vecs: list of query vectors truncated to the same dim.
        gold: full-dim top-k sets produced by ``gold_topk``.
        k: number of top results to compare.

    Returns:
        Mean recall@k as a float in [0.0, 1.0].
    """
    total = 0.0
    for q, gold_set in zip(truncated_query_vecs, gold):
        got = set(rank(q, truncated_doc_vecs)[:k].tolist())
        total += len(got & gold_set) / k
    return total / len(gold)


def search_latency_ms(
    truncated_doc_vecs: np.ndarray,
    truncated_query_vecs: list,
) -> float:
    """Measure the average time (ms) to score one query against the corpus.

    Runs ``LATENCY_REPEATS × n_queries`` search calls using
    ``time.perf_counter`` and returns the per-call average in milliseconds.

    At 35 docs, latency is dominated by Python overhead and is sub-ms at
    every dim.  The number becomes meaningful at corpus sizes of millions.

    Args:
        truncated_doc_vecs: current-dim doc matrix.
        truncated_query_vecs: current-dim query vectors.

    Returns:
        Average search latency in milliseconds as a float.
    """
    start = time.perf_counter()
    for _ in range(LATENCY_REPEATS):
        for q in truncated_query_vecs:
            _ = rank(q, truncated_doc_vecs)
    elapsed = time.perf_counter() - start
    n = LATENCY_REPEATS * len(truncated_query_vecs)
    return (elapsed / n) * 1000.0


def sanity_check(
    doc_ids: list[str],
    doc_vecs: np.ndarray,
    query_vecs: list,
) -> bool:
    """Verify the full-dim reference ranking against hand-labelled expectations.

    Before trusting the recall curve, confirm the full-dim model returns
    obviously-relevant docs for the 4 SANITY queries.  A failure here means
    the reference is flawed (e.g. wrong task prefix, wrong model) and the
    recall numbers cannot be trusted.

    Prints one ``[OK ]`` or ``[XX ]`` line per sanity query.

    Args:
        doc_ids: list of string doc ids in corpus order.
        doc_vecs: full-dim doc matrix, shape ``(35, 768)``.
        query_vecs: full-dim query vectors, parallel to QUERIES.

    Returns:
        ``True`` if every sanity query returns an expected doc at rank 1,
        ``False`` otherwise.
    """
    print("Sanity check (full-dim top-1 vs known-relevant):")
    q_by_text = {q[0]: qv for q, qv in zip(QUERIES, query_vecs)}
    all_ok = True
    for qtext, relevant_ids in SANITY.items():
        top_idx = int(rank(q_by_text[qtext], doc_vecs)[0])
        top_id = doc_ids[top_idx]
        ok = top_id in relevant_ids
        all_ok = all_ok and ok
        mark = "OK " if ok else "XX "
        print(f"  [{mark}] '{qtext[:45]}...' -> {top_id} (want {sorted(relevant_ids)})")
    print(f"  => reference {'looks sane' if all_ok else 'FAILED — investigate'}\n")
    return all_ok


def run() -> list[dict]:
    """Run the full MRL benchmark and return results as a list of dicts.

    Embeds all corpus docs and queries once at full dim, verifies the
    reference ranking with ``sanity_check``, then sweeps over DIMS measuring
    recall@K, search latency, and memory for each truncation level.

    Prints a formatted table of results to stdout.

    Returns:
        A list of dicts, one per dim in DIMS::

            [{"dim": 64, "recall": 0.733, "latency_ms": 0.003, "mem_kb": 8.75}, ...]
    """
    print("Embedding 35 docs + 9 queries at full dim (once)...\n")
    doc_ids, doc_vecs, query_vecs = embed_all()

    sanity_check(doc_ids, doc_vecs, query_vecs)

    gold = gold_topk(doc_vecs, query_vecs, K)

    results = []
    for dim in DIMS:
        td = np.vstack([truncate(v, dim) for v in doc_vecs])
        tq = [truncate(v, dim) for v in query_vecs]
        recall = recall_at_k(td, tq, gold, K)
        latency = search_latency_ms(td, tq)
        mem_kb = td.nbytes / 1024.0
        results.append({"dim": dim, "recall": recall, "latency_ms": latency, "mem_kb": mem_kb})

    print(f"Results (k={K}, corpus={len(CORPUS)} docs):")
    print(f"{'dim':>5} | {'recall@'+str(K):>9} | {'search ms':>10} | {'memory KB':>10}")
    print("-" * 46)
    for r in results:
        print(f"{r['dim']:>5} | {r['recall']:>9.3f} | {r['latency_ms']:>10.4f} | {r['mem_kb']:>10.2f}")

    return results


if __name__ == "__main__":
    run()
