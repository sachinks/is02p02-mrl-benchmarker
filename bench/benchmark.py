"""MRL benchmark engine.

Embed the corpus + queries ONCE at full dim, then sweep truncation
dimensions measuring three things per dim:

  - recall@k   : quality, vs the full-dim ranking as ground truth
  - search ms  : avg time to score one query against the whole corpus
  - memory KB  : bytes to store the truncated corpus matrix

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
    """Return doc indices sorted best-first.

    Vectors are unit-length, so cosine similarity == dot product.
    """
    scores = doc_matrix @ query_vec
    return np.argsort(-scores)


def embed_all():
    """Embed every doc and query once, at full 768 dim."""
    doc_ids = [d[0] for d in CORPUS]
    doc_vecs = np.vstack([embed(d[1], kind="document") for d in CORPUS])
    query_vecs = [embed(q[0], kind="query") for q in QUERIES]
    return doc_ids, doc_vecs, query_vecs


def gold_topk(doc_vecs, query_vecs, k):
    """Full-dim top-k doc index sets — the reference ground truth."""
    return [set(rank(q, doc_vecs)[:k].tolist()) for q in query_vecs]


def recall_at_k(truncated_doc_vecs, truncated_query_vecs, gold, k):
    """Mean recall@k of truncated rankings vs the gold (full-dim) sets."""
    total = 0.0
    for q, gold_set in zip(truncated_query_vecs, gold):
        got = set(rank(q, truncated_doc_vecs)[:k].tolist())
        total += len(got & gold_set) / k
    return total / len(gold)


def search_latency_ms(truncated_doc_vecs, truncated_query_vecs):
    """Avg ms to score one query against the whole corpus at this dim."""
    start = time.perf_counter()
    for _ in range(LATENCY_REPEATS):
        for q in truncated_query_vecs:
            _ = rank(q, truncated_doc_vecs)
    elapsed = time.perf_counter() - start
    n = LATENCY_REPEATS * len(truncated_query_vecs)
    return (elapsed / n) * 1000.0


def sanity_check(doc_ids, doc_vecs, query_vecs):
    """Confirm the full-dim reference returns obviously-relevant docs."""
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


def run():
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
