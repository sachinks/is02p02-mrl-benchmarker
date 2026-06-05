# IS02P02 — MRL Benchmarker

> *"Most embedding models give you one fixed-size vector — use all 768 dimensions or none. Matryoshka Representation Learning changes the deal: the first k dimensions of an MRL embedding are themselves a valid, high-quality embedding. Train once. Truncate anywhere. Benchmark to find the smallest dimension that still meets your recall target."*

---

## What this project builds

A benchmark that measures, on a local `nomic-embed-text` model, how much retrieval quality you lose when you truncate MRL embeddings — and how much memory you save. It embeds a labelled corpus once at full dimension, then sweeps truncation sizes and records recall@k, search latency, and memory cost at each size.

The practical value: before deploying an embedding-based retrieval system you want to know the smallest dimension that keeps quality above your threshold. This project gives you that curve. It also teaches the one mandatory implementation detail — re-normalising after truncation — that almost everyone gets wrong the first time.

---

## Matryoshka Representation Learning

`nomic-embed-text` (v1.5) is trained with Matryoshka Representation Learning. A normally-trained embedding model packs meaning evenly across all its dimensions — there is no reason for dimension 1 to be more informative than dimension 500. MRL changes the training loss so the model is penalised at several nested prefix lengths simultaneously (e.g. at 64, 128, 256, 512, and 768 dims). To drive the 64-d loss down, the model is forced to pack as much discriminative signal as possible into the first 64 coordinates; the next slice refines, and so on outward.

The result: a 768-d MRL vector contains a usable 512-d, 256-d, 128-d, and 64-d embedding nested inside it — like Russian dolls. Slicing works. Slicing a normally-trained vector gives noise.

---

## Task prefixes

`nomic-embed-text` expects each input to be tagged by its purpose. Corpus items must be prefixed with `search_document:` and queries with `search_query:`. The model uses these tags to apply different transformations optimised for each role — a document representation is built to be *found*, a query representation is built to *find*.

Skipping the prefix does not throw an error; it quietly degrades retrieval quality. This is a common silent bug in production systems.

---

## L2 normalisation

Ollama returns raw, un-normalised vectors (values well outside [−1, 1]). Every vector must be scaled to unit length before any comparison:

```python
vec = vec / np.linalg.norm(vec)
```

Once normalised, **cosine similarity becomes a plain dot product** — `docs @ query` followed by an argsort. This is cheap, correct, and the same operation used in IS02P01.

---

## Truncate, then re-normalise — the MRL move

```python
def truncate(vec, dim):
    head = vec[:dim]            # keep the first dim numbers
    return head / np.linalg.norm(head)   # put it back on the unit sphere
```

Slicing a unit vector shortens it — the head no longer has length 1. **Re-normalising after truncation is mandatory.** Without it, truncated vectors live on different-radius spheres and their cosine scores are silently wrong. This is the single most common MRL implementation bug. The self-test in `bench/embed.py` proves every truncated vector has L2 norm ≈ 1.0 before the benchmark runs.

---

## Ground truth and recall@k

The reference "correct" ranking is the **full 768-d top-k** for each query. recall@k then asks: *how many of a truncated dimension's top-k appear in the full-dim top-k?* This cleanly isolates the truncation effect — at dim=768 recall is 1.000 by construction, which also serves as a correctness check. A small hand-labelled sanity set (`SANITY_PAIRS`) confirms the full-dim reference is itself sensible before the curve is trusted.

---

## How to install & run

Requires a local [Ollama](https://ollama.com) instance serving `nomic-embed-text`:

```bash
ollama pull nomic-embed-text
```

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Self-test: verify truncation norms are all ~1.0
python -m bench.embed

# 4. Data-load check: 35 docs / 9 queries
python -m bench.corpus

# 5. Run the benchmark — prints the recall/latency/memory table
python -m bench.benchmark

# 6. Generate the visualisation — prints table + saves mrl_benchmark.png
python -m bench.visualise
```

Config (`config.py`, overrideable via `.env`): `OLLAMA_URL` (default `http://127.0.0.1:11434`), `EMBED_MODEL` (default `nomic-embed-text`).

---

## Project structure

```
is02p02-mrl-benchmarker/
  bench/
    embed.py        embed(text, kind) + truncate(vec, dim) + self-test
    corpus.py       DOCS (35 texts, 6 topics), QUERIES (9), SANITY_PAIRS (4)
    benchmark.py    recall@k + latency + memory sweep across dims
    visualise.py    2-panel PNG: recall vs dim | memory vs dim
  config.py         pydantic-settings: OLLAMA_URL, EMBED_MODEL
  requirements.txt
  README.md
```

---

## Algorithm & code flow

### 1. Configuration (`config.py`)

`Settings` is a pydantic-settings class that reads `OLLAMA_URL` and `EMBED_MODEL` from the environment or a `.env` file, with sensible defaults. All bench modules import a single `settings` instance from here.

### 2. Corpus (`bench/corpus.py`)

Defines three module-level constants:
- `DOCS` — 35 sentences across 6 topics (machine learning, databases, web development, devops, security, algorithms). Each is a `(text, topic)` tuple.
- `QUERIES` — 9 natural-language queries, one or two per topic.
- `SANITY_PAIRS` — 4 hand-labelled `(query, expected_top_doc)` pairs used to verify the full-dim reference ranking is sensible.

Running as `__main__` prints a data-load report (doc count, topic distribution, sanity check results).

### 3. Embedding and truncation (`bench/embed.py`)

`embed(text, kind)` — prepends `search_document:` or `search_query:` prefix based on `kind`, POSTs to Ollama's `/api/embeddings` endpoint, extracts the vector, and L2-normalises it. Returns a 768-d numpy unit vector.

`truncate(vec, dim)` — slices `vec[:dim]` and renormalises. Returns a `dim`-d unit vector.

Self-test (runs on `python -m bench.embed`): embeds 5 texts, truncates to all benchmark dims, asserts every norm is within 1e-6 of 1.0. Fails loudly if any vector is not on the unit sphere.

### 4. Benchmark (`bench/benchmark.py`)

The sweep runs in three phases:

**Phase 1 — embed corpus once.** All 35 docs are embedded at full 768-d and stored as a `(35, 768)` matrix. This is done once; truncations are derived from these full-dim vectors.

**Phase 2 — build ground truth.** For each of the 9 queries, embed at 768-d, truncate to 768 (no-op), cosine-search the full corpus, record the top-k indices as ground truth.

**Phase 3 — sweep dimensions.** For each dim in `[64, 128, 256, 512, 768]`:
- Truncate all 35 doc vectors to `dim`-d.
- For each query: truncate query vector to `dim`-d, cosine-search, get top-k indices.
- recall@k = |truncated top-k ∩ full-dim top-k| / k.
- Memory = `35 × dim × 4` bytes (float32).
- Latency = `timeit` over 100 search calls.

Prints a formatted table at the end.

### 5. Visualisation (`bench/visualise.py`)

Calls `benchmark.run()`, then builds a 2-panel matplotlib figure:
- Left panel: recall@k vs dimension (line + point markers).
- Right panel: memory (KB) vs dimension (bar chart).

Saves as `mrl_benchmark.png` in the project root.

---

## Observed

Corpus: 35 docs across 6 topics · 9 queries · k = 5

| dim | recall@5 | search ms | memory KB |
|----:|---------:|----------:|----------:|
|  64 |    0.733 |    ~0.003 |      8.75 |
| 128 |    0.711 |    ~0.003 |     17.50 |
| 256 |    0.756 |    ~0.005 |     35.00 |
| 512 |    0.867 |    ~0.005 |     70.00 |
| 768 |    1.000 |    ~0.005 |    105.00 |

![MRL quality/cost tradeoff](mrl_benchmark.png)

At **512 dims** you keep ~87% of full-dim quality for 2/3 the memory. At **256 dims** — one third the storage — you still hold ~76%. The knee sits between 256 and 512: quality declines gently down to 256, then there is a sharp drop toward 64. Where you set the dial is a recall budget, not a fixed answer.

**The curve is not perfectly monotonic** — 128 (0.711) dips slightly below 64 (0.733). On a 35-doc corpus with k=5, one query swapping a single doc moves recall by ~0.02. The low-dim end is within sampling noise; the overall trend (higher dim → higher recall, sharp gain from 256 to 768) is solid.

**Latency is flat and sub-microsecond at every dimension.** On 35 docs the dot-product cost is dwarfed by overhead, so truncation buys almost nothing in time at this scale. Memory is the real lever here. Latency would diverge on a corpus of millions of vectors — the place MRL's speed story actually pays off.

**Memory scales exactly linearly** (`n_docs × dim × 4` bytes, float32): 105 → 70 → 35 → 17.5 → 8.75 KB. Storing 256-d instead of 768-d cuts memory by 67% with a 24% recall cost — a trade-off worth knowing before any deployment decision.

---

## BENEATH

**Why does MRL training front-load information into the early dimensions — what actually forces that to happen?**

A normal contrastive embedding loss only cares about the *full* vector: it pulls related pairs together and pushes unrelated ones apart using all 768 dims at once. Nothing about that loss says the first 64 numbers must be useful on their own. The model is free to scatter meaning arbitrarily across dimensions, and it does.

MRL changes the *loss*, not the architecture. During training it computes the contrastive loss at **several nested dimensions simultaneously** — e.g. at 64, 128, 256, 512, and 768 — and sums them all. To drive the 64-d loss down, the model must pack as much discriminative signal as possible into the first 64 coordinates. To then drive the 128-d loss down further, it refines the next 64. Each outer slice adds detail on top of what the inner slice already captured. The nesting is a direct consequence of optimising every prefix simultaneously.

This is why you can slice an MRL vector and still get a valid retrieval embedding, and why slicing a normally-trained vector gives you noise: the ordinary model had no incentive to make any prefix self-sufficient. The cost of MRL is a slight degradation at full dimension compared to a model trained only on the full-dim loss — the shared capacity has to serve multiple prefix objectives simultaneously. In practice that cost is small enough to be worth the flexibility.

---

MIT © [Sachin Kolige](https://github.com/sachinks)
