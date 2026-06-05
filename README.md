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

The reference "correct" ranking is the **full 768-d top-k** for each query. recall@k then asks: *how many of a truncated dimension's top-k appear in the full-dim top-k?* This cleanly isolates the truncation effect — at dim=768 recall is 1.000 by construction, which also serves as a correctness check. A small hand-labelled sanity dict (`SANITY`) confirms the full-dim reference is itself sensible before the curve is trusted.

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
    corpus.py       CORPUS (35 3-tuples of id/text/topic), QUERIES (9), SANITY (4-entry dict)
    benchmark.py    recall@k + latency + memory sweep across dims
    visualise.py    2-panel PNG: recall vs dim | memory vs dim
  config.py         pydantic-settings: OLLAMA_URL, EMBED_MODEL
  requirements.txt
  README.md
```

---

## Algorithm & code flow

### 1. `config.py` — settings

`Settings` is a pydantic-settings `BaseSettings` subclass. `SettingsConfigDict(env_file=".env", extra="ignore")` means values are read from environment variables or a `.env` file; unrecognised keys are silently ignored. A module-level singleton `settings = Settings()` is imported by all bench modules.

Fields with defaults: `ollama_url = "http://127.0.0.1:11434"`, `embed_model = "nomic-embed-text"`. Override at the shell with e.g. `OLLAMA_URL=http://remote:11434 python -m bench.benchmark`.

### 2. `bench/corpus.py` — dataset

Three module-level constants:

- **`CORPUS`** — list of 35 `(id, text, topic)` 3-tuples across 6 topics (programming, finance, cooking, animals, sports, health). The `id` field (e.g. `"p1"`, `"h4"`) is used exclusively by the sanity checker to identify expected top results — the embedding engine never sees it.
- **`QUERIES`** — list of 9 `(query_text, expected_topic)` tuples. Benchmark accesses `q[0]` for the text string. `expected_topic` is metadata only.
- **`SANITY`** — `dict[str, set[str]]` mapping 4 query strings to sets of doc ids known to be obviously relevant. Used by `sanity_check()` in `benchmark.py` before the recall sweep begins.

`__main__` block counts docs by topic and prints a data-load report.

### 3. `bench/embed.py` — embedding and truncation

**`_PREFIX` dict** — `{"document": "search_document: ", "query": "search_query: "}`. Prefixes are prepended to the text before sending to Ollama. Never add them yourself — `embed()` does it automatically.

**`_l2_normalize(vec)`** — divides by `np.linalg.norm(vec)`; returns `vec` unchanged if norm is 0.0 (zero-vector guard). Called by both `embed()` and `truncate()`.

**`embed(text, kind="document") -> np.ndarray`** — validates `kind`, posts `{"model": settings.embed_model, "prompt": prefix + text}` to `{settings.ollama_url}/api/embeddings` with `timeout=60`, calls `resp.raise_for_status()`, extracts `resp.json()["embedding"]`, casts to `float32`, and passes through `_l2_normalize`. Returns shape `(768,)`.

**`truncate(vec, dim) -> np.ndarray`** — raises `ValueError` if `dim > vec.shape[0]`, then returns `_l2_normalize(vec[:dim])`. Shape `(dim,)`.

**Self-test** (`python -m bench.embed`): embeds 5 texts (3 document, 2 query), asserts full-dim norm and all 5 truncated-dim norms are within `NORM_TOL = 1e-5` of 1.0. Total: 25 hard `AssertionError` checks. Prints one `OK` line per text showing all norms; ends with `"All norm checks passed."`.

### 4. `bench/benchmark.py` — sweep engine

**Constants:** `DIMS = [64, 128, 256, 512, 768]`, `K = 5`, `LATENCY_REPEATS = 50`.

**`rank(query_vec, doc_matrix)`** — `doc_matrix @ query_vec` gives a `(n,)` scores array, then `np.argsort(-scores)` returns indices best-first (negation avoids a reverse-slice).

**`embed_all()`** — embeds every `CORPUS` doc (`kind="document"`) and every `QUERIES` query (`kind="query"`) at full 768-d. Docs are stacked with `np.vstack(...)` into shape `(35, 768)`. Returns `(doc_ids, doc_vecs, query_vecs)`.

**`gold_topk(doc_vecs, query_vecs, k)`** — runs `rank()` at full dim for each query, takes `[:k]`, returns a list of `set[int]` — the ground-truth index sets.

**`recall_at_k(td, tq, gold, k)`** — for each query: `got = set(rank(tq[i], td)[:k])`, recall = `len(got & gold[i]) / k`. Returns the mean across all 9 queries.

**`search_latency_ms(td, tq)`** — `time.perf_counter` around `LATENCY_REPEATS × 9` `rank()` calls; divides total elapsed by `50 × 9` and multiplies by 1000 for milliseconds.

**`sanity_check(doc_ids, doc_vecs, query_vecs)`** — builds `q_by_text = {query_text: vec}` from `QUERIES`/`query_vecs`, then for each of the 4 `SANITY` entries checks that `rank()[0]` returns a doc id in the expected set. Prints `[OK ]` or `[XX ]` per entry; returns `True` if all pass.

**`run()`** — calls `embed_all()` → `sanity_check()` → `gold_topk()` → per-dim sweep (truncate docs+queries, `recall_at_k`, `search_latency_ms`, `mem_kb = td.nbytes / 1024.0`) → prints formatted table → returns results list.

### 5. `bench/visualise.py` — chart

`visualise(results, output_path="mrl_benchmark.png")`:
- Unpacks `dim`, `recall`, `mem_kb` from each result dict.
- `plt.subplots(1, 2, figsize=(12, 5))` — two side-by-side panels.
- **Left panel** (`ax_q`): `plot(dims, recall, "o-", color="#2a7")`, dashed `axhline(1.0)` for the full-dim reference, `ylim(0, 1.05)`, value labels at `xytext=(0, 8)` offset.
- **Right panel** (`ax_c`): `plot(dims, mem, "s-", color="#c63")`, value labels same offset.
- `fig.tight_layout()`, `fig.savefig(output_path, dpi=120, bbox_inches="tight")`.

`__main__` block calls `run()` then `visualise(results)`.

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

## License

MIT © [Sachin Kolige](https://github.com/sachinks)
