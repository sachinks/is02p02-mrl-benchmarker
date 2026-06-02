"""Visualise the MRL benchmark: quality vs cost across truncation dims.

Produces one figure with two side-by-side panels:
  left  : recall@k vs dim  (quality)
  right : memory KB vs dim  (cost)

Run:  python -m bench.visualise
"""

import matplotlib

matplotlib.use("Agg")  # headless WSL — no display needed

import matplotlib.pyplot as plt

from bench.benchmark import K, run


def visualise(results, output_path="mrl_benchmark.png"):
    dims = [r["dim"] for r in results]
    recall = [r["recall"] for r in results]
    mem = [r["mem_kb"] for r in results]

    fig, (ax_q, ax_c) = plt.subplots(1, 2, figsize=(12, 5))

    # --- left: quality ---
    ax_q.plot(dims, recall, "o-", color="#2a7", linewidth=2, markersize=8)
    ax_q.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="full-dim reference")
    ax_q.set_title(f"Quality: recall@{K} vs truncation dim")
    ax_q.set_xlabel("embedding dimensions")
    ax_q.set_ylabel(f"recall@{K}")
    ax_q.set_xticks(dims)
    ax_q.set_ylim(0, 1.05)
    ax_q.grid(True, alpha=0.3)
    ax_q.legend()
    for x, y in zip(dims, recall):
        ax_q.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    # --- right: cost ---
    ax_c.plot(dims, mem, "s-", color="#c63", linewidth=2, markersize=8)
    ax_c.set_title("Cost: corpus memory vs truncation dim")
    ax_c.set_xlabel("embedding dimensions")
    ax_c.set_ylabel("memory (KB)")
    ax_c.set_xticks(dims)
    ax_c.grid(True, alpha=0.3)
    for x, y in zip(dims, mem):
        ax_c.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    fig.suptitle("Matryoshka Representation Learning — quality/cost tradeoff (nomic-embed-text)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"saved {output_path}")


if __name__ == "__main__":
    results = run()
    print()
    visualise(results)
