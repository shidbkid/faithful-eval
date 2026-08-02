"""Run every scorer over the dataset and print a results table.

Usage:
    python run.py

Adding a scorer = appending to SCORERS. Nothing else changes.
"""

import argparse
import json
import statistics
import time

import data
from scorers import RandomScorer

SCORERS = [
    RandomScorer(),
]


def peak_vram_gb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
    except ImportError:
        pass
    return None


def reset_vram_stats():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def evaluate(scorer, examples):
    """Score every example, timing each call. Returns a per-scorer row."""
    from scipy.stats import spearmanr
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    reset_vram_stats()
    preds, latencies = [], []
    for ex in examples:
        t0 = time.perf_counter()
        preds.append(scorer.score(ex["source"], ex["summary"]))
        latencies.append(time.perf_counter() - t0)

    labels = [ex["label"] for ex in examples]
    rho = spearmanr(preds, labels).statistic

    # Binary faithfulness: an example is "faithful" iff every annotator gave
    # it a perfect consistency score (label == 5.0). See README.
    binary = [1 if l >= 5.0 else 0 for l in labels]
    if 0 < sum(binary) < len(binary):
        auc = roc_auc_score(binary, preds)
        # Balanced accuracy at the best threshold over observed scores
        # (oracle threshold — same treatment for every scorer).
        bacc = max(
            balanced_accuracy_score(binary, [1 if p >= t else 0 for p in preds])
            for t in sorted(set(preds))
        )
    else:
        auc, bacc = float("nan"), float("nan")

    return {
        "scorer": scorer.name,
        "spearman": rho,
        "balanced_acc": bacc,
        "roc_auc": auc,
        "median_latency_ms": statistics.median(latencies) * 1e3,
        "peak_vram_gb": peak_vram_gb(),
        "n": len(examples),
        "preds": preds,
    }


def fmt(x, spec=".3f"):
    if x is None:
        return "n/a"
    return format(x, spec)


def print_table(rows):
    header = f"{'scorer':<14}{'spearman':>10}{'bal_acc':>10}{'roc_auc':>10}{'med_ms':>10}{'vram_gb':>10}{'n':>6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['scorer']:<14}"
            f"{fmt(r['spearman']):>10}"
            f"{fmt(r['balanced_acc']):>10}"
            f"{fmt(r['roc_auc']):>10}"
            f"{fmt(r['median_latency_ms'], '.1f'):>10}"
            f"{fmt(r['peak_vram_gb']):>10}"
            f"{r['n']:>6}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of (source, summary) pairs")
    parser.add_argument("--out", default="results.json",
                        help="where to dump raw per-scorer results")
    args = parser.parse_args()

    examples = data.load_toy()
    if args.limit:
        examples = examples[: args.limit]
    print(f"{len(examples)} (source, summary) pairs\n")

    rows = []
    for scorer in SCORERS:
        rows.append(evaluate(scorer, examples))

    print_table(rows)

    with open(args.out, "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "preds"} for r in rows],
                  f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
