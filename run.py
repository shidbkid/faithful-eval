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
from scorers import (BERTScoreScorer, LLMJudgeScorer, NLIScorer,
                     RandomScorer, RougeLScorer)

# Zero-arg constructors; instantiated one at a time in main() so a heavy
# model is only loaded while its scorer is running.
SCORERS = [
    RandomScorer,
    RougeLScorer,
    BERTScoreScorer,
    NLIScorer,
    LLMJudgeScorer,
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

    # Binary labels come from the dataset loader (SummEval: consistency==5;
    # RAGTruth: zero hallucination spans). See data.py / README.
    binary = [ex["binary"] for ex in examples]
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
    parser.add_argument("--dataset", default="summeval",
                        choices=list(data.DATASETS),
                        help="benchmark to evaluate (default: summeval)")
    parser.add_argument("--task", default="Summary",
                        help="RAGTruth task filter: Summary|QA|Data2txt|all "
                             "(default: Summary; ignored for summeval)")
    parser.add_argument("--limit", type=int, default=None,
                        help="random (seeded) subsample of pairs, for quick runs")
    parser.add_argument("--only", default=None,
                        help="comma-separated scorer names to run, e.g. "
                             "--only random,rouge-l")
    parser.add_argument("--out", default="results.json",
                        help="where to dump raw per-scorer results")
    args = parser.parse_args()

    load_kwargs = {}
    if args.dataset == "ragtruth":
        load_kwargs["task"] = None if args.task == "all" else args.task
    examples = data.load(args.dataset, **load_kwargs)
    if args.limit and args.limit < len(examples):
        import random as _random
        examples = _random.Random(0).sample(examples, args.limit)
    n_pos = sum(ex["binary"] for ex in examples)
    print(f"dataset={args.dataset}  {len(examples)} pairs  "
          f"faithful={n_pos} ({n_pos / len(examples):.1%})\n")

    only = set(args.only.split(",")) if args.only else None
    rows = []
    for cls in SCORERS:
        if only is not None and cls.name not in only:
            continue
        scorer = cls()
        rows.append(evaluate(scorer, examples))
        # Reprint the table and rewrite results after every scorer, so a
        # crash in scorer N doesn't lose scorers 1..N-1.
        print_table(rows)
        print()
        with open(args.out, "w") as f:
            json.dump([{k: v for k, v in r.items() if k != "preds"}
                       for r in rows], f, indent=2)

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
