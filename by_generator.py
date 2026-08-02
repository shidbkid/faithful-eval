"""Path 1: detector AUC vs generating-model strength (zero GPU).

Slices saved RAGTruth preds by `system` (generator). Hypothesis: detector
AUC falls as the generating model gets stronger — same annotators, same
task, within one dataset.

    python by_generator.py
    python by_generator.py --task Summary,QA,Data2txt
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import data
import metrics

ROOT = os.path.dirname(os.path.abspath(__file__))

# Rough capability order for plotting / monotonicity check.
GENERATOR_ORDER = [
    "llama-2-7b-chat",
    "mistral-7B-instruct",
    "llama-2-13b-chat",
    "llama-2-70b-chat",
    "gpt-3.5-turbo-0613",
    "gpt-4-0613",
]

TASK_PREDS = {
    "Summary": ("results-ragtruth.preds.json", "results-scale-7b.preds.json"),
    "QA": ("results-ragtruth-qa.preds.json", "results-ragtruth-qa-7b.preds.json"),
    "Data2txt": ("results-ragtruth-d2t.preds.json", None),
}

FOCUS = ("rouge-l", "nli-deberta", "llm-judge", "llm-judge-7b-4bit")


def load_aligned(task: str):
    examples = data.load_ragtruth(task=task)
    main_path, extra_path = TASK_PREDS[task]
    blob = json.load(open(os.path.join(ROOT, main_path)))
    assert [e["binary"] for e in examples] == blob["binary"], task
    scorers = dict(blob["scorers"])
    if extra_path and os.path.exists(os.path.join(ROOT, extra_path)):
        extra = json.load(open(os.path.join(ROOT, extra_path)))
        assert extra["binary"] == blob["binary"], extra_path
        for k, v in extra["scorers"].items():
            scorers[k] = v
    return examples, scorers


def slice_auc(examples, scorers, n_boot: int = 1000):
    by_gen = defaultdict(list)
    for i, ex in enumerate(examples):
        by_gen[ex["system"]].append(i)

    rows = []
    for gen in GENERATOR_ORDER:
        idxs = by_gen.get(gen)
        if not idxs:
            continue
        binary = [examples[i]["binary"] for i in idxs]
        faith = sum(binary) / len(binary)
        row = {
            "generator": gen,
            "n": len(idxs),
            "faithful_rate": faith,
            "scorers": {},
        }
        for name in FOCUS:
            if name not in scorers:
                continue
            preds = [scorers[name][i] for i in idxs]
            a = metrics.auc(binary, preds)
            ci = metrics.bootstrap_auc_ci(binary, preds, n_boot=n_boot)
            row["scorers"][name] = {"roc_auc": a, "roc_auc_ci": ci}
        rows.append(row)
    return rows


def is_stable(row: dict, min_per_class: int = 10) -> bool:
    """AUC is unreliable when a slice has almost no positives or negatives."""
    n = row["n"]
    pos = int(round(row["faithful_rate"] * n))
    neg = n - pos
    return neg >= min_per_class and pos >= min_per_class


def spearman_vs_rank(rows, scorer: str, stable_only: bool = False) -> float | None:
    """Correlation of detector AUC with generator rank (0=weak … 5=strong)."""
    xs, ys = [], []
    for rank, row in enumerate(rows):
        if stable_only and not is_stable(row):
            continue
        if scorer not in row["scorers"]:
            continue
        a = row["scorers"][scorer]["roc_auc"]
        if a != a:  # NaN
            continue
        xs.append(rank)
        ys.append(a)
    if len(xs) < 3:
        return None
    from scipy.stats import spearmanr
    return float(spearmanr(xs, ys).statistic)


def print_task(task: str, rows: list):
    print(f"\n### {task}\n")
    scorers = [s for s in FOCUS if any(s in r["scorers"] for r in rows)]
    header = "| generator | n | faithful | " + " | ".join(scorers) + " |"
    sep = "|---|---:|---:|" + "|".join(["---:" for _ in scorers]) + "|"
    print(header)
    print(sep)
    for r in rows:
        cells = []
        for s in scorers:
            if s not in r["scorers"]:
                cells.append("—")
                continue
            a = r["scorers"][s]["roc_auc"]
            cells.append(f"{a:.3f}" if a == a else "n/a")
        print(f"| {r['generator']} | {r['n']} | {r['faithful_rate']:.1%} | "
              + " | ".join(cells) + " |")
    unstable = [r["generator"] for r in rows if not is_stable(r)]
    if unstable:
        print(f"\nUnstable slices (<10 pos or neg): {', '.join(unstable)}")
    print()
    for s in scorers:
        rho = spearman_vs_rank(rows, s)
        rho_s = spearman_vs_rank(rows, s, stable_only=True)
        if rho is None and rho_s is None:
            continue

        def _dir(r):
            if r is None:
                return "n/a"
            if r < -0.3:
                return "falls"
            if r > 0.3:
                return "rises"
            return "flat"

        print(f"- {s}: Spearman all={rho:+.3f} ({_dir(rho)}); "
              f"stable-only={rho_s if rho_s is None else f'{rho_s:+.3f}'} "
              f"({_dir(rho_s)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Summary,QA,Data2txt")
    parser.add_argument("--out", default="results-by-generator.json")
    args = parser.parse_args()

    out = {"hypothesis": (
        "Detector AUC falls as the generating model gets stronger "
        "(within RAGTruth, same annotators)."
    ), "generator_order_weak_to_strong": GENERATOR_ORDER, "tasks": {}}

    print(out["hypothesis"])
    print("Generator order (weak → strong): " + " → ".join(GENERATOR_ORDER))

    for task in args.task.split(","):
        task = task.strip()
        examples, scorers = load_aligned(task)
        rows = slice_auc(examples, scorers)
        print_task(task, rows)
        out["tasks"][task] = {
            "rows": rows,
            "unstable_generators": [r["generator"] for r in rows
                                    if not is_stable(r)],
            "spearman_auc_vs_gen_rank": {
                s: spearman_vs_rank(rows, s)
                for s in FOCUS if any(s in r["scorers"] for r in rows)
            },
            "spearman_auc_vs_gen_rank_stable_only": {
                s: spearman_vs_rank(rows, s, stable_only=True)
                for s in FOCUS if any(s in r["scorers"] for r in rows)
            },
        }

    path = os.path.join(ROOT, args.out)
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
