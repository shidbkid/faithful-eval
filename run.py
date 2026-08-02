"""Run every scorer over the dataset and print a results table.

Usage:
    python run.py
    python run.py --dataset ragtruth --save-preds
    python run.py --dataset ragtruth --only llm-judge --judge-model 0.5b,1.5b,3b,7b-4bit

Adding a scorer = appending to SCORERS. Nothing else changes.
"""

import argparse
import json
import statistics
import time

import data
import metrics
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

# Named judge variants for the scaling curve.
JUDGE_MODELS = {
    "0.5b": ("Qwen/Qwen2.5-0.5B-Instruct", False, "llm-judge-0.5b"),
    "1.5b": ("Qwen/Qwen2.5-1.5B-Instruct", False, "llm-judge-1.5b"),
    "3b": ("Qwen/Qwen2.5-3B-Instruct", False, "llm-judge-3b"),
    "7b-4bit": ("Qwen/Qwen2.5-7B-Instruct", True, "llm-judge-7b-4bit"),
}


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
            torch.cuda.empty_cache()
    except ImportError:
        pass


def evaluate(scorer, examples):
    """Score every example, timing each call. Returns a per-scorer row."""
    from scipy.stats import spearmanr

    reset_vram_stats()
    preds, latencies = [], []
    for ex in examples:
        t0 = time.perf_counter()
        preds.append(scorer.score(ex["source"], ex["summary"]))
        latencies.append(time.perf_counter() - t0)

    labels = [ex["label"] for ex in examples]
    binary = [ex["binary"] for ex in examples]
    rho = spearmanr(preds, labels).statistic
    auc = metrics.auc(binary, preds)
    bacc = metrics.oracle_bal_acc(binary, preds)
    ci = metrics.bootstrap_auc_ci(binary, preds)

    return {
        "scorer": scorer.name,
        "spearman": rho,
        "balanced_acc": bacc,
        "roc_auc": auc,
        "roc_auc_ci": ([round(ci[0], 4), round(ci[1], 4)] if ci else None),
        "median_latency_ms": statistics.median(latencies) * 1e3,
        "peak_vram_gb": peak_vram_gb(),
        "n": len(examples),
        "preds": preds,
    }


def fmt(x, spec=".3f"):
    if x is None:
        return "n/a"
    try:
        if x != x:  # NaN
            return "n/a"
    except TypeError:
        pass
    return format(x, spec)


def print_table(rows):
    header = (f"{'scorer':<18}{'spearman':>10}{'bal_acc':>10}"
              f"{'roc_auc':>10}{'med_ms':>10}{'vram_gb':>10}{'n':>6}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['scorer']:<18}"
            f"{fmt(r['spearman']):>10}"
            f"{fmt(r['balanced_acc']):>10}"
            f"{fmt(r['roc_auc']):>10}"
            f"{fmt(r['median_latency_ms'], '.1f'):>10}"
            f"{fmt(r['peak_vram_gb']):>10}"
            f"{r['n']:>6}"
        )
        ci = r.get("roc_auc_ci")
        if ci:
            print(f"{'':18}  AUC 95% CI  {ci[0]:.3f}-{ci[1]:.3f}")


def build_scorer_list(only, judge_models):
    """Return a list of zero-arg callables that construct scorers."""
    factories = []
    if judge_models:
        for key in judge_models:
            if key not in JUDGE_MODELS:
                raise SystemExit(
                    f"unknown --judge-model {key!r}; "
                    f"choose from {list(JUDGE_MODELS)}")
            model_name, load_4bit, name = JUDGE_MODELS[key]

            def make(mn=model_name, q=load_4bit, n=name):
                return LLMJudgeScorer(model_name=mn, load_in_4bit=q, name=n)
            factories.append(make)
        return factories

    only_set = set(only.split(",")) if only else None
    for cls in SCORERS:
        if only_set is not None and cls.name not in only_set:
            continue
        factories.append(cls)
    return factories


def dump(rows, path, save_preds):
    slim = []
    preds_blob = {"binary": None, "label": None, "scorers": {}}
    for r in rows:
        slim.append({k: v for k, v in r.items() if k != "preds"})
        if save_preds:
            preds_blob["scorers"][r["scorer"]] = r["preds"]
    with open(path, "w") as f:
        json.dump(slim, f, indent=2)
    if save_preds:
        pred_path = path.replace(".json", "") + ".preds.json"
        # binary/label filled by caller once
        with open(pred_path, "w") as f:
            json.dump(preds_blob, f)
        return pred_path
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="summeval",
                        choices=list(data.DATASETS),
                        help="benchmark to evaluate (default: summeval)")
    parser.add_argument("--task", default="Summary",
                        help="RAGTruth: Summary|QA|Data2txt|all. "
                             "TofuEval: MeetB|MediaS|all (default both). "
                             "Ignored for summeval.")
    parser.add_argument("--limit", type=int, default=None,
                        help="random (seeded) subsample of pairs, for quick runs")
    parser.add_argument("--only", default=None,
                        help="comma-separated scorer names to run, e.g. "
                             "--only random,rouge-l")
    parser.add_argument("--judge-model", default=None,
                        help="comma-separated judge sizes for scaling curve: "
                             "0.5b,1.5b,3b,7b-4bit (replaces default SCORERS)")
    parser.add_argument("--out", default="results.json",
                        help="where to dump aggregate results")
    parser.add_argument("--save-preds", action="store_true",
                        help="also write <out>.preds.json with per-example scores")
    args = parser.parse_args()

    load_kwargs = {}
    if args.dataset == "ragtruth":
        load_kwargs["task"] = None if args.task == "all" else args.task
    elif args.dataset == "tofueval":
        # reuse --task: all/Summary-default -> both subsets
        if args.task in ("all", "Summary"):
            load_kwargs["subset"] = None
        else:
            load_kwargs["subset"] = args.task
    examples = data.load(args.dataset, **load_kwargs)
    if args.limit and args.limit < len(examples):
        import random as _random
        examples = _random.Random(0).sample(examples, args.limit)
    n_pos = sum(ex["binary"] for ex in examples)
    print(f"dataset={args.dataset}  {len(examples)} pairs  "
          f"faithful={n_pos} ({n_pos / len(examples):.1%})\n")

    judge_keys = (args.judge_model.split(",") if args.judge_model else None)
    factories = build_scorer_list(args.only, judge_keys)

    rows = []
    pred_path = None
    for factory in factories:
        scorer = factory()
        print(f"running {scorer.name} ...")
        rows.append(evaluate(scorer, examples))
        # free GPU before next heavy model
        del scorer
        reset_vram_stats()
        print_table(rows)
        print()
        # rewrite after every scorer
        out_rows = []
        for r in rows:
            out_rows.append({k: v for k, v in r.items() if k != "preds"})
        with open(args.out, "w") as f:
            json.dump(out_rows, f, indent=2)
        if args.save_preds:
            pred_path = args.out.replace(".json", "") + ".preds.json"
            blob = {
                "dataset": args.dataset,
                "task": load_kwargs.get("task", "Summary"),
                "binary": [ex["binary"] for ex in examples],
                "label": [ex["label"] for ex in examples],
                "scorers": {r["scorer"]: r["preds"] for r in rows},
            }
            with open(pred_path, "w") as f:
                json.dump(blob, f)

    print(f"wrote {args.out}"
          + (f" and {pred_path}" if pred_path else ""))


if __name__ == "__main__":
    main()
