"""Judge prompt-variance: 1.5B on RAGTruth Summary with 3 paraphrases.

Skips 7B-4bit (3x ~20 min) unless --also-7b is passed.

    python judge_prompt_variance.py
"""

from __future__ import annotations

import argparse
import json
import os

import data
from run import evaluate, print_table, reset_vram_stats
from scorers import LLMJudgeScorer

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--also-7b", action="store_true",
                        help="also run 7B-4bit x3 (~45-60 min extra)")
    args = parser.parse_args()

    examples = data.load_ragtruth(task="Summary")
    print(f"dataset=ragtruth task=Summary  n={len(examples)}\n")

    configs = [
        ("1.5b", "Qwen/Qwen2.5-1.5B-Instruct", False),
    ]
    if args.also_7b:
        configs.append(("7b-4bit", "Qwen/Qwen2.5-7B-Instruct", True))
    else:
        print("skipping 7B-4bit (use --also-7b to run; ~45-60 min)\n")

    all_rows = []
    for tag, model_name, q4 in configs:
        for i, prompt in enumerate(LLMJudgeScorer.PROMPTS):
            name = f"llm-judge-{tag}-p{i + 1}"
            print(f"running {name} ...")
            scorer = LLMJudgeScorer(
                model_name=model_name, load_in_4bit=q4,
                name=name, prompt=prompt)
            row = evaluate(scorer, examples)
            all_rows.append(row)
            del scorer
            reset_vram_stats()
            print_table(all_rows)
            print()

    # Summarize spreads per size
    summary = {}
    for tag, _, _ in configs:
        aucs = [r["roc_auc"] for r in all_rows
                if r["scorer"].startswith(f"llm-judge-{tag}-p")]
        summary[tag] = {
            "aucs": aucs,
            "min": min(aucs),
            "max": max(aucs),
            "spread": max(aucs) - min(aucs),
        }
        print(f"{tag}: AUC {aucs}  min={min(aucs):.3f} max={max(aucs):.3f} "
              f"spread={max(aucs) - min(aucs):.3f}")

    out = {
        "rows": [{k: v for k, v in r.items() if k != "preds"} for r in all_rows],
        "summary": summary,
        "skipped_7b": not args.also_7b,
    }
    path = os.path.join(ROOT, "results-judge-prompt-variance.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
