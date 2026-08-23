"""NLI source-chunk size sensitivity on RAGTruth QA.

    python nli_chunk_sweep.py

Writes results-nli-chunk-qa.json (new file only). Default chunk_size=2
behavior unchanged in NLIScorer.
"""

import json
import os

import data
import metrics
from run import evaluate, print_table, reset_vram_stats
from scorers import NLIScorer

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    examples = data.load_ragtruth(task="QA")
    print(f"dataset=ragtruth task=QA  n={len(examples)}\n")
    rows = []
    for cs in (1, 2, 3):
        scorer = NLIScorer(chunk_size=cs)
        print(f"running {scorer.name} (chunk_size={cs}) ...")
        rows.append(evaluate(scorer, examples))
        del scorer
        reset_vram_stats()
        print_table(rows)
        print()

    out = [{k: v for k, v in r.items() if k != "preds"} for r in rows]
    path = os.path.join(ROOT, "results-nli-chunk-qa.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
