"""Add approximate Hanley–McNeil 95% CIs for ROC-AUC into results JSON.

Does not need per-example scores — uses (AUC, n_pos, n_neg) only.
Good enough for the claim gate; bootstrap CIs can replace these later
if preds are persisted.
"""

import json
import math
import sys


def hanley_ci(auc, n_pos, n_neg, z=1.96):
    if not (0 < auc < 1) or n_pos < 2 or n_neg < 2:
        return None
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    se2 = (auc * (1 - auc)
           + (n_pos - 1) * (q1 - auc ** 2)
           + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    se = math.sqrt(max(se2, 0.0))
    return [max(0.0, auc - z * se), min(1.0, auc + z * se)]


# (path, n_pos, n_neg) — faithful counts from the dataset loaders
TARGETS = [
    ("results.json", 1306, 294),
    ("results-ragtruth.json", 696, 204),
]


def main():
    for path, n_pos, n_neg in TARGETS:
        rows = json.load(open(path))
        print(f"=== {path} (pos={n_pos}, neg={n_neg}) ===")
        for r in rows:
            ci = hanley_ci(r["roc_auc"], n_pos, n_neg)
            r["roc_auc_ci"] = ([round(ci[0], 4), round(ci[1], 4)]
                               if ci else None)
            print(f"  {r['scorer']:12} auc={r['roc_auc']:.3f}  "
                  f"ci={r['roc_auc_ci']}")
        json.dump(rows, open(path, "w"), indent=2)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
