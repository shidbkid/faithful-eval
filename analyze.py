"""Post-hoc analysis from saved predictions.

    python analyze.py --preds results-ragtruth.preds.json

Writes:
  analysis-ragtruth.json   — bootstrap CIs, complementarity, cascade
and prints a markdown summary for the README / post.
"""

import argparse
import json
import os

import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", required=True,
                        help="*.preds.json from run.py --save-preds")
    parser.add_argument("--out", default=None,
                        help="analysis JSON path (default: alongside preds)")
    args = parser.parse_args()

    blob = json.load(open(args.preds))
    binary = blob["binary"]
    scorers = blob["scorers"]
    out = {"n": len(binary), "faithful": sum(binary),
           "bootstrap_auc": {}, "disagreement": {}, "cascade": {}}

    print(f"n={out['n']}  faithful={out['faithful']} "
          f"({out['faithful']/out['n']:.1%})\n")

    print("### Bootstrap AUC 95% CIs\n")
    print("| scorer | ROC-AUC | AUC 95% CI |")
    print("|---|---|---|")
    aucs = {}
    for name, preds in scorers.items():
        a = metrics.auc(binary, preds)
        ci = metrics.bootstrap_auc_ci(binary, preds)
        aucs[name] = (a, ci)
        out["bootstrap_auc"][name] = {
            "roc_auc": a,
            "roc_auc_ci": ci,
        }
        ci_s = f"{ci[0]:.3f}-{ci[1]:.3f}" if ci else "n/a"
        print(f"| {name} | {a:.3f} | {ci_s} |")

    if "nli-deberta" in aucs and any(k.startswith("llm-judge") for k in aucs):
        judge_key = ("llm-judge" if "llm-judge" in aucs
                     else next(k for k in aucs if k.startswith("llm-judge")))
        nli_ci = aucs["nli-deberta"][1]
        j_ci = aucs[judge_key][1]
        overlap = metrics.cis_overlap(nli_ci, j_ci)
        out["flip_gate"] = {
            "nli_ci": nli_ci,
            "judge_ci": j_ci,
            "overlap": overlap,
            "claim": ("directional" if overlap
                      else "solid — CIs do not overlap"),
        }
        print(f"\nFlip gate (NLI vs {judge_key}): "
              f"{'OVERLAP → directional' if overlap else 'SEPARATED → solid'}")

    if "nli-deberta" in scorers and "llm-judge" in scorers:
        d = metrics.disagreement(
            binary, scorers["nli-deberta"], scorers["llm-judge"],
            name_a="nli", name_b="judge")
        out["disagreement"]["nli_vs_judge"] = d
        print("\n### Failure complementarity (NLI vs judge)\n")
        print(f"- both correct: {d['both_correct']}")
        print(f"- both wrong:   {d['both_wrong']}")
        print(f"- only NLI wrong (judge saves): {d['only_nli_wrong']}")
        print(f"- only judge wrong (NLI saves): {d['only_judge_wrong']}")
        print(f"- complementarity among errors: "
              f"{d['complementarity']:.1%}")

    need = ("rouge-l", "nli-deberta", "llm-judge")
    if all(k in scorers for k in need):
        clo, chi = metrics.cascade_bands(scorers["rouge-l"])
        mlo, mhi = metrics.cascade_bands(scorers["nli-deberta"])
        scores, stages = metrics.cascade_score(
            scorers["rouge-l"], scorers["nli-deberta"], scorers["llm-judge"],
            clo, chi, mlo, mhi)
        from collections import Counter
        stage_counts = Counter(stages)
        n = len(stages)
        # latency proxy: median ms from a sibling results JSON if present
        lat = {"rouge-l": 8.0, "nli-deberta": 400.0, "llm-judge": 1800.0}
        sibling = args.preds.replace(".preds.json", ".json")
        if os.path.exists(sibling):
            for r in json.load(open(sibling)):
                if r["scorer"] in lat:
                    lat[r["scorer"]] = r["median_latency_ms"]
        mean_ms = sum(lat[s] for s in stages) / n
        judge_ms = lat["llm-judge"]
        a = metrics.auc(binary, scores)
        b = metrics.oracle_bal_acc(binary, scores)
        judge_auc = metrics.auc(binary, scorers["llm-judge"])
        out["cascade"] = {
            "bands": {"rouge": [clo, chi], "nli": [mlo, mhi]},
            "stage_frac": {k: v / n for k, v in stage_counts.items()},
            "roc_auc": a,
            "balanced_acc": b,
            "mean_latency_ms": mean_ms,
            "judge_auc": judge_auc,
            "judge_latency_ms": judge_ms,
            "auc_ratio_vs_judge": a / judge_auc if judge_auc else None,
            "latency_ratio_vs_judge": mean_ms / judge_ms if judge_ms else None,
        }
        print("\n### Cascade (ROUGE → NLI → judge on uncertain band)\n")
        print(f"- stage mix: " + ", ".join(
            f"{k}={v/n:.0%}" for k, v in stage_counts.items()))
        print(f"- cascade AUC={a:.3f}  bal_acc={b:.3f}  "
              f"mean {mean_ms:.0f} ms/doc")
        print(f"- vs judge alone: "
              f"{100*a/judge_auc:.0f}% of judge AUC at "
              f"{100*mean_ms/judge_ms:.0f}% of judge latency")

    out_path = args.out or args.preds.replace(".preds.json", "") + ".analysis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
