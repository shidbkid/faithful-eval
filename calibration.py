"""Scorer self-knowledge: risk-coverage, cascade gates, cross-dataset ECE.

Does a scorer's own confidence tell you when it is wrong?

Leakage protocol
----------------
Thresholds and isotonic maps are fit on one dataset and applied to another.
Default directions: fit RAGTruth Summary -> eval TofuEval, and the swap.
Never fit and evaluate on the same examples.

Judge size variants (1.5B / 7B-4bit) exist only on RAGTruth Summary preds;
for those we use a seeded 50/50 within-RAGTruth split and label it clearly.

    python calibration.py
    python calibration.py --figure-only
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss

import metrics
from cascade import _load_summary_judges, load_latencies, soft_cascade

ROOT = os.path.dirname(os.path.abspath(__file__))

# Scorers requested for Task 1 (names as reported).
WANTED = (
    "rouge-l",
    "bertscore",
    "nli-deberta",
    "minicheck",
    "alignscore",
    "llm-judge-1.5b",
    "llm-judge-3b",
    "llm-judge-7b-4bit",
)

# Colors from plot.py (identity follows the entity).
COLORS = {
    "random": "#8a8984",
    "rouge-l": "#2a78d6",
    "bertscore": "#eb6834",
    "nli-deberta": "#1baf7a",
    "llm-judge": "#4a3aa7",
    "llm-judge-1.5b": "#4a3aa7",
    "llm-judge-3b": "#6b5bc7",
    "llm-judge-7b-4bit": "#2f2470",
    "minicheck": "#c45c26",
    "alignscore": "#0e8a9a",
}
FALLBACK = "#e87ba4"


def _load_bundle(name: str) -> dict:
    """Merge preds for a named corpus into {binary, scorers}."""
    if name == "ragtruth-summary":
        base = json.load(open(os.path.join(ROOT, "results-ragtruth.preds.json")))
        scorers = dict(base["scorers"])
        # Prefer named judge sizes from the scale runs; keep llm-judge as 3b alias.
        scale = json.load(open(os.path.join(ROOT, "results-scale.preds.json")))
        s7 = json.load(open(os.path.join(ROOT, "results-scale-7b.preds.json")))
        assert scale["binary"] == base["binary"]
        assert s7["binary"] == base["binary"]
        scorers["llm-judge-1.5b"] = scale["scorers"]["llm-judge-1.5b"]
        scorers["llm-judge-3b"] = scale["scorers"]["llm-judge-3b"]
        scorers["llm-judge-7b-4bit"] = s7["scorers"]["llm-judge-7b-4bit"]
        # Alias: original llm-judge row == 3B on this card.
        if "llm-judge" in scorers and "llm-judge-3b" not in scorers:
            scorers["llm-judge-3b"] = scorers["llm-judge"]
        ns = os.path.join(ROOT, "results-newscorers-ragtruth.preds.json")
        if os.path.exists(ns):
            extra = json.load(open(ns))
            assert extra["binary"] == base["binary"]
            scorers.update(extra["scorers"])
        return {"name": name, "binary": base["binary"], "scorers": scorers}

    if name == "tofueval":
        base = json.load(open(os.path.join(ROOT, "results-tofueval.preds.json")))
        scorers = dict(base["scorers"])
        # TofuEval llm-judge is the default 3B.
        if "llm-judge" in scorers:
            scorers["llm-judge-3b"] = scorers["llm-judge"]
        ns = os.path.join(ROOT, "results-newscorers-tofueval.preds.json")
        if os.path.exists(ns):
            extra = json.load(open(ns))
            assert extra["binary"] == base["binary"]
            scorers.update(extra["scorers"])
        return {"name": name, "binary": base["binary"], "scorers": scorers}

    raise ValueError(name)


def fit_threshold(binary, preds) -> float:
    """Train-only threshold maximizing balanced accuracy."""
    best_t, best = 0.5, -1.0
    for t in sorted(set(preds)):
        acc = balanced_accuracy_score(
            binary, [1 if p >= t else 0 for p in preds])
        if acc > best:
            best, best_t = acc, float(t)
    return best_t


def confidence(preds, t) -> np.ndarray:
    """c = |s - t|, min-max normalized on the evaluation split."""
    raw = np.abs(np.asarray(preds, dtype=float) - t)
    lo, hi = float(raw.min()), float(raw.max())
    if hi - lo < 1e-12:
        return np.zeros_like(raw)
    return (raw - lo) / (hi - lo)


def is_correct(binary, preds, t) -> np.ndarray:
    pred_bin = np.asarray([1 if p >= t else 0 for p in preds])
    return (pred_bin == np.asarray(binary)).astype(int)


def bal_acc_at_coverage(binary, preds, t, conf, coverage: float) -> float:
    """Balanced accuracy on the top-coverage% most confident examples."""
    n = len(binary)
    k = max(1, int(round(n * coverage)))
    order = np.argsort(-conf)  # descending confidence
    idx = order[:k]
    y = np.asarray(binary)[idx]
    s = np.asarray(preds, dtype=float)[idx]
    if len(set(y.tolist())) < 2:
        # Degenerate slice - fall back to accuracy.
        pred = (s >= t).astype(int)
        return float((pred == y).mean())
    return float(balanced_accuracy_score(y, (s >= t).astype(int)))


def risk_coverage_curve(binary, preds, t, conf):
    coverages = [i / 10 for i in range(1, 11)]  # 0.1 .. 1.0
    accs = [bal_acc_at_coverage(binary, preds, t, conf, c) for c in coverages]
    # AURC: integrate risk = 1 - bal_acc over coverage (lower better).
    risks = [1.0 - a for a in accs]
    aurc = float(np.trapezoid(risks, coverages)) if hasattr(np, "trapezoid") \
        else float(np.trapz(risks, coverages))
    acc50 = bal_acc_at_coverage(binary, preds, t, conf, 0.5)
    return {
        "coverages": coverages,
        "balanced_acc": accs,
        "acc_at_50": acc50,
        "aurc": aurc,
    }


def bootstrap_rc_ci(binary, preds, t, n_boot=1000, seed=0):
    """Bootstrap CIs for acc@50% and AURC (paired resampling of examples)."""
    y = np.asarray(binary)
    s = np.asarray(preds, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    a50s, aurcs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, sb = y[idx].tolist(), s[idx].tolist()
        # Threshold stays the train-fit value (no refit on eval bootstrap).
        conf = confidence(sb, t)
        curve = risk_coverage_curve(yb, sb, t, conf)
        a50s.append(curve["acc_at_50"])
        aurcs.append(curve["aurc"])
    return {
        "acc_at_50_ci": [float(np.percentile(a50s, 2.5)),
                         float(np.percentile(a50s, 97.5))],
        "aurc_ci": [float(np.percentile(aurcs, 2.5)),
                    float(np.percentile(aurcs, 97.5))],
    }


def ece_score(y_true, y_prob, n_bins=10) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if not np.any(mask):
            continue
        conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)


def within_ragtruth_split(binary, preds, seed=0):
    """50/50 fit/eval split for scorers that lack a second corpus."""
    n = len(binary)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    mid = n // 2
    fit_i, eval_i = idx[:mid], idx[mid:]
    return (
        [binary[i] for i in fit_i],
        [preds[i] for i in fit_i],
        [binary[i] for i in eval_i],
        [preds[i] for i in eval_i],
    )


# ---------------------------------------------------------------------------
# Task 1 - risk-coverage
# ---------------------------------------------------------------------------

def run_risk_coverage(n_boot=1000) -> dict:
    rt = _load_bundle("ragtruth-summary")
    tf = _load_bundle("tofueval")
    directions = [
        ("ragtruth-summary", "tofueval", rt, tf),
        ("tofueval", "ragtruth-summary", tf, rt),
    ]

    results = {
        "protocol": (
            "Threshold fit on fit_set, confidence |s-t| min-max normalized "
            "on eval_set. Directions: RT-Summary↔TofuEval. Judge 1.5B/7B "
            "lack TofuEval preds -> seeded 50/50 within-RAGTruth split."
        ),
        "directions": {},
    }

    print("### Task 1 - Risk-coverage (fit -> eval)\n")
    for fit_name, eval_name, fit_b, eval_b in directions:
        key = f"{fit_name}->{eval_name}"
        rows = []
        print(f"#### {key}\n")
        print("| scorer | full AUC | acc@50% | AURC | acc@50% CI | AURC CI | note |")
        print("|---|---:|---:|---:|---|---|---|")

        for scorer in WANTED:
            note = ""
            if scorer not in eval_b["scorers"]:
                # Need within-split on RAGTruth
                if scorer not in rt["scorers"]:
                    continue
                y_fit, s_fit, y_eval, s_eval = within_ragtruth_split(
                    rt["binary"], rt["scorers"][scorer])
                note = "within-RT 50/50 (no TF preds)"
                protocol_label = "within-ragtruth-summary-50/50"
            elif scorer not in fit_b["scorers"]:
                if scorer not in rt["scorers"]:
                    continue
                y_fit, s_fit, y_eval, s_eval = within_ragtruth_split(
                    rt["binary"], rt["scorers"][scorer])
                note = "within-RT 50/50 (no TF preds for fit)"
                protocol_label = "within-ragtruth-summary-50/50"
            else:
                y_fit = fit_b["binary"]
                s_fit = fit_b["scorers"][scorer]
                y_eval = eval_b["binary"]
                s_eval = eval_b["scorers"][scorer]
                protocol_label = key

            # Skip duplicate within-RT rows on the second direction
            if note and fit_name != "ragtruth-summary":
                continue

            t = fit_threshold(y_fit, s_fit)
            conf = confidence(s_eval, t)
            curve = risk_coverage_curve(y_eval, s_eval, t, conf)
            cis = bootstrap_rc_ci(y_eval, s_eval, t, n_boot=n_boot)
            full_auc = metrics.auc(y_eval, s_eval)
            row = {
                "scorer": scorer,
                "protocol": protocol_label,
                "threshold": t,
                "full_auc": full_auc,
                "acc_at_50": curve["acc_at_50"],
                "aurc": curve["aurc"],
                "acc_at_50_ci": cis["acc_at_50_ci"],
                "aurc_ci": cis["aurc_ci"],
                "curve": curve,
                "note": note,
            }
            rows.append(row)
            a_ci = f"{cis['acc_at_50_ci'][0]:.3f}-{cis['acc_at_50_ci'][1]:.3f}"
            u_ci = f"{cis['aurc_ci'][0]:.3f}-{cis['aurc_ci'][1]:.3f}"
            print(f"| {scorer} | {full_auc:.3f} | {curve['acc_at_50']:.3f} | "
                  f"{curve['aurc']:.3f} | {a_ci} | {u_ci} | {note or '-'} |")

        # Rankings
        by_auc = sorted(rows, key=lambda r: -r["full_auc"])
        by_aurc = sorted(rows, key=lambda r: r["aurc"])  # lower better
        auc_rank = [r["scorer"] for r in by_auc]
        aurc_rank = [r["scorer"] for r in by_aurc]
        match = auc_rank == aurc_rank
        print(f"\nAUC ranking (best->worst): {auc_rank}")
        print(f"AURC ranking (best->worst): {aurc_rank}")
        print(f"Rankings match: {match}\n")

        results["directions"][key] = {
            "rows": rows,
            "auc_ranking": auc_rank,
            "aurc_ranking": aurc_rank,
            "rankings_match": match,
        }

    # Also store within-RT-only block for judges if not already under first dir
    return results


# ---------------------------------------------------------------------------
# Task 2 - cascade gate sweep (RAGTruth Summary -> 7B)
# ---------------------------------------------------------------------------

def run_gate_sweep() -> dict:
    """Reuse cascade.soft_cascade; parameterize the cheap gate only."""
    binary, scorers = _load_summary_judges()
    # Merge minicheck / alignscore
    ns = os.path.join(ROOT, "results-newscorers-ragtruth.preds.json")
    extra = json.load(open(ns))
    assert extra["binary"] == binary
    scorers.update(extra["scorers"])

    lat = load_latencies("Summary")
    # Pull MiniCheck / AlignScore latencies from the newscorers aggregate JSON.
    for path in ("results-newscorers-ragtruth.json",):
        for r in json.load(open(os.path.join(ROOT, path))):
            lat[r["scorer"]] = r["median_latency_ms"]

    dear_key = "llm-judge-7b-4bit"
    dear = scorers[dear_key]
    gates = ("rouge-l", "nli-deberta", "minicheck", "alignscore",
             "llm-judge-1.5b")

    print("### Task 2 - Cascade gate sweep (RAGTruth Summary -> 7B-4bit)\n")
    print("| gate | cascade AUC | mean ms | escalated% | "
          "% of 7B AUC | % of 7B latency |")
    print("|---|---:|---:|---:|---:|---:|")

    rows = []
    target_auc = metrics.auc(binary, dear)
    target_ms = lat[dear_key]

    for gate in gates:
        scores, stages, band = soft_cascade(scorers[gate], dear)
        stage_names = [gate if s == "cheap" else dear_key for s in stages]
        n = len(stages)
        mean_ms = sum(lat[s] for s in stage_names) / n
        esc = sum(1 for s in stages if s == "dear") / n
        auc = metrics.auc(binary, scores)
        # Soft-keep mixes scales; also report z-calibrated cascade for clarity.
        # Primary table uses existing soft_cascade protocol (raw soft-keep).
        row = {
            "gate": gate,
            "policy": f"{gate} -> mid-band {dear_key} (soft-keep)",
            "roc_auc": auc,
            "roc_auc_ci": metrics.bootstrap_auc_ci(binary, scores, n_boot=1000),
            "mean_latency_ms": mean_ms,
            "escalated_frac": esc,
            "band": list(band),
            "auc_ratio_vs_7b": auc / target_auc,
            "latency_ratio_vs_7b": mean_ms / target_ms,
            "target_auc": target_auc,
            "target_latency_ms": target_ms,
            "stage_frac": {k: v / n for k, v in Counter(stage_names).items()},
        }
        rows.append(row)
        print(f"| {gate} | {auc:.3f} | {mean_ms:.0f} | {100*esc:.0f}% | "
              f"{100*row['auc_ratio_vs_7b']:.0f}% | "
              f"{100*row['latency_ratio_vs_7b']:.0f}% |")

    nli = next(r for r in rows if r["gate"] == "nli-deberta")
    mc = next(r for r in rows if r["gate"] == "minicheck")
    j15 = next(r for r in rows if r["gate"] == "llm-judge-1.5b")
    print(f"\nMiniCheck-gated AUC={mc['roc_auc']:.3f} vs NLI-gated "
          f"{nli['roc_auc']:.3f} (Δ={mc['roc_auc']-nli['roc_auc']:+.3f}); "
          f"1.5B-gated={j15['roc_auc']:.3f}.")

    return {
        "protocol": (
            "RAGTruth Summary; soft_cascade from cascade.py (uncertain "
            "tertile band, soft-keep cheap score); target=llm-judge-7b-4bit. "
            "No change to cascade.py defaults."
        ),
        "rows": rows,
        "comparison": {
            "minicheck_auc": mc["roc_auc"],
            "nli_auc": nli["roc_auc"],
            "judge_1.5b_auc": j15["roc_auc"],
            "minicheck_minus_nli": mc["roc_auc"] - nli["roc_auc"],
        },
    }


# ---------------------------------------------------------------------------
# Task 3 - ECE / Brier after cross-dataset isotonic
# ---------------------------------------------------------------------------

def run_ece() -> dict:
    rt = _load_bundle("ragtruth-summary")
    tf = _load_bundle("tofueval")
    directions = [
        ("ragtruth-summary", "tofueval", rt, tf),
        ("tofueval", "ragtruth-summary", tf, rt),
    ]
    shared = [s for s in WANTED
              if s in rt["scorers"] and s in tf["scorers"]]

    print("### Task 3 - Cross-dataset isotonic ECE / Brier\n")
    out = {"protocol": (
        "IsotonicRegression(score->P(faithful)) fit on fit_set, applied to "
        "eval_set. ECE: 10 equal-width bins. Shared scorers only."
    ), "directions": {}}

    for fit_name, eval_name, fit_b, eval_b in directions:
        key = f"{fit_name}->{eval_name}"
        print(f"#### {key}\n")
        print("| scorer | full AUC | AURC (from T1 dir) | ECE | Brier | "
              "failure mode |")
        print("|---|---:|---:|---:|---:|---|")
        rows = []
        for scorer in shared:
            y_fit = np.asarray(fit_b["binary"])
            s_fit = np.asarray(fit_b["scorers"][scorer], dtype=float)
            y_eval = np.asarray(eval_b["binary"])
            s_eval = np.asarray(eval_b["scorers"][scorer], dtype=float)

            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(s_fit, y_fit)
            p_eval = iso.predict(s_eval)

            ece = ece_score(y_eval, p_eval, n_bins=10)
            brier = float(brier_score_loss(y_eval, p_eval))
            full_auc = metrics.auc(y_eval.tolist(), s_eval.tolist())

            # Threshold from fit set for AURC (ordering quality)
            t = fit_threshold(y_fit.tolist(), s_fit.tolist())
            conf = confidence(s_eval.tolist(), t)
            aurc = risk_coverage_curve(
                y_eval.tolist(), s_eval.tolist(), t, conf)["aurc"]

            # Failure-mode heuristic: compare to median AURC/ECE in this block
            rows.append({
                "scorer": scorer,
                "full_auc": full_auc,
                "aurc": aurc,
                "ece": ece,
                "brier": brier,
            })

        # Label failure modes relative to median within the direction
        med_aurc = float(np.median([r["aurc"] for r in rows]))
        med_ece = float(np.median([r["ece"] for r in rows]))
        for r in rows:
            bad_order = r["aurc"] > med_aurc
            bad_prob = r["ece"] > med_ece
            if bad_order and not bad_prob:
                mode = "ordering uninformative (bad AURC)"
            elif not bad_order and bad_prob:
                mode = "ordering OK, probs miscalibrated (bad ECE)"
            elif bad_order and bad_prob:
                mode = "both ordering and probability calibration weak"
            else:
                mode = "relatively well-calibrated (ordering + probs)"
            r["failure_mode"] = mode
            print(f"| {r['scorer']} | {r['full_auc']:.3f} | {r['aurc']:.3f} | "
                  f"{r['ece']:.3f} | {r['brier']:.3f} | {mode} |")
        print()
        out["directions"][key] = {"rows": rows,
                                  "median_aurc": med_aurc,
                                  "median_ece": med_ece}

    return out


# ---------------------------------------------------------------------------
# Task 4 - figure
# ---------------------------------------------------------------------------

def plot_risk_coverage(rc_results, outfile="calibration.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Use RT->TF direction (primary); fall back to first available.
    key = "ragtruth-summary->tofueval"
    block = rc_results["directions"].get(key) or next(
        iter(rc_results["directions"].values()))
    rows = [r for r in block["rows"] if not r.get("note")]
    if not rows:
        rows = block["rows"]

    SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd"
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=8)

    for r in rows:
        cov = r["curve"]["coverages"]
        # Plot coverage 100%->10% left-to-right as requested
        cov_plot = [c * 100 for c in reversed(cov)]
        acc_plot = list(reversed(r["curve"]["balanced_acc"]))
        c = COLORS.get(r["scorer"], FALLBACK)
        ax.plot(cov_plot, acc_plot, color=c, linewidth=2, label=r["scorer"],
                zorder=3)
        ax.scatter([cov_plot[0], cov_plot[-1]], [acc_plot[0], acc_plot[-1]],
                   s=25, color=c, zorder=4)

    ax.set_xlabel("coverage (% most-confident kept)", color=INK_2, fontsize=9)
    ax.set_ylabel("balanced accuracy", color=INK_2, fontsize=9)
    ax.set_title(
        "Risk-coverage: do scorers know when they're wrong?\n"
        "(threshold fit on RAGTruth Summary -> eval TofuEval)",
        color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_xlim(105, 5)
    ax.margins(y=0.12)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, outfile), facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {outfile}")


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-only", action="store_true")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--skip-boot", action="store_true",
                        help="faster smoke: n_boot=100")
    args = parser.parse_args()
    n_boot = 100 if args.skip_boot else args.n_boot

    cal_path = os.path.join(ROOT, "results-calibration.json")
    if args.figure_only and os.path.exists(cal_path):
        rc = json.load(open(cal_path))
        plot_risk_coverage(rc)
        return

    rc = run_risk_coverage(n_boot=n_boot)
    json.dump(_jsonable(rc), open(cal_path, "w"), indent=2)
    print(f"wrote {cal_path}\n")

    gates = run_gate_sweep()
    gate_path = os.path.join(ROOT, "results-cascade-gates.json")
    json.dump(_jsonable(gates), open(gate_path, "w"), indent=2)
    print(f"wrote {gate_path}\n")

    ece = run_ece()
    ece_path = os.path.join(ROOT, "results-ece.json")
    json.dump(_jsonable(ece), open(ece_path, "w"), indent=2)
    print(f"wrote {ece_path}\n")

    plot_risk_coverage(rc)

    # Headline summary for Claude
    d = rc["directions"]["ragtruth-summary->tofueval"]
    print("=== HEADLINE ===")
    print(f"AURC ranking: {d['aurc_ranking']}")
    print(f"AUC ranking:  {d['auc_ranking']}")
    print(f"Match: {d['rankings_match']}")
    print(f"Gate sweep: MiniCheck {gates['comparison']['minicheck_auc']:.3f} "
          f"vs NLI {gates['comparison']['nli_auc']:.3f} vs 1.5B "
          f"{gates['comparison']['judge_1.5b_auc']:.3f}")


if __name__ == "__main__":
    main()
