"""Scorer self-knowledge: risk-coverage, regimes, transferability, ECE.

Work order #3 corrects the degenerate |s-t| confidence measure and separates
in-distribution (CV) from cross-dataset (SHIFT) regimes.

    python calibration.py
    python calibration.py --skip-boot          # n_boot=100 smoke
    python calibration.py --figure-only
    python calibration.py --task guards       # Task 1 only (legacy |s-t|)
    python calibration.py --task confidence
    python calibration.py --task regimes
    python calibration.py --task transfer
    python calibration.py --task ece
    python calibration.py --task figure
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold

import metrics
from cascade import _load_summary_judges, load_latencies, soft_cascade

ROOT = os.path.dirname(os.path.abspath(__file__))

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

COVERAGES = [i / 10 for i in range(1, 11)]  # 0.1 .. 1.0
MIN_AURC_POINTS = 6


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_bundle(name: str) -> dict:
    """Merge preds for a named corpus into {binary, scorers}."""
    if name == "ragtruth-summary":
        base = json.load(open(os.path.join(ROOT, "results-ragtruth.preds.json")))
        scorers = dict(base["scorers"])
        scale = json.load(open(os.path.join(ROOT, "results-scale.preds.json")))
        s7 = json.load(open(os.path.join(ROOT, "results-scale-7b.preds.json")))
        assert scale["binary"] == base["binary"]
        assert s7["binary"] == base["binary"]
        scorers["llm-judge-1.5b"] = scale["scorers"]["llm-judge-1.5b"]
        scorers["llm-judge-3b"] = scale["scorers"]["llm-judge-3b"]
        scorers["llm-judge-7b-4bit"] = s7["scorers"]["llm-judge-7b-4bit"]
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
        if "llm-judge" in scorers:
            scorers["llm-judge-3b"] = scorers["llm-judge"]
        ns = os.path.join(ROOT, "results-newscorers-tofueval.preds.json")
        if os.path.exists(ns):
            extra = json.load(open(ns))
            assert extra["binary"] == base["binary"]
            scorers.update(extra["scorers"])
        # Optional 1.5B / 7B-4bit from work-order #3 GPU run.
        judges = os.path.join(ROOT, "results-tofueval-judges.preds.json")
        if os.path.exists(judges):
            extra = json.load(open(judges))
            assert extra["binary"] == base["binary"], (
                "tofueval-judges binary length/order mismatch")
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


def fit_isotonic(binary, preds) -> IsotonicRegression:
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(np.asarray(preds, dtype=float), np.asarray(binary))
    return iso


# ---------------------------------------------------------------------------
# Confidence measures
# ---------------------------------------------------------------------------

def confidence_abs_threshold(preds, t) -> np.ndarray:
    """LEGACY (buggy for off-centre t): c = |s - t|, min-max on eval."""
    raw = np.abs(np.asarray(preds, dtype=float) - t)
    lo, hi = float(raw.min()), float(raw.max())
    if hi - lo < 1e-12:
        return np.zeros_like(raw)
    return (raw - lo) / (hi - lo)


def confidence_calibrated(preds, iso: IsotonicRegression) -> np.ndarray:
    """(a) c = |P(faithful) - 0.5| after isotonic fit on the fit split."""
    p = iso.predict(np.asarray(preds, dtype=float))
    return np.abs(p - 0.5)


def confidence_twosided_rank(preds, t) -> np.ndarray:
    """(b) Percentile within the same side of t (threshold-symmetric)."""
    s = np.asarray(preds, dtype=float)
    c = np.zeros(len(s), dtype=float)
    above = s >= t
    below = ~above
    if above.any():
        vals = s[above]
        # percentile of each value among those on the same side
        order = vals.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(vals) + 1) / len(vals)
        c[above] = ranks
    if below.any():
        dists = t - s[below]
        order = dists.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(dists) + 1) / len(dists)
        c[below] = ranks
    return c


# ---------------------------------------------------------------------------
# Degeneracy-aware risk-coverage
# ---------------------------------------------------------------------------

def _subset_stats(binary, preds, t, conf, coverage: float) -> dict:
    n = len(binary)
    k = max(1, int(round(n * coverage)))
    order = np.argsort(-np.asarray(conf))
    idx = order[:k]
    y = np.asarray(binary)[idx]
    s = np.asarray(preds, dtype=float)[idx]
    pred = (s >= t).astype(int)
    n_pos_true = int((y == 1).sum())
    n_neg_true = int((y == 0).sum())
    n_pos_pred = int((pred == 1).sum())
    n_neg_pred = int((pred == 0).sum())
    degenerate = (
        n_pos_pred == 0 or n_neg_pred == 0
        or n_pos_true == 0 or n_neg_true == 0
    )
    bal = None if degenerate else float(balanced_accuracy_score(y, pred))
    return {
        "coverage": coverage,
        "n": int(k),
        "n_pos_true": n_pos_true,
        "n_neg_true": n_neg_true,
        "n_pos_pred": n_pos_pred,
        "n_neg_pred": n_neg_pred,
        "degenerate": degenerate,
        "balanced_acc": bal,
    }


def risk_coverage_curve(binary, preds, t, conf, scorer="", direction="",
                        warnings: list | None = None) -> dict:
    """Risk-coverage with degeneracy guards.

    Degenerate coverage points get balanced_acc=null and are excluded from AURC.
    AURC is null if fewer than MIN_AURC_POINTS valid coverage points.
    """
    points = []
    for cov in COVERAGES:
        pt = _subset_stats(binary, preds, t, conf, cov)
        if pt["degenerate"] and warnings is not None:
            warnings.append({
                "scorer": scorer,
                "direction": direction,
                "coverage": cov,
                "n_pos_true": pt["n_pos_true"],
                "n_neg_true": pt["n_neg_true"],
                "n_pos_pred": pt["n_pos_pred"],
                "n_neg_pred": pt["n_neg_pred"],
            })
        points.append(pt)

    valid = [p for p in points if not p["degenerate"]]
    if len(valid) < MIN_AURC_POINTS:
        aurc = None
        aurc_reason = (
            f"only {len(valid)}/{len(COVERAGES)} non-degenerate coverage "
            f"points (need >={MIN_AURC_POINTS})"
        )
    else:
        covs = [p["coverage"] for p in valid]
        risks = [1.0 - p["balanced_acc"] for p in valid]
        aurc = float(np.trapezoid(risks, covs)) if hasattr(np, "trapezoid") \
            else float(np.trapz(risks, covs))
        aurc_reason = None

    # acc@50%: null if that specific point is degenerate
    pt50 = next(p for p in points if abs(p["coverage"] - 0.5) < 1e-9)
    acc50 = pt50["balanced_acc"]

    return {
        "coverages": [p["coverage"] for p in points],
        "balanced_acc": [p["balanced_acc"] for p in points],
        "points": points,
        "n_valid_points": len(valid),
        "acc_at_50": acc50,
        "aurc": aurc,
        "aurc_reason": aurc_reason,
    }


def bootstrap_rc_ci(binary, preds, t, conf_fn, n_boot=1000, seed=0):
    """Bootstrap CIs; nulls skipped in percentile (may yield null CI)."""
    y = np.asarray(binary)
    s = np.asarray(preds, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    a50s, aurcs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, sb = y[idx].tolist(), s[idx].tolist()
        conf = conf_fn(sb, t) if conf_fn.__code__.co_argcount == 2 \
            else conf_fn(sb)  # noqa: unlikely
        # Always call with (preds, t) or use a lambda wrapper from callers.
        curve = risk_coverage_curve(yb, sb, t, conf)
        if curve["acc_at_50"] is not None:
            a50s.append(curve["acc_at_50"])
        if curve["aurc"] is not None:
            aurcs.append(curve["aurc"])

    def _ci(vals):
        if len(vals) < max(10, n_boot // 20):
            return None
        return [float(np.percentile(vals, 2.5)),
                float(np.percentile(vals, 97.5))]

    return {"acc_at_50_ci": _ci(a50s), "aurc_ci": _ci(aurcs)}


def bootstrap_rc_ci_with_conf(binary, preds, t, make_conf, n_boot=1000, seed=0):
    y = np.asarray(binary)
    s = np.asarray(preds, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    a50s, aurcs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx].tolist()
        sb = s[idx].tolist()
        conf = make_conf(sb)
        curve = risk_coverage_curve(yb, sb, t, conf)
        if curve["acc_at_50"] is not None:
            a50s.append(curve["acc_at_50"])
        if curve["aurc"] is not None:
            aurcs.append(curve["aurc"])

    def _ci(vals):
        if len(vals) < max(10, n_boot // 20):
            return None
        return [float(np.percentile(vals, 2.5)),
                float(np.percentile(vals, 97.5))]

    return {"acc_at_50_ci": _ci(a50s), "aurc_ci": _ci(aurcs)}


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


def _fmt(x, nd=3):
    if x is None:
        return "null"
    return f"{x:.{nd}f}"


def _rank_rows(rows, conf_key: str):
    """Rank by AUC and by AURC under conf_key; skip null AURC for AURC rank."""
    by_auc = sorted(rows, key=lambda r: -r["full_auc"])
    with_aurc = [r for r in rows if r[conf_key]["aurc"] is not None]
    by_aurc = sorted(with_aurc, key=lambda r: r[conf_key]["aurc"])
    auc_rank = [r["scorer"] for r in by_auc]
    aurc_rank = [r["scorer"] for r in by_aurc]
    # Match only among scorers that have AURC
    auc_among = [s for s in auc_rank if s in aurc_rank]
    match = auc_among == aurc_rank and len(aurc_rank) == len(rows)
    return auc_rank, aurc_rank, match


# ---------------------------------------------------------------------------
# Task 1 — degeneracy guards on legacy |s-t| (diagnostic)
# ---------------------------------------------------------------------------

def run_guards_legacy(n_boot=1000) -> dict:
    """Apply degeneracy guards to the old |s-t| measure (SHIFT shared only)."""
    rt = _load_bundle("ragtruth-summary")
    tf = _load_bundle("tofueval")
    warnings: list = []
    results = {
        "protocol": (
            "LEGACY confidence |s-t| with degeneracy guards. SHIFT only; "
            "scorers without both corpora omitted. Degenerate coverage "
            "points -> balanced_acc=null; AURC null if <6 valid points."
        ),
        "confidence": "abs_threshold_legacy",
        "warnings": warnings,
        "directions": {},
    }
    print("### Task 1 - Degeneracy guards on legacy |s-t|\n")
    for fit_b, eval_b, key in (
        (rt, tf, "ragtruth-summary->tofueval"),
        (tf, rt, "tofueval->ragtruth-summary"),
    ):
        rows = []
        print(f"#### {key}\n")
        print("| scorer | AUC | acc@50% | AURC | n_valid | note |")
        print("|---|---:|---:|---:|---:|---|")
        for scorer in WANTED:
            if scorer not in fit_b["scorers"] or scorer not in eval_b["scorers"]:
                continue
            y_fit, s_fit = fit_b["binary"], fit_b["scorers"][scorer]
            y_eval, s_eval = eval_b["binary"], eval_b["scorers"][scorer]
            t = fit_threshold(y_fit, s_fit)
            conf = confidence_abs_threshold(s_eval, t)
            curve = risk_coverage_curve(
                y_eval, s_eval, t, conf, scorer=scorer, direction=key,
                warnings=warnings)
            cis = bootstrap_rc_ci_with_conf(
                y_eval, s_eval, t,
                lambda sb, _t=t: confidence_abs_threshold(sb, _t),
                n_boot=n_boot)
            full_auc = metrics.auc(y_eval, s_eval)
            row = {
                "scorer": scorer,
                "regime": "SHIFT",
                "direction": key,
                "threshold": t,
                "full_auc": full_auc,
                "acc_at_50": curve["acc_at_50"],
                "aurc": curve["aurc"],
                "aurc_reason": curve["aurc_reason"],
                "n_valid_points": curve["n_valid_points"],
                "acc_at_50_ci": cis["acc_at_50_ci"],
                "aurc_ci": cis["aurc_ci"],
                "curve": curve,
            }
            rows.append(row)
            print(f"| {scorer} | {full_auc:.3f} | {_fmt(curve['acc_at_50'])} | "
                  f"{_fmt(curve['aurc'])} | {curve['n_valid_points']} | "
                  f"{curve['aurc_reason'] or '-'} |")
        results["directions"][key] = {"rows": rows}
        print()
    print(f"warnings: {len(warnings)} degenerate (scorer, coverage) points\n")
    return results


# ---------------------------------------------------------------------------
# Shared eval helpers for confidence (a)/(b) under a fixed fit/eval split
# ---------------------------------------------------------------------------

def _eval_split(y_fit, s_fit, y_eval, s_eval, scorer, direction, warnings,
                n_boot, conf_names=("calibrated", "twosided_rank")):
    t = fit_threshold(y_fit, s_fit)
    iso = fit_isotonic(y_fit, s_fit)
    full_auc = metrics.auc(y_eval, s_eval)
    out = {
        "scorer": scorer,
        "direction": direction,
        "threshold": t,
        "full_auc": full_auc,
        "measures": {},
    }
    makers = {
        "calibrated": lambda sb, _iso=iso: confidence_calibrated(sb, _iso),
        "twosided_rank": lambda sb, _t=t: confidence_twosided_rank(sb, _t),
    }
    for name in conf_names:
        conf = makers[name](s_eval)
        curve = risk_coverage_curve(
            y_eval, s_eval, t, conf, scorer=scorer,
            direction=f"{direction}:{name}", warnings=warnings)
        cis = bootstrap_rc_ci_with_conf(
            y_eval, s_eval, t, makers[name], n_boot=n_boot)
        out["measures"][name] = {
            "acc_at_50": curve["acc_at_50"],
            "aurc": curve["aurc"],
            "aurc_reason": curve["aurc_reason"],
            "n_valid_points": curve["n_valid_points"],
            "acc_at_50_ci": cis["acc_at_50_ci"],
            "aurc_ci": cis["aurc_ci"],
            "curve": curve,
        }
    return out


# ---------------------------------------------------------------------------
# Task 2 — two confidence measures (SHIFT, shared scorers)
# ---------------------------------------------------------------------------

def run_confidence_measures(n_boot=1000) -> dict:
    rt = _load_bundle("ragtruth-summary")
    tf = _load_bundle("tofueval")
    warnings: list = []
    results = {
        "protocol": (
            "SHIFT: threshold + isotonic fit on fit_set, eval on other. "
            "Confidence (a) calibrated |p-0.5|; (b) two-sided rank within "
            "side of t. Degeneracy guards applied. Scorers missing either "
            "corpus omitted."
        ),
        "warnings": warnings,
        "directions": {},
    }
    print("### Task 2 - Confidence measures (a)/(b) under SHIFT\n")
    for fit_b, eval_b, key in (
        (rt, tf, "ragtruth-summary->tofueval"),
        (tf, rt, "tofueval->ragtruth-summary"),
    ):
        rows = []
        print(f"#### {key}\n")
        print("| scorer | AUC | acc@50%(a) | AURC(a) | acc@50%(b) | AURC(b) |")
        print("|---|---:|---:|---:|---:|---:|")
        for scorer in WANTED:
            if scorer not in fit_b["scorers"] or scorer not in eval_b["scorers"]:
                continue
            row = _eval_split(
                fit_b["binary"], fit_b["scorers"][scorer],
                eval_b["binary"], eval_b["scorers"][scorer],
                scorer, key, warnings, n_boot)
            row["regime"] = "SHIFT"
            rows.append(row)
            a, b = row["measures"]["calibrated"], row["measures"]["twosided_rank"]
            print(f"| {scorer} | {row['full_auc']:.3f} | "
                  f"{_fmt(a['acc_at_50'])} | {_fmt(a['aurc'])} | "
                  f"{_fmt(b['acc_at_50'])} | {_fmt(b['aurc'])} |")

        # Rankings per measure
        rank_info = {}
        for m in ("calibrated", "twosided_rank"):
            # Adapt _rank_rows shape
            fake = [{"scorer": r["scorer"], "full_auc": r["full_auc"],
                     m: r["measures"][m]} for r in rows]
            auc_r, aurc_r, match = _rank_rows(fake, m)
            rank_info[m] = {
                "auc_ranking": auc_r,
                "aurc_ranking": aurc_r,
                "rankings_match": match,
            }
            print(f"\n[{m}] AUC:  {auc_r}")
            print(f"[{m}] AURC: {aurc_r}")
            print(f"[{m}] match: {match}")
        a_rank = rank_info["calibrated"]["aurc_ranking"]
        b_rank = rank_info["twosided_rank"]["aurc_ranking"]
        print(f"\n(a) vs (b) AURC ranking agree: {a_rank == b_rank}\n")
        results["directions"][key] = {
            "rows": rows,
            "rankings": rank_info,
            "measures_aurc_agree": a_rank == b_rank,
        }
    print(f"warnings: {len(warnings)}\n")
    return results


# ---------------------------------------------------------------------------
# Task 3 — Regime IN (5-fold CV) and Regime SHIFT
# ---------------------------------------------------------------------------

def run_regime_in(bundle: dict, n_splits=5, n_boot=1000, seed=0) -> dict:
    """5-fold CV: fit t + isotonic on 4 folds, eval on held-out; concat."""
    y = np.asarray(bundle["binary"])
    warnings: list = []
    rows = []
    print(f"#### IN — {bundle['name']} ({n_splits}-fold CV)\n")
    print("| scorer | AUC | acc@50%(a) | AURC(a) | acc@50%(b) | AURC(b) |")
    print("|---|---:|---:|---:|---:|---:|")

    for scorer in WANTED:
        if scorer not in bundle["scorers"]:
            continue
        s = np.asarray(bundle["scorers"][scorer], dtype=float)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        # Collect OOF preds, fitted thresholds applied per fold, and confidences
        oof_y, oof_s, oof_t = [], [], []
        oof_conf_a, oof_conf_b = [], []
        fold_thresholds = []
        for train_idx, test_idx in skf.split(s, y):
            y_tr, s_tr = y[train_idx].tolist(), s[train_idx].tolist()
            y_te, s_te = y[test_idx], s[test_idx]
            t = fit_threshold(y_tr, s_tr)
            iso = fit_isotonic(y_tr, s_tr)
            fold_thresholds.append(t)
            oof_y.extend(y_te.tolist())
            oof_s.extend(s_te.tolist())
            # Per-example threshold from the fold that produced it
            oof_t.extend([t] * len(test_idx))
            oof_conf_a.extend(confidence_calibrated(s_te, iso).tolist())
            oof_conf_b.extend(confidence_twosided_rank(s_te, t).tolist())

        # For binary decisions use each example's fold threshold.
        # Risk-coverage needs a single t for pred labels in the subset —
        # use per-example thresholds via a custom path:
        full_auc = metrics.auc(oof_y, oof_s)
        measures = {}
        for name, conf in (("calibrated", oof_conf_a),
                           ("twosided_rank", oof_conf_b)):
            curve = _rc_with_per_example_t(
                oof_y, oof_s, oof_t, conf, scorer=scorer,
                direction=f"IN:{bundle['name']}:{name}", warnings=warnings)
            # Bootstrap with a pooled threshold (median fold t) as approximation
            t_pool = float(np.median(fold_thresholds))
            if name == "calibrated":
                # Refit iso on all data would leak; bootstrap uses fold-pooled
                # confidence recomputed with global iso on OOF only as approx.
                iso_all = fit_isotonic(oof_y, oof_s)  # OOF labels only — mild
                make = lambda sb, _iso=iso_all: confidence_calibrated(sb, _iso)
            else:
                make = lambda sb, _t=t_pool: confidence_twosided_rank(sb, _t)
            cis = bootstrap_rc_ci_with_conf(
                oof_y, oof_s, t_pool, make, n_boot=n_boot)
            measures[name] = {
                "acc_at_50": curve["acc_at_50"],
                "aurc": curve["aurc"],
                "aurc_reason": curve["aurc_reason"],
                "n_valid_points": curve["n_valid_points"],
                "acc_at_50_ci": cis["acc_at_50_ci"],
                "aurc_ci": cis["aurc_ci"],
                "curve": curve,
            }
        row = {
            "scorer": scorer,
            "regime": "IN",
            "dataset": bundle["name"],
            "full_auc": full_auc,
            "fold_thresholds": fold_thresholds,
            "measures": measures,
        }
        rows.append(row)
        a, b = measures["calibrated"], measures["twosided_rank"]
        print(f"| {scorer} | {full_auc:.3f} | "
              f"{_fmt(a['acc_at_50'])} | {_fmt(a['aurc'])} | "
              f"{_fmt(b['acc_at_50'])} | {_fmt(b['aurc'])} |")

    rank_info = {}
    for m in ("calibrated", "twosided_rank"):
        fake = [{"scorer": r["scorer"], "full_auc": r["full_auc"],
                 m: r["measures"][m]} for r in rows]
        auc_r, aurc_r, match = _rank_rows(fake, m)
        rank_info[m] = {
            "auc_ranking": auc_r,
            "aurc_ranking": aurc_r,
            "rankings_match": match,
        }
        print(f"\n[{m}] AUC:  {auc_r}")
        print(f"[{m}] AURC: {aurc_r}")
        print(f"[{m}] match: {match}")
    print()
    return {"rows": rows, "rankings": rank_info, "warnings": warnings}


def _rc_with_per_example_t(binary, preds, thresholds, conf, scorer="",
                           direction="", warnings=None) -> dict:
    """Like risk_coverage_curve but each example has its own decision threshold."""
    y = np.asarray(binary)
    s = np.asarray(preds, dtype=float)
    th = np.asarray(thresholds, dtype=float)
    conf = np.asarray(conf, dtype=float)
    points = []
    for cov in COVERAGES:
        n = len(y)
        k = max(1, int(round(n * cov)))
        idx = np.argsort(-conf)[:k]
        yy, ss, tt = y[idx], s[idx], th[idx]
        pred = (ss >= tt).astype(int)
        n_pos_true = int((yy == 1).sum())
        n_neg_true = int((yy == 0).sum())
        n_pos_pred = int((pred == 1).sum())
        n_neg_pred = int((pred == 0).sum())
        degenerate = (
            n_pos_pred == 0 or n_neg_pred == 0
            or n_pos_true == 0 or n_neg_true == 0
        )
        if degenerate and warnings is not None:
            warnings.append({
                "scorer": scorer, "direction": direction, "coverage": cov,
                "n_pos_true": n_pos_true, "n_neg_true": n_neg_true,
                "n_pos_pred": n_pos_pred, "n_neg_pred": n_neg_pred,
            })
        bal = None if degenerate else float(balanced_accuracy_score(yy, pred))
        points.append({
            "coverage": cov, "n": int(k),
            "n_pos_true": n_pos_true, "n_neg_true": n_neg_true,
            "n_pos_pred": n_pos_pred, "n_neg_pred": n_neg_pred,
            "degenerate": degenerate, "balanced_acc": bal,
        })
    valid = [p for p in points if not p["degenerate"]]
    if len(valid) < MIN_AURC_POINTS:
        aurc, reason = None, (
            f"only {len(valid)}/{len(COVERAGES)} non-degenerate coverage "
            f"points (need >={MIN_AURC_POINTS})")
    else:
        covs = [p["coverage"] for p in valid]
        risks = [1.0 - p["balanced_acc"] for p in valid]
        aurc = float(np.trapezoid(risks, covs)) if hasattr(np, "trapezoid") \
            else float(np.trapz(risks, covs))
        reason = None
    pt50 = next(p for p in points if abs(p["coverage"] - 0.5) < 1e-9)
    return {
        "coverages": [p["coverage"] for p in points],
        "balanced_acc": [p["balanced_acc"] for p in points],
        "points": points,
        "n_valid_points": len(valid),
        "acc_at_50": pt50["balanced_acc"],
        "aurc": aurc,
        "aurc_reason": reason,
    }


def run_regimes(n_boot=1000) -> dict:
    rt = _load_bundle("ragtruth-summary")
    tf = _load_bundle("tofueval")
    print("### Task 3 - Regimes IN and SHIFT\n")

    in_rt = run_regime_in(rt, n_boot=n_boot)
    # TofuEval IN for scorers that have TF preds
    in_tf = run_regime_in(tf, n_boot=n_boot)

    # SHIFT: reuse confidence-measures path (shared scorers only)
    shift = run_confidence_measures(n_boot=n_boot)

    omitted = {
        "ragtruth-summary->tofueval": [
            s for s in WANTED
            if s not in rt["scorers"] or s not in tf["scorers"]
        ],
        "tofueval->ragtruth-summary": [
            s for s in WANTED
            if s not in rt["scorers"] or s not in tf["scorers"]
        ],
    }
    print("SHIFT omitted (missing preds):", omitted)

    return {
        "protocol": (
            "IN: 5-fold stratified CV within dataset (fit t+isotonic on 4 "
            "folds, eval held-out, concat). SHIFT: fit on one corpus, eval "
            "on the other; scorers lacking either side omitted (no within-"
            "split substitution). Confidence (a)/(b) with degeneracy guards."
        ),
        "IN": {
            "ragtruth-summary": in_rt,
            "tofueval": in_tf,
        },
        "SHIFT": shift,
        "SHIFT_omitted": omitted,
        "available_tofueval_scorers": sorted(tf["scorers"].keys()),
    }


# ---------------------------------------------------------------------------
# Task 4 — threshold transferability
# ---------------------------------------------------------------------------

def run_transferability() -> dict:
    rt = _load_bundle("ragtruth-summary")
    tf = _load_bundle("tofueval")
    print("### Task 4 - Threshold transferability\n")
    print("| scorer | direction | t_A | t_B | %pos(A) | %pos(B) | "
          "balacc(A) | balacc(B) | penalty |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")

    rows = []
    for fit_b, eval_b, key in (
        (rt, tf, "ragtruth-summary->tofueval"),
        (tf, rt, "tofueval->ragtruth-summary"),
    ):
        for scorer in WANTED:
            if scorer not in fit_b["scorers"] or scorer not in eval_b["scorers"]:
                continue
            y_a = fit_b["binary"]
            s_a = fit_b["scorers"][scorer]
            y_b = np.asarray(eval_b["binary"])
            s_b = np.asarray(eval_b["scorers"][scorer], dtype=float)
            t_a = fit_threshold(y_a, s_a)
            t_b = fit_threshold(y_b.tolist(), s_b.tolist())
            pred_a = (s_b >= t_a).astype(int)
            pred_b = (s_b >= t_b).astype(int)
            pct_a = float(pred_a.mean())
            pct_b = float(pred_b.mean())
            # Balanced acc may be undefined if single class preds
            def _bal(y, pred):
                if len(set(pred.tolist())) < 2 or len(set(y.tolist())) < 2:
                    return None
                return float(balanced_accuracy_score(y, pred))
            ba_a = _bal(y_b, pred_a)
            ba_b = _bal(y_b, pred_b)
            penalty = None if (ba_a is None or ba_b is None) else ba_b - ba_a
            row = {
                "scorer": scorer,
                "direction": key,
                "t_A": t_a,
                "t_B": t_b,
                "pct_pos_under_tA": pct_a,
                "pct_pos_under_tB": pct_b,
                "bal_acc_under_tA": ba_a,
                "bal_acc_under_tB": ba_b,
                "transfer_penalty": penalty,  # how much better B's own t is
            }
            rows.append(row)
            print(f"| {scorer} | {key} | {t_a:.4f} | {t_b:.4f} | "
                  f"{100*pct_a:.1f}% | {100*pct_b:.1f}% | "
                  f"{_fmt(ba_a)} | {_fmt(ba_b)} | {_fmt(penalty)} |")
        print()

    return {
        "protocol": (
            "t_A = threshold fit on dataset A; t_B = oracle threshold on B. "
            "Apply both to B's scores. transfer_penalty = bal_acc(t_B) - "
            "bal_acc(t_A) (positive = A-threshold hurts)."
        ),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Task 5 — ECE under corrected regimes
# ---------------------------------------------------------------------------

def run_ece_corrected(n_splits=5, seed=0) -> dict:
    """ECE/Brier under IN (OOF isotonic) and SHIFT (cross-dataset isotonic)."""
    rt = _load_bundle("ragtruth-summary")
    tf = _load_bundle("tofueval")
    print("### Task 5 - ECE / Brier (corrected regimes)\n")
    out = {
        "protocol": (
            "IN: isotonic fit on CV train folds, predict on held-out; "
            "concat OOF probs -> ECE/Brier. SHIFT: isotonic fit on A, "
            "apply to B. Degeneracy not applicable to ECE. Old "
            "results-ece.json retained; this file supersedes it."
        ),
        "IN": {},
        "SHIFT": {},
    }

    for bundle in (rt, tf):
        y = np.asarray(bundle["binary"])
        rows = []
        print(f"#### IN — {bundle['name']}\n")
        print("| scorer | ECE | Brier | AUC |")
        print("|---|---:|---:|---:|")
        for scorer in WANTED:
            if scorer not in bundle["scorers"]:
                continue
            s = np.asarray(bundle["scorers"][scorer], dtype=float)
            skf = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=seed)
            oof_y, oof_p, oof_s = [], [], []
            for tr, te in skf.split(s, y):
                iso = fit_isotonic(y[tr].tolist(), s[tr].tolist())
                oof_y.extend(y[te].tolist())
                oof_p.extend(iso.predict(s[te]).tolist())
                oof_s.extend(s[te].tolist())
            ece = ece_score(oof_y, oof_p)
            brier = float(brier_score_loss(oof_y, oof_p))
            auc = metrics.auc(oof_y, oof_s)
            rows.append({
                "scorer": scorer, "ece": ece, "brier": brier, "full_auc": auc,
            })
            print(f"| {scorer} | {ece:.3f} | {brier:.3f} | {auc:.3f} |")
        print()
        out["IN"][bundle["name"]] = {"rows": rows}

    print("#### SHIFT\n")
    for fit_b, eval_b, key in (
        (rt, tf, "ragtruth-summary->tofueval"),
        (tf, rt, "tofueval->ragtruth-summary"),
    ):
        rows = []
        print(f"##### {key}\n")
        print("| scorer | ECE | Brier | AUC |")
        print("|---|---:|---:|---:|")
        for scorer in WANTED:
            if scorer not in fit_b["scorers"] or scorer not in eval_b["scorers"]:
                continue
            iso = fit_isotonic(fit_b["binary"], fit_b["scorers"][scorer])
            y_e = np.asarray(eval_b["binary"])
            s_e = np.asarray(eval_b["scorers"][scorer], dtype=float)
            p = iso.predict(s_e)
            ece = ece_score(y_e, p)
            brier = float(brier_score_loss(y_e, p))
            auc = metrics.auc(y_e.tolist(), s_e.tolist())
            rows.append({
                "scorer": scorer, "ece": ece, "brier": brier, "full_auc": auc,
            })
            print(f"| {scorer} | {ece:.3f} | {brier:.3f} | {auc:.3f} |")
        print()
        out["SHIFT"][key] = {"rows": rows}
    return out


# ---------------------------------------------------------------------------
# Cascade gate sweep (unchanged; kept for confirmation)
# ---------------------------------------------------------------------------

def run_gate_sweep() -> dict:
    binary, scorers = _load_summary_judges()
    ns = os.path.join(ROOT, "results-newscorers-ragtruth.preds.json")
    extra = json.load(open(ns))
    assert extra["binary"] == binary
    scorers.update(extra["scorers"])
    lat = load_latencies("Summary")
    for path in ("results-newscorers-ragtruth.json",):
        for r in json.load(open(os.path.join(ROOT, path))):
            lat[r["scorer"]] = r["median_latency_ms"]
    dear_key = "llm-judge-7b-4bit"
    dear = scorers[dear_key]
    gates = ("rouge-l", "nli-deberta", "minicheck", "alignscore",
             "llm-judge-1.5b")
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
        rows.append({
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
        })
    nli = next(r for r in rows if r["gate"] == "nli-deberta")
    mc = next(r for r in rows if r["gate"] == "minicheck")
    j15 = next(r for r in rows if r["gate"] == "llm-judge-1.5b")
    return {
        "protocol": (
            "RAGTruth Summary; soft_cascade from cascade.py (uncertain "
            "tertile band, soft-keep cheap score); target=llm-judge-7b-4bit. "
            "Unchanged from work order #2."
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
# Figure
# ---------------------------------------------------------------------------

def plot_risk_coverage(rc_block, outfile="calibration.png", title_extra=""):
    """Plot from a list of rows with measures.twosided_rank.curve (preferred)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd"
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=8)

    for r in rc_block:
        # Prefer twosided_rank; fall back to calibrated / legacy curve
        if "measures" in r:
            curve = r["measures"].get("twosided_rank", {}).get("curve") \
                or r["measures"].get("calibrated", {}).get("curve")
        else:
            curve = r.get("curve")
        if not curve:
            continue
        cov = curve["coverages"]
        acc = curve["balanced_acc"]
        # Drop nulls — show gaps (break the line)
        xs, ys = [], []
        segments_x, segments_y = [], []
        for c, a in zip(reversed(cov), reversed(acc)):
            if a is None:
                if xs:
                    segments_x.append(xs)
                    segments_y.append(ys)
                    xs, ys = [], []
                continue
            xs.append(c * 100)
            ys.append(a)
        if xs:
            segments_x.append(xs)
            segments_y.append(ys)
        color = COLORS.get(r["scorer"], FALLBACK)
        for i, (sx, sy) in enumerate(zip(segments_x, segments_y)):
            ax.plot(sx, sy, color=color, linewidth=2,
                    label=r["scorer"] if i == 0 else None, zorder=3)
            ax.scatter(sx, sy, s=18, color=color, zorder=4)

    ax.set_xlabel("coverage (% most-confident kept)", color=INK_2, fontsize=9)
    ax.set_ylabel("balanced accuracy", color=INK_2, fontsize=9)
    ax.set_title(
        "Risk-coverage (corrected: two-sided rank confidence)\n" + title_extra,
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
    if o is None:
        return None
    return o


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-only", action="store_true")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--skip-boot", action="store_true")
    parser.add_argument(
        "--task", default="all",
        choices=("all", "guards", "confidence", "regimes", "transfer",
                 "ece", "figure", "gates"),
        help="run a single work-order task")
    args = parser.parse_args()
    n_boot = 100 if args.skip_boot else args.n_boot
    task = args.task

    if args.figure_only or task == "figure":
        # Prefer corrected regimes file
        path = os.path.join(ROOT, "results-calibration-regimes.json")
        if not os.path.exists(path):
            path = os.path.join(ROOT, "results-calibration.json")
        data = json.load(open(path))
        rows = None
        title = ""
        if "SHIFT" in data:
            block = data["SHIFT"]["directions"].get(
                "ragtruth-summary->tofueval")
            if block:
                rows = block["rows"]
                title = "Regime SHIFT: fit RAGTruth Summary -> eval TofuEval"
        elif "directions" in data:
            block = data["directions"].get("ragtruth-summary->tofueval")
            if block:
                rows = block["rows"]
                title = "fit RAGTruth Summary -> eval TofuEval"
        if rows:
            plot_risk_coverage(rows, title_extra=title)
        return

    if task in ("all", "guards"):
        guards = run_guards_legacy(n_boot=n_boot)
        path = os.path.join(ROOT, "results-calibration-guards.json")
        json.dump(_jsonable(guards), open(path, "w"), indent=2)
        print(f"wrote {path}\n")
        # Also refresh primary results-calibration.json with guarded legacy
        # for Task 1 commit continuity (superseded later by regimes file).
        if task == "guards":
            json.dump(_jsonable(guards), open(
                os.path.join(ROOT, "results-calibration.json"), "w"), indent=2)

    if task in ("all", "confidence"):
        conf = run_confidence_measures(n_boot=n_boot)
        path = os.path.join(ROOT, "results-calibration-confidence.json")
        json.dump(_jsonable(conf), open(path, "w"), indent=2)
        print(f"wrote {path}\n")

    if task in ("all", "regimes"):
        regimes = run_regimes(n_boot=n_boot)
        path = os.path.join(ROOT, "results-calibration-regimes.json")
        json.dump(_jsonable(regimes), open(path, "w"), indent=2)
        print(f"wrote {path}\n")
        # Primary corrected calibration artifact
        json.dump(_jsonable(regimes), open(
            os.path.join(ROOT, "results-calibration.json"), "w"), indent=2)

    if task in ("all", "transfer"):
        transfer = run_transferability()
        path = os.path.join(ROOT, "results-threshold-transfer.json")
        json.dump(_jsonable(transfer), open(path, "w"), indent=2)
        print(f"wrote {path}\n")

    if task in ("all", "ece"):
        ece = run_ece_corrected()
        path = os.path.join(ROOT, "results-ece-corrected.json")
        json.dump(_jsonable(ece), open(path, "w"), indent=2)
        print(f"wrote {path}\n")

    if task in ("all", "gates"):
        gates = run_gate_sweep()
        print("Cascade confirmation: MiniCheck "
              f"{gates['comparison']['minicheck_auc']:.3f} "
              f"({100*gates['rows'][2]['auc_ratio_vs_7b']:.1f}% of 7B AUC, "
              f"{100*gates['rows'][2]['latency_ratio_vs_7b']:.0f}% latency)")

    if task == "all":
        # Figure from regimes SHIFT
        reg_path = os.path.join(ROOT, "results-calibration-regimes.json")
        if os.path.exists(reg_path):
            data = json.load(open(reg_path))
            rows = data["SHIFT"]["directions"]["ragtruth-summary->tofueval"]["rows"]
            plot_risk_coverage(
                rows,
                title_extra="Regime SHIFT: fit RAGTruth Summary -> eval TofuEval")


if __name__ == "__main__":
    main()
