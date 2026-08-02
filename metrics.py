"""Shared metric helpers: bootstrap AUC CIs, oracle bal-acc, cascade, etc."""

from __future__ import annotations

import random
from typing import Iterable

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def oracle_bal_acc(binary: list[int], preds: list[float]) -> float:
    if len(set(binary)) < 2:
        return float("nan")
    return max(
        balanced_accuracy_score(binary, [1 if p >= t else 0 for p in preds])
        for t in sorted(set(preds))
    )


def auc(binary: list[int], preds: list[float]) -> float:
    if len(set(binary)) < 2:
        return float("nan")
    return float(roc_auc_score(binary, preds))


def bootstrap_auc_ci(binary: list[int], preds: list[float],
                     n_boot: int = 2000, seed: int = 0,
                     alpha: float = 0.05) -> list[float] | None:
    """Percentile bootstrap 95% CI for ROC-AUC (paired resampling)."""
    y = np.asarray(binary)
    s = np.asarray(preds, dtype=float)
    if y.min() == y.max():
        return None
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, sb = y[idx], s[idx]
        if yb.min() == yb.max():
            continue
        aucs.append(roc_auc_score(yb, sb))
    if len(aucs) < 100:
        return None
    lo = float(np.quantile(aucs, alpha / 2))
    hi = float(np.quantile(aucs, 1 - alpha / 2))
    return [lo, hi]


def cis_overlap(a: list[float] | None, b: list[float] | None) -> bool | None:
    if not a or not b:
        return None
    return not (a[1] < b[0] or b[1] < a[0])


def disagreement(binary: list[int], preds_a: list[float], preds_b: list[float],
                 name_a: str = "a", name_b: str = "b") -> dict:
    """Compare binary errors at each scorer's own oracle threshold."""
    def preds_to_hat(preds):
        best_t, best = None, -1.0
        for t in sorted(set(preds)):
            hat = [1 if p >= t else 0 for p in preds]
            b = balanced_accuracy_score(binary, hat)
            if b > best:
                best, best_t = b, t
        return [1 if p >= best_t else 0 for p in preds], best_t

    hat_a, t_a = preds_to_hat(preds_a)
    hat_b, t_b = preds_to_hat(preds_b)
    n = len(binary)
    both_wrong = both_right = a_only = b_only = 0
    for y, a, b in zip(binary, hat_a, hat_b):
        ea, eb = (a != y), (b != y)
        if ea and eb:
            both_wrong += 1
        elif (not ea) and (not eb):
            both_right += 1
        elif ea and (not eb):
            a_only += 1
        else:
            b_only += 1
    return {
        "threshold_a": t_a,
        "threshold_b": t_b,
        "both_correct": both_right,
        "both_wrong": both_wrong,
        f"only_{name_a}_wrong": a_only,
        f"only_{name_b}_wrong": b_only,
        "complementarity": (a_only + b_only) / max(1, a_only + b_only + both_wrong),
        "n": n,
    }


def cascade_score(preds_cheap: list[float], preds_mid: list[float],
                  preds_dear: list[float],
                  cheap_lo: float, cheap_hi: float,
                  mid_lo: float, mid_hi: float) -> tuple[list[float], list[str]]:
    """ROUGE -> NLI -> judge cascade.

    Outside [lo, hi] the cheaper scorer's score is kept; inside the band
    we escalate. Returns (scores, stage_used per example).
    """
    out, stage = [], []
    for c, m, d in zip(preds_cheap, preds_mid, preds_dear):
        if c <= cheap_lo or c >= cheap_hi:
            out.append(c)
            stage.append("rouge-l")
        elif m <= mid_lo or m >= mid_hi:
            out.append(m)
            stage.append("nli-deberta")
        else:
            out.append(d)
            stage.append("llm-judge")
    return out, stage


def cascade_bands(preds: list[float], low_q: float = 0.33,
                  high_q: float = 0.67) -> tuple[float, float]:
    return float(np.quantile(preds, low_q)), float(np.quantile(preds, high_q))
