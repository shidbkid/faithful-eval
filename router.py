"""Learned router: pick which detector to trust per example.

Cheap features only (lengths, ROUGE, NLI) — the judge is a *candidate*,
not a feature, so routing can skip it at inference.

Leakage guard: train on one corpus, test on another (or leave-one-task-out).

    python router.py
"""

from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import data
import metrics

ROOT = os.path.dirname(os.path.abspath(__file__))

# Preference among correct scorers: cheaper first.
CANDIDATES = ("rouge-l", "nli-deberta", "llm-judge")
CHEAP = ("rouge-l", "nli-deberta")  # always computed for features


def _load_bundle(name: str):
    """Return dict with examples, scorers, task label, binary."""
    if name == "ragtruth-summary":
        examples = data.load_ragtruth(task="Summary")
        blob = json.load(open(os.path.join(ROOT, "results-ragtruth.preds.json")))
        task = "Summary"
    elif name == "ragtruth-qa":
        examples = data.load_ragtruth(task="QA")
        blob = json.load(open(os.path.join(ROOT, "results-ragtruth-qa.preds.json")))
        task = "QA"
    elif name == "ragtruth-d2t":
        examples = data.load_ragtruth(task="Data2txt")
        blob = json.load(open(os.path.join(ROOT, "results-ragtruth-d2t.preds.json")))
        task = "Data2txt"
    elif name == "tofueval":
        examples = data.load_tofueval()
        blob = json.load(open(os.path.join(ROOT, "results-tofueval.preds.json")))
        task = "Dialogue"
    else:
        raise ValueError(name)

    assert [e["binary"] for e in examples] == blob["binary"], name
    scorers = {k: blob["scorers"][k] for k in CANDIDATES}
    return {
        "name": name,
        "task": task,
        "examples": examples,
        "scorers": scorers,
        "binary": blob["binary"],
    }


def _oracle_threshold(binary, preds) -> float:
    """Train-only threshold maximizing balanced accuracy."""
    from sklearn.metrics import balanced_accuracy_score
    best_t, best = 0.5, -1.0
    for t in sorted(set(preds)):
        acc = balanced_accuracy_score(binary, [1 if p >= t else 0 for p in preds])
        if acc > best:
            best, best_t = acc, t
    return best_t


def _features(examples, scorers, task: str) -> np.ndarray:
    """Cheap features — no judge scores."""
    task_ids = {"Summary": 0, "QA": 1, "Data2txt": 2, "Dialogue": 3}
    tid = task_ids[task]
    rows = []
    rouge = scorers["rouge-l"]
    nli = scorers["nli-deberta"]
    for i, ex in enumerate(examples):
        src_len = max(1, len(ex["source"]))
        sum_len = max(1, len(ex["summary"]))
        r, n = rouge[i], nli[i]
        rows.append([
            np.log1p(src_len),
            np.log1p(sum_len),
            sum_len / src_len,
            r,
            n,
            abs(n - 0.5),          # NLI uncertainty
            abs(r - 0.5),          # ROUGE uncertainty (weak)
            float(tid == 0),
            float(tid == 1),
            float(tid == 2),
            float(tid == 3),
        ])
    return np.asarray(rows, dtype=float)


def _route_labels(binary, scorers, thresholds) -> np.ndarray:
    """Which candidate to trust: cheapest correct under train thresholds."""
    y = []
    for i in range(len(binary)):
        gold = binary[i]
        correct = []
        for j, name in enumerate(CANDIDATES):
            pred = 1 if scorers[name][i] >= thresholds[name] else 0
            if pred == gold:
                correct.append(j)
        if correct:
            y.append(min(correct))  # cheapest among correct
        else:
            # none correct: pick judge (last) — escalate when cheap fails
            y.append(len(CANDIDATES) - 1)
    return np.asarray(y, dtype=int)


def _fit_calibration(train_bundles):
    """Z-score each candidate from TRAIN only so routed scores share a scale."""
    stats = {}
    for name in CANDIDATES:
        vals = []
        for b in train_bundles:
            vals.extend(b["scorers"][name])
        arr = np.asarray(vals, dtype=float)
        stats[name] = (float(arr.mean()), float(arr.std() + 1e-8))
    return stats


def _calibrate(scorers, stats):
    out = {}
    for name in CANDIDATES:
        mu, sd = stats[name]
        out[name] = [(p - mu) / sd for p in scorers[name]]
    return out


def _apply_route(choices, scorers) -> list[float]:
    out = []
    for i, c in enumerate(choices):
        out.append(scorers[CANDIDATES[c]][i])
    return out


def _soft_route(proba, scorers, judge_gate: float | None = None) -> tuple[list[float], list[bool]]:
    """Mixture of calibrated scores.

    If judge_gate is set, the judge score is only mixed in when
    p(judge) >= gate (and we mark the example as a judge call). Otherwise
    probability mass is renormalized over cheap scorers only.
    """
    out, called = [], []
    mats = [scorers[name] for name in CANDIDATES]
    judge_k = CANDIDATES.index("llm-judge")
    for i in range(len(mats[0])):
        p = proba[i].copy()
        use_judge = True
        if judge_gate is not None and p[judge_k] < judge_gate:
            p[judge_k] = 0.0
            s = p.sum()
            p = p / s if s > 0 else np.array([0.5, 0.5, 0.0])
            use_judge = False
        out.append(float(sum(p[k] * mats[k][i] for k in range(len(CANDIDATES)))))
        called.append(use_judge)
    return out, called


def _task_heuristic(task: str, n: int) -> np.ndarray:
    """QA→ROUGE; else→judge. Matches the deployable recipe without learning."""
    if task == "QA":
        return np.zeros(n, dtype=int)  # rouge-l
    return np.full(n, 2, dtype=int)    # llm-judge


def evaluate_split(train_names: list[str], test_name: str) -> dict:
    trains = [_load_bundle(n) for n in train_names]
    test = _load_bundle(test_name)

    # Fit thresholds on TRAIN only
    thresholds = {}
    train_binary = []
    for b in trains:
        train_binary.extend(b["binary"])
    for name in CANDIDATES:
        preds = []
        for b in trains:
            preds.extend(b["scorers"][name])
        thresholds[name] = _oracle_threshold(train_binary, preds)

    # Build train matrices
    X_parts, y_parts = [], []
    for b in trains:
        X_parts.append(_features(b["examples"], b["scorers"], b["task"]))
        y_parts.append(_route_labels(b["binary"], b["scorers"], thresholds))
    X_train = np.vstack(X_parts)
    y_train = np.concatenate(y_parts)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xs, y_train)

    cal = _fit_calibration(trains)
    # Labels use raw thresholds; emitted route scores are z-scored on train.
    test_cal = _calibrate(test["scorers"], cal)

    # Test
    X_test = scaler.transform(
        _features(test["examples"], test["scorers"], test["task"]))
    pred_choice = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    # align proba columns to class indices 0..K-1
    full_proba = np.zeros((len(pred_choice), len(CANDIDATES)))
    for col, cls in enumerate(clf.classes_):
        full_proba[:, int(cls)] = proba[:, col]

    routed = _apply_route(pred_choice, test_cal)
    # Cost-aware soft: only pay for judge when p(judge) >= 1/3.
    soft, soft_called = _soft_route(full_proba, test_cal, judge_gate=1.0 / 3)
    soft_always, _ = _soft_route(full_proba, test_cal, judge_gate=None)
    binary = test["binary"]

    heur = _task_heuristic(test["task"], len(binary))
    heur_scores = _apply_route(heur, test_cal)

    oracle_choice = _route_labels(binary, test["scorers"], thresholds)
    oracle_scores = _apply_route(oracle_choice, test_cal)

    def pack(scores, choices=None, judge_frac=None):
        row = {
            "roc_auc": metrics.auc(binary, scores),
            "roc_auc_ci": metrics.bootstrap_auc_ci(binary, scores, n_boot=1000),
            "balanced_acc": metrics.oracle_bal_acc(binary, scores),
        }
        if choices is not None:
            frac = Counter(int(c) for c in choices)
            n = len(choices)
            row["choice_frac"] = {
                CANDIDATES[k]: frac.get(k, 0) / n for k in range(len(CANDIDATES))
            }
            row["judge_frac"] = frac.get(2, 0) / n
        if judge_frac is not None:
            row["judge_frac"] = judge_frac
        return row

    singles = {
        name: pack(test["scorers"][name]) for name in CANDIDATES
    }

    return {
        "train": train_names,
        "test": test_name,
        "test_task": test["task"],
        "n_train": len(y_train),
        "n_test": len(binary),
        "train_label_frac": {
            CANDIDATES[k]: float(np.mean(y_train == k))
            for k in range(len(CANDIDATES))
        },
        "thresholds": thresholds,
        "singles": singles,
        "task_heuristic": pack(heur_scores, heur),
        "learned_router": pack(routed, pred_choice),
        "learned_soft_gated": pack(
            soft, judge_frac=float(np.mean(soft_called))),
        "learned_soft_always": pack(soft_always, judge_frac=1.0),
        "oracle_router": pack(oracle_scores, oracle_choice),
    }


def _fmt(row):
    ci = row.get("roc_auc_ci")
    ci_s = f"{ci[0]:.3f}-{ci[1]:.3f}" if ci else "n/a"
    extra = ""
    if "judge_frac" in row:
        extra = f"  judge={row['judge_frac']:.0%}"
    return f"AUC {row['roc_auc']:.3f} [{ci_s}]{extra}"


def main():
    splits = [
        (["ragtruth-summary", "ragtruth-qa", "ragtruth-d2t"], "tofueval"),
        (["tofueval"], "ragtruth-summary"),
        (["ragtruth-summary", "ragtruth-d2t"], "ragtruth-qa"),
        (["ragtruth-qa", "ragtruth-d2t"], "ragtruth-summary"),
        (["ragtruth-summary", "ragtruth-qa"], "ragtruth-d2t"),
    ]

    results = []
    print("| train → test | rouge | nli | judge | task-heur | "
          "hard | soft-gated | soft-all | oracle | judge% (hard/soft) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for train_names, test_name in splits:
        r = evaluate_split(train_names, test_name)
        results.append(r)
        train_s = "+".join(t.replace("ragtruth-", "") for t in train_names)
        test_s = test_name.replace("ragtruth-", "")
        print(
            f"| {train_s} → {test_s} | "
            f"{r['singles']['rouge-l']['roc_auc']:.3f} | "
            f"{r['singles']['nli-deberta']['roc_auc']:.3f} | "
            f"{r['singles']['llm-judge']['roc_auc']:.3f} | "
            f"{r['task_heuristic']['roc_auc']:.3f} | "
            f"{r['learned_router']['roc_auc']:.3f} | "
            f"**{r['learned_soft_gated']['roc_auc']:.3f}** | "
            f"{r['learned_soft_always']['roc_auc']:.3f} | "
            f"{r['oracle_router']['roc_auc']:.3f} | "
            f"{r['learned_router']['judge_frac']:.0%}/"
            f"{r['learned_soft_gated']['judge_frac']:.0%} |"
        )
        print(f"  hard choices: " + ", ".join(
            f"{k}={v:.0%}" for k, v in r["learned_router"]["choice_frac"].items()))

    headline = results[0]
    hard = headline["learned_router"]["roc_auc"]
    soft_g = headline["learned_soft_gated"]["roc_auc"]
    soft_a = headline["learned_soft_always"]["roc_auc"]
    judge = headline["singles"]["llm-judge"]["roc_auc"]
    heur = headline["task_heuristic"]["roc_auc"]
    qa = next(r for r in results if r["test"] == "ragtruth-qa")
    takeaway = (
        f"RAGTruth→TofuEval: hard={hard:.3f} soft-gated={soft_g:.3f} "
        f"(judge {headline['learned_soft_gated']['judge_frac']:.0%}) "
        f"soft-all={soft_a:.3f} vs judge={judge:.3f} / heuristic={heur:.3f}. "
        f"Leave-one-task QA: soft-gated={qa['learned_soft_gated']['roc_auc']:.3f} "
        f"vs rouge={qa['singles']['rouge-l']['roc_auc']:.3f}."
    )
    if soft_g >= judge - 0.015 and headline["learned_soft_gated"]["judge_frac"] < 0.85:
        takeaway += " Soft-gated nearly matches judge at lower judge rate."
    elif qa["learned_soft_gated"]["roc_auc"] > qa["singles"]["rouge-l"]["roc_auc"] + 0.015:
        takeaway += " Clearest win: QA leave-one-task beats always-ROUGE."
    else:
        takeaway += (" No clean dominate-the-tools win cross-dataset; "
                     "task heuristic remains the practical recipe; "
                     "oracle gap (~0.9) says a better router is possible.")

    out = {"results": results, "takeaway": takeaway, "candidates": CANDIDATES}
    path = os.path.join(ROOT, "results-router.json")
    # numpy-safe dump
    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, list):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o
    json.dump(conv(out), open(path, "w"), indent=2)
    print(f"\n{takeaway}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
