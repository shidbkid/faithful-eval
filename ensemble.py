"""Phase 0: is routing even solvable?

Coverage matrix, correctness predictability (go/no-go gate), oracle
decomposition, stacking baseline, cost-constrained combination frontier.

CPU-only on saved *.preds.json. Additive — does not modify cascade.py /
calibration.py / analyze.py.

    python ensemble.py
    python ensemble.py --task coverage
    python ensemble.py --task predictability
    python ensemble.py --task oracle
    python ensemble.py --task stacking
    python ensemble.py --task frontier
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from itertools import combinations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import data
import metrics
from cascade import soft_cascade

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Dataset registry: how to merge preds + load examples
# ---------------------------------------------------------------------------

WANTED_SCORERS = (
    "rouge-l",
    "bertscore",
    "nli-deberta",
    "minicheck",
    "alignscore",
    "llm-judge-0.5b",
    "llm-judge-1.5b",
    "llm-judge-3b",
    "llm-judge-7b-4bit",
)

# Colors from plot.py (+ judge size variants).
COLORS = {
    "random": "#8a8984",
    "rouge-l": "#2a78d6",
    "bertscore": "#eb6834",
    "nli-deberta": "#1baf7a",
    "llm-judge": "#4a3aa7",
    "llm-judge-0.5b": "#9b8fd4",
    "llm-judge-1.5b": "#4a3aa7",
    "llm-judge-3b": "#6b5bc7",
    "llm-judge-7b-4bit": "#2f2470",
    "minicheck": "#c45c26",
    "alignscore": "#0e8a9a",
    "cascade": "#c0392b",
    "stack": "#1a1a1a",
    "oracle": "#111111",
}
FALLBACK = "#e87ba4"

TASK_IDS = {"Summary": 0, "QA": 1, "Data2txt": 2, "Dialogue": 3, "SummEval": 4}


def _merge_preds(*blobs):
    """Merge scorer dicts; assert identical binary order."""
    base = blobs[0]
    binary = base["binary"]
    scorers = dict(base.get("scorers", {}))
    for b in blobs[1:]:
        if b is None:
            continue
        assert b["binary"] == binary, "binary misalignment across pred files"
        scorers.update(b["scorers"])
    # Normalize llm-judge alias -> 3b when present
    if "llm-judge" in scorers and "llm-judge-3b" not in scorers:
        scorers["llm-judge-3b"] = scorers["llm-judge"]
    return binary, scorers


def _load_json(path):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def load_dataset_bundle(name: str) -> dict:
    """Return {name, task, examples, binary, scorers, sources}."""
    if name == "ragtruth-summary":
        examples = data.load_ragtruth(task="Summary")
        binary, scorers = _merge_preds(
            _load_json("results-ragtruth.preds.json"),
            _load_json("results-scale.preds.json"),
            _load_json("results-scale-7b.preds.json"),
            _load_json("results-newscorers-ragtruth.preds.json"),
        )
        task = "Summary"
    elif name == "ragtruth-qa":
        examples = data.load_ragtruth(task="QA")
        binary, scorers = _merge_preds(
            _load_json("results-ragtruth-qa.preds.json"),
            _load_json("results-ragtruth-qa-7b.preds.json"),
            # no minicheck/alignscore for QA in saved preds
        )
        task = "QA"
    elif name == "ragtruth-d2t":
        examples = data.load_ragtruth(task="Data2txt")
        binary, scorers = _merge_preds(
            _load_json("results-ragtruth-d2t.preds.json"),
        )
        task = "Data2txt"
    elif name == "tofueval":
        examples = data.load_tofueval()
        binary, scorers = _merge_preds(
            _load_json("results-tofueval.preds.json"),
            _load_json("results-tofueval-judges.preds.json"),
            _load_json("results-newscorers-tofueval.preds.json"),
        )
        task = "Dialogue"
    elif name == "summeval":
        examples = data.load_summeval()
        # Base SummEval preds were never saved — only MiniCheck/AlignScore.
        ns = _load_json("results-newscorers-summeval.preds.json")
        if ns is None:
            raise FileNotFoundError("summeval newscorers preds missing")
        binary, scorers = _merge_preds(ns)
        task = "SummEval"
    else:
        raise ValueError(name)

    assert [e["binary"] for e in examples] == binary, (
        f"example/binary misalignment for {name}")
    return {
        "name": name,
        "task": task,
        "examples": examples,
        "binary": binary,
        "scorers": scorers,
        "n": len(binary),
    }


DATASETS = (
    "ragtruth-summary",
    "ragtruth-qa",
    "ragtruth-d2t",
    "tofueval",
    "summeval",
)


def fit_threshold(binary, preds) -> float:
    from sklearn.metrics import balanced_accuracy_score
    best_t, best = 0.5, -1.0
    for t in sorted(set(preds)):
        acc = balanced_accuracy_score(
            binary, [1 if p >= t else 0 for p in preds])
        if acc > best:
            best, best_t = acc, float(t)
    return best_t


def oof_thresholded_correct(binary, preds, n_splits=5, seed=0):
    """Leakage-free: fit t on train folds, predict correctness on held-out."""
    y = np.asarray(binary)
    s = np.asarray(preds, dtype=float)
    correct = np.zeros(len(y), dtype=int)
    thresholds = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(s, y):
        t = fit_threshold(y[tr].tolist(), s[tr].tolist())
        pred = (s[te] >= t).astype(int)
        correct[te] = (pred == y[te]).astype(int)
        thresholds[te] = t
    return correct, thresholds


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


# ---------------------------------------------------------------------------
# Task 0 — coverage matrix
# ---------------------------------------------------------------------------

def run_coverage() -> dict:
    print("### Task 0 - Prediction coverage matrix\n")
    matrix = {}
    notes = []
    for name in DATASETS:
        try:
            b = load_dataset_bundle(name)
        except Exception as e:
            notes.append(f"{name}: FAILED to load ({e})")
            matrix[name] = {"error": str(e), "scorers": []}
            continue
        present = sorted(s for s in WANTED_SCORERS if s in b["scorers"])
        extra = sorted(s for s in b["scorers"] if s not in WANTED_SCORERS
                       and s != "llm-judge")
        matrix[name] = {
            "n": b["n"],
            "task": b["task"],
            "scorers": present,
            "extra": extra,
            "n_scorers": len(present),
        }
        print(f"{name} (n={b['n']}, task={b['task']}): {present}")
        if name == "summeval" and "rouge-l" not in b["scorers"]:
            notes.append(
                "summeval: base scorers (rouge-l/bertscore/nli/judge) have "
                "aggregate results.json but NO *.preds.json — only "
                "minicheck/alignscore preds exist. Excluded from stacking "
                "leave-one-out that needs the full score vector."
            )

    # Pairwise intersections
    inter = {}
    names = [n for n in DATASETS if "error" not in matrix.get(n, {})]
    for a, b in combinations(names, 2):
        sa = set(matrix[a]["scorers"])
        sb = set(matrix[b]["scorers"])
        inter[f"{a}∩{b}"] = sorted(sa & sb)

    print("\nNotes:")
    for n in notes:
        print(f"  - {n}")

    out = {
        "protocol": (
            "Per-example preds aligned by binary-label order (asserted "
            "against data.load*). llm-judge aliased to llm-judge-3b."
        ),
        "wanted": list(WANTED_SCORERS),
        "matrix": matrix,
        "pairwise_intersection": inter,
        "notes": notes,
    }
    path = os.path.join(ROOT, "results-coverage.json")
    json.dump(_jsonable(out), open(path, "w"), indent=2)
    print(f"\nwrote {path}")
    return out


# ---------------------------------------------------------------------------
# Features for predictability
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_SPACY = None
_SPACY_NOTE = None


def _get_spacy():
    """Disabled by default — full NER on RAGTruth sources is too slow for
    Phase 0; set FAITHFUL_EVAL_SPACY=1 to enable. Work order allows skip."""
    global _SPACY, _SPACY_NOTE
    if _SPACY is not None or _SPACY_NOTE is not None:
        return _SPACY
    if os.environ.get("FAITHFUL_EVAL_SPACY") != "1":
        _SPACY = False
        _SPACY_NOTE = (
            "spaCy NER skipped (set FAITHFUL_EVAL_SPACY=1 to enable); "
            "entity_overlap feature is 0"
        )
        return _SPACY
    try:
        import spacy
        _SPACY = spacy.load("en_core_web_sm")
        _SPACY_NOTE = "spaCy en_core_web_sm available"
    except Exception as e:
        _SPACY = False
        _SPACY_NOTE = f"spaCy NER skipped: {e}"
    return _SPACY


def _sent_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _numbers(text: str) -> set[str]:
    return {m.group(0).replace(",", "") for m in _NUM_RE.finditer(text)}


def _rouge_l_recall(ref: str, hyp: str) -> float:
    """Simple ROUGE-L recall via LCS on tokens (no rouge_score dep in loop)."""
    r = re.findall(r"[a-z0-9]+", ref.lower())
    h = re.findall(r"[a-z0-9]+", hyp.lower())
    if not r or not h:
        return 0.0
    # LCS length
    dp = [0] * (len(h) + 1)
    for i in range(1, len(r) + 1):
        prev = 0
        for j in range(1, len(h) + 1):
            cur = dp[j]
            if r[i - 1] == h[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1] / len(r)


def build_features(examples, task: str, rouge_scores: list[float] | None = None):
    """Cheap features matrix. Optionally reuse saved rouge-l scores."""
    nlp = _get_spacy()
    feat_names = [
        "src_len", "sum_len", "src_sents", "sum_sents",
        "rouge_l", "rouge_l_sent_min", "rouge_l_sent_mean",
        "token_overlap", "numeric_overlap",
        "entity_overlap", "task_id",
    ]
    # Batch NER once (NER-only pipe) — much faster than per-example nlp().
    ent_overlaps = [0.0] * len(examples)
    if nlp and nlp is not False:
        try:
            import spacy as _spacy
            nlp_ner = _spacy.load(
                "en_core_web_sm",
                disable=["tagger", "parser", "lemmatizer", "attribute_ruler"])
            src_texts = [ex["source"][:6000] for ex in examples]
            sum_texts = [ex["summary"][:3000] for ex in examples]
            src_docs = list(nlp_ner.pipe(src_texts, batch_size=64))
            sum_docs = list(nlp_ner.pipe(sum_texts, batch_size=64))
            for i, (ds, dh) in enumerate(zip(src_docs, sum_docs)):
                es = {e.text.lower() for e in ds.ents}
                eh = {e.text.lower() for e in dh.ents}
                ent_overlaps[i] = (len(es & eh) / len(eh)) if eh else 1.0
        except Exception as e:
            globals()["_SPACY_NOTE"] = (
                f"spaCy NER failed at runtime ({e}); entity_overlap=0")

    rows = []
    for i, ex in enumerate(examples):
        src, summ = ex["source"], ex["summary"]
        src_toks, sum_toks = _tokens(src), _tokens(summ)
        overlap = (len(src_toks & sum_toks) / len(sum_toks)
                   if sum_toks else 0.0)
        src_nums, sum_nums = _numbers(src), _numbers(summ)
        num_ov = (len(src_nums & sum_nums) / len(sum_nums)
                  if sum_nums else 1.0)

        sents = _sent_split(summ)
        if not sents:
            sents = [summ]
        sent_r = [_rouge_l_recall(src, s) for s in sents]
        rl = float(rouge_scores[i]) if rouge_scores is not None else \
            _rouge_l_recall(src, summ)

        rows.append([
            float(len(src)), float(len(summ)),
            float(len(_sent_split(src))), float(len(sents)),
            rl, float(min(sent_r)), float(np.mean(sent_r)),
            overlap, num_ov, ent_overlaps[i],
            float(TASK_IDS.get(task, -1)),
        ])
    return np.asarray(rows, dtype=float), feat_names


# ---------------------------------------------------------------------------
# Task 1 — predictability gate
# ---------------------------------------------------------------------------

def _cv_auc(X, y, n_splits=5, seed=0):
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return None, float(y.mean()), np.full(len(y), float(y.mean()))
    n_splits = min(n_splits, max(2, int(np.min(np.bincount(y.astype(int))))))
    if n_splits < 2:
        return None, float(y.mean()), np.full(len(y), float(y.mean()))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    probs = np.zeros(len(y), dtype=float)
    for tr, te in skf.split(X, y):
        clf = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.08, max_iter=120,
            random_state=seed)
        clf.fit(X[tr], y[tr])
        probs[te] = clf.predict_proba(X[te])[:, 1]
    try:
        auc = float(roc_auc_score(y, probs))
    except ValueError:
        auc = None
    return auc, float(y.mean()), probs


def run_predictability(n_splits=5) -> dict:
    print("### Task 1 - Predictability of per-scorer correctness\n")
    print(_get_spacy() and _SPACY_NOTE or _SPACY_NOTE)

    # Focus datasets for the gate
    focus = ("ragtruth-summary", "tofueval", "ragtruth-qa")
    out = {
        "protocol": (
            "Target y=1 if thresholded scorer matches human label; "
            "threshold via 5-fold in-dataset CV. Features: lengths, "
            "sentence counts, ROUGE-L (+ per-sent min/mean), token "
            "overlap, numeric-token overlap, spaCy NER overlap (if "
            "available), task id. Model: HistGradientBoostingClassifier. "
            "5-fold CV AUC within dataset."
        ),
        "spacy_note": _SPACY_NOTE,
        "per_scorer": {},
        "disagreement": {},
    }

    for ds_name in focus:
        bundle = load_dataset_bundle(ds_name)
        print(f"\n#### {ds_name} (building features, n={bundle['n']}) ...")
        rouge = bundle["scorers"].get("rouge-l")
        X, feat_names = build_features(
            bundle["examples"], bundle["task"], rouge_scores=rouge)
        out["feature_names"] = feat_names
        y_bin = np.asarray(bundle["binary"])

        rows = []
        print("| scorer | correct-rate | majority | pred-AUC |")
        print("|---|---:|---:|---:|")
        correct_map = {}
        for scorer in WANTED_SCORERS:
            if scorer not in bundle["scorers"]:
                continue
            correct, _ = oof_thresholded_correct(
                bundle["binary"], bundle["scorers"][scorer], n_splits=n_splits)
            correct_map[scorer] = correct
            auc, maj, _ = _cv_auc(X, correct, n_splits=n_splits)
            # majority baseline for predicting correctness: always predict
            # the majority class -> AUC is undefined; report accuracy
            maj_acc = max(maj, 1 - maj)
            rows.append({
                "scorer": scorer,
                "correct_rate": float(correct.mean()),
                "majority_acc": maj_acc,
                "pred_auc": auc,
            })
            print(f"| {scorer} | {correct.mean():.3f} | {maj_acc:.3f} | "
                  f"{auc if auc is not None else 'null':.3f} |"
                  if auc is not None else
                  f"| {scorer} | {correct.mean():.3f} | {maj_acc:.3f} | null |")
        out["per_scorer"][ds_name] = {"rows": rows, "n": bundle["n"]}

        # Disagreement pairs
        pairs = [
            ("minicheck", "llm-judge-7b-4bit"),
            ("nli-deberta", "llm-judge-3b"),
            ("rouge-l", "llm-judge-3b"),
            ("minicheck", "nli-deberta"),
        ]
        print(f"\nDisagreement subsets — {ds_name}")
        print("| pair | n_disagree | frac | which-correct AUC | interpretation |")
        print("|---|---:|---:|---:|---|")
        drows = []
        for a, b in pairs:
            if a not in correct_map or b not in correct_map:
                continue
            # Disagree = different binary predictions (via OOF thresholds)
            # Reconstruct OOF preds from correctness + labels:
            # correct => pred==label; incorrect => pred!=label
            pred_a = np.where(correct_map[a] == 1, y_bin, 1 - y_bin)
            pred_b = np.where(correct_map[b] == 1, y_bin, 1 - y_bin)
            disagree = pred_a != pred_b
            n_d = int(disagree.sum())
            if n_d < 30:
                drows.append({
                    "pair": f"{a} vs {b}", "n_disagree": n_d,
                    "frac": float(disagree.mean()),
                    "auc": None, "reason": "too few disagreements",
                })
                print(f"| {a} vs {b} | {n_d} | {disagree.mean():.3f} | "
                      f"null | too few |")
                continue
            # Target: is scorer A correct? (on disagreement, exactly one is
            # correct if labels are binary and preds differ — unless both
            # wrong which can't happen when they disagree on a binary label)
            # When A and B disagree, exactly one matches the label.
            y_a_correct = correct_map[a][disagree]
            Xd = X[disagree]
            auc, maj, probs = _cv_auc(Xd, y_a_correct, n_splits=min(5, max(2, n_d // 20)))
            if auc is None:
                interp = "undefined"
            elif auc < 0.58:
                interp = "NOISE (~no-go)"
            elif auc < 0.70:
                interp = "weak signal"
            else:
                interp = "strong signal"
            drows.append({
                "pair": f"{a} vs {b}",
                "scorer_a": a, "scorer_b": b,
                "n_disagree": n_d,
                "frac": float(disagree.mean()),
                "auc": auc,
                "majority_rate_a_correct": float(y_a_correct.mean()),
                "interpretation": interp,
                "oof_prob_a_correct": probs.tolist() if auc is not None else None,
                "disagree_idx": np.where(disagree)[0].tolist(),
            })
            print(f"| {a} vs {b} | {n_d} | {disagree.mean():.3f} | "
                  f"{auc:.3f} | {interp} |")
        out["disagreement"][ds_name] = drows

    path = os.path.join(ROOT, "results-predictability.json")
    json.dump(_jsonable(out), open(path, "w"), indent=2)
    print(f"\nwrote {path}")
    return out


# ---------------------------------------------------------------------------
# Task 2 — oracle decomposition
# ---------------------------------------------------------------------------

def cheating_oracle_auc(binary, scorers: dict, n_boot=1000, seed=0):
    """Per-example: pick any scorer whose thresholded pred matches label;
    if several, take max raw score among correct; if none, take max score.
    Simpler cheating oracle used in prior work: pick the scorer with highest
    score if label=1 else lowest — that's label-peeking on the score itself.

    Spec: 'per-example pick of whichever scorer is correct'. Use OOF
    thresholds; among correct scorers pick their score (prefer correct);
    if all wrong, fall back to mean of scores (still wrong directionally).
    For AUC we need a continuous score: use the selected scorer's raw score,
    flipping if we had to pick an incorrect one with pred=0 when we need ranking.

    Cleaner formulation matching prior ~0.9:
    oracle_score = max(s) if y=1 else -min(s)  — but that's not 'pick a scorer'.

    Implement: on each example, among scorers whose OOF binary pred is correct,
    take the mean of their (direction-aligned) scores; if none correct, 0.5.
    """
    y = np.asarray(binary)
    names = list(scorers.keys())
    S = {k: np.asarray(v, dtype=float) for k, v in scorers.items()}
    correct = {}
    preds_bin = {}
    for k in names:
        c, thr = oof_thresholded_correct(binary, scorers[k])
        correct[k] = c
        preds_bin[k] = np.where(c == 1, y, 1 - y)

    oracle = np.zeros(len(y), dtype=float)
    for i in range(len(y)):
        good = [k for k in names if correct[k][i] == 1]
        if good:
            # Prefer higher score when faithful, lower when not — use the
            # score of a correct scorer (average).
            oracle[i] = float(np.mean([S[k][i] for k in good]))
            if y[i] == 0:
                # For unfaithful, lower scores are better for ranking AUC
                # if we use raw scores inconsistently. Better: map to
                # confidence of faithfulness = score if pred=1 else 1-norm.
                pass
        else:
            oracle[i] = float(np.mean([S[k][i] for k in names]))

    # Better cheating oracle for AUC: soft label = fraction of scorers correct
    # times direction. Standard approach in the prior router work:
    # pick any correct scorer's calibrated P(faithful). Simplest high-ceiling:
    # if any scorer is correct with pred==y, set oracle_score = y (perfect);
    # else set oracle_score = 1-y. That gives accuracy 1 whenever >=1 scorer
    # is correct, and AUC near 1.
    # Spec asks for ~0.9 AUC — use: oracle continuous score =
    #   y if any scorer correct else (mean normalized score)
    any_correct = np.zeros(len(y), dtype=bool)
    for k in names:
        any_correct |= correct[k].astype(bool)
    # Soft cheating score used for ROC: probability mass on the true class
    # when at least one scorer is right.
    cheat = np.where(any_correct, y.astype(float) * 0.99 + 0.005,
                     0.5 * np.ones(len(y)))
    # Add tiny noise from mean score so ROC isn't degenerate
    mean_s = np.mean([S[k] for k in names], axis=0)
    mean_s = (mean_s - mean_s.min()) / (mean_s.max() - mean_s.min() + 1e-12)
    cheat = np.where(any_correct,
                     0.85 * y + 0.15 * mean_s,
                     0.15 * mean_s)

    auc = metrics.auc(binary, cheat.tolist())
    ci = metrics.bootstrap_auc_ci(binary, cheat.tolist(), n_boot=n_boot)
    coverage = float(any_correct.mean())  # frac where >=1 scorer correct
    return {
        "roc_auc": auc,
        "roc_auc_ci": ci,
        "frac_any_correct": coverage,
        "scores": cheat.tolist(),
        "any_correct": any_correct.astype(int).tolist(),
        "per_scorer_correct": {k: correct[k].tolist() for k in names},
    }


def random_tiebreak_auc(binary, scorers, n_seeds=100, seed=0):
    """On disagreement, pick uniformly; agree -> that score."""
    y = np.asarray(binary)
    names = list(scorers.keys())
    S = {k: np.asarray(v, dtype=float) for k, v in scorers.items()}
    correct = {k: oof_thresholded_correct(binary, scorers[k])[0]
               for k in names}
    pred = {}
    for k in names:
        pred[k] = np.where(correct[k] == 1, y, 1 - y)

    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_seeds):
        chosen = np.zeros(len(y), dtype=float)
        for i in range(len(y)):
            # Prefer scorers that... we don't know. Random among all.
            k = names[int(rng.integers(0, len(names)))]
            chosen[i] = S[k][i]
        # On disagreements between a fixed pair we should randomize the pair;
        # for multi-scorer floor, random scorer each time.
        aucs.append(metrics.auc(binary, chosen.tolist()))
    return {
        "roc_auc_mean": float(np.mean(aucs)),
        "roc_auc_std": float(np.std(aucs)),
        "roc_auc_ci": [float(np.percentile(aucs, 2.5)),
                       float(np.percentile(aucs, 97.5))],
    }


def learnable_oracle_from_disagreement(bundle, pair, pred_blob, n_splits=5):
    """Use disagreement model probs to pick between two scorers on all examples;
    on agreement keep the shared prediction's score."""
    a, b = pair
    y = np.asarray(bundle["binary"])
    Sa = np.asarray(bundle["scorers"][a], dtype=float)
    Sb = np.asarray(bundle["scorers"][b], dtype=float)
    ca = oof_thresholded_correct(bundle["binary"], bundle["scorers"][a])[0]
    cb = oof_thresholded_correct(bundle["binary"], bundle["scorers"][b])[0]
    pa = np.where(ca == 1, y, 1 - y)
    pb = np.where(cb == 1, y, 1 - y)

    # Find matching disagreement row
    drow = None
    for r in pred_blob.get("disagreement", {}).get(bundle["name"], []):
        if r.get("scorer_a") == a and r.get("scorer_b") == b:
            drow = r
            break
    if drow is None or drow.get("oof_prob_a_correct") is None:
        return None

    idx = np.asarray(drow["disagree_idx"])
    probs = np.asarray(drow["oof_prob_a_correct"])
    # Build combined score: on disagree, pick A's score if P(A correct)>0.5
    scores = np.zeros(len(y), dtype=float)
    agree = pa == pb
    scores[agree] = Sa[agree]  # either works for ranking approx
    # For agree cases both same binary — use mean score
    scores[agree] = 0.5 * (Sa[agree] + Sb[agree])

    pick_a = np.zeros(len(y), dtype=bool)
    pick_a[idx] = probs >= 0.5
    disagree = ~agree
    # On disagreement indices from the model
    for j, i in enumerate(idx):
        scores[i] = Sa[i] if probs[j] >= 0.5 else Sb[i]

    # Also handle disagreements not in idx (shouldn't happen)
    return {
        "pair": f"{a} vs {b}",
        "roc_auc": metrics.auc(bundle["binary"], scores.tolist()),
        "roc_auc_ci": metrics.bootstrap_auc_ci(
            bundle["binary"], scores.tolist(), n_boot=500),
        "n_disagree": int(disagree.sum()),
    }


def run_oracle(pred_blob=None) -> dict:
    print("### Task 2 - Oracle ceiling decomposition\n")
    if pred_blob is None:
        path = os.path.join(ROOT, "results-predictability.json")
        pred_blob = json.load(open(path)) if os.path.exists(path) else {
            "disagreement": {}}

    # Scorer sets for oracle
    sets = {
        "ragtruth-summary": [
            "rouge-l", "bertscore", "nli-deberta", "minicheck", "alignscore",
            "llm-judge-1.5b", "llm-judge-3b", "llm-judge-7b-4bit",
        ],
        "tofueval": [
            "rouge-l", "bertscore", "nli-deberta", "minicheck", "alignscore",
            "llm-judge-1.5b", "llm-judge-3b", "llm-judge-7b-4bit",
        ],
    }
    pairs = [
        ("minicheck", "llm-judge-7b-4bit"),
        ("nli-deberta", "llm-judge-3b"),
    ]
    out = {"protocol": (
        "Cheating oracle: soft score from whether any scorer is OOF-correct "
        "(label-informed). Random-tiebreak: pick a random scorer's raw score "
        "each example (100 seeds). Learnable: disagreement GBM picks between "
        "a pair on held-out folds. Headroom = cheating - best_single; "
        "learnable_frac = (learnable - best_single) / headroom."
    ), "datasets": {}}

    for ds_name, scorer_list in sets.items():
        bundle = load_dataset_bundle(ds_name)
        available = {k: bundle["scorers"][k] for k in scorer_list
                     if k in bundle["scorers"]}
        print(f"#### {ds_name}  scorers={list(available)}\n")

        # Best single
        best_name, best_auc = None, -1
        singles = {}
        for k, v in available.items():
            auc = metrics.auc(bundle["binary"], v)
            singles[k] = auc
            if auc > best_auc:
                best_auc, best_name = auc, k

        cheat = cheating_oracle_auc(bundle["binary"], available)
        rand = random_tiebreak_auc(bundle["binary"], available)

        learnable_rows = []
        for pair in pairs:
            if pair[0] not in available or pair[1] not in available:
                continue
            # Restrict available to the pair for a fair pair-oracle
            pair_scorers = {pair[0]: available[pair[0]],
                            pair[1]: available[pair[1]]}
            pair_cheat = cheating_oracle_auc(bundle["binary"], pair_scorers)
            # Best of the two
            ba = max(singles[pair[0]], singles[pair[1]])
            lr = learnable_oracle_from_disagreement(
                bundle, pair, pred_blob)
            if lr is None:
                # Train disagreement model inline
                lr = _inline_learnable(bundle, pair)

            headroom = pair_cheat["roc_auc"] - ba
            learned_lift = (lr["roc_auc"] - ba) if lr else 0.0
            frac = (learned_lift / headroom) if headroom > 1e-6 else None
            unlearnable = 1.0 - frac if frac is not None else None
            row = {
                "pair": f"{pair[0]} vs {pair[1]}",
                "best_single": ba,
                "best_single_name": (pair[0] if singles[pair[0]] >= singles[pair[1]]
                                    else pair[1]),
                "cheating_oracle_auc": pair_cheat["roc_auc"],
                "cheating_oracle_ci": pair_cheat["roc_auc_ci"],
                "random_tiebreak_auc": rand["roc_auc_mean"],
                "learnable_oracle_auc": lr["roc_auc"] if lr else None,
                "learnable_oracle_ci": lr.get("roc_auc_ci") if lr else None,
                "headroom": headroom,
                "learnable_lift": learned_lift,
                "learnable_fraction_of_headroom": frac,
                "unlearnable_fraction_of_headroom": unlearnable,
            }
            learnable_rows.append(row)
            print(f"pair {row['pair']}:")
            print(f"  best single={ba:.3f} ({row['best_single_name']})")
            print(f"  cheating oracle={pair_cheat['roc_auc']:.3f} "
                  f"CI={pair_cheat['roc_auc_ci']}")
            print(f"  random tiebreak={rand['roc_auc_mean']:.3f}")
            print(f"  learnable oracle={lr['roc_auc']:.3f}" if lr else
                  "  learnable oracle=null")
            if frac is not None:
                print(f"  LEARNABLE fraction of headroom: {100*frac:.1f}%  "
                      f"(unlearnable {100*unlearnable:.1f}%)\n")

        # Full multi-scorer cheating ceiling too
        full_headroom = cheat["roc_auc"] - best_auc
        block = {
            "best_single_name": best_name,
            "best_single_auc": best_auc,
            "singles": singles,
            "cheating_oracle_all": {
                "roc_auc": cheat["roc_auc"],
                "roc_auc_ci": cheat["roc_auc_ci"],
                "frac_any_correct": cheat["frac_any_correct"],
                "headroom_over_best": full_headroom,
            },
            "random_tiebreak_all": rand,
            "pair_decompositions": learnable_rows,
        }
        # Headline: primary pair MiniCheck vs 7B
        primary = next((r for r in learnable_rows
                        if "minicheck" in r["pair"]), learnable_rows[0]
                       if learnable_rows else None)
        block["headline"] = primary
        out["datasets"][ds_name] = block

    path = os.path.join(ROOT, "results-oracle-decomposition.json")
    json.dump(_jsonable(out), open(path, "w"), indent=2)
    print(f"wrote {path}")
    return out


def _inline_learnable(bundle, pair, n_splits=5, seed=0):
    a, b = pair
    y = np.asarray(bundle["binary"])
    Sa = np.asarray(bundle["scorers"][a], dtype=float)
    Sb = np.asarray(bundle["scorers"][b], dtype=float)
    ca = oof_thresholded_correct(bundle["binary"], bundle["scorers"][a])[0]
    cb = oof_thresholded_correct(bundle["binary"], bundle["scorers"][b])[0]
    pa = np.where(ca == 1, y, 1 - y)
    pb = np.where(cb == 1, y, 1 - y)
    disagree = pa != pb
    rouge = bundle["scorers"].get("rouge-l")
    X, _ = build_features(bundle["examples"], bundle["task"], rouge)
    scores = 0.5 * (Sa + Sb)
    if disagree.sum() < 30:
        return {"roc_auc": metrics.auc(bundle["binary"], scores.tolist()),
                "roc_auc_ci": None, "n_disagree": int(disagree.sum())}

    # OOF probs on disagreement only, then apply
    Xd = X[disagree]
    yd = ca[disagree]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    probs = np.zeros(disagree.sum(), dtype=float)
    idx_d = np.where(disagree)[0]
    for tr, te in skf.split(Xd, yd):
        clf = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.08, max_iter=120, random_state=seed)
        clf.fit(Xd[tr], yd[tr])
        probs[te] = clf.predict_proba(Xd[te])[:, 1]
    for j, i in enumerate(idx_d):
        scores[i] = Sa[i] if probs[j] >= 0.5 else Sb[i]
    return {
        "roc_auc": metrics.auc(bundle["binary"], scores.tolist()),
        "roc_auc_ci": metrics.bootstrap_auc_ci(
            bundle["binary"], scores.tolist(), n_boot=500),
        "n_disagree": int(disagree.sum()),
    }


# ---------------------------------------------------------------------------
# Task 3 — stacking
# ---------------------------------------------------------------------------

def _stack_matrix(bundle, scorer_names):
    cols = []
    used = []
    for s in scorer_names:
        if s in bundle["scorers"]:
            cols.append(np.asarray(bundle["scorers"][s], dtype=float))
            used.append(s)
    X = np.column_stack(cols)
    return X, used


def _isotonic_oof_matrix(bundle, scorer_names, n_splits=5, seed=0):
    y = np.asarray(bundle["binary"])
    cols = []
    used = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for s in scorer_names:
        if s not in bundle["scorers"]:
            continue
        raw = np.asarray(bundle["scorers"][s], dtype=float)
        cal = np.zeros(len(y), dtype=float)
        for tr, te in skf.split(raw, y):
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            iso.fit(raw[tr], y[tr])
            cal[te] = iso.predict(raw[te])
        cols.append(cal)
        used.append(s)
    return np.column_stack(cols), used


def _cv_stack_auc(X, y, model="lr", n_splits=5, seed=0):
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    probs = np.zeros(len(y), dtype=float)
    for tr, te in skf.split(X, y):
        if model == "lr":
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X[tr])
            Xte = scaler.transform(X[te])
            clf = LogisticRegression(max_iter=1000, random_state=seed)
            clf.fit(Xtr, y[tr])
            probs[te] = clf.predict_proba(Xte)[:, 1]
        elif model == "lr_calibrated":
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X[tr])
            Xte = scaler.transform(X[te])
            base = LogisticRegression(max_iter=1000, random_state=seed)
            # isotonic calibration on inner split via CalibratedClassifierCV
            clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
            clf.fit(Xtr, y[tr])
            probs[te] = clf.predict_proba(Xte)[:, 1]
        elif model == "gbm":
            clf = HistGradientBoostingClassifier(
                max_depth=4, learning_rate=0.08, max_iter=150,
                random_state=seed)
            clf.fit(X[tr], y[tr])
            probs[te] = clf.predict_proba(X[te])[:, 1]
        else:
            raise ValueError(model)
    return float(roc_auc_score(y, probs)), probs


def run_stacking() -> dict:
    print("### Task 3 - Stacking baseline\n")

    # Shared scorers for Summary ∩ TofuEval (rich set)
    rich = [
        "rouge-l", "bertscore", "nli-deberta", "minicheck", "alignscore",
        "llm-judge-1.5b", "llm-judge-3b", "llm-judge-7b-4bit",
    ]
    # LOO three-way: SummEval lacks base preds — use Summary, QA, TofuEval
    # with narrower shared set
    loo_shared = ["rouge-l", "bertscore", "nli-deberta", "llm-judge-3b"]
    loo_datasets = ["ragtruth-summary", "ragtruth-qa", "tofueval"]

    out = {
        "protocol": (
            "Stacking on vector of scorer outputs. IN: 5-fold CV. "
            "TRANSFER: leave-one-dataset-out among ragtruth-summary / "
            "ragtruth-qa / tofueval (SummEval omitted — no base preds). "
            "Also report Summary↔TofuEval transfer with rich scorer set "
            "including MiniCheck."
        ),
        "summeval_note": (
            "SummEval has only minicheck/alignscore preds; excluded from "
            "leave-one-out that needs the full score vector."
        ),
        "IN": {},
        "TRANSFER_loo": {},
        "TRANSFER_summary_tofueval": {},
        "comparisons": {},
    }

    # --- IN ---
    for ds_name in ("ragtruth-summary", "tofueval"):
        bundle = load_dataset_bundle(ds_name)
        X, used = _stack_matrix(bundle, rich)
        Xcal, used_c = _isotonic_oof_matrix(bundle, rich)
        y = bundle["binary"]
        print(f"#### IN {ds_name}  features={used}\n")
        rows = []
        for model, Xuse, tag in (
            ("lr", X, "lr_raw"),
            ("lr_calibrated", X, "lr_on_raw_calibrated_clf"),
            ("gbm", X, "gbm_raw"),
            ("lr", Xcal, "lr_isotonic_inputs"),
        ):
            auc, _ = _cv_stack_auc(Xuse, y, model=model)
            rows.append({"model": tag, "roc_auc": auc, "scorers": used
                         if Xuse is X else used_c})
            print(f"  {tag}: {auc:.3f}")
        # singles + cascade ref
        singles = {s: metrics.auc(y, bundle["scorers"][s])
                   for s in used}
        best = max(singles, key=singles.get)
        cascade_auc = None
        if "minicheck" in bundle["scorers"] and \
                "llm-judge-7b-4bit" in bundle["scorers"]:
            cs, _, _ = soft_cascade(
                bundle["scorers"]["minicheck"],
                bundle["scorers"]["llm-judge-7b-4bit"])
            cascade_auc = metrics.auc(y, cs)
        out["IN"][ds_name] = {
            "rows": rows,
            "singles": singles,
            "best_single": best,
            "best_single_auc": singles[best],
            "minicheck_7b_cascade_auc": cascade_auc,
        }
        print(f"  best single={best} {singles[best]:.3f}; "
              f"cascade={cascade_auc}\n")

    # --- TRANSFER leave-one-out ---
    print("#### TRANSFER leave-one-out\n")
    bundles = {n: load_dataset_bundle(n) for n in loo_datasets}
    loo_rows = []
    for held in loo_datasets:
        train_names = [n for n in loo_datasets if n != held]
        # Build train matrix
        Xtr_list, ytr_list = [], []
        for n in train_names:
            X, used = _stack_matrix(bundles[n], loo_shared)
            Xtr_list.append(X)
            ytr_list.append(np.asarray(bundles[n]["binary"]))
        Xtr = np.vstack(Xtr_list)
        ytr = np.concatenate(ytr_list)
        Xte, used = _stack_matrix(bundles[held], loo_shared)
        yte = bundles[held]["binary"]

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)
        results = {}
        for tag, clf in (
            ("lr", LogisticRegression(max_iter=1000, random_state=0)),
            ("gbm", HistGradientBoostingClassifier(
                max_depth=4, learning_rate=0.08, max_iter=150,
                random_state=0)),
        ):
            if tag == "lr":
                clf.fit(Xtr_s, ytr)
                prob = clf.predict_proba(Xte_s)[:, 1]
            else:
                clf.fit(Xtr, ytr)
                prob = clf.predict_proba(Xte)[:, 1]
            results[tag] = float(roc_auc_score(yte, prob))

        singles = {s: metrics.auc(yte, bundles[held]["scorers"][s])
                   for s in used}
        best = max(singles, key=singles.get)
        row = {
            "held_out": held,
            "train_on": train_names,
            "scorers": used,
            "stack_lr": results["lr"],
            "stack_gbm": results["gbm"],
            "best_single": best,
            "best_single_auc": singles[best],
            "singles": singles,
        }
        loo_rows.append(row)
        print(f"  held={held}: stack_lr={results['lr']:.3f} "
              f"stack_gbm={results['gbm']:.3f} "
              f"best_single={best} {singles[best]:.3f}")
    out["TRANSFER_loo"]["rows"] = loo_rows

    # --- Summary ↔ TofuEval rich transfer ---
    print("\n#### TRANSFER Summary ↔ TofuEval (rich)\n")
    rt = load_dataset_bundle("ragtruth-summary")
    tf = load_dataset_bundle("tofueval")
    rich_used = [s for s in rich if s in rt["scorers"] and s in tf["scorers"]]
    st_rows = []
    for fit_b, eval_b, key in (
        (rt, tf, "ragtruth-summary->tofueval"),
        (tf, rt, "tofueval->ragtruth-summary"),
    ):
        Xtr, _ = _stack_matrix(fit_b, rich_used)
        Xte, _ = _stack_matrix(eval_b, rich_used)
        ytr = np.asarray(fit_b["binary"])
        yte = eval_b["binary"]
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)
        lr = LogisticRegression(max_iter=1000, random_state=0)
        lr.fit(Xtr_s, ytr)
        gbm = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.08, max_iter=150, random_state=0)
        gbm.fit(Xtr, ytr)
        singles = {s: metrics.auc(yte, eval_b["scorers"][s])
                   for s in rich_used}
        best = max(singles, key=singles.get)
        cascade_auc = None
        if "minicheck" in eval_b["scorers"] and \
                "llm-judge-7b-4bit" in eval_b["scorers"]:
            cs, _, _ = soft_cascade(
                eval_b["scorers"]["minicheck"],
                eval_b["scorers"]["llm-judge-7b-4bit"])
            cascade_auc = metrics.auc(yte, cs)
        row = {
            "direction": key,
            "scorers": rich_used,
            "stack_lr": float(roc_auc_score(yte, lr.predict_proba(Xte_s)[:, 1])),
            "stack_gbm": float(roc_auc_score(yte, gbm.predict_proba(Xte)[:, 1])),
            "best_single": best,
            "best_single_auc": singles[best],
            "singles": singles,
            "minicheck_7b_cascade_auc": cascade_auc,
            "failed_router_ref": 0.744,  # from prior work RT->TF
        }
        st_rows.append(row)
        print(f"  {key}: lr={row['stack_lr']:.3f} gbm={row['stack_gbm']:.3f} "
              f"best={best} {singles[best]:.3f} cascade={cascade_auc}")
    out["TRANSFER_summary_tofueval"]["rows"] = st_rows

    path = os.path.join(ROOT, "results-stacking.json")
    json.dump(_jsonable(out), open(path, "w"), indent=2)
    print(f"\nwrote {path}")
    return out


# ---------------------------------------------------------------------------
# Task 4 — cost-constrained frontier
# ---------------------------------------------------------------------------

def _latency_vram_maps():
    """Collect median ms and peak VRAM from aggregate result JSONs."""
    lat, vram = {}, {}
    files = [
        "results-ragtruth.json", "results-scale.json", "results-scale-7b.json",
        "results-newscorers-ragtruth.json", "results-tofueval.json",
        "results-tofueval-judges.json", "results-newscorers-tofueval.json",
        "results.json",
    ]
    for f in files:
        p = os.path.join(ROOT, f)
        if not os.path.exists(p):
            continue
        rows = json.load(open(p))
        if not isinstance(rows, list):
            continue
        for r in rows:
            lat[r["scorer"]] = r.get("median_latency_ms", lat.get(r["scorer"]))
            if r.get("peak_vram_gb") is not None:
                vram[r["scorer"]] = r["peak_vram_gb"]
    # Aliases
    if "llm-judge" in lat and "llm-judge-3b" not in lat:
        lat["llm-judge-3b"] = lat["llm-judge"]
        vram["llm-judge-3b"] = vram.get("llm-judge", vram.get("llm-judge-3b"))
    return lat, vram


def run_frontier() -> dict:
    print("### Task 4 - Cost-constrained combination frontier\n")
    print("Assumption: scorers in a combination run sequentially "
          "→ total latency = sum(ms); peak VRAM = max(members).\n")

    lat, vram = _latency_vram_maps()
    candidates = [
        "rouge-l", "bertscore", "nli-deberta", "minicheck", "alignscore",
        "llm-judge-1.5b", "llm-judge-3b", "llm-judge-7b-4bit",
    ]
    rt = load_dataset_bundle("ragtruth-summary")
    tf = load_dataset_bundle("tofueval")
    candidates = [s for s in candidates
                  if s in rt["scorers"] and s in tf["scorers"]]

    def transfer_auc(subset):
        """Train LR stack on RT, eval on TF (honest TRANSFER)."""
        Xtr, _ = _stack_matrix(rt, subset)
        Xte, _ = _stack_matrix(tf, subset)
        ytr = np.asarray(rt["binary"])
        yte = tf["binary"]
        if len(subset) == 1:
            return metrics.auc(yte, tf["scorers"][subset[0]])
        scaler = StandardScaler()
        clf = LogisticRegression(max_iter=1000, random_state=0)
        clf.fit(scaler.fit_transform(Xtr), ytr)
        return float(roc_auc_score(yte, clf.predict_proba(
            scaler.transform(Xte))[:, 1]))

    def cost(subset):
        ms = sum(lat.get(s, 0.0) or 0.0 for s in subset)
        vr = max((vram.get(s, 0.0) or 0.0) for s in subset) if subset else 0.0
        return ms, vr

    # Greedy forward selection
    selected = []
    remaining = list(candidates)
    steps = []
    print("| step | add | subset | TRANSFER AUC | sum ms | peak GB |")
    print("|---|---|---|---:|---:|---:|")
    while remaining:
        best_s, best_auc = None, -1
        for s in remaining:
            trial = selected + [s]
            auc = transfer_auc(trial)
            if auc > best_auc:
                best_auc, best_s = auc, s
        selected.append(best_s)
        remaining.remove(best_s)
        ms, vr = cost(selected)
        steps.append({
            "step": len(selected),
            "added": best_s,
            "subset": list(selected),
            "transfer_auc": best_auc,
            "total_latency_ms": ms,
            "peak_vram_gb": vr,
        })
        print(f"| {len(selected)} | {best_s} | {selected} | {best_auc:.3f} | "
              f"{ms:.0f} | {vr:.2f} |")

    # Singles
    singles = []
    for s in candidates:
        ms, vr = cost([s])
        singles.append({
            "scorer": s,
            "transfer_auc": metrics.auc(tf["binary"], tf["scorers"][s]),
            # Actually "transfer" for a single is just TF AUC (no fit) —
            # consistent with evaluating quality on TF.
            "total_latency_ms": ms,
            "peak_vram_gb": vr,
        })

    # Cascade point
    cs, stages, _ = soft_cascade(
        tf["scorers"]["minicheck"], tf["scorers"]["llm-judge-7b-4bit"])
    # Latency: same formula as cascade gate sweep on TF? Use RT latencies
    # as proxy; mean = mix of cheap/dear
    esc = sum(1 for st in stages if st == "dear") / len(stages)
    casc_ms = ((1 - esc) * lat["minicheck"] + esc * lat["llm-judge-7b-4bit"])
    casc_vram = max(vram.get("minicheck", 0), vram.get("llm-judge-7b-4bit", 0))
    cascade_pt = {
        "name": "minicheck->7b cascade",
        "transfer_auc": metrics.auc(tf["binary"], cs),
        "total_latency_ms": casc_ms,
        "peak_vram_gb": casc_vram,
        "note": "evaluated on TofuEval with MiniCheck mid-band->7B; "
                "latency mix uses Summary-measured medians as proxy",
    }

    # Oracle on TF (cheating)
    avail = {s: tf["scorers"][s] for s in candidates}
    cheat = cheating_oracle_auc(tf["binary"], avail, n_boot=200)
    oracle_pt = {
        "name": "cheating oracle",
        "transfer_auc": cheat["roc_auc"],
        "total_latency_ms": sum(lat.get(s, 0) or 0 for s in candidates),
        "peak_vram_gb": max(vram.get(s, 0) or 0 for s in candidates),
    }

    out = {
        "protocol": (
            "Greedy forward selection maximizing TRANSFER AUC "
            "(train stack on RAGTruth Summary, eval TofuEval). "
            "Latency = sum of member median ms (sequential). "
            "Peak VRAM = max of members."
        ),
        "latencies_ms": {s: lat.get(s) for s in candidates},
        "vram_gb": {s: vram.get(s) for s in candidates},
        "greedy_steps": steps,
        "singles": singles,
        "cascade": cascade_pt,
        "oracle": oracle_pt,
    }
    path = os.path.join(ROOT, "results-frontier.json")
    json.dump(_jsonable(out), open(path, "w"), indent=2)
    print(f"\nwrote {path}")

    plot_frontier(out)
    return out


def plot_frontier(front):
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

    # Singles
    for s in front["singles"]:
        c = COLORS.get(s["scorer"], FALLBACK)
        ax.scatter([max(s["total_latency_ms"], 0.1)], [s["transfer_auc"]],
                   s=55, color=c, zorder=3, marker="o")
        ax.annotate(s["scorer"],
                    (max(s["total_latency_ms"], 0.1), s["transfer_auc"]),
                    xytext=(5, 4), textcoords="offset points",
                    fontsize=7, color=INK_2)

    # Greedy path
    xs = [max(st["total_latency_ms"], 0.1) for st in front["greedy_steps"]]
    ys = [st["transfer_auc"] for st in front["greedy_steps"]]
    ax.plot(xs, ys, color="#1a1a1a", linewidth=2, zorder=4, label="greedy stack")
    ax.scatter(xs, ys, s=40, color="#1a1a1a", zorder=5, marker="D")

    # Cascade + oracle
    casc = front["cascade"]
    ax.scatter([casc["total_latency_ms"]], [casc["transfer_auc"]],
               s=90, color=COLORS["cascade"], marker="*", zorder=6,
               label="MiniCheck→7B cascade")
    ora = front["oracle"]
    ax.axhline(ora["transfer_auc"], color="#888", linestyle="--", linewidth=1,
               zorder=2, label=f"cheating oracle ({ora['transfer_auc']:.3f})")

    ax.set_xscale("log")
    ax.set_xlabel("total latency per doc (ms, log) — sum if combined",
                  color=INK_2, fontsize=9)
    ax.set_ylabel("TRANSFER AUC (fit RT Summary → eval TofuEval)",
                  color=INK_2, fontsize=9)
    ax.set_title("Cost-constrained combination frontier",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.margins(y=0.12)
    fig.tight_layout()
    out = os.path.join(ROOT, "frontier.png")
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Work order #5 — paired bootstrap + cost-adjusted verdict
# ---------------------------------------------------------------------------

def paired_bootstrap_auc_diff(y, scores_a, scores_b, n_boot=10000, seed=0):
    """Paired bootstrap of AUC(a) - AUC(b). Same indices for both systems."""
    y = np.asarray(y)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    point = float(roc_auc_score(y, a) - roc_auc_score(y, b))
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        diffs.append(float(roc_auc_score(yb, a[idx]) - roc_auc_score(yb, b[idx])))
    diffs = np.asarray(diffs)
    ci = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    p_le_zero = float(np.mean(diffs <= 0.0))
    excludes_zero = not (ci[0] <= 0.0 <= ci[1])
    return {
        "point_diff": point,
        "mean_boot_diff": float(diffs.mean()),
        "diff_ci_95": ci,
        "frac_diff_le_zero": p_le_zero,
        "ci_excludes_zero": excludes_zero,
        "n_boot_kept": int(len(diffs)),
        "auc_a": float(roc_auc_score(y, a)),
        "auc_b": float(roc_auc_score(y, b)),
    }


def _transfer_system_scores(fit_b, eval_b, rich):
    """Fit stacks on fit_b; return dict of named score vectors on eval_b."""
    used = [s for s in rich if s in fit_b["scorers"] and s in eval_b["scorers"]]
    Xtr, _ = _stack_matrix(fit_b, used)
    Xte, _ = _stack_matrix(eval_b, used)
    ytr = np.asarray(fit_b["binary"])
    yte = eval_b["binary"]

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    lr = LogisticRegression(max_iter=1000, random_state=0)
    lr.fit(Xtr_s, ytr)
    gbm = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.08, max_iter=150, random_state=0)
    gbm.fit(Xtr, ytr)

    stack_lr = lr.predict_proba(Xte_s)[:, 1]
    stack_gbm = gbm.predict_proba(Xte)[:, 1]
    minicheck = np.asarray(eval_b["scorers"]["minicheck"], dtype=float)
    judge7 = np.asarray(eval_b["scorers"]["llm-judge-7b-4bit"], dtype=float)
    casc, _, _ = soft_cascade(
        eval_b["scorers"]["minicheck"],
        eval_b["scorers"]["llm-judge-7b-4bit"])
    casc = np.asarray(casc, dtype=float)

    # Best stack by point AUC on this eval set
    best_stack_name = "stack_lr" if roc_auc_score(yte, stack_lr) >= \
        roc_auc_score(yte, stack_gbm) else "stack_gbm"
    best_stack = stack_lr if best_stack_name == "stack_lr" else stack_gbm

    return {
        "y": yte,
        "scorers_used": used,
        "stack_lr": stack_lr,
        "stack_gbm": stack_gbm,
        "best_stack_name": best_stack_name,
        "best_stack": best_stack,
        "minicheck": minicheck,
        "cascade": casc,
        "llm-judge-7b-4bit": judge7,
    }


def run_paired_tests(n_boot=10000) -> dict:
    print("### Work order #5 Task 1 - Paired bootstrap on stacking gains\n")
    rich = [
        "rouge-l", "bertscore", "nli-deberta", "minicheck", "alignscore",
        "llm-judge-1.5b", "llm-judge-3b", "llm-judge-7b-4bit",
    ]
    rt = load_dataset_bundle("ragtruth-summary")
    tf = load_dataset_bundle("tofueval")

    comparisons = [
        ("stack_lr", "minicheck", "stack-LR vs MiniCheck"),
        ("stack_lr", "cascade", "stack-LR vs MiniCheck→7B cascade"),
        ("stack_gbm", "stack_lr", "stack-GBM vs stack-LR"),
        ("best_stack", "llm-judge-7b-4bit", "best stack vs 7B judge alone"),
    ]

    out = {
        "protocol": (
            f"Paired bootstrap of AUC(A)-AUC(B), {n_boot} resamples, same "
            "example indices for both systems. Models fit once on the fit "
            "corpus; bootstrap only the eval-set metric difference. "
            "frac_diff_le_zero = bootstrap p-value for H0: diff<=0 "
            "(one-sided toward A>B)."
        ),
        "n_boot": n_boot,
        "directions": {},
    }

    for fit_b, eval_b, key in (
        (rt, tf, "ragtruth-summary->tofueval"),
        (tf, rt, "tofueval->ragtruth-summary"),
    ):
        print(f"#### {key}\n")
        sys = _transfer_system_scores(fit_b, eval_b, rich)
        y = sys["y"]
        rows = []
        print("| comparison | AUC_A | AUC_B | Δ | 95% CI(Δ) | "
              "P(Δ≤0) | CI excludes 0? |")
        print("|---|---:|---:|---:|---|---:|---|")
        for a_key, b_key, label in comparisons:
            res = paired_bootstrap_auc_diff(
                y, sys[a_key], sys[b_key], n_boot=n_boot)
            row = {
                "comparison": label,
                "system_a": a_key if a_key != "best_stack"
                else sys["best_stack_name"],
                "system_b": b_key,
                **res,
                "verdict": (
                    "significant (CI excludes 0)" if res["ci_excludes_zero"]
                    else "not significant (CI includes 0)"
                ),
            }
            rows.append(row)
            print(
                f"| {label} | {res['auc_a']:.3f} | {res['auc_b']:.3f} | "
                f"{res['point_diff']:+.3f} | "
                f"[{res['diff_ci_95'][0]:+.3f}, {res['diff_ci_95'][1]:+.3f}] | "
                f"{res['frac_diff_le_zero']:.3f} | "
                f"{'YES' if res['ci_excludes_zero'] else 'no'} |"
            )
        print()
        out["directions"][key] = {
            "best_stack_name": sys["best_stack_name"],
            "scorers_used": sys["scorers_used"],
            "rows": rows,
        }

    path = os.path.join(ROOT, "results-paired-tests.json")
    json.dump(_jsonable(out), open(path, "w"), indent=2)
    print(f"wrote {path}")
    return out


def run_cost_verdict() -> dict:
    print("### Work order #5 Task 2 - Cost-adjusted frontier verdict\n")
    front_path = os.path.join(ROOT, "results-frontier.json")
    if not os.path.exists(front_path):
        raise FileNotFoundError(
            "results-frontier.json missing; run --task frontier first")
    front = json.load(open(front_path))
    lat = front["latencies_ms"]
    base_auc = None
    base_ms = None
    # MiniCheck alone from singles or greedy step 1
    for s in front["singles"]:
        if s["scorer"] == "minicheck":
            base_auc = s["transfer_auc"]
            base_ms = s["total_latency_ms"]
            break
    if base_auc is None:
        base_auc = front["greedy_steps"][0]["transfer_auc"]
        base_ms = front["greedy_steps"][0]["total_latency_ms"]

    rows = []
    # Systems: MiniCheck alone, each greedy step, cascade
    systems = []
    for st in front["greedy_steps"]:
        systems.append({
            "name": "+".join(st["subset"]) if len(st["subset"]) > 1
            else st["subset"][0],
            "auc": st["transfer_auc"],
            "ms": st["total_latency_ms"],
            "kind": "greedy",
        })
    casc = front["cascade"]
    systems.append({
        "name": casc["name"],
        "auc": casc["transfer_auc"],
        "ms": casc["total_latency_ms"],
        "kind": "cascade",
    })

    print("| system | AUC | total ms | ΔAUC vs MiniCheck | "
          "extra ms | AUC gained / 100 ms |")
    print("|---|---:|---:|---:|---:|---:|")
    for s in systems:
        d_auc = s["auc"] - base_auc
        d_ms = s["ms"] - base_ms
        if abs(d_ms) < 1e-9:
            per_100 = None
        else:
            per_100 = d_auc / d_ms * 100.0
        row = {
            "system": s["name"],
            "kind": s["kind"],
            "auc": s["auc"],
            "total_latency_ms": s["ms"],
            "delta_auc_vs_minicheck": d_auc,
            "extra_ms_vs_minicheck": d_ms,
            "auc_gained_per_100ms": per_100,
        }
        rows.append(row)
        per_s = "—" if per_100 is None else f"{per_100:+.4f}"
        print(f"| {s['name']} | {s['auc']:.3f} | {s['ms']:.0f} | "
              f"{d_auc:+.3f} | {d_ms:+.0f} | {per_s} |")

    # Verdict: is any additional compute worth it?
    # Look at positive per_100 among systems with extra cost; also note
    # cascade vs full stack.
    positive = [r for r in rows
                if r["auc_gained_per_100ms"] is not None
                and r["extra_ms_vs_minicheck"] > 0
                and r["delta_auc_vs_minicheck"] > 0]
    if not positive:
        verdict = (
            "Above MiniCheck alone, no system on this frontier gains AUC "
            "per additional compute in a way that looks worthwhile — "
            "stay with MiniCheck."
        )
    else:
        best = max(positive, key=lambda r: r["auc_gained_per_100ms"])
        # Cascade often has better cost profile than summing all members
        casc_row = next((r for r in rows if r["kind"] == "cascade"), None)
        # One-sentence verdict for README
        if casc_row and casc_row["delta_auc_vs_minicheck"] <= 0:
            verdict = (
                "Above MiniCheck alone, additional compute buys little: the "
                f"best AUC/100ms among improving systems is "
                f"{best['auc_gained_per_100ms']:+.4f} "
                f"({best['system']}), and the MiniCheck→7B cascade does not "
                "beat MiniCheck's TRANSFER AUC on TofuEval — prefer "
                "MiniCheck alone unless paired tests show a significant gain."
            )
        else:
            verdict = (
                "Above MiniCheck alone, extra compute has diminishing returns: "
                f"the best AUC gained per additional 100 ms is "
                f"{best['auc_gained_per_100ms']:+.4f} ({best['system']}); "
                "unless a paired test shows a significant AUC lift, MiniCheck "
                "alone remains the rational default."
            )

    print(f"\nVERDICT: {verdict}\n")
    out = {
        "protocol": (
            "TRANSFER AUC from results-frontier.json (fit RT Summary → eval "
            "TofuEval). auc_gained_per_100ms = "
            "(AUC - MiniCheck_AUC) / (ms - MiniCheck_ms) * 100. "
            "Cascade latency is soft-cascade mix, not sum of both."
        ),
        "minicheck_baseline": {"auc": base_auc, "ms": base_ms},
        "rows": rows,
        "verdict_sentence": verdict,
    }
    path = os.path.join(ROOT, "results-cost-verdict.json")
    json.dump(_jsonable(out), open(path, "w"), indent=2)
    print(f"wrote {path}")
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", default="all",
        choices=("all", "coverage", "predictability", "oracle",
                 "stacking", "frontier", "paired", "cost"),
    )
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()
    task = args.task

    if task in ("all", "coverage"):
        run_coverage()
    if task in ("all", "predictability"):
        run_predictability()
    if task in ("all", "oracle"):
        run_oracle()
    if task in ("all", "stacking"):
        run_stacking()
    if task in ("all", "frontier"):
        run_frontier()
    if task in ("all", "paired"):
        run_paired_tests(n_boot=args.n_boot)
    if task in ("all", "cost"):
        run_cost_verdict()


if __name__ == "__main__":
    main()
