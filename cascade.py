"""Task-aware cascade — offline eval + live scorer.

Primary policy (one row per RAGTruth task)
------------------------------------------
- QA        → ROUGE-L only
- Summary   → NLI, escalate mid-confidence band to 7B-4bit judge
- Data2txt  → NLI, escalate mid-confidence band to 3B judge

NLI extremes keep the NLI score (soft); mid-band uses the judge. Soft-keep
avoids destroying AUC by hard-clipping, but NLI is still a weak gate on
LLM-era data — see the Summary ablation (1.5B→7B) for a cascade that
actually approaches 7B quality at ~half the latency.

    python cascade.py
"""

from __future__ import annotations

import json
import os
from collections import Counter

import metrics

ROOT = os.path.dirname(os.path.abspath(__file__))

TASK_PREDS = {
    "Summary": "results-ragtruth.preds.json",
    "QA": "results-ragtruth-qa.preds.json",
    "Data2txt": "results-ragtruth-d2t.preds.json",
}

DEFAULT_LAT = {
    "rouge-l": 8.0,
    "nli-deberta": 110.0,
    "llm-judge": 535.0,
    "llm-judge-1.5b": 309.0,
    "llm-judge-3b": 541.0,
    "llm-judge-7b-4bit": 1174.0,
}


def load_latencies(task: str | None = None) -> dict:
    """Latencies are task-dependent (Data2txt NLI is ~8× Summary NLI)."""
    lat = dict(DEFAULT_LAT)
    task_files = {
        "Summary": ("results-ragtruth.json", "results-scale.json",
                    "results-scale-7b.json"),
        "QA": ("results-ragtruth-qa.json", "results-ragtruth-qa-7b.json"),
        "Data2txt": ("results-ragtruth-d2t.json",),
    }
    files = task_files.get(task, (
        "results-ragtruth.json",
        "results-scale.json",
        "results-scale-7b.json",
    ))
    for path in files:
        p = os.path.join(ROOT, path)
        if os.path.exists(p):
            for r in json.load(open(p)):
                lat[r["scorer"]] = r["median_latency_ms"]
    return lat


def soft_cascade(cheap_preds, dear_preds, low_q=0.33, high_q=0.67):
    """Keep cheap score outside [lo,hi]; escalate inside the band."""
    lo, hi = metrics.cascade_bands(cheap_preds, low_q, high_q)
    scores, stages = [], []
    for c, d in zip(cheap_preds, dear_preds):
        if c <= lo or c >= hi:
            scores.append(c)
            stages.append("cheap")
        else:
            scores.append(d)
            stages.append("dear")
    return scores, stages, (lo, hi)


def _load_summary_judges():
    base = json.load(open(os.path.join(ROOT, TASK_PREDS["Summary"])))
    scorers = dict(base["scorers"])
    binary = base["binary"]
    scale = json.load(open(os.path.join(ROOT, "results-scale.preds.json")))
    s7 = json.load(open(os.path.join(ROOT, "results-scale-7b.preds.json")))
    scorers["llm-judge-1.5b"] = scale["scorers"]["llm-judge-1.5b"]
    scorers["llm-judge-3b"] = scale["scorers"]["llm-judge-3b"]
    scorers["llm-judge-7b-4bit"] = s7["scorers"]["llm-judge-7b-4bit"]
    return binary, scorers


def _pack(task, policy, binary, scores, stages, lat, stage_map, band,
          target_key, refs_src):
    stages = [stage_map[s] for s in stages]
    n = len(stages)
    mean_ms = sum(lat[s] for s in stages) / n
    auc = metrics.auc(binary, scores)
    ci = metrics.bootstrap_auc_ci(binary, scores)
    refs = {}
    for name, preds in refs_src.items():
        refs[name] = {
            "roc_auc": metrics.auc(binary, preds),
            "median_latency_ms": lat.get(name),
        }
    target = refs[target_key]
    return {
        "task": task,
        "policy": policy,
        "n": n,
        "roc_auc": auc,
        "roc_auc_ci": ci,
        "balanced_acc": metrics.oracle_bal_acc(binary, scores),
        "mean_latency_ms": mean_ms,
        "stage_frac": {k: v / n for k, v in Counter(stages).items()},
        "band": list(band) if band else None,
        "target": target_key,
        "auc_ratio_vs_target": auc / target["roc_auc"],
        "latency_ratio_vs_target": mean_ms / target["median_latency_ms"],
        "refs": refs,
    }


def evaluate_primary() -> list[dict]:
    rows = []

    # --- Summary: NLI -> 7B ---
    lat = load_latencies("Summary")
    binary, scorers = _load_summary_judges()
    scores, stages, band = soft_cascade(
        scorers["nli-deberta"], scorers["llm-judge-7b-4bit"])
    rows.append(_pack(
        "Summary",
        "NLI -> mid-band 7B-4bit (soft-keep)",
        binary, scores, stages, lat,
        {"cheap": "nli-deberta", "dear": "llm-judge-7b-4bit"},
        band, "llm-judge-7b-4bit",
        {k: scorers[k] for k in (
            "rouge-l", "nli-deberta", "llm-judge", "llm-judge-7b-4bit")},
    ))

    # --- QA: ROUGE only ---
    lat = load_latencies("QA")
    qa = json.load(open(os.path.join(ROOT, TASK_PREDS["QA"])))
    scores = qa["scorers"]["rouge-l"]
    stages = ["cheap"] * len(scores)
    rows.append(_pack(
        "QA",
        "ROUGE-L only",
        qa["binary"], scores, stages, lat,
        {"cheap": "rouge-l", "dear": "rouge-l"},
        None, "rouge-l",
        {k: qa["scorers"][k] for k in (
            "rouge-l", "nli-deberta", "llm-judge")},
    ))

    # --- Data2txt: NLI -> 3B ---
    lat = load_latencies("Data2txt")
    d2t = json.load(open(os.path.join(ROOT, TASK_PREDS["Data2txt"])))
    scores, stages, band = soft_cascade(
        d2t["scorers"]["nli-deberta"], d2t["scorers"]["llm-judge"])
    rows.append(_pack(
        "Data2txt",
        "NLI -> mid-band 3B judge (soft-keep)",
        d2t["binary"], scores, stages, lat,
        {"cheap": "nli-deberta", "dear": "llm-judge"},
        band, "llm-judge",
        {k: d2t["scorers"][k] for k in (
            "rouge-l", "nli-deberta", "llm-judge")},
    ))
    return rows


def evaluate_ablations() -> list[dict]:
    """Summary-only: cascades that actually approach 7B quality."""
    lat = load_latencies("Summary")
    binary, scorers = _load_summary_judges()
    rows = []
    for cheap_key, label, lq, hq in (
        ("llm-judge-1.5b",
         "1.5B -> mid-band 7B-4bit (soft-keep)", 0.33, 0.67),
        ("nli-deberta",
         "NLI -> mid-band 7B-4bit @ q[0.1,0.9]", 0.1, 0.9),
    ):
        scores, stages, band = soft_cascade(
            scorers[cheap_key], scorers["llm-judge-7b-4bit"], lq, hq)
        rows.append(_pack(
            "Summary",
            label,
            binary, scores, stages, lat,
            {"cheap": cheap_key, "dear": "llm-judge-7b-4bit"},
            band, "llm-judge-7b-4bit",
            {k: scorers[k] for k in (
                cheap_key, "llm-judge-7b-4bit", "nli-deberta")},
        ))
    return rows


class TaskAwareCascade:
    """Live scorer. QA→ROUGE; Summary/Data2txt→NLI then judge on mid-band."""

    name = "cascade"

    def __init__(self, task: str = "Summary",
                 low_q: float = 0.33, high_q: float = 0.67,
                 judge_model_name: str = None,
                 load_in_4bit: bool = None):
        if task not in ("Summary", "QA", "Data2txt"):
            raise ValueError(task)
        self.task = task
        self.low_q = low_q
        self.high_q = high_q
        if judge_model_name is None:
            judge_model_name = (
                "Qwen/Qwen2.5-7B-Instruct" if task == "Summary"
                else "Qwen/Qwen2.5-3B-Instruct")
        if load_in_4bit is None:
            load_in_4bit = "7B" in judge_model_name
        self.judge_model_name = judge_model_name
        self.load_in_4bit = load_in_4bit
        self._rouge = self._nli = self._judge = None
        self._nli_scores_seen: list[float] = []

    def _ensure(self):
        from scorers import LLMJudgeScorer, NLIScorer, RougeLScorer
        if self.task == "QA" and self._rouge is None:
            self._rouge = RougeLScorer()
        if self.task != "QA" and self._nli is None:
            self._nli = NLIScorer()
            self._judge = LLMJudgeScorer(
                model_name=self.judge_model_name,
                load_in_4bit=self.load_in_4bit)

    def score(self, source: str, summary: str) -> float:
        self._ensure()
        if self.task == "QA":
            return self._rouge.score(source, summary)
        n = self._nli.score(source, summary)
        self._nli_scores_seen.append(n)
        if len(self._nli_scores_seen) < 30:
            return self._judge.score(source, summary)
        lo, hi = metrics.cascade_bands(
            self._nli_scores_seen, self.low_q, self.high_q)
        if n <= lo or n >= hi:
            return n
        return self._judge.score(source, summary)


def _print_row(r):
    ci = r["roc_auc_ci"]
    ci_s = f"{ci[0]:.3f}-{ci[1]:.3f}" if ci else "n/a"
    print(f"| {r['task']} | {r['policy']} | {r['roc_auc']:.3f} | {ci_s} | "
          f"{r['balanced_acc']:.3f} | {r['mean_latency_ms']:.0f} | "
          f"{100*r['auc_ratio_vs_target']:.0f}% of {r['target']} | "
          f"{100*r['latency_ratio_vs_target']:.0f}% |")
    stages = ", ".join(f"{k}={v:.0%}" for k, v in r["stage_frac"].items())
    print(f"  stages: {stages}")


def main():
    primary = evaluate_primary()
    ablations = evaluate_ablations()

    print("### Task-aware cascade (primary)\n")
    print("| task | policy | ROC-AUC | AUC 95% CI | bal acc | "
          "mean ms | % target AUC | % target latency |")
    print("|---|---|---|---|---|---|---|---|")
    for r in primary:
        _print_row(r)

    print("\n### Summary ablations (when NLI is a weak gate)\n")
    print("| task | policy | ROC-AUC | AUC 95% CI | bal acc | "
          "mean ms | % target AUC | % target latency |")
    print("|---|---|---|---|---|---|---|---|")
    for r in ablations:
        _print_row(r)

    out = {
        "primary": primary,
        "ablations": ablations,
        "takeaway": (
            "Task routing is the win (QA→ROUGE). NLI-gated escalation on "
            "Summary/Data2txt does not approach full-judge AUC at tertiary "
            "bands — NLI confidence is poorly structured on LLM-era data. "
            "On Summary, a 1.5B→7B judge cascade recovers ~95% of 7B AUC "
            "at roughly half the latency."
        ),
    }
    path = os.path.join(ROOT, "results-cascade.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"\n{out['takeaway']}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
