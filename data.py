"""Dataset loading.

Every loader yields the same example dict:
    {"doc_id": str, "system": str, "source": str, "summary": str,
     "label": float, "binary": int}

`label` is higher = more faithful (used for Spearman).
`binary` is 1 = faithful / 0 = unfaithful (used for bal_acc / ROC-AUC).

Datasets
--------
summeval  — 2019-era system summaries with 1–5 expert consistency
            (Fabbri et al.; fetched via BARTScore's vendored pickle).
ragtruth  — LLM-era RAG / summary / QA responses with human hallucination
            spans (Niu et al. 2024; Hugging Face `wandb/RAGTruth-processed`).
"""

import os
import pickle
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

SUMMEVAL_URL = ("https://raw.githubusercontent.com/neulab/BARTScore/main/"
                "SUM/SummEval/data.pkl")
SUMMEVAL_CACHE = os.path.join(DATA_DIR, "summeval.pkl")

DATASETS = ("summeval", "ragtruth")


def load_summeval():
    if not os.path.exists(SUMMEVAL_CACHE):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"downloading SummEval -> {SUMMEVAL_CACHE}")
        urllib.request.urlretrieve(SUMMEVAL_URL, SUMMEVAL_CACHE)

    with open(SUMMEVAL_CACHE, "rb") as f:
        raw = pickle.load(f)

    examples = []
    for doc_id, doc in sorted(raw.items()):
        for system, s in sorted(doc["sys_summs"].items()):
            label = float(s["scores"]["consistency"])
            examples.append({
                "doc_id": str(doc_id),
                "system": system,
                "source": doc["src"],
                "summary": s["sys_summ"],
                "label": label,
                # Perfect expert agreement on consistency == 5.0.
                "binary": 1 if label >= 5.0 else 0,
            })
    return examples


def _ragtruth_source(context: str) -> str:
    # The processed HF dump appends a literal "\n\noutput:" prompt tail.
    if context.endswith("\n\noutput:"):
        return context[: -len("\n\noutput:")].rstrip()
    return context


def load_ragtruth(split: str = "test", task: str | None = "Summary"):
    """Load RAGTruth examples from Hugging Face.

    Parameters
    ----------
    split : "test" | "train"
    task  : "Summary" | "QA" | "Data2txt" | None
        If set, keep only that task_type. Default Summary keeps the
        comparison closest to SummEval. Pass None for all tasks.
    """
    from datasets import load_dataset

    ds = load_dataset("wandb/RAGTruth-processed", split=split)
    examples = []
    for row in ds:
        if task is not None and row["task_type"] != task:
            continue
        h = row["hallucination_labels_processed"]
        n_hallu = int(h["evident_conflict"]) + int(h["baseless_info"])
        binary = 1 if n_hallu == 0 else 0
        # Soft label for Spearman: 1.0 if clean, else decays with span count.
        label = 1.0 if binary else 1.0 / (1.0 + n_hallu)
        examples.append({
            "doc_id": str(row["id"]),
            "system": row["model"],
            "source": _ragtruth_source(row["context"]),
            "summary": row["output"],
            "label": label,
            "binary": binary,
        })
    if not examples:
        raise ValueError(
            f"no RAGTruth examples for split={split!r} task={task!r}")
    return examples


def load(name: str, **kwargs):
    if name == "summeval":
        return load_summeval()
    if name == "ragtruth":
        return load_ragtruth(**kwargs)
    raise ValueError(
        f"unknown dataset {name!r}; choose from {DATASETS}")


if __name__ == "__main__":
    for name in DATASETS:
        if name == "ragtruth":
            ex = load(name, task="Summary")
        else:
            ex = load(name)
        n_pos = sum(e["binary"] for e in ex)
        print(f"{name}: {len(ex)} pairs, "
              f"{len({e['doc_id'] for e in ex})} docs, "
              f"{len({e['system'] for e in ex})} systems, "
              f"faithful={n_pos} ({n_pos / len(ex):.1%})")
