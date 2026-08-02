"""Dataset loading.

Each example is a dict:
    {"doc_id": str, "system": str, "source": str, "summary": str, "label": float}
where label is the mean expert consistency rating from SummEval (1-5,
higher = more faithful).

SummEval = 100 CNN/DailyMail articles x 16 system summaries, each rated by
3 experts. We fetch the processed copy vendored in the BARTScore repo
(github.com/neulab/BARTScore, Apache-2.0), which pairs every summary with
its source article and the expert scores. Cached locally after first
download.
"""

import os
import pickle
import urllib.request

DATA_URL = ("https://raw.githubusercontent.com/neulab/BARTScore/main/"
            "SUM/SummEval/data.pkl")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "data", "summeval.pkl")


def load_summeval():
    if not os.path.exists(CACHE):
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        print(f"downloading SummEval -> {CACHE}")
        urllib.request.urlretrieve(DATA_URL, CACHE)

    with open(CACHE, "rb") as f:
        raw = pickle.load(f)

    examples = []
    for doc_id, doc in sorted(raw.items()):
        for system, s in sorted(doc["sys_summs"].items()):
            examples.append({
                "doc_id": doc_id,
                "system": system,
                "source": doc["src"],
                "summary": s["sys_summ"],
                "label": s["scores"]["consistency"],
            })
    return examples


if __name__ == "__main__":
    ex = load_summeval()
    print(f"{len(ex)} pairs, {len({e['doc_id'] for e in ex})} documents, "
          f"{len({e['system'] for e in ex})} systems")
