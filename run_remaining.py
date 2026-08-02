"""Finish the study: QA + Data2txt full suites, then 7B-4bit on QA."""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
ENV = os.environ.copy()
ENV["NLTK_DISABLE_IMPORT_SECURITY"] = "1"
ENV["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
ENV["PYTHONIOENCODING"] = "utf-8"


def run(args):
    print("\n>>>", " ".join(args), flush=True)
    r = subprocess.run([PY, *args], cwd=ROOT, env=ENV)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main():
    run(["run.py", "--dataset", "ragtruth", "--task", "QA",
         "--out", "results-ragtruth-qa.json", "--save-preds"])
    run(["analyze.py", "--preds", "results-ragtruth-qa.preds.json"])

    run(["run.py", "--dataset", "ragtruth", "--task", "Data2txt",
         "--out", "results-ragtruth-d2t.json", "--save-preds"])
    run(["analyze.py", "--preds", "results-ragtruth-d2t.preds.json"])

    run(["run.py", "--dataset", "ragtruth", "--task", "QA",
         "--judge-model", "7b-4bit",
         "--out", "results-ragtruth-qa-7b.json", "--save-preds"])

    print("\nALL REMAINING RUNS DONE", flush=True)


if __name__ == "__main__":
    main()
