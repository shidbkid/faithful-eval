"""Quality-vs-cost scatter from a results JSON.

    python plot.py                              # results.json -> results.png
    python plot.py --in results-ragtruth.json --out results-ragtruth.png \
                   --title "Faithfulness scoring: quality vs cost (RAGTruth Summary)"

X = median latency per (source, summary) pair (log scale), Y = Spearman
correlation with human labels. Each point is one scorer; VRAM is noted in
the point label when the scorer used a GPU.
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3dd"
POINT = "#2a78d6"


def label(r):
    if r.get("peak_vram_gb"):
        return f"{r['scorer']} ({r['peak_vram_gb']:.1f} GB)"
    return r["scorer"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", default="results.json",
                        help="results JSON from run.py")
    parser.add_argument("--out", default="results.png",
                        help="output plot path")
    parser.add_argument("--title",
                        default="Faithfulness scoring: quality vs cost",
                        help="plot title")
    args = parser.parse_args()

    rows = json.load(open(args.infile))

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    xs = [max(r["median_latency_ms"], 0.01) for r in rows]
    ys = [r["spearman"] for r in rows]
    ax.scatter(xs, ys, s=70, color=POINT, zorder=3)
    for x, y, r in zip(xs, ys, rows):
        ax.annotate(label(r), (x, y), xytext=(7, 5),
                    textcoords="offset points", fontsize=9, color=INK)

    ax.set_xscale("log")
    ax.set_xlabel("median latency per document pair (ms, log scale)",
                  color=INK_2, fontsize=9)
    ax.set_ylabel("Spearman vs human label", color=INK_2, fontsize=9)
    ax.set_title(args.title, color=INK, fontsize=11, loc="left")
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.margins(x=0.15, y=0.15)

    fig.tight_layout()
    fig.savefig(args.out, facecolor=SURFACE)
    print(f"wrote {args.out}\n")

    def f(v, spec=".3f"):
        return "n/a" if v is None else format(v, spec)

    print("| scorer | spearman | balanced acc | ROC-AUC "
          "| median ms/doc | peak VRAM (GB) |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['scorer']} | {f(r['spearman'])} "
              f"| {f(r['balanced_acc'])} | {f(r['roc_auc'])} "
              f"| {f(r['median_latency_ms'], '.1f')} "
              f"| {f(r['peak_vram_gb'], '.1f')} |")


if __name__ == "__main__":
    main()
