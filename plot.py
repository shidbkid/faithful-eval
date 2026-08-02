"""Plots + markdown tables from results JSON files.

    python plot.py

Always: results.json -> results.png (quality-vs-latency scatter) + table.
If results-ragtruth.json also exists: -> comparison.png, a slope chart of
ROC-AUC on SummEval vs RAGTruth — the ranking reversal is the picture.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3dd"

# Fixed color per scorer (identity follows the entity across every chart).
COLORS = {
    "random": "#8a8984",
    "rouge-l": "#2a78d6",
    "bertscore": "#eb6834",
    "nli-deberta": "#1baf7a",
    "llm-judge": "#4a3aa7",
}
FALLBACK = "#e87ba4"


def color(name):
    return COLORS.get(name, FALLBACK)


def label(r):
    if r.get("peak_vram_gb"):
        return f"{r['scorer']} ({r['peak_vram_gb']:.1f} GB)"
    return r["scorer"]


def new_axes(width=7, height=4.5):
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=8)
    return fig, ax


def scatter(rows, outfile, title):
    fig, ax = new_axes()
    xs = [max(r["median_latency_ms"], 0.01) for r in rows]
    ys = [r["spearman"] for r in rows]
    for x, y, r in zip(xs, ys, rows):
        ax.scatter([x], [y], s=70, color=color(r["scorer"]), zorder=3)
        ax.annotate(label(r), (x, y), xytext=(7, 5),
                    textcoords="offset points", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("median latency per document pair (ms, log scale)",
                  color=INK_2, fontsize=9)
    ax.set_ylabel("Spearman vs human label", color=INK_2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.margins(x=0.15, y=0.15)
    fig.tight_layout()
    fig.savefig(outfile, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {outfile}")


def slope(rows_a, rows_b, name_a, name_b, outfile):
    """ROC-AUC per scorer on two datasets; crossing lines = ranking flip."""
    fig, ax = new_axes(width=6.5, height=4.5)
    by_a = {r["scorer"]: r for r in rows_a}
    by_b = {r["scorer"]: r for r in rows_b}
    scorers = [s for s in by_a if s in by_b]

    for s in scorers:
        ya, yb = by_a[s]["roc_auc"], by_b[s]["roc_auc"]
        c = color(s)
        ax.plot([0, 1], [ya, yb], color=c, linewidth=2, zorder=3)
        ax.scatter([0, 1], [ya, yb], s=45, color=c, zorder=4)
        ax.annotate(f"{s}  {yb:.3f}", (1, yb), xytext=(8, -3),
                    textcoords="offset points", fontsize=9, color=INK)
        ax.annotate(f"{ya:.3f}", (0, ya), xytext=(-8, -3), ha="right",
                    textcoords="offset points", fontsize=8, color=INK_2)

    ax.set_xlim(-0.35, 1.75)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([name_a, name_b], fontsize=10, color=INK)
    ax.set_ylabel("ROC-AUC (unsupported-claim detection)",
                  color=INK_2, fontsize=9)
    ax.set_title("Same scorers, different era of hallucination",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(outfile, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {outfile}")


def fmt(v, spec=".3f"):
    return "n/a" if v is None else format(v, spec)


def table(rows, heading):
    print(f"\n### {heading}\n")
    has_ci = any(r.get("roc_auc_ci") for r in rows)
    ci_col = " AUC 95% CI |" if has_ci else ""
    print("| scorer | spearman | balanced acc | ROC-AUC |"
          f"{ci_col} median ms/doc | peak VRAM (GB) |")
    print("|---|---|---|---|---|---|" + ("---|" if has_ci else ""))
    for r in rows:
        ci = ""
        if has_ci:
            lo_hi = r.get("roc_auc_ci")
            ci = (f" {lo_hi[0]:.3f}–{lo_hi[1]:.3f} |" if lo_hi
                  else " n/a |")
        print(f"| {r['scorer']} | {fmt(r['spearman'])} "
              f"| {fmt(r['balanced_acc'])} | {fmt(r['roc_auc'])} |{ci} "
              f"{fmt(r['median_latency_ms'], '.1f')} "
              f"| {fmt(r['peak_vram_gb'], '.1f')} |")


def scale_curve(judge_rows, nli_auc, outfile):
    """Judge quality vs VRAM, with NLI as a horizontal reference."""
    fig, ax = new_axes(width=7, height=4.5)
    xs, ys, labels = [], [], []
    for r in judge_rows:
        xs.append(max(r.get("peak_vram_gb") or 0.1, 0.1))
        ys.append(r["roc_auc"])
        labels.append(r["scorer"].replace("llm-judge-", ""))
    ax.plot(xs, ys, color=color("llm-judge"), linewidth=2, zorder=3)
    ax.scatter(xs, ys, s=70, color=color("llm-judge"), zorder=4)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), xytext=(6, 5),
                    textcoords="offset points", fontsize=9, color=INK)
    if nli_auc is not None:
        ax.axhline(nli_auc, color=color("nli-deberta"), linestyle="--",
                   linewidth=1.5, zorder=2)
        ax.annotate(f"nli-deberta  {nli_auc:.3f}",
                    (xs[-1], nli_auc), xytext=(-4, 6),
                    textcoords="offset points", ha="right",
                    fontsize=9, color=color("nli-deberta"))
    ax.set_xlabel("peak VRAM (GB)", color=INK_2, fontsize=9)
    ax.set_ylabel("ROC-AUC", color=INK_2, fontsize=9)
    ax.set_title("How many GB of judge to beat a 1.5 GB NLI model?",
                 color=INK, fontsize=11, loc="left")
    ax.margins(x=0.15, y=0.15)
    fig.tight_layout()
    fig.savefig(outfile, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {outfile}")


def main():
    rows = json.load(open("results.json"))
    scatter(rows, "results.png",
            "Faithfulness scoring: quality vs cost (SummEval, n=1600)")
    table(rows, "SummEval (2019-era summaries, n=1600)")

    if os.path.exists("results-ragtruth.json"):
        rt = json.load(open("results-ragtruth.json"))
        scatter(rt, "results-ragtruth.png",
                "Quality vs cost (RAGTruth summaries)")
        slope(rows, rt, "SummEval\n(2019-era)", "RAGTruth\n(LLM-era)",
              "comparison.png")
        table(rt, "RAGTruth (LLM-era summaries)")

    if os.path.exists("results-scale.json"):
        scale = json.load(open("results-scale.json"))
        nli_auc = None
        if os.path.exists("results-ragtruth.json"):
            for r in json.load(open("results-ragtruth.json")):
                if r["scorer"] == "nli-deberta":
                    nli_auc = r["roc_auc"]
        scale_curve(scale, nli_auc, "scale.png")
        table(scale, "Judge scaling curve (RAGTruth)")


if __name__ == "__main__":
    main()

