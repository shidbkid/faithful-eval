# faithful-eval

How well do summary-faithfulness scorers that run **entirely on local
hardware** detect unsupported claims — and what do they cost in latency and
VRAM?

Faithfulness benchmarks usually report correlation with human judgment and
stop there, assuming an API-based frontier judge. Nobody reports
quality-per-millisecond-per-gigabyte for scorers you can run inside a
customer's firewall. This repo measures that trade-off.

This is an independent, personal-time project built only on public datasets,
public models, and public tooling.

## Results

**Answer:** it depends on the era of the hallucination — and that's the
finding. On 2019-era summaries (SummEval), a ~1.5 GB NLI model beats a 3B LLM
judge on every quality metric at ~9× less latency. On LLM-era summaries
(RAGTruth), the ranking flips: NLI degrades sharply and the judge takes the
lead. The cheap detector stops working as the hallucinations get more fluent.

![Same scorers, different era of hallucination](comparison.png)

The judge leads and NLI clearly degrades, though their RAGTruth AUC confidence
intervals overlap (NLI 0.615–0.695 vs judge 0.680–0.752) — the flip is
**directional, not yet decisive**. CIs are Hanley–McNeil 95% approximations
from (AUC, n_pos, n_neg); see `_add_auc_ci.py`.

### SummEval — 2019-era system summaries (n=1600)

| scorer | spearman | balanced acc | ROC-AUC | AUC 95% CI | median ms/doc | peak VRAM (GB) |
|---|---|---|---|---|---|---|
| random | 0.013 | 0.516 | 0.506 | 0.470–0.542 | 0.0 | 0.0 |
| rouge-l | 0.386 | 0.687 | 0.705 | 0.676–0.735 | 4.7 | 0.0 |
| bertscore | 0.353 | 0.710 | 0.750 | 0.723–0.777 | 48.0 | 1.1 |
| nli-deberta | 0.403 | 0.743 | 0.786 | 0.761–0.810 | 77.3 | 1.5 |
| llm-judge | 0.379 | 0.704 | 0.771 | 0.745–0.796 | 719.2 | 6.6 |

![quality vs cost (SummEval)](results.png)

### RAGTruth — LLM-era summaries (n=900)

| scorer | spearman | balanced acc | ROC-AUC | AUC 95% CI | median ms/doc | peak VRAM (GB) |
|---|---|---|---|---|---|---|
| random | 0.087 | 0.552 | 0.558 | 0.514–0.602 | 0.0 | 0.0 |
| rouge-l | 0.227 | 0.638 | 0.658 | 0.618–0.698 | 7.9 | 0.0 |
| bertscore | 0.202 | 0.632 | 0.640 | 0.599–0.681 | 48.2 | 1.1 |
| nli-deberta | 0.227 | 0.619 | 0.655 | 0.615–0.695 | 403.2 | 0.8 |
| llm-judge | 0.312 | 0.661 | 0.716 | 0.680–0.752 | 1853.6 | 6.8 |

![quality vs cost (RAGTruth)](results-ragtruth.png)

Both runs on a single RTX 4000 Ada (~12 GB); the judge auto-selected
`Qwen/Qwen2.5-3B-Instruct`. Latencies comparable only within one machine.

### What this means

If your summaries come from older, weaker models — or your hallucinations are
of the clumsy kind (wrong entities, garbled repetition) — a small NLI model is
the clear on-prem choice. If your summaries come from modern LLMs, whose
fabrications are fluent and plausible, the shallow entailment check loses its
edge and a local LLM judge becomes worth its ~4× VRAM and ~9× latency.
BERTScore is dominated on both datasets, and ROUGE-L remains an embarrassingly
strong floor for a CPU-only metric from 2004.

## Benchmark

Two datasets, same scorer interface. Pick with `--dataset`:

| flag | what it is | label |
|---|---|---|
| `--dataset summeval` (default) | [SummEval](https://github.com/Yale-LILY/SummEval) — 100 CNN/DM articles × 16 **2019-era** system summaries, 3 expert consistency ratings | continuous 1–5; binary = consistency == 5.0 |
| `--dataset ragtruth` | [RAGTruth](https://arxiv.org/abs/2401.00396) (via `wandb/RAGTruth-processed`) — responses from GPT-4 / GPT-3.5 / Llama-2 / Mistral with human hallucination spans | soft label from span count; binary = zero spans |

RAGTruth defaults to the **Summary** task (900 test pairs) so the comparison
stays summary-faithfulness; pass `--task all` for QA + Data2txt too. SummEval
is fetched via the [BARTScore](https://github.com/neulab/BARTScore) vendored
pickle (Apache-2.0) and cached under `data/`.

**Metrics per scorer:**

- **spearman** — rank correlation of scorer output with the dataset's continuous label.
- **balanced acc / ROC-AUC** — binary faithfulness (see table above). Balanced accuracy uses an oracle threshold over the scorer's own outputs — same treatment for every scorer.
- **median ms/doc** — median wall-clock per `score(source, summary)` call.
- **peak VRAM** — `torch.cuda.max_memory_allocated`, reset before each scorer; `0.0` means CPU-only.

## Scorers

Everything implements one interface (`scorers.py`):

```python
class Scorer:
    name: str
    def score(self, source: str, summary: str) -> float: ...
```

| scorer | approach |
|---|---|
| `random` | uniform noise; floor for every metric |
| `rouge-l` | ROUGE-Lsum recall of the summary *against the source* — fraction of the summary supported lexically (note: not summary-vs-reference ROUGE, which measures relevance) |
| `bertscore` | BERTScore precision of summary vs source (roberta-large embeddings) |
| `nli-deberta` | SummaC-style zero-shot NLI (DeBERTa-v3 MNLI): min over summary sentences of max entailment over overlapping 2-sentence source chunks |
| `llm-judge` | Qwen2.5-Instruct (7B/3B/1.5B, auto-sized to detected VRAM) verifies each summary sentence against the article; score = mean P("yes") read from first-token logits |

## Reproducing

Needs Python 3.10+ and a CUDA GPU for the model scorers. Install a CUDA build
of PyTorch for your platform ([pytorch.org](https://pytorch.org)), then:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -r requirements.txt          # or: pip install -r requirements.lock.txt
python run.py            # full benchmark -> table on stdout + results.json
python plot.py           # results.json -> results.png + markdown table
```

`python run.py` downloads the dataset on first use and runs every scorer in
`SCORERS` (model weights fetched from Hugging Face on first use). The NLI
scorer needs ~2–4 GB of VRAM; the LLM judge picks the largest Qwen2.5 Instruct
model that fits your card (7B needs ~16 GB, 3B ~7 GB, 1.5B ~4 GB — pass
`model_name` to override). If NLTK raises an import-security error with a
project-local `.venv` on Python 3.14+, set
`NLTK_DISABLE_IMPORT_SECURITY=1` before running. Useful during development:

```bash
python run.py --only random,rouge-l      # subset of scorers
python run.py --limit 200                # seeded random subsample of pairs
python run.py --dataset ragtruth         # LLM-era RAGTruth Summary (900 pairs)
python run.py --dataset ragtruth --task all --limit 200
python smoke_test.py                     # model-free tests of scorer logic
```

The table is rewritten and `results.json` re-dumped after every scorer, so a
crash in scorer N never loses scorers 1..N-1.

Adding a scorer = implement the interface, append the class to `SCORERS` in
`run.py`. Nothing else changes.
