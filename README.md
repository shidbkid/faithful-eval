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

Full SummEval run (n=1600) on a local NVIDIA RTX 4000 Ada (~12 GB). The LLM
judge auto-selected `Qwen/Qwen2.5-3B-Instruct`.

| scorer | spearman | balanced acc | ROC-AUC | median ms/doc | peak VRAM (GB) |
|---|---|---|---|---|---|
| random | 0.013 | 0.516 | 0.506 | 0.0 | 0.0 |
| rouge-l | 0.386 | 0.687 | 0.705 | 4.7 | 0.0 |
| bertscore | 0.353 | 0.710 | 0.750 | 48.0 | 1.1 |
| nli-deberta | 0.403 | 0.743 | 0.786 | 77.3 | 1.5 |
| llm-judge | 0.379 | 0.704 | 0.771 | 719.2 | 6.6 |

![quality vs cost](results.png)

NLI wins on Spearman, balanced accuracy, and ROC-AUC while using ~4× less
VRAM and ~9× less latency than the local LLM judge. BERTScore underperforms
ROUGE-L on Spearman despite higher AUC cost. Latencies are only comparable
within a single machine's run.

## Benchmark

**Dataset:** [SummEval](https://github.com/Yale-LILY/SummEval) — 100
CNN/DailyMail articles × 16 system summaries, each rated 1–5 for
*consistency* by 3 experts. Mean expert consistency is the faithfulness
label. The loader fetches the processed copy vendored in the
[BARTScore](https://github.com/neulab/BARTScore) repo (Apache-2.0), which
pairs every summary with its source article; it is cached under `data/`.

**Metrics per scorer:**

- **spearman** — rank correlation of scorer output with mean expert consistency (1600 pairs).
- **balanced acc / ROC-AUC** — binary task: *faithful* iff consistency = 5.0 (81.6% of pairs; all three experts rated it perfectly consistent). Balanced accuracy is reported at the best threshold over the scorer's own outputs — an oracle threshold, applied identically to every scorer.
- **median ms/doc** — median wall-clock per `score(source, summary)` call.
- **peak VRAM** — `torch.cuda.max_memory_allocated`, reset before each scorer; `n/a` on CPU.

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

```bash
pip install -r requirements.txt
python run.py            # full benchmark -> table on stdout + results.json
python plot.py           # results.json -> results.png + markdown table
```

`python run.py` downloads the dataset on first use and runs every scorer in
`SCORERS` (model weights fetched from Hugging Face on first use; the two
model scorers want a CUDA GPU). The NLI scorer needs ~2–4 GB of VRAM; the
LLM judge picks the largest Qwen2.5 Instruct model that fits your card
(7B needs ~16 GB, 3B ~7 GB, 1.5B ~4 GB — pass `model_name` to override).
Useful during development:

```bash
python run.py --only random,rouge-l      # subset of scorers
python run.py --limit 200                # seeded random subsample of pairs
python smoke_test.py                     # model-free tests of scorer logic
```

The table is rewritten and `results.json` re-dumped after every scorer, so a
crash in scorer N never loses scorers 1..N-1.

Adding a scorer = implement the interface, append the class to `SCORERS` in
`run.py`. Nothing else changes.
