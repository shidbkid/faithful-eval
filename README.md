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

**Answer:** it depends on the *era* and the *task* — and that's the finding.

1. **Era (two LLM-era datasets).** On 2019-era SummEval, NLI beats a 3B judge.
   On LLM-era **RAGTruth Summary**, NLI collapses; **7B-4bit** clears it with
   non-overlapping CIs. On LLM-era **TofuEval** (dialogue summaries), a **3B**
   judge already clears NLI solidly (0.763–0.821 vs 0.675–0.732) — the flip
   replicates off RAGTruth.
2. **Task.** The flip is **not universal** across RAGTruth:
   - **Summary** — judge wins (3B directional, 7B-4bit solid vs NLI)
   - **Data2txt** — judge wins **solidly** (3B CI 0.706–0.771 vs NLI 0.556–0.633)
   - **QA** — ranking reverses again: **ROUGE-L wins** (0.729); even 7B-4bit
     (0.669) fails to beat cheap lexical/NLI baselines
3. **Cascade.** Complementarity is high (~70–75%), but NLI is a weak cheap gate
   on LLM-era text. Task-route QA→ROUGE; on Summary use **1.5B→7B** (95% of
   7B AUC @ 51% latency), not NLI→judge.

![Same scorers, different era of hallucination](comparison.png)

![How many GB of judge to beat NLI on Summary?](scale.png)

![Does the flip hold across RAGTruth tasks?](multitask.png)

Bootstrap AUC CIs from paired resampling of saved predictions
(`run.py --save-preds` → `analyze.py`). All numbers: RTX 4000 Ada (~12 GB).

### SummEval — 2019-era system summaries (n=1600)

| scorer | spearman | balanced acc | ROC-AUC | AUC 95% CI | median ms/doc | peak VRAM (GB) |
|---|---|---|---|---|---|---|
| random | 0.013 | 0.516 | 0.506 | 0.470–0.542 | 0.0 | 0.0 |
| rouge-l | 0.386 | 0.687 | 0.705 | 0.676–0.735 | 4.7 | 0.0 |
| bertscore | 0.353 | 0.710 | 0.750 | 0.723–0.777 | 48.0 | 1.1 |
| nli-deberta | 0.403 | 0.743 | 0.786 | 0.761–0.810 | 77.3 | 1.5 |
| llm-judge (3B) | 0.379 | 0.704 | 0.771 | 0.745–0.796 | 719.2 | 6.6 |

![quality vs cost (SummEval)](results.png)

### RAGTruth — LLM-era summaries (n=900)

| scorer | spearman | balanced acc | ROC-AUC | AUC 95% CI | median ms/doc | peak VRAM (GB) |
|---|---|---|---|---|---|---|
| random | 0.087 | 0.552 | 0.558 | 0.514–0.601 | 0.0 | 0.0 |
| rouge-l | 0.227 | 0.638 | 0.658 | 0.615–0.699 | 7.4 | 0.0 |
| bertscore | 0.202 | 0.632 | 0.640 | 0.598–0.682 | 44.6 | 1.1 |
| nli-deberta | 0.227 | 0.619 | 0.655 | 0.613–0.697 | 108.9 | 0.8 |
| llm-judge (3B) | 0.312 | 0.661 | 0.716 | 0.680–0.751 | 535.3 | 6.8 |

![quality vs cost (RAGTruth)](results-ragtruth.png)

### Judge scaling curve (RAGTruth, n=900)

| scorer | spearman | balanced acc | ROC-AUC | AUC 95% CI | median ms/doc | peak VRAM (GB) |
|---|---|---|---|---|---|---|
| llm-judge-0.5b | 0.177 | 0.592 | 0.622 | 0.578–0.666 | 215 | 1.5 |
| llm-judge-1.5b | 0.319 | 0.669 | **0.720** | 0.679–0.761 | 309 | 3.6 |
| llm-judge-3b | 0.312 | 0.661 | 0.716 | 0.680–0.751 | 541 | 6.8 |
| llm-judge-7b-4bit | **0.432** | **0.739** | **0.786** | **0.750–0.821** | 1174 | 6.7 |

0.5B loses to NLI. 1.5B is the first size whose point estimate clears NLI.
7B-4bit is the first whose CI sits entirely above NLI’s — and it fits in the
same ~7 GB envelope as the 3B via 4-bit quantization.

### RAGTruth by task (n=900 each, 3B judge unless noted)

| task | rouge-l | nli-deberta | llm-judge 3B | llm-judge 7B-4bit | flip vs NLI |
|---|---:|---:|---:|---:|---|
| Summary | 0.658 | 0.655 | 0.716 | **0.786** | 3B directional / 7B **solid** |
| QA | **0.729** | 0.689 | 0.626 | 0.669 | judge **loses** (even at 7B) |
| Data2txt | 0.610 | 0.594 | **0.738** | — | 3B **solid** |

AUC 95% CIs (bootstrap): Summary NLI 0.613–0.697 vs 7B 0.750–0.821
(separated); Data2txt NLI 0.556–0.633 vs 3B 0.706–0.771 (separated); QA
ROUGE 0.691–0.765 vs 3B judge 0.582–0.668 (judge below).

### TofuEval — LLM-era dialogue summaries (n=1498)

Second modern dataset (MeetB + MediaS via gated `lytang/LLM-AggreFact`).
Needs `hf auth login` once.

| scorer | spearman | balanced acc | ROC-AUC | AUC 95% CI | median ms/doc | peak VRAM (GB) |
|---|---|---|---|---|---|---|
| random | -0.014 | 0.504 | 0.490 | 0.454–0.525 | 0.0 | 0.0 |
| rouge-l | 0.319 | 0.679 | 0.724 | 0.692–0.760 | 5.8 | 0.0 |
| bertscore | 0.280 | 0.674 | 0.697 | 0.666–0.730 | 46.3 | 1.1 |
| nli-deberta | 0.290 | 0.672 | 0.704 | 0.675–0.732 | 56.5 | 0.9 |
| llm-judge (3B) | **0.416** | **0.717** | **0.792** | **0.763–0.821** | 207 | 6.8 |

Flip gate NLI vs 3B: **solid** (CIs do not overlap). NLI fails to beat ROUGE
here too — same “cheap detectors stall on fluent LLM text” pattern as RAGTruth
Summary, and this time 3B is enough without a 7B.

### Failure complementarity + cascade (Summary)

At each scorer’s own oracle threshold, NLI vs 3B judge on Summary:

| | count |
|---|---:|
| both correct | 344 |
| both wrong | 138 |
| only NLI wrong (judge saves) | 267 |
| only judge wrong (NLI saves) | 151 |
| complementarity among errors | **75%** |

QA / Data2txt complementarity: 69% / 73%. Naive Summary cascade (ROUGE→NLI→judge
tertile bands): AUC 0.639 at 92 ms/doc — fast, but does not beat NLI alone.

### Task-aware cascade (`cascade.py`)

Complementarity says a cascade should help; the naive one didn't. Next question:
route by **task**, escalate by **confidence**, report one cascade row per task.

| task | policy | ROC-AUC | mean ms | vs target |
|---|---|---|---:|---|
| QA | ROUGE-L only | **0.729** | 4 | = best (judge loses here) |
| Summary | NLI → mid-band 7B-4bit | 0.645 | 471 | 82% of 7B AUC @ 40% latency |
| Data2txt | NLI → mid-band 3B | 0.574 | 950 | 78% of 3B AUC @ 79% latency |

**Task routing works; NLI-gated escalation does not.** On Summary/Data2txt the
NLI confidence band is the wrong cheap gate (skewed near zero on LLM-era text) —
you keep ~⅔ of docs on NLI and the cascade underperforms the full judge.

The cascade that *does* approach 7B quality is judge→judge on Summary:

| policy | ROC-AUC | mean ms | vs 7B-4bit |
|---|---|---:|---|
| **1.5B → mid-band 7B-4bit** | **0.748** | **597** | **95% AUC @ 51% latency** |
| NLI → 7B @ wide q[0.1,0.9] | 0.747 | 961 | 95% AUC but 82% latency (barely a cascade) |

Ceiling is high; the cheap gate has to be a smaller judge, not NLI.

### What this means

- Clumsy / older hallucinations → ship **NLI**.
- Fluent LLM **summaries** (RAGTruth + TofuEval) and **data-to-text** → a
  local judge pays off; on TofuEval **3B is already solid**, on RAGTruth
  Summary **1.5B leads / 7B-4bit decides**. Want most of 7B quality cheaper
  → **1.5B→7B cascade**, not NLI→judge.
- Fluent LLM **QA** → don't bother with a judge; **ROUGE-L** is best here.
- One scorer never wins everywhere — measure the task, then pick.

## Benchmark

Three datasets, same scorer interface. Pick with `--dataset`:

| flag | what it is | label |
|---|---|---|
| `--dataset summeval` (default) | [SummEval](https://github.com/Yale-LILY/SummEval) — 100 CNN/DM articles × 16 **2019-era** system summaries, 3 expert consistency ratings | continuous 1–5; binary = consistency == 5.0 |
| `--dataset ragtruth` | [RAGTruth](https://arxiv.org/abs/2401.00396) (via `wandb/RAGTruth-processed`) — responses from GPT-4 / GPT-3.5 / Llama-2 / Mistral with human hallucination spans | soft label from span count; binary = zero spans |
| `--dataset tofueval` | [TofuEval](https://arxiv.org/abs/2402.13249) via gated [`lytang/LLM-AggreFact`](https://huggingface.co/datasets/lytang/LLM-AggreFact) — MeetB + MediaS dialogue summary claims | binary human label (1 = faithful) |

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
python run.py --only random,rouge-l
python run.py --limit 200
python run.py --dataset ragtruth --save-preds   # writes *.preds.json
python analyze.py --preds results-ragtruth.preds.json
python run.py --dataset ragtruth --judge-model 0.5b,1.5b,3b,7b-4bit \
    --out results-scale.json --save-preds
python cascade.py              # task-aware cascade from saved preds
python run.py --dataset tofueval --out results-tofueval.json --save-preds
python plot.py                 # also builds comparison.png + scale.png
python smoke_test.py
```

TofuEval / LLM-AggreFact is gated: request access on the Hub, then
`hf auth login` before the first `--dataset tofueval` run.

`--judge-model 7b-4bit` needs `bitsandbytes` + `accelerate`. The table and
JSON rewrite after every scorer, so a crash never loses finished rows.

Adding a scorer = implement the interface, append to `SCORERS` in `run.py`.
