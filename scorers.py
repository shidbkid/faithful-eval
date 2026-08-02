"""Faithfulness scorers. One interface, no more.

Every scorer maps (source, summary) -> float where higher = more faithful.
Heavy dependencies are imported lazily inside each scorer so that running
the cheap scorers never loads torch.
"""

import random


class Scorer:
    name: str = "base"

    def score(self, source: str, summary: str) -> float:
        raise NotImplementedError


class RandomScorer(Scorer):
    """Dummy scorer. Exists to prove the pipeline works end to end."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def score(self, source: str, summary: str) -> float:
        return self.rng.random()


class RougeLScorer(Scorer):
    """Lexical baseline: ROUGE-L recall of the summary against the source.

    Measures how much of the summary is copied from the article. Cheap,
    CPU-only, and expected to be a weak-but-nonzero faithfulness signal.
    """

    name = "rouge-l"

    def __init__(self):
        import nltk
        from rouge_score import rouge_scorer
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        self._scorer = rouge_scorer.RougeScorer(["rougeLsum"],
                                                use_stemmer=True,
                                                split_summaries=True)

    def score(self, source: str, summary: str) -> float:
        # target=summary, prediction=source is deliberate: recall is
        # computed against the target, so this returns the fraction of the
        # summary's longest-common-subsequence covered by the article --
        # i.e. how much of the summary is supported by the source. That is
        # the faithfulness direction (plain summary-vs-reference ROUGE is
        # a relevance measure, not a faithfulness one).
        result = self._scorer.score(target=summary, prediction=source)
        return result["rougeLsum"].recall


class BERTScoreScorer(Scorer):
    """Embedding baseline: BERTScore precision of the summary vs the source.

    Precision is the faithfulness direction: it asks how well each summary
    token is matched by some source token in embedding space.
    """

    name = "bertscore"

    def __init__(self, model_type: str = "roberta-large", device: str = None):
        from bert_score import BERTScorer
        self._scorer = BERTScorer(model_type=model_type, lang="en",
                                  device=device)

    def score(self, source: str, summary: str) -> float:
        p, r, f = self._scorer.score(cands=[summary], refs=[source])
        return p[0].item()


def _sent_split(text: str):
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]


class NLIScorer(Scorer):
    """SummaC-style zero-shot NLI scorer.

    Split the summary into sentences and the source into overlapping
    2-sentence chunks. For each summary sentence take the max entailment
    probability over all source chunks (its best supporting evidence),
    then take the min over summary sentences: a summary is only as
    faithful as its least-supported claim.
    """

    name = "nli-deberta"

    def __init__(self,
                 model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                 device: str = None,
                 batch_size: int = 32,
                 max_length: int = 512):
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name).to(self.device).eval()
        # Find the "entailment" logit index from the model config instead
        # of hardcoding it -- MNLI checkpoints disagree on label order.
        self.entail_idx = next(
            i for i, lbl in self.model.config.id2label.items()
            if lbl.lower().startswith("entail"))

    def _entail_probs(self, pairs):
        """pairs: list of (premise, hypothesis) -> list of P(entailment)."""
        import torch
        probs = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            enc = self.tokenizer([p for p, h in batch],
                                 [h for p, h in batch],
                                 truncation=True,
                                 max_length=self.max_length,
                                 padding=True,
                                 return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
            probs.extend(torch.softmax(logits, dim=-1)[:, self.entail_idx]
                         .tolist())
        return probs

    def score(self, source: str, summary: str) -> float:
        src_sents = _sent_split(source)
        # Overlapping 2-sentence chunks so claims supported across a
        # sentence boundary still find their evidence.
        chunks = [" ".join(src_sents[i:i + 2])
                  for i in range(max(1, len(src_sents) - 1))]
        claims = _sent_split(summary) or [summary]

        pairs = [(c, claim) for claim in claims for c in chunks]
        probs = self._entail_probs(pairs)

        per_claim = []
        n = len(chunks)
        for j in range(len(claims)):
            per_claim.append(max(probs[j * n:(j + 1) * n]))
        return min(per_claim)


class LLMJudgeScorer(Scorer):
    """Local LLM judge: claim-level verification with a small instruct model.

    Splits the summary into sentences (claims). For each claim, asks the
    model whether the article fully supports it, and reads the probability
    of the "yes" token from the first-token logits (no sampling, no
    parsing). Score = mean P(yes) across claims -- graded, monotone, and
    robust to the model refusing to emit clean JSON.
    """

    name = "llm-judge"

    PROMPT = (
        "You are verifying a summary claim against a news article.\n\n"
        "ARTICLE:\n{source}\n\n"
        "CLAIM:\n{claim}\n\n"
        "Is the claim fully supported by the article, with no invented or "
        "contradicted details? Answer with exactly one word, yes or no."
    )

    def __init__(self,
                 model_name: str = None,
                 device: str = None,
                 max_source_chars: int = 6000):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model_name is None:
            model_name = self._pick_model()
            print(f"llm-judge: auto-selected {model_name}")
        self.max_source_chars = max_source_chars
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        ).to(self.device).eval()
        self.yes_id = self.tokenizer.encode("yes",
                                            add_special_tokens=False)[0]
        self.no_id = self.tokenizer.encode("no", add_special_tokens=False)[0]

    def _pick_model(self) -> str:
        """Largest Qwen2.5 instruct model that fits the detected VRAM.

        bf16 rule of thumb: weights ~2 bytes/param + activations/cache.
        Explicitly pass model_name to override.
        """
        import torch
        vram_gb = 0.0
        if self.device.startswith("cuda") and torch.cuda.is_available():
            vram_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9)
        if vram_gb >= 20:
            return "Qwen/Qwen2.5-7B-Instruct"     # ~16 GB in bf16
        if vram_gb >= 10:
            return "Qwen/Qwen2.5-3B-Instruct"     # ~7 GB
        return "Qwen/Qwen2.5-1.5B-Instruct"       # ~4 GB, also the CPU pick

    def _p_yes(self, source: str, claim: str) -> float:
        import torch
        messages = [{"role": "user",
                     "content": self.PROMPT.format(
                         source=source[: self.max_source_chars],
                         claim=claim)}]
        ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(ids).logits[0, -1]
        two = torch.softmax(
            torch.stack([logits[self.yes_id], logits[self.no_id]]), dim=0)
        return two[0].item()

    def score(self, source: str, summary: str) -> float:
        claims = _sent_split(summary) or [summary]
        return sum(self._p_yes(source, c) for c in claims) / len(claims)
