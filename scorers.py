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
