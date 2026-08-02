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
