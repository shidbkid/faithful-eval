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
