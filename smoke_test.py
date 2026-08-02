"""Model-free smoke tests for the aggregation logic in the model scorers.

The NLI and LLM-judge scorers were developed on a machine that could not
download model weights, so the parts that don't need weights -- sentence
splitting, chunking, claim aggregation, label-index lookup -- are tested
here with stubbed models.

    python smoke_test.py
"""

from scorers import LLMJudgeScorer, NLIScorer, _sent_split


class StubNLI(NLIScorer):
    """NLIScorer with the transformer swapped for token overlap."""

    def __init__(self):  # no model load
        self.batch_size = 8

    def _entail_probs(self, pairs):
        out = []
        for premise, hyp in pairs:
            h = set(hyp.lower().split())
            p = set(premise.lower().split())
            out.append(len(h & p) / max(1, len(h)))
        return out


class StubJudge(LLMJudgeScorer):
    def __init__(self):  # no model load
        self.max_source_chars = 6000

    def _p_yes(self, source, claim):
        c = set(claim.lower().split())
        s = set(source.lower().split())
        return len(c & s) / max(1, len(c))


def main():
    src = ("The council met on Tuesday. It voted 7-2 to approve the bike "
           "lane network. Construction begins in March.")
    faithful = "The council voted 7-2 to approve the bike lane network."
    mixed = ("The council voted 7-2 to approve the bike lane network. "
             "Zebras invaded the chamber afterwards.")

    assert len(_sent_split(src)) == 3
    assert _sent_split("") == []

    nli = StubNLI()
    s_faithful = nli.score(src, faithful)
    s_mixed = nli.score(src, mixed)
    # min-aggregation: one bad claim must drag the whole summary down
    assert s_faithful > 0.9, s_faithful
    assert s_mixed < 0.5, s_mixed

    judge = StubJudge()
    j_faithful = judge.score(src, faithful)
    j_mixed = judge.score(src, mixed)
    # mean-aggregation: bad claim lowers but doesn't zero the score
    assert j_faithful > 0.9, j_faithful
    assert s_mixed < j_mixed < j_faithful, (s_mixed, j_mixed, j_faithful)

    # single-sentence summary handled (no claim split possible)
    assert 0.0 <= nli.score(src, "Construction begins in March.") <= 1.0

    print("smoke tests passed")


if __name__ == "__main__":
    main()
