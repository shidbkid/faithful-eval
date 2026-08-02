"""Dataset loading.

Each example is a dict: {"source": str, "summary": str, "label": float}
where label is the human consistency rating (1-5, higher = more faithful).
"""

TOY = [
    {
        "source": "The city council voted 7-2 on Tuesday to approve the new "
                  "bike lane network downtown. Construction begins in March "
                  "and is expected to cost $4.2 million.",
        "summary": "The council approved a downtown bike lane network in a "
                   "7-2 vote; construction starts in March.",
        "label": 5.0,
    },
    {
        "source": "The city council voted 7-2 on Tuesday to approve the new "
                  "bike lane network downtown. Construction begins in March "
                  "and is expected to cost $4.2 million.",
        "summary": "The council unanimously rejected the bike lane proposal, "
                   "citing its $10 million price tag.",
        "label": 1.0,
    },
    {
        "source": "Researchers at the university published a study showing "
                  "that the local bat population has declined 40% since 2019, "
                  "likely due to white-nose syndrome.",
        "summary": "A university study found bat numbers have fallen 40% "
                   "since 2019, probably because of white-nose syndrome.",
        "label": 5.0,
    },
    {
        "source": "Researchers at the university published a study showing "
                  "that the local bat population has declined 40% since 2019, "
                  "likely due to white-nose syndrome.",
        "summary": "Bats in the region are thriving, with the university "
                   "reporting record population growth this year.",
        "label": 1.0,
    },
]


def load_toy():
    """Tiny hardcoded set used while the harness was being built."""
    return list(TOY)
