"""Sentence templates.

Statements are compositional: prefix x tail. Some (prefix, tail) combinations are held out, but every
prefix and every tail also appears in training, so held-out templates contain no unseen words.
"""

# {a} = attribute surface word, {list} = rendered option list
PREFIXES: dict[str, list[str]] = {
    "neg": ["i dont like", "i hate", "i cannot stand", "i dislike", "no", "please avoid", "nothing"],
    "pos": ["i want", "i like", "i love", "i prefer", "i crave", "give me", "i need"],
}
TAILS: list[str] = ["drinks", "beverages", "things", "stuff"]


def _is_held_out(prefix_idx: int, tail_idx: int) -> bool:
    return (prefix_idx + tail_idx) % 4 == 3


def _build(held_out: bool) -> dict[str, list[str]]:
    return {
        pol: [
            f"{prefix} {{a}} {tail}"
            for i, prefix in enumerate(prefixes)
            for j, tail in enumerate(TAILS)
            if _is_held_out(i, j) == held_out
        ]
        for pol, prefixes in PREFIXES.items()
    }


TRAIN_STATEMENTS: dict[str, list[str]] = _build(held_out=False)
HELD_OUT_STATEMENTS: dict[str, list[str]] = _build(held_out=True)

TRAIN_QUESTIONS: list[str] = [
    "should i take {list}",
    "which one {list}",
    "i can have {list}",
    "what should i choose {list}",
]

HELD_OUT_QUESTIONS: list[str] = [
    "which should i take {list}",
    "should i choose {list}",
]

FILLERS: list[str] = [
    "hello there",
    "it is friday evening",
    "i am at the party",
    "what a day",
]
