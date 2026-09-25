"""Rule-based synthetic dataset generator.

Splits:
  train / val / test  - disjoint drink option-sets (a set is held out if any of its pairs is held out)
  test_templates      - test option-sets rendered with held-out sentence templates
"""
import itertools
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from . import kb, templates

SPLITS = ("train", "val", "test", "test_templates")
DEFAULT_SIZES = {"train": 20000, "val": 2000, "test": 2000, "test_templates": 2000}
P_THREE_OPTIONS = 0.2
P_FILLER = 0.4
VAL_PAIR_FRACTION = 0.10
TEST_PAIR_FRACTION = 0.15


@dataclass
class Sample:
    context: str
    options: list[str]
    label: int
    attr: str
    polarity: str
    statement_tpl: str
    question_tpl: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Sample":
        return cls(**d)


def pair_splits(seed: int) -> dict[frozenset, str]:
    pairs = [frozenset(p) for p in itertools.combinations(sorted(kb.DRINKS), 2)]
    random.Random(f"pairs-{seed}").shuffle(pairs)
    n_test = round(len(pairs) * TEST_PAIR_FRACTION)
    n_val = round(len(pairs) * VAL_PAIR_FRACTION)
    out = {}
    for i, p in enumerate(pairs):
        out[p] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return out


def option_set_split(options: tuple[str, ...], ps: dict[frozenset, str]) -> str:
    kinds = {ps[frozenset(p)] for p in itertools.combinations(options, 2)}
    if "test" in kinds:
        return "test"
    if "val" in kinds:
        return "val"
    return "train"


def is_valid(options: tuple[str, ...], attr: str, polarity: str) -> bool:
    n_with = sum(kb.has_attr(o, attr) for o in options)
    return n_with == 1 if polarity == "pos" else len(options) - n_with == 1


def correct_option(options: list[str] | tuple[str, ...], attr: str, polarity: str) -> str:
    want = polarity == "pos"
    hits = [o for o in options if kb.has_attr(o, attr) == want]
    if len(hits) != 1:
        raise ValueError(f"ambiguous: {options} {attr} {polarity}")
    return hits[0]


def enumerate_combos(seed: int) -> dict[tuple[str, int, str], list[tuple[tuple[str, ...], str]]]:
    """(base_split, n_options, polarity) -> [(sorted option tuple, attr)] with exactly one correct option."""
    ps = pair_splits(seed)
    out: dict = {}
    for k in (2, 3):
        for opts in itertools.combinations(sorted(kb.DRINKS), k):
            split = option_set_split(opts, ps)
            for attr in kb.ATTRIBUTES:
                for pol in ("pos", "neg"):
                    if is_valid(opts, attr, pol):
                        out.setdefault((split, k, pol), []).append((opts, attr))
    return out


def render_list(rng: random.Random, options: list[str]) -> str:
    items = [f"a {o}" for o in options] if rng.random() < 0.5 else list(options)
    return ", ".join(items[:-1]) + " or " + items[-1]


def make_sample(
    rng: random.Random,
    combos: dict,
    base_split: str,
    stmt_tpls: dict[str, list[str]],
    q_tpls: list[str],
) -> Sample:
    k = 3 if rng.random() < P_THREE_OPTIONS else 2
    pol = rng.choice(("pos", "neg"))
    opts, attr = rng.choice(combos[(base_split, k, pol)])
    options = list(opts)
    rng.shuffle(options)
    label = options.index(correct_option(options, attr, pol))

    stmt_tpl = rng.choice(stmt_tpls[pol])
    q_tpl = rng.choice(q_tpls)
    statement = stmt_tpl.replace("{a}", rng.choice(kb.ATTRIBUTES[attr]))
    question = q_tpl.replace("{list}", render_list(rng, options))
    parts = [statement, question] if rng.random() < 0.5 else [question, statement]
    if rng.random() < P_FILLER:
        parts.insert(0, rng.choice(templates.FILLERS))
    return Sample(" ".join(parts), options, label, attr, pol, stmt_tpl, q_tpl)


def generate_split(split: str, n: int, seed: int = 0) -> list[Sample]:
    if split not in SPLITS:
        raise ValueError(split)
    combos = enumerate_combos(seed)
    rng = random.Random(f"{seed}-{split}")
    if split == "test_templates":
        base, stmts, qs = "test", templates.HELD_OUT_STATEMENTS, templates.HELD_OUT_QUESTIONS
    else:
        base, stmts, qs = split, templates.TRAIN_STATEMENTS, templates.TRAIN_QUESTIONS
    return [make_sample(rng, combos, base, stmts, qs) for _ in range(n)]


def generate_all(seed: int = 0, sizes: dict[str, int] | None = None) -> dict[str, list[Sample]]:
    sizes = {**DEFAULT_SIZES, **(sizes or {})}
    return {s: generate_split(s, sizes[s], seed) for s in SPLITS}


def write_jsonl(path: Path, samples: list[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict()) + "\n")


def read_jsonl(path: Path) -> list[Sample]:
    with open(path) as f:
        return [Sample.from_dict(json.loads(line)) for line in f if line.strip()]


def write_all(out_dir: Path, data: dict[str, list[Sample]]) -> None:
    for split, samples in data.items():
        write_jsonl(Path(out_dir) / f"{split}.jsonl", samples)


def read_all(data_dir: Path) -> dict[str, list[Sample]]:
    return {s: read_jsonl(Path(data_dir) / f"{s}.jsonl") for s in SPLITS}
