from collections import Counter

import pytest

from optscore.data import generator, kb, templates
from optscore.data.generator import Sample
from optscore.data.vocab import tokenize


@pytest.fixture(scope="module")
def data():
    return generator.generate_all(seed=0, sizes={s: 1500 for s in generator.SPLITS})


def test_labels_follow_the_rules(data):
    for samples in data.values():
        for s in samples:
            attr_words = kb.ATTRIBUTES[s.attr]
            assert any(w in tokenize(s.context) for w in attr_words), s.context
            with_attr = [o for o in s.options if kb.has_attr(o, s.attr)]
            without = [o for o in s.options if not kb.has_attr(o, s.attr)]
            if s.polarity == "neg":
                assert len(without) == 1 and s.options[s.label] == without[0]
            else:
                assert len(with_attr) == 1 and s.options[s.label] == with_attr[0]


def test_options_are_distinct_known_drinks(data):
    for samples in data.values():
        for s in samples:
            assert 2 <= len(s.options) <= 3
            assert len(set(s.options)) == len(s.options)
            assert set(s.options) <= set(kb.DRINKS)
            assert 0 <= s.label < len(s.options)


def test_options_appear_in_context(data):
    for s in data["train"]:
        for o in s.options:
            assert o in tokenize(s.context)


def test_option_sets_are_disjoint_across_train_val_test(data):
    sets = {sp: {frozenset(s.options) for s in data[sp]} for sp in ("train", "val", "test")}
    assert not sets["train"] & sets["val"]
    assert not sets["train"] & sets["test"]
    assert not sets["val"] & sets["test"]


def test_test_templates_use_test_option_sets_and_held_out_phrasing(data):
    train_stmts = {t for ts in templates.TRAIN_STATEMENTS.values() for t in ts}
    for s in data["test_templates"]:
        assert s.question_tpl in templates.HELD_OUT_QUESTIONS
        assert s.statement_tpl not in train_stmts
    tt_sets = {frozenset(s.options) for s in data["test_templates"]}
    assert not tt_sets & {frozenset(s.options) for s in data["train"]}
    assert not tt_sets & {frozenset(s.options) for s in data["val"]}


def test_train_never_uses_held_out_templates(data):
    for s in data["train"] + data["val"] + data["test"]:
        assert s.question_tpl in templates.TRAIN_QUESTIONS
        assert s.statement_tpl in templates.TRAIN_STATEMENTS[s.polarity]


def test_held_out_statements_are_disjoint_from_train_and_nonempty():
    for pol in ("pos", "neg"):
        train, held = set(templates.TRAIN_STATEMENTS[pol]), set(templates.HELD_OUT_STATEMENTS[pol])
        assert train and held and not train & held


def test_every_prefix_and_tail_appears_in_training():
    for pol, prefixes in templates.PREFIXES.items():
        for p in prefixes:
            assert any(t.startswith(p + " ") for t in templates.TRAIN_STATEMENTS[pol]), p
    for tail in templates.TAILS:
        assert any(t.endswith(" " + tail) for ts in templates.TRAIN_STATEMENTS.values() for t in ts), tail


def test_generation_is_deterministic_and_seed_sensitive():
    a = generator.generate_split("train", 200, seed=3)
    b = generator.generate_split("train", 200, seed=3)
    c = generator.generate_split("train", 200, seed=4)
    assert a == b
    assert a != c


def test_label_positions_are_balanced(data):
    for n in (2, 3):
        pos = Counter(s.label for s in data["train"] if len(s.options) == n)
        total = sum(pos.values())
        assert total > 50
        for i in range(n):
            assert abs(pos[i] / total - 1 / n) < 0.1, (n, pos)


def test_both_polarities_and_option_counts_present(data):
    train = data["train"]
    assert {s.polarity for s in train} == {"pos", "neg"}
    assert {len(s.options) for s in train} == {2, 3}
    assert {s.attr for s in train} == set(kb.ATTRIBUTES)


def test_negation_is_not_a_shortcut(data):
    """Same attribute with opposite polarity must give opposite answers on the same option set."""
    by_key = {}
    for s in data["train"]:
        by_key.setdefault((frozenset(s.options), s.attr), set()).add((s.polarity, s.options[s.label]))
    both = [v for v in by_key.values() if {p for p, _ in v} == {"pos", "neg"}]
    assert both, "dataset must contain contrastive polarity pairs"
    for v in both:
        answers = {p: a for p, a in v}
        if len(v) == 2:
            assert answers["pos"] != answers["neg"]


def test_jsonl_roundtrip(tmp_path, data):
    generator.write_all(tmp_path, data)
    back = generator.read_all(tmp_path)
    assert back == data
    assert isinstance(back["train"][0], Sample)


def test_every_split_has_combos_for_every_size_and_polarity():
    combos = generator.enumerate_combos(0)
    for base in ("train", "val", "test"):
        for k in (2, 3):
            for pol in ("pos", "neg"):
                assert combos.get((base, k, pol)), (base, k, pol)


def test_unknown_split_rejected():
    with pytest.raises(ValueError):
        generator.generate_split("bogus", 1)
