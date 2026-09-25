from optscore.data import generator, templates
from optscore.data.vocab import CLS_ID, PAD_ID, UNK_ID, Vocab, build_default_vocab, tokenize, train_template_words


def test_tokenize_handles_apostrophes_and_punctuation():
    assert tokenize("Should I take a beer or a wine? I don't like carbonated beverages.") == [
        "should", "i", "take", "a", "beer", "or", "a", "wine", "i", "dont", "like", "carbonated", "beverages",
    ]


def test_special_token_ids_are_fixed():
    v = build_default_vocab()
    assert (v.tokens[PAD_ID], v.tokens[CLS_ID], v.tokens[UNK_ID]) == ("<pad>", "<cls>", "<unk>")


def test_encode_prepends_cls_maps_unknown_and_truncates():
    v = build_default_vocab()
    ids = v.encode("beer zzzunknown wine")
    assert ids[0] == CLS_ID and ids[2] == UNK_ID and len(ids) == 4
    assert len(v.encode("beer " * 100, max_len=10)) == 10


def test_vocab_requires_specials_first():
    try:
        Vocab(["beer"])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_no_generated_sample_has_unknown_tokens():
    v = build_default_vocab()
    for split, samples in generator.generate_all(0, {k: 300 for k in generator.SPLITS}).items():
        for s in samples:
            assert UNK_ID not in v.encode(s.context), (split, s.context)
            for o in s.options:
                assert UNK_ID not in v.encode(o)


def test_held_out_templates_only_use_words_seen_in_training():
    seen = train_template_words()
    held = list(templates.HELD_OUT_QUESTIONS)
    for tpls in templates.HELD_OUT_STATEMENTS.values():
        held.extend(tpls)
    for tpl in held:
        words = set(tokenize(tpl.replace("{a}", " ").replace("{list}", " ")))
        assert words <= seen, (tpl, words - seen)
