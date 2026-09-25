import torch

from optscore.data.batching import collate
from optscore.data.vocab import CLS_ID, PAD_ID, build_default_vocab
from optscore.model.scorer import ModelConfig, OptionScorer


def make_model(**kw):
    torch.manual_seed(0)
    v = build_default_vocab()
    return OptionScorer(ModelConfig(vocab_size=len(v), dropout=0.0, **kw)), v


def batch_of(v, queries):
    items = []
    for ctx, opts in queries:
        items.append((v.encode(ctx), [v.encode(o) for o in opts], 0))
    return collate(items)


def test_output_shape_and_unit_norm_encoding():
    m, v = make_model()
    m.eval()
    b = batch_of(v, [("i want something fizzy should i take a beer or a wine", ["beer", "wine"])])
    assert m(b["ctx"], b["opts"], b["valid"]).shape == (1, 2)
    enc = m.encode(b["ctx"])
    assert torch.allclose(enc.norm(dim=-1), torch.ones(1), atol=1e-5)


def test_logits_bounded_by_inverse_temperature():
    m, v = make_model(tau=0.1)
    m.eval()
    b = batch_of(v, [("beer or wine", ["beer", "wine"])])
    assert m(b["ctx"], b["opts"], b["valid"]).abs().max() <= 10.0 + 1e-4


def test_variable_option_counts_are_masked():
    m, v = make_model()
    m.eval()
    b = batch_of(v, [("a", ["beer", "wine"]), ("a", ["beer", "wine", "tea"])])
    logits = m(b["ctx"], b["opts"], b["valid"])
    assert logits[0, 2] == float("-inf")
    probs = logits.softmax(-1)
    assert probs[0, 2] == 0
    assert torch.allclose(probs.sum(-1), torch.ones(2))
    assert not torch.isnan(logits).any()


def test_padding_does_not_change_result():
    m, v = make_model()
    m.eval()
    ctx = torch.tensor([v.encode("i want something fizzy")])
    opts = torch.tensor([[v.encode("beer"), v.encode("wine")]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    base = m(ctx, opts, valid)
    ctx_p = torch.cat([ctx, torch.full((1, 7), PAD_ID)], dim=1)
    opts_p = torch.cat([opts, torch.full((1, 2, 5), PAD_ID)], dim=2)
    assert torch.allclose(base, m(ctx_p, opts_p, valid), atol=1e-5)


def test_batch_composition_does_not_change_result():
    m, v = make_model()
    m.eval()
    q1 = ("i want something fizzy should i take a beer or a wine", ["beer", "wine"])
    q2 = ("hello there no sweet drinks which one cola tea milk", ["cola", "tea", "milk"])
    alone = m(**{k: t for k, t in batch_of(v, [q1]).items() if k != "label"})
    both = m(**{k: t for k, t in batch_of(v, [q1, q2]).items() if k != "label"})
    assert torch.allclose(alone[0], both[0, :2], atol=1e-5)


def test_permuting_options_permutes_probabilities():
    m, v = make_model()
    m.eval()
    ctx = "no fizzy drinks please should i take beer or wine or tea"
    a = batch_of(v, [(ctx, ["beer", "wine", "tea"])])
    b = batch_of(v, [(ctx, ["tea", "beer", "wine"])])
    pa = m(a["ctx"], a["opts"], a["valid"]).softmax(-1)[0]
    pb = m(b["ctx"], b["opts"], b["valid"]).softmax(-1)[0]
    assert torch.allclose(pa[[2, 0, 1]], pb, atol=1e-5)


def test_gradients_reach_every_parameter():
    m, v = make_model()
    m.train()
    b = batch_of(v, [("i want something fizzy should i take a beer or a wine", ["beer", "wine"])])
    loss = torch.nn.functional.cross_entropy(m(b["ctx"], b["opts"], b["valid"]), b["label"])
    loss.backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, missing


def test_encoder_is_bidirectional():
    """Changing a later token must change the <cls> representation (no causal mask)."""
    m, v = make_model()
    m.eval()
    a = torch.tensor([v.encode("beer wine tea")])
    b = torch.tensor([v.encode("beer wine cola")])
    assert not torch.allclose(m.encode(a), m.encode(b), atol=1e-6)


def test_collate_pads_missing_options_with_lone_cls():
    v = build_default_vocab()
    b = batch_of(v, [("a", ["beer", "wine"]), ("a", ["beer", "wine", "tea"])])
    assert b["valid"].tolist() == [[True, True, False], [True, True, True]]
    assert b["opts"][0, 2].tolist()[0] == CLS_ID
    assert (b["opts"][0, 2, 1:] == PAD_ID).all()
