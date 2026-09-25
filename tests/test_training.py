import json

import pytest

from optscore.checkpoint import load_checkpoint
from optscore.data import generator
from optscore.inference import Scorer
from optscore.training.evaluate import evaluate
from optscore.training.train import TrainConfig, train

SPEC_CONTEXT = "Should I take a beer or a wine? I don't like carbonated beverages"


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    data = generator.generate_all(0, {"train": 1500, "val": 300, "test": 300, "test_templates": 300})
    ckpt = tmp_path_factory.mktemp("ckpt") / "m.pt"
    model, vocab, history = train(TrainConfig(epochs=3, ckpt_path=str(ckpt)), data, log=lambda _: None)
    return model, vocab, history, data, ckpt


def test_loss_decreases(tiny):
    _, _, history, *_ = tiny
    assert history[-1]["train_loss"] < history[0]["train_loss"]


def test_evaluate_reports_expected_keys(tiny):
    model, vocab, _, data, _ = tiny
    r = evaluate(model, data["val"], vocab)
    assert {"accuracy", "nll", "n", "acc_pol_pos", "acc_pol_neg", "acc_n_2", "acc_n_3"} <= set(r)
    assert 0 <= r["accuracy"] <= 1 and r["n"] == len(data["val"])


def test_checkpoint_roundtrip_gives_identical_outputs(tiny):
    model, vocab, _, _, ckpt = tiny
    model2, vocab2 = load_checkpoint(ckpt)
    assert vocab2.tokens == vocab.tokens
    a = Scorer(model, vocab).probabilities(SPEC_CONTEXT, ["beer", "wine"])
    b = Scorer(model2, vocab2).probabilities(SPEC_CONTEXT, ["beer", "wine"])
    assert a == pytest.approx(b, abs=1e-6)


def test_inference_returns_json_probabilities_over_given_options(tiny):
    model, vocab, *_ = tiny
    out = Scorer(model, vocab).probabilities(SPEC_CONTEXT, ["beer", "wine"])
    assert list(out) == ["beer", "wine"]
    assert sum(out.values()) == pytest.approx(1.0)
    assert all(0 <= p <= 1 for p in out.values())
    json.dumps(out)


def test_inference_accepts_any_option_count_and_rejects_bad_input(tiny):
    model, vocab, *_ = tiny
    s = Scorer(model, vocab)
    assert len(s.probabilities("x", ["beer", "wine", "tea", "milk", "cola"])) == 5
    with pytest.raises(ValueError):
        s.probabilities("x", ["beer"])
    with pytest.raises(ValueError):
        s.probabilities("x", ["beer", "beer"])


def test_training_is_reproducible():
    data = generator.generate_all(0, {"train": 600, "val": 100, "test": 100, "test_templates": 100})
    cfg = TrainConfig(epochs=1, seed=5, eval_splits=())
    _, _, h1 = train(cfg, data, log=lambda _: None)
    _, _, h2 = train(cfg, data, log=lambda _: None)
    assert h1[0]["train_loss"] == pytest.approx(h2[0]["train_loss"], rel=1e-6)


@pytest.fixture(scope="module")
def poc(tmp_path_factory):
    data = generator.generate_all(0)
    ckpt = tmp_path_factory.mktemp("poc") / "m.pt"
    model, vocab, _ = train(TrainConfig(epochs=12, ckpt_path=str(ckpt)), data, log=lambda _: None)
    return model, vocab, data


@pytest.mark.slow
def test_poc_generalizes_to_unseen_option_sets(poc):
    model, vocab, data = poc
    r = evaluate(model, data["test"], vocab)
    assert r["accuracy"] > 0.95, r
    assert r["acc_pol_neg"] > 0.9 and r["acc_pol_pos"] > 0.9, r


@pytest.mark.slow
def test_poc_generalizes_to_unseen_phrasings(poc):
    model, vocab, data = poc
    r = evaluate(model, data["test_templates"], vocab)
    assert r["accuracy"] > 0.85, r


@pytest.mark.slow
def test_poc_answers_the_spec_example(poc):
    model, vocab, _ = poc
    s = Scorer(model, vocab)
    out = s.probabilities(SPEC_CONTEXT, ["beer", "wine"])
    assert out["wine"] > 0.8, out
    flipped = s.probabilities("Should I take a beer or a wine? I want something carbonated", ["beer", "wine"])
    assert flipped["beer"] > 0.8, flipped
