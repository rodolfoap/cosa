#!.venv/bin/python3
import argparse
import json
from pathlib import Path

from optscore.checkpoint import load_checkpoint
from optscore.data import generator
from optscore.inference import Scorer
from optscore.training.evaluate import evaluate
from optscore.training.train import TrainConfig, train

DATA_DIR = "datasets"
CKPT = "checkpoints/model.pt"


def cmd_gen(a):
    data = generator.generate_all(a.seed, {"train": a.n_train})
    generator.write_all(Path(a.data), data)
    for split, samples in data.items():
        print(f"{split:15s} {len(samples):6d} -> {a.data}/{split}.jsonl")
    print("example:", json.dumps(data["train"][0].to_dict()))


def cmd_train(a):
    if not (Path(a.data) / "train.jsonl").exists():
        cmd_gen(a)
    data = generator.read_all(a.data)
    cfg = TrainConfig(epochs=a.epochs, seed=a.seed, ckpt_path=a.ckpt, eval_splits=("val",))
    model, vocab, _ = train(cfg, data)
    cmd_eval(a, model, vocab, data)


def cmd_eval(a, model=None, vocab=None, data=None):
    if model is None:
        model, vocab = load_checkpoint(a.ckpt)
        data = generator.read_all(a.data)
    for split in ("val", "test", "test_templates"):
        r = evaluate(model, data[split], vocab)
        print(split, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}))


def cmd_ask(a):
    print(json.dumps(Scorer.from_checkpoint(a.ckpt).probabilities(a.context, a.options), indent=2))


def main():
    p = argparse.ArgumentParser(description="Attention-based option scoring (proof of concept)")
    sub = p.add_subparsers(dest="cmd")

    def common(sp):
        sp.add_argument("--data", default=DATA_DIR)
        sp.add_argument("--ckpt", default=CKPT)
        sp.add_argument("--seed", type=int, default=0)

    g = sub.add_parser("gen", help="generate the synthetic dataset")
    common(g)
    g.add_argument("--n-train", type=int, default=generator.DEFAULT_SIZES["train"])
    g.set_defaults(fn=cmd_gen)

    t = sub.add_parser("train", help="train (generates the dataset if missing) and evaluate")
    common(t)
    t.add_argument("--n-train", type=int, default=generator.DEFAULT_SIZES["train"])
    t.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("eval", help="evaluate a saved checkpoint on val/test/test_templates")
    common(e)
    e.set_defaults(fn=cmd_eval)

    q = sub.add_parser("ask", help='e.g. ask "should i take a beer or a wine? i dont like carbonated beverages" beer wine')
    common(q)
    q.add_argument("context")
    q.add_argument("options", nargs="+")
    q.set_defaults(fn=cmd_ask)

    a = p.parse_args()
    if not a.cmd:
        p.print_help()
        return
    a.fn(a)


if __name__ == "__main__":
    main()
