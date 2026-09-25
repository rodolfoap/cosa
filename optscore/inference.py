from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .data.batching import collate, encode_query
from .data.vocab import Vocab
from .model.scorer import OptionScorer


class Scorer:
    def __init__(self, model: OptionScorer, vocab: Vocab):
        self.model = model.eval()
        self.vocab = vocab

    @classmethod
    def from_checkpoint(cls, path: Path | str) -> "Scorer":
        return cls(*load_checkpoint(path))

    @torch.no_grad()
    def probabilities(self, context: str, options: list[str]) -> dict[str, float]:
        if len(options) < 2:
            raise ValueError("need at least two options")
        if len(set(options)) != len(options):
            raise ValueError("options must be distinct")
        ctx, opts = encode_query(self.vocab, context, options, self.model.cfg.max_len)
        batch = collate([(ctx, opts, 0)])
        probs = self.model(batch["ctx"], batch["opts"], batch["valid"]).softmax(-1)[0]
        return {o: p for o, p in zip(options, probs.tolist())}
