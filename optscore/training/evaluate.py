from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.batching import OptionDataset, collate
from ..data.generator import Sample
from ..data.vocab import Vocab
from ..model.scorer import OptionScorer


@torch.no_grad()
def evaluate(model: OptionScorer, samples: list[Sample], vocab: Vocab, batch_size: int = 512) -> dict:
    """Accuracy and NLL overall, plus accuracy by polarity and option count (polarity split exposes shortcut learning)."""
    was_training = model.training
    model.eval()
    ds = OptionDataset(samples, vocab, model.cfg.max_len)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate)

    total_nll, correct = 0.0, 0
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    i = 0
    for batch in loader:
        logits = model(batch["ctx"], batch["opts"], batch["valid"])
        total_nll += F.cross_entropy(logits, batch["label"], reduction="sum").item()
        hit = (logits.argmax(-1) == batch["label"]).tolist()
        for h in hit:
            s = samples[i]
            i += 1
            correct += h
            for key in (f"pol_{s.polarity}", f"n_{len(s.options)}"):
                groups[key][0] += h
                groups[key][1] += 1
    if was_training:
        model.train()
    n = len(samples)
    out = {"accuracy": correct / n, "nll": total_nll / n, "n": n}
    out.update({f"acc_{k}": c / t for k, (c, t) in sorted(groups.items())})
    return out
