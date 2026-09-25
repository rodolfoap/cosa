from pathlib import Path

import torch

from .data.vocab import Vocab
from .model.scorer import ModelConfig, OptionScorer


def save_checkpoint(path: Path | str, model: OptionScorer, vocab: Vocab) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state": model.state_dict(), "config": model.cfg.to_dict(), "vocab": vocab.tokens},
        path,
    )


def load_checkpoint(path: Path | str) -> tuple[OptionScorer, Vocab]:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    model = OptionScorer(ModelConfig(**blob["config"]))
    model.load_state_dict(blob["state"])
    model.eval()
    return model, Vocab(blob["vocab"])
