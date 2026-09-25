import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..checkpoint import save_checkpoint
from ..data import generator
from ..data.batching import OptionDataset, collate
from ..data.vocab import Vocab, build_default_vocab
from ..model.scorer import ModelConfig, OptionScorer
from .evaluate import evaluate


@dataclass
class TrainConfig:
    epochs: int = 12
    batch_size: int = 128
    lr: float = 2e-3
    weight_decay: float = 0.01
    seed: int = 0
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    tau: float = 0.1
    ckpt_path: str | None = None
    eval_splits: tuple[str, ...] = ("val",)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(
    cfg: TrainConfig,
    data: dict[str, list[generator.Sample]],
    log: Callable[[str], None] = print,
) -> tuple[OptionScorer, Vocab, list[dict]]:
    set_seed(cfg.seed)
    vocab = build_default_vocab()
    model = OptionScorer(
        ModelConfig(
            vocab_size=len(vocab), d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
            d_ff=cfg.d_ff, dropout=cfg.dropout, tau=cfg.tau,
        )
    )
    ds = OptionDataset(data["train"], vocab, model.cfg.max_len)
    gen = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate, generator=gen)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = cfg.epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * s / total))
    )

    history: list[dict] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for batch in loader:
            logits = model(batch["ctx"], batch["opts"], batch["valid"])
            loss = F.cross_entropy(logits, batch["label"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(batch["label"])
            n += len(batch["label"])
        row = {"epoch": epoch, "train_loss": loss_sum / n}
        for split in cfg.eval_splits:
            if split in data:
                row[split] = evaluate(model, data[split], vocab)
        history.append(row)
        extra = " ".join(f"{s}_acc={row[s]['accuracy']:.3f}" for s in cfg.eval_splits if s in row)
        log(f"epoch {epoch:2d} train_loss={row['train_loss']:.4f} {extra}")

    model.eval()
    if cfg.ckpt_path:
        save_checkpoint(Path(cfg.ckpt_path), model, vocab)
        log(f"saved {cfg.ckpt_path}")
    return model, vocab, history
