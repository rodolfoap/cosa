import torch
from torch.utils.data import Dataset

from .generator import Sample
from .vocab import CLS_ID, PAD_ID, Vocab

Item = tuple[list[int], list[list[int]], int]


def encode_query(vocab: Vocab, context: str, options: list[str], max_len: int = 48) -> tuple[list[int], list[list[int]]]:
    return vocab.encode(context, max_len), [vocab.encode(o, max_len) for o in options]


class OptionDataset(Dataset):
    def __init__(self, samples: list[Sample], vocab: Vocab, max_len: int = 48):
        self.items: list[Item] = []
        for s in samples:
            ctx, opts = encode_query(vocab, s.context, s.options, max_len)
            self.items.append((ctx, opts, s.label))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Item:
        return self.items[i]


def _pad(seqs: list[list[int]], length: int) -> list[list[int]]:
    return [s + [PAD_ID] * (length - len(s)) for s in seqs]


def collate(items: list[Item]) -> dict[str, torch.Tensor]:
    """Pads contexts and options; missing option slots become a lone <cls> and are masked out via `valid`."""
    b = len(items)
    n_max = max(len(opts) for _, opts, _ in items)
    t_ctx = max(len(ctx) for ctx, _, _ in items)
    t_opt = max(len(o) for _, opts, _ in items for o in opts)

    ctx = torch.tensor(_pad([c for c, _, _ in items], t_ctx), dtype=torch.long)
    opt_rows, valid = [], torch.zeros(b, n_max, dtype=torch.bool)
    for i, (_, opts, _) in enumerate(items):
        valid[i, : len(opts)] = True
        padded = opts + [[CLS_ID]] * (n_max - len(opts))
        opt_rows.append(_pad(padded, t_opt))
    return {
        "ctx": ctx,
        "opts": torch.tensor(opt_rows, dtype=torch.long),
        "valid": valid,
        "label": torch.tensor([lab for _, _, lab in items], dtype=torch.long),
    }
