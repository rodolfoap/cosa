from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..data.vocab import PAD_ID


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    max_len: int = 48
    dropout: float = 0.1
    tau: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


class OptionScorer(nn.Module):
    """Shared bidirectional attention encoder; softmax over dot-product scores of context vs each option."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=PAD_ID)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: [B, T] starting with <cls>. Returns unit vectors [B, d]."""
        pos = torch.arange(ids.size(1), device=ids.device)
        h = self.tok(ids) + self.pos(pos)
        h = self.encoder(h, src_key_padding_mask=ids == PAD_ID)
        return F.normalize(self.proj(h[:, 0]), dim=-1)

    def forward(self, ctx: torch.Tensor, opts: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """ctx [B,Tc], opts [B,N,To], valid [B,N] -> logits [B,N] (-inf on invalid slots)."""
        b, n, t = opts.shape
        c = self.encode(ctx)
        o = self.encode(opts.reshape(b * n, t)).view(b, n, -1)
        logits = torch.einsum("bd,bnd->bn", c, o) / self.cfg.tau
        return logits.masked_fill(~valid, float("-inf"))
