from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class GNNEmbeddedTransformerCaptioner(nn.Module):
    """Three encoder and decoder layers for image caption generation."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.positional = SinusoidalPositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.vocab_projection = nn.Linear(d_model, vocab_size)

    def forward(self, gnn_nodes: torch.Tensor, caption_input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # gnn_nodes: [batch, nodes, d_model]
        # caption_input_ids: [batch, target_length]
        tgt = self.positional(self.token_embedding(caption_input_ids))
        memory = self.transformer.encoder(gnn_nodes)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(caption_input_ids.shape[1]).to(caption_input_ids.device)
        decoded = self.transformer.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
        return self.vocab_projection(decoded), memory

    @torch.no_grad()
    def greedy_decode(self, gnn_nodes: torch.Tensor, bos_idx: int, eos_idx: int, max_len: int) -> torch.Tensor:
        batch = gnn_nodes.shape[0]
        generated = torch.full((batch, 1), bos_idx, dtype=torch.long, device=gnn_nodes.device)
        memory = self.transformer.encoder(gnn_nodes)
        for _ in range(max_len - 1):
            tgt = self.positional(self.token_embedding(generated))
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(generated.shape[1]).to(gnn_nodes.device)
            decoded = self.transformer.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
            next_token = self.vocab_projection(decoded[:, -1]).argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if torch.all(next_token.squeeze(1) == eos_idx):
                break
        return generated

