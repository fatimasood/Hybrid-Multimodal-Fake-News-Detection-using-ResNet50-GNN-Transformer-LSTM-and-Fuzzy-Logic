from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMTextClassifier(nn.Module):
    """LSTM text classifier with sigmoid output, as specified in Table 7."""

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        trainable_embeddings: bool = True,
    ):
        super().__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(embedding_matrix)
        self.embedding.weight.requires_grad = trainable_embeddings
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.pool_projection = nn.Linear(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim * 2, 1)
        self.activation = nn.Sigmoid()

    def forward(self, text_ids: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # text_ids: [batch, sequence_length]
        embedded = self.embedding(text_ids)
        if lengths is not None:
            packed = pack_padded_sequence(
                embedded,
                lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (hidden, _) = self.lstm(packed)
        else:
            _, (hidden, _) = self.lstm(embedded)
        # Concatenate the final LSTM state with masked mean pooling.
        # This keeps the LSTM architecture while improving short-title robustness.
        if lengths is not None:
            mask = (text_ids != 0).unsqueeze(-1)
            summed = (embedded * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1)
            pooled = summed / denom
        else:
            pooled = embedded.mean(dim=1)
        pooled = torch.tanh(self.pool_projection(pooled))
        features = torch.cat([hidden[-1], pooled], dim=1)
        logits = self.output(features).squeeze(-1)
        return self.activation(logits)
