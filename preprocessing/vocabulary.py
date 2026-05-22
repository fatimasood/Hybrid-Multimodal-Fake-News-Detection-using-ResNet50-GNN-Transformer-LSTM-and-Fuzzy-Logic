from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"


@dataclass
class Vocabulary:
    token_to_idx: dict[str, int]

    @property
    def idx_to_token(self) -> list[str]:
        items = sorted(self.token_to_idx.items(), key=lambda item: item[1])
        return [token for token, _ in items]

    def __len__(self) -> int:
        return len(self.token_to_idx)

    def encode(self, tokens: list[str], max_length: int, add_bos_eos: bool = False) -> list[int]:
        if add_bos_eos:
            tokens = [BOS_TOKEN, *tokens, EOS_TOKEN]
        ids = [self.token_to_idx.get(token, self.token_to_idx[UNK_TOKEN]) for token in tokens]
        ids = ids[:max_length]
        ids.extend([self.token_to_idx[PAD_TOKEN]] * (max_length - len(ids)))
        return ids


def build_vocabulary(tokenized_texts: list[list[str]], min_freq: int = 1, max_size: int | None = None) -> Vocabulary:
    counter: Counter[str] = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)
    specials = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
    token_to_idx = {token: idx for idx, token in enumerate(specials)}
    most_common = counter.most_common(max_size)
    for token, freq in most_common:
        if freq >= min_freq and token not in token_to_idx:
            token_to_idx[token] = len(token_to_idx)
    return Vocabulary(token_to_idx)


def load_embedding_matrix(vocab: Vocabulary, embedding_dim: int, path: str | None, seed: int = 42) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=0.0, scale=0.05, size=(len(vocab), embedding_dim)).astype("float32")
    matrix[vocab.token_to_idx[PAD_TOKEN]] = 0.0
    if not path:
        return torch.tensor(matrix)

    embedding_path = Path(path)
    with embedding_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip().split()
            if len(parts) != embedding_dim + 1:
                continue
            token = parts[0]
            idx = vocab.token_to_idx.get(token)
            if idx is not None:
                matrix[idx] = np.asarray(parts[1:], dtype="float32")
    return torch.tensor(matrix)

