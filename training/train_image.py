from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from data.datasets import TrustifyMultimodalDataset
from models.image.image_classifier import TrustifyImageBranch
from preprocessing.image import build_image_transform
from preprocessing.text import build_text_preprocessor
from preprocessing.vocabulary import Vocabulary
from training.common import build_loader, save_checkpoint
from utils.config import load_config
from utils.reproducibility import seed_everything


def train_image(config_path: str) -> None:
    cfg = load_config(config_path)
    seed_everything(cfg["experiment"]["seed"], cfg["experiment"]["deterministic"])
    processed = Path(cfg["paths"]["processed_dir"]) / cfg["dataset"]["active_dataset"]
    vocab = Vocabulary(json.loads((processed / "vocab.json").read_text(encoding="utf-8")))
    train_df = pd.read_csv(processed / "train.csv")
    text_preprocessor = build_text_preprocessor(cfg["preprocessing"]["text"])
    transform = build_image_transform(cfg["preprocessing"]["image"])
    train_ds = TrustifyMultimodalDataset(
        train_df,
        text_column=cfg["dataset"]["text_column"],
        image_column=cfg["dataset"]["image_column"],
        caption_column=cfg["dataset"]["caption_column"],
        label_column=cfg["dataset"]["label_column"],
        vocab=vocab,
        text_preprocessor=text_preprocessor.clean,
        image_transform=transform,
        max_text_length=cfg["preprocessing"]["text"]["max_sequence_length"],
        max_caption_length=cfg["models"]["image"]["transformer"]["max_caption_length"],
        allow_missing_images=False,
    )
    image_cfg = cfg["models"]["image"]
    tr_cfg = image_cfg["transformer"]
    model = TrustifyImageBranch(
        vocab_size=len(vocab),
        avgpool_dim=image_cfg["avgpool_dim"],
        graph_nodes=image_cfg["graph_nodes"],
        node_feature_dim=image_cfg["node_feature_dim"],
        transformer_heads=tr_cfg["attention_heads"],
        transformer_ff_dim=tr_cfg["feedforward_dim"],
        transformer_dropout=tr_cfg["dropout"],
        max_caption_length=tr_cfg["max_caption_length"],
    )
    device = torch.device(cfg["experiment"]["device"] if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["models"]["text_lstm"]["initial_learning_rate"])
    criterion_cls = nn.BCELoss()
    criterion_cap = nn.CrossEntropyLoss(ignore_index=0)
    loader = build_loader(train_ds, cfg["models"]["text_lstm"]["batch_size"], cfg["experiment"]["seed"], shuffle=True)

    for epoch in range(1, cfg["models"]["text_lstm"]["epochs"] + 1):
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            caption = batch["caption_ids"].to(device)
            outputs = model(batch["image"].to(device), caption[:, :-1])
            cls_loss = criterion_cls(outputs["image_confidence"], batch["label"].to(device))
            cap_logits = outputs["caption_logits"].reshape(-1, len(vocab))
            cap_target = caption[:, 1:].reshape(-1)
            caption_loss = criterion_cap(cap_logits, cap_target)
            loss = cls_loss + caption_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["models"]["text_lstm"]["gradient_threshold"])
            optimizer.step()
        save_checkpoint(Path(cfg["paths"]["checkpoints_dir"]) / "image_gnn_transformer.pt", model, optimizer, epoch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/trustify_reproduction.yaml")
    args = parser.parse_args()
    train_image(args.config)


if __name__ == "__main__":
    main()

