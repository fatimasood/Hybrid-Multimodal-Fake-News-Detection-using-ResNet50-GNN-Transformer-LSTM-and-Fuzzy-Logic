from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from data.datasets import TrustifyMultimodalDataset
from models.text.lstm_classifier import LSTMTextClassifier
from preprocessing.image import build_image_transform
from preprocessing.text import build_text_preprocessor
from preprocessing.vocabulary import Vocabulary, load_embedding_matrix
from training.common import build_loader, save_checkpoint
from utils.config import load_config
from utils.reproducibility import seed_everything


def train_text(config_path: str) -> None:
    cfg = load_config(config_path)
    seed_everything(cfg["experiment"]["seed"], cfg["experiment"]["deterministic"])
    processed = Path(cfg["paths"]["processed_dir"]) / cfg["dataset"]["active_dataset"]
    vocab = Vocabulary(json.loads((processed / "vocab.json").read_text(encoding="utf-8")))
    train_df = pd.read_csv(processed / "train.csv")
    val_df = pd.read_csv(processed / "validation.csv")
    text_preprocessor = build_text_preprocessor(cfg["preprocessing"]["text"])
    transform = build_image_transform(cfg["preprocessing"]["image"])
    common_kwargs = dict(
        text_column=cfg["dataset"]["text_column"],
        image_column=cfg["dataset"]["image_column"],
        caption_column=cfg["dataset"]["caption_column"],
        label_column=cfg["dataset"]["label_column"],
        vocab=vocab,
        text_preprocessor=text_preprocessor.clean,
        image_transform=transform,
        max_text_length=cfg["preprocessing"]["text"]["max_sequence_length"],
        max_caption_length=cfg["models"]["image"]["transformer"]["max_caption_length"],
        allow_missing_images=True,
    )
    train_ds = TrustifyMultimodalDataset(train_df, **common_kwargs)
    val_ds = TrustifyMultimodalDataset(val_df, **common_kwargs)
    model_cfg = cfg["models"]["text_lstm"]
    embedding_matrix = load_embedding_matrix(
        vocab,
        cfg["preprocessing"]["text"]["embedding"]["dimension"],
        cfg["paths"].get("embedding_path"),
        cfg["experiment"]["seed"],
    )
    model = LSTMTextClassifier(
        embedding_matrix=embedding_matrix,
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        trainable_embeddings=cfg["preprocessing"]["text"]["embedding"]["trainable"],
    )
    device = torch.device(cfg["experiment"]["device"] if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg["initial_learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=model_cfg["learn_rate_drop_factor"])
    criterion = nn.BCELoss()
    train_loader = build_loader(train_ds, model_cfg["batch_size"], cfg["experiment"]["seed"], shuffle=True)
    val_loader = build_loader(val_ds, model_cfg["batch_size"], cfg["experiment"]["seed"], shuffle=False)

    for epoch in range(1, model_cfg["epochs"] + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            preds = model(batch["text_ids"].to(device), batch["text_length"].to(device))
            loss = criterion(preds, batch["label"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), model_cfg["gradient_threshold"])
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        save_checkpoint(Path(cfg["paths"]["checkpoints_dir"]) / "text_lstm.pt", model, optimizer, epoch)
        print({"epoch": epoch, "train_bce": sum(train_losses) / max(len(train_losses), 1)})

    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            preds = model(batch["text_ids"].to(device), batch["text_length"].to(device))
            val_losses.append(float(criterion(preds, batch["label"].to(device)).cpu()))
    print({"validation_bce": sum(val_losses) / max(len(val_losses), 1)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/trustify_reproduction.yaml")
    args = parser.parse_args()
    train_text(args.config)


if __name__ == "__main__":
    main()
