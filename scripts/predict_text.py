from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.datasets import TrustifyMultimodalDataset
from evaluation.metrics import binary_metrics
from models.text.lstm_classifier import LSTMTextClassifier
from preprocessing.image import build_image_transform
from preprocessing.text import build_text_preprocessor
from preprocessing.vocabulary import Vocabulary, load_embedding_matrix
from training.common import build_loader
from utils.config import load_config
from utils.logging import write_json
from utils.reproducibility import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the text-only LSTM branch.")
    parser.add_argument("--config", default="configs/trustify_reproduction.yaml")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--checkpoint", default="checkpoints/text_lstm.pt")
    parser.add_argument("--out-dir", default="logs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["experiment"]["seed"], cfg["experiment"]["deterministic"])
    processed = Path(cfg["paths"]["processed_dir"]) / cfg["dataset"]["active_dataset"]
    vocab = Vocabulary(json.loads((processed / "vocab.json").read_text(encoding="utf-8")))
    frame = pd.read_csv(processed / f"{args.split}.csv")

    text_preprocessor = build_text_preprocessor(cfg["preprocessing"]["text"])
    dataset = TrustifyMultimodalDataset(
        frame,
        text_column=cfg["dataset"]["text_column"],
        image_column=cfg["dataset"]["image_column"],
        caption_column=cfg["dataset"]["caption_column"],
        label_column=cfg["dataset"]["label_column"],
        vocab=vocab,
        text_preprocessor=text_preprocessor.clean,
        image_transform=build_image_transform(cfg["preprocessing"]["image"]),
        max_text_length=cfg["preprocessing"]["text"]["max_sequence_length"],
        max_caption_length=cfg["models"]["image"]["transformer"]["max_caption_length"],
        allow_missing_images=True,
    )

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
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(cfg["experiment"]["device"] if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loader = build_loader(dataset, model_cfg["batch_size"], cfg["experiment"]["seed"], shuffle=False)
    probabilities: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            probs = model(batch["text_ids"].to(device)).detach().cpu().numpy()
            probabilities.extend(probs.tolist())
            labels.extend(batch["label"].numpy().astype(int).tolist())

    y_prob = np.asarray(probabilities)
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = np.asarray(labels)
    metrics = binary_metrics(y_true, y_pred)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"label": y_true, "text_confidence": y_prob, "prediction": y_pred}).to_csv(
        out_dir / f"text_{args.split}_predictions.csv",
        index=False,
    )
    write_json(out_dir / f"text_{args.split}_metrics.json", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
