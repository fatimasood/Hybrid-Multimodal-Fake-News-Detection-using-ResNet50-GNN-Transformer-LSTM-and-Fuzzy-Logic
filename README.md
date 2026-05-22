# Trustify Strict Reproduction

This repository is a strict reproduction scaffold for:

**Bharati, Bharti, and Dehalwar. "A novel fuzzy logic-based hybrid framework for detecting fake news." Discover Computing 29, 58 (2026). DOI: 10.1007/s10791-025-09882-x**

The implementation follows the paper as closely as possible and records every missing or contradictory detail in [configs/assumptions.yaml](configs/assumptions.yaml).

## Paper Dataset Link

The paper's Data Availability section gives one dataset link:

https://www.kaggle.com/datasets/sudishbasnet/truthseekertwitterdataset2023

The paper also reports experiments on Twitter, BuzzFeed, and PolitiFact, but does not provide separate download URLs for BuzzFeed or PolitiFact. It additionally gives inconsistent sample counts between Table 4 and Table 5.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet omw-1.4 punkt
```

For CUDA 12.1, install the matching PyTorch 2.2 wheel from the official PyTorch index if needed:

```powershell
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
```

## Download Dataset

Configure Kaggle credentials first, then run:

```powershell
python scripts/download_dataset.py --dataset sudishbasnet/truthseekertwitterdataset2023 --out data/raw/truthseekertwitterdataset2023
```

## Expected Project Flow

```powershell
python reproduce.py --config configs/trustify_reproduction.yaml --stage prepare
python reproduce.py --config configs/trustify_reproduction.yaml --stage train_text
python reproduce.py --config configs/trustify_reproduction.yaml --stage train_image
python reproduce.py --config configs/trustify_reproduction.yaml --stage train_hybrid
python reproduce.py --config configs/trustify_reproduction.yaml --stage evaluate
python scripts/run_ablation.py --config configs/trustify_reproduction.yaml
```

## Reported Paper Metrics

| Dataset | Method | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|
| Twitter | LSTM text | 0.968 +- 0.041 | 0.944 +- 0.039 | 0.955 +- 0.038 | 0.938 +- 0.038 |
| Twitter | GNN image | 0.987 +- 0.007 | 0.960 +- 0.007 | 0.973 +- 0.006 | 0.965 +- 0.005 |
| Twitter | Hybrid | 0.984 +- 0.0079 | 0.972 +- 0.0074 | 0.978 +- 0.0063 | 0.961 +- 0.007 |
| BuzzFeed | LSTM text | 0.907 +- 0.068 | 0.912 +- 0.072 | 0.910 +- 0.061 | 0.928 +- 0.058 |
| BuzzFeed | GNN image | 0.977 +- 0.013 | 0.920 +- 0.013 | 0.957 +- 0.011 | 0.962 +- 0.018 |
| BuzzFeed | Hybrid | 0.949 +- 0.013 | 0.922 +- 0.110 | 0.935 +- 0.011 | 0.959 +- 0.015 |
| PolitiFact | LSTM text | 0.920 +- 0.050 | 0.914 +- 0.054 | 0.916 +- 0.052 | 0.922 +- 0.047 |
| PolitiFact | GNN image | 0.957 +- 0.012 | 0.920 +- 0.012 | 0.946 +- 0.011 | 0.924 +- 0.014 |
| PolitiFact | Hybrid | 0.949 +- 0.0092 | 0.923 +- 0.0091 | 0.939 +- 0.0091 | 0.922 +- 0.0087 |

## Known Reproduction Gaps

The paper does not provide an official repository, exact vocabulary, pretrained embedding file, LSTM hidden dimension, Transformer attention heads, image-caption labels, caption decoding policy, full fuzzy membership parameters, or exact GA settings. This repository makes those choices explicit in config and keeps them isolated.

