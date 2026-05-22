# Data Directory

Expected folders:

```text
data/
  raw/
  processed/
```

The paper states that the analyzed datasets are available at:

https://www.kaggle.com/datasets/sudishbasnet/truthseekertwitterdataset2023

Run:

```powershell
python scripts/download_dataset.py --dataset sudishbasnet/truthseekertwitterdataset2023 --out data/raw/truthseekertwitterdataset2023
```

For full multimodal reproduction, each prepared CSV must expose:

| Column | Meaning |
|---|---|
| `text` | Original post/article text |
| `image_path` | Path to associated image |
| `caption` | Ground-truth or training caption for the image-caption decoder |
| `label` | Real/fake label |

If the downloaded Kaggle archive lacks image paths or image files, the repository will still prepare text records but hybrid/image training cannot be a faithful reproduction.

