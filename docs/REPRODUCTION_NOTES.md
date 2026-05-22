# Phase 2: Full Implementation

Implemented modules:

- `preprocessing/`: paper-aligned text cleaning, vocabulary, image resizing/normalization
- `data/`: CSV loader and deterministic 80/10/10 split
- `models/text/`: LSTM + sigmoid classifier
- `models/image/`: ResNet-50 avgpool extractor and image branch
- `models/gnn/`: feature slicing, learned adjacency, GNN message passing
- `models/transformer/`: 3-encoder/3-decoder image caption Transformer
- `models/fuzzy/`: Gaussian MFs, centroid defuzzification, FCS, MFCE grid and GA threshold tuning
- `training/`: text, image, and fuzzy-threshold training entry points
- `evaluation/`: paper metric computation and paper-result deltas
- `visualization/`: metric plotting
- `scripts/`: dataset download, dataset preparation, ablation launcher

# Phase 3: Experimental Replication

To replicate:

1. Download the paper dataset:
   `python scripts/download_dataset.py --dataset sudishbasnet/truthseekertwitterdataset2023 --out data/raw/truthseekertwitterdataset2023`
2. Convert the downloaded files to a CSV exposing `text`, `image_path`, `caption`, and `label`.
3. Prepare deterministic splits:
   `python reproduce.py --config configs/trustify_reproduction.yaml --stage prepare --csv path/to/prepared.csv`
4. Train text:
   `python reproduce.py --config configs/trustify_reproduction.yaml --stage train_text`
5. Train image:
   `python reproduce.py --config configs/trustify_reproduction.yaml --stage train_image`
6. Generate validation/test confidence CSVs with columns `label`, `text_confidence`, `image_confidence`, optional `caption_text_similarity`.
7. Tune fuzzy thresholds:
   `python reproduce.py --config configs/trustify_reproduction.yaml --stage train_hybrid --predictions logs/validation_predictions.csv`
8. Evaluate:
   `python reproduce.py --config configs/trustify_reproduction.yaml --stage evaluate --predictions logs/test_predictions.csv`

# Phase 4: Result Comparison With Paper

The expected metrics are embedded in `evaluation/compare_paper.py` and summarized in `README.md`. Evaluation output includes metric deltas against the paper's Table 8.

# Phase 5: Error Analysis and Failure Cases

The paper identifies:

- satirical posts misidentified as fake, approximately 6%
- satirical posts with real images and misleading captions creating modality conflict, approximately 4%
- a flood-relief post from Kerala incorrectly labeled fake due to dataset bias around disaster imagery
- ambiguous fuzzy outputs when real/fake memberships are nearly balanced

The implemented fuzzy system returns per-rule traces so these cases can be audited by firing strength, consequent centroid, and fuzzy confidence score.

