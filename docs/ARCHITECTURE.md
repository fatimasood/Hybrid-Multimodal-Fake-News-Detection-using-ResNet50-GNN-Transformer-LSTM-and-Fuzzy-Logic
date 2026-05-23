# Trustify-Style Architecture Used For Fakeddit

This implementation keeps the paper's architecture shape but evaluates it on a Fakeddit subset because the paper-linked TruthSeeker Kaggle dataset has no image files or captions.

```mermaid
flowchart LR
    A["Fakeddit title/text"] --> B["Paper-style text preprocessing<br/>lowercase, punctuation removal,<br/>stopwords, stemming, lemmatization"]
    B --> C["Vocabulary + embedding layer"]
    C --> D["LSTM text classifier"]
    D --> E["Text confidence Ct"]

    F["Fakeddit image URL"] --> G["Download image"]
    G --> H["Resize 224x224 + ImageNet normalization"]
    H --> I["ResNet-50 avgpool features<br/>2048 dims"]
    I --> J["Slice into 16 graph nodes<br/>128 dims each"]
    J --> K["GNN learned adjacency<br/>sigmoid Conv1d(|vi-vj|)"]
    K --> L["3-layer Transformer encoder-decoder"]
    L --> M["Image confidence Ci"]

    E --> N["Gaussian fuzzy inference system"]
    M --> N
    O["Caption-text similarity proxy"] --> N
    N --> P["Centroid defuzzification"]
    P --> Q["Final fake/real prediction + rule trace"]
```

## Dataset-Specific Notes

- Fakeddit `clean_title` is used as text input.
- Fakeddit `image_url` is downloaded into local JPEG files for ResNet-50.
- Fakeddit `clean_title` is also used as the decoder caption target because Fakeddit does not provide separate human image captions.
- Fakeddit observed binary convention is `2_way_label=1` for real/true-style posts and `0` for fake-style posts.
- This is a complete multimodal experiment, not an exact reproduction of the paper's claimed Twitter/BuzzFeed/PolitiFact experiments.

