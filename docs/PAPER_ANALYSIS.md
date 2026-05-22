# Phase 1: Paper Analysis

## Dataset Pipeline

The paper reports experiments on three benchmark datasets: Twitter, BuzzFeed, and PolitiFact. The Data Availability section provides only one explicit URL:

https://www.kaggle.com/datasets/sudishbasnet/truthseekertwitterdataset2023

It does not provide separate URLs for BuzzFeed or PolitiFact. Table 4 and Table 5 disagree:

| Dataset | Table 4 | Table 5 |
|---|---:|---:|
| Twitter | about 17k posts, about 3.4k test | 17,000 total, 13,600 train, 1,700 validation, 1,700 test |
| BuzzFeed | about 2.7k articles, about 600 test | 220 total, 176 train, 22 validation, 22 test |
| PolitiFact | about 11k posts, about 2.2k test | 3,565 total, 2,852 train, 357 validation, 356 test |

The paper states randomized 80/10/10 train/validation/test splits, seed 42, ten repeated runs, and comparable but omitted five-fold cross-validation.

## Text Pipeline

The text pipeline states:

- remove inappropriate words, stop words, punctuation, and other inappropriate components
- apply stemming and lemmatization
- tokenize into words or subwords
- encode using word embeddings such as Word2Vec or GloVe
- feed vectors into an LSTM
- Table 7 settings: Adam, batch size 20, epochs 20, sigmoid activation, initial LR 0.005, gradient threshold 1, LR drop factor 0.2

Missing: exact tokenizer, inappropriate word list, embedding choice, embedding dimension, vocabulary construction, sequence length, LSTM hidden size, dropout, layer count, scheduler step interval.

## Image Pipeline

The paper states:

- input images are sliced to focus on regions
- a pre-trained ResNet-50 initialized with ImageNet weights extracts features
- average pooling layer features are used
- feature vectors are sliced and distributed as GNN nodes
- non-diagonal adjacency is initialized to 1
- learned relation: `Adj_i,j = sigmoid(Cl(|vertex_i - vertex_j|))`, where `Cl` is a 1x1 convolution
- vertex transform: `vertex_a = tanh(Weight_a * vertex_a + bias_a)`
- initial hidden state: `h_a^0 = ReLU(vertex_a)`
- message aggregation: `x_a^t = sum_{(d,a) in B}(Weight_g * h_d^{t-1} + bias_g)`
- GNN output feeds a Transformer encoder-decoder
- Transformer has three encoder layers and three decoder layers
- decoder uses self-attention over prior tokens, cross-attention over encoder output, layer normalization, and a linear vocabulary projection

Missing: image resize, number of slices/nodes, node dimensions, attention heads, d_model, feed-forward dimension, dropout, caption vocabulary, caption targets, decoding strategy, optimizer settings for image branch.

## Fuzzy Logic System

The paper states:

- confidence scores from text `Ct`, image `Ci`, and fusion `Cf` are aggregated
- linguistic variables include Low, Medium, High
- Gaussian membership functions are used
- fuzzy rule example: `IF Ct is High AND Ci is Medium THEN Verdict = Real with Confidence mu_k`
- defuzzification: `D_f = sum(mu_k * z_k) / sum(mu_k)`
- threshold membership Eq. 6 uses a linear ramp between `T_low` and `T_high`
- threshold tuning minimizes `MFCE = (1/N) sum |mu_hat_i - y_i|`
- FCS: `sum(w_i * mu_i) / sum(w_i)`
- Table 2 rules: Real/Real -> Real; Fake/Real -> Fake; Real/Fake -> Fake; Fake/Fake -> Fake
- ablation text says each input has two membership functions optimized with a genetic algorithm

Contradiction: Scenario 2 says Text Fake + Image Real -> Real, conflicting with Table 2.

## Evaluation

Metrics are accuracy, precision, recall, F1-score, and confusion matrix. Results are reported as averages over ten trials with standard deviations in Table 8. Decision-layer comparison reports Softmax, MLP, and FIS.

## Runtime and Hardware

The paper reports RTX 4090, 64 GB RAM, Intel Core i9, PyTorch 2.2, CUDA 12.1, seed 42. It also separately mentions Intel i5-1235U at 1.30 GHz. Inference time is reported as 14.2 ms per multimodal sample: LSTM 3.1 ms, GNN 2.8 ms, Transformer 7.5 ms, Fuzzy layer 0.8 ms.

## Reproducibility Risks

- No official repository found from the paper text or web search.
- Dataset source is under-specified for BuzzFeed and PolitiFact.
- Multimodal image-caption alignment is not described.
- Fuzzy rules and scenario narrative conflict.
- Several architecture dimensions are absent.
- Exact pretrained embeddings and ResNet checkpoint are absent.
- Caption generation training objective is not specified.
- The paper alternates between image captioning, image classification, and multimodal fake-news classification without a complete end-to-end loss definition.

