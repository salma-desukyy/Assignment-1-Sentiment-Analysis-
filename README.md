# Assignment 2 -- Sentiment Analysis: Results and Comparisons

## 1. Task Overview

Ternary sentiment classification (positive / negative / neutral) on two domains:

- **SST-3**: Movie review sentences from the Stanford Sentiment Treebank.
- **Bakeoff**: Restaurant review sentences (new domain, tests generalization).

Primary metric: **Mean of macro-F1** scores across both datasets.

---

## 2. Dataset Statistics

| Split | Source | Examples |
|-------|--------|----------|
| SST-3 Train | Movie reviews | 8,544 |
| SST-3 Dev | Movie reviews | 1,101 |
| Bakeoff Dev | Restaurant reviews | 2,361 |
| SST-3 Test (bakeoff submission) | Movie reviews | 2,210 |
| Bakeoff Test (bakeoff submission) | Restaurant reviews | 2,362 |

### Bakeoff Dev Label Distribution

| Label | Count | % |
|-------|-------|---|
| neutral | 1,019 | 43.2% |
| positive | 777 | 32.9% |
| negative | 565 | 23.9% |

---

## 3. Model Comparison

Five systems were evaluated, each progressively more powerful:

### 3.1 Results Summary

| # | Model | Features | SST-3 Dev Macro-F1 | Bakeoff Dev Macro-F1 | Mean Macro-F1 |
|---|-------|----------|--------------------|----------------------|---------------|
| 1 | Logistic Regression (Softmax) | Unigram counts | 0.518 | 0.315 | **0.416** |
| 2 | RNN Classifier | Learned embeddings | 0.501 | 0.342 | **0.415** (Note: 0.432 in prior run) |
| 3 | Shallow NN + GloVe | GloVe 300d averaging | 0.470 | -- | **0.470** (SST-3 only) |
| 4 | Shallow NN + BERT CLS | BERT [CLS] vector | 0.605 | -- | **0.605** (SST-3 only) |
| 5 | **Bidirectional RNN + BERT tokens** (Original) | BERT token embeddings | 0.604 | 0.675 | **0.640** |

### 3.2 Per-Class Breakdown (SST-3 Dev)

| Model | Neg P | Neg R | Neg F1 | Neu P | Neu R | Neu F1 | Pos P | Pos R | Pos F1 | Accuracy |
|-------|-------|-------|--------|-------|-------|--------|-------|-------|--------|----------|
| Softmax | 0.628 | 0.689 | 0.657 | 0.343 | 0.153 | 0.211 | 0.629 | 0.750 | 0.684 | 0.602 |
| RNN | 0.556 | 0.584 | 0.569 | 0.259 | 0.279 | 0.269 | 0.653 | 0.595 | 0.623 | 0.525 |
| Shallow NN + GloVe | 0.613 | 0.717 | 0.661 | 0.412 | 0.031 | 0.057 | 0.611 | 0.802 | 0.693 | 0.609 |
| Shallow NN + BERT | 0.704 | 0.801 | 0.750 | 0.461 | 0.205 | 0.284 | 0.730 | 0.842 | 0.782 | 0.694 |
| **Original System** | **0.717** | **0.759** | **0.738** | **0.424** | **0.231** | **0.299** | **0.717** | **0.845** | **0.776** | **0.684** |

### 3.3 Per-Class Breakdown (Bakeoff Dev -- Restaurant Reviews)

| Model | Neg P | Neg R | Neg F1 | Neu P | Neu R | Neu F1 | Pos P | Pos R | Pos F1 | Accuracy |
|-------|-------|-------|--------|-------|-------|--------|-------|-------|--------|----------|
| Softmax | 0.272 | 0.690 | 0.390 | 0.429 | 0.113 | 0.179 | 0.409 | 0.346 | 0.375 | 0.328 |
| RNN | 0.272 | 0.577 | 0.370 | 0.438 | 0.281 | 0.342 | 0.398 | 0.261 | 0.315 | 0.345 |
| **Original System** | **0.650** | **0.550** | **0.596** | **0.692** | **0.777** | **0.732** | **0.719** | **0.679** | **0.698** | **0.692** |

---

## 4. Key Comparisons and Insights

### 4.1 Softmax vs. RNN Baseline

- Both achieve similar mean macro-F1 (~0.42).
- The softmax model has **higher accuracy** (0.602 vs. 0.525 on SST-3) due to better precision on polar classes.
- The RNN slightly improves **neutral recall** (0.279 vs. 0.153), but at the cost of polar-class performance.
- Neither model handles domain shift well (bakeoff macro-F1 drops to ~0.32-0.34).

### 4.2 GloVe vs. BERT Representations

- BERT CLS encoding dramatically outperforms GloVe averaging: **0.605 vs. 0.470** macro-F1 on SST-3 Dev.
- BERT's contextual representations capture sentiment-bearing patterns that static GloVe vectors miss.
- The GloVe model nearly fails on `neutral` (F1 = 0.057, recall = 0.031), suggesting that averaged static vectors cannot distinguish the neutral class.

### 4.3 Impact of Mixed-Domain Training

- Adding 1,000 bakeoff examples to training improves bakeoff macro-F1 from **0.315 to 0.518** (+64%) for the softmax model.
- SST-3 performance remains stable (0.518 vs. 0.518), confirming no degradation from mixed training.
- This shows that even a small amount of in-domain data can substantially reduce domain shift.

### 4.4 Original System -- Best Model

The original system combines three key innovations:

1. **BERT token-level embeddings** (not just [CLS]) fed into a bidirectional 2-layer RNN.
2. **Mixed-domain training** (SST-3 + half of bakeoff dev).
3. **Hyperparameter search** over hidden dim, bidirectionality, number of layers, and activation.

Best hyperparameters found: `{bidirectional: True, classifier_activation: ReLU(), hidden_dim: 200, num_layers: 2}`

This achieves the highest performance across all models:
- SST-3 Dev: **0.604** macro-F1
- Bakeoff Dev: **0.675** macro-F1
- **Mean: 0.640** (a 54% improvement over the softmax baseline)

### 4.5 The Neutral Class Problem

Across all models, `neutral` is consistently the hardest class:

| Model | Neutral F1 (SST-3) | Neutral F1 (Bakeoff) |
|-------|---------------------|----------------------|
| Softmax | 0.211 | 0.179 |
| RNN | 0.269 | 0.342 |
| GloVe + Shallow NN | 0.057 | -- |
| BERT + Shallow NN | 0.284 | -- |
| Original System | 0.299 | **0.732** |

The original system's mixed-domain training dramatically improves neutral classification on restaurant reviews (0.732), since the bakeoff data contains many neutral examples that teach the model domain-specific neutral patterns.

---

## 5. Error Analysis

### 5.1 Joint Errors (Both Softmax and RNN Wrong)

- **1,534 examples** where both models predicted incorrectly.
- Gold label distribution of hard cases: neutral (799), positive (528), negative (207).
- The neutral class dominates hard cases, confirming it is the primary source of errors.

### 5.2 Common Error Patterns

| Pattern | Example | Gold | Both Predicted |
|---------|---------|------|----------------|
| Sarcasm / implicit sentiment | "WE HAD A RESERVATION." | neutral | positive/negative |
| Domain-specific vocabulary | "Scallops were devine." | positive | negative/neutral |
| Ambiguous phrasing | "Come hungry." | positive | negative/neutral |
| Implicit comparison | "The steaks there at least equal, in some cases superior to Bob's." | positive | negative/neutral |
| Factual but loaded | "We finally stopped a busboy and asked for help." | neutral | negative |

### 5.3 Domain Shift Effects

- Both baseline models show severe performance drops on restaurant reviews.
- The vocabulary of restaurant reviews (menu items, service terms, ambiance descriptions) is absent from the SST-3 movie-review training set.
- Mixed-domain training is the single most effective strategy for closing this gap.

---

## 6. Bakeoff Submission Summary

The final bakeoff CSV contains **4,572 predictions** across both test sets:

| Dataset | Examples | Negative | Positive | Neutral |
|---------|----------|----------|----------|---------|
| Bakeoff (restaurant) | 2,362 | 650 (27.5%) | 686 (29.0%) | 1,026 (43.4%) |
| SST-3 (movie) | 2,210 | 1,087 (49.2%) | 959 (43.4%) | 164 (7.4%) |
| **Total** | **4,572** | **1,737** | **1,645** | **1,190** |

The prediction distributions reflect the domain differences:
- **SST-3 test**: Strongly polar (92.6% positive or negative), very few neutral -- consistent with movie review sentiment.
- **Bakeoff test**: More balanced with a neutral majority (43.4%) -- consistent with restaurant review patterns seen in the dev set.

---

## 7. Submission Files

| File | Description |
|------|-------------|
| `hw_sentiment.ipynb` | Complete notebook with all code, outputs, and answers |
| `cs224u-sentiment-bakeoff-entry.csv` | Bakeoff predictions (4,572 rows) |
| `ANALYSIS.md` | This analysis document |
