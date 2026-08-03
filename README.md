# Disaster Tweet Classification: From Bag-of-Words to Transformers

Binary text classification on ~7,600 tweets, predicting whether a tweet describes a **real disaster** ("Forest fire near La Ronge Sask. Canada") or uses disaster language **figuratively** ("My exam is a disaster").

This project is deliberately structured as a **ladder of three models**, where each rung exists to expose a specific limitation that the next rung addresses. The goal was not to maximize a leaderboard score, but to understand *why* each architecture behaves the way it does — and where added complexity does and doesn't pay off.

---

## Results

All three models were evaluated on an **identical stratified 80/20 validation split** (fixed seed) using a **single shared evaluation function**, so the comparison isolates the model rather than split luck.

| Model | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.765 | 0.835 | 0.704| 0.817 |
| GloVe + BiLSTM | 0.760 | 0.777 | 0.743| 0.798 |
| **DistilBERT (fine-tuned)** | **0.799** | 0.792 | 0.807 | 0.826 |

> *Fill in the blank precision/recall/accuracy cells from your saved `results` list.*

**Headline finding:** the bag-of-words baseline and the BiLSTM performed **equivalently** (0.765 vs 0.760). Only the pretrained transformer produced a meaningful improvement, and even then by a bounded margin (+4 F1). Both observations point to the same conclusion about this task — discussed below.

---

## The ladder

### Rung 1 — TF-IDF + Logistic Regression (F1 0.760)

**Approach.** Tweets are converted to sparse vectors over a learned vocabulary. TF-IDF weights each word by how frequent it is *in this tweet* against how rare it is *across the corpus*, so uninformative words like "the" collapse toward zero weight while discriminative words like "wildfire" survive. A linear classifier is fit on top.

**Why this baseline.** It is fast, and — critically — **interpretable**. Because the model is linear, every vocabulary word carries a single readable weight, so the learned decision rule can be inspected directly. Words most predictive of *disaster* were exactly what domain intuition suggests (`wildfire`, `earthquake`, `evacuation`, `debris`), confirming the model learned real signal rather than an artifact.

**Limitation surfaced.** Bag-of-words discards **word order entirely**. `"fire spreading through hills"` and `"hills spreading fire through"` produce identical feature vectors. Negation breaks completely: `"not a disaster"` becomes two independent features, and the model cannot bind `not` to `disaster`. Any relationship *between* words is invisible.

*Leakage note:* the vectorizer was fit on training data only and merely applied to validation, so no validation statistics leaked into the learned vocabulary or IDF values.

### Rung 2 — GloVe + BiLSTM (F1 0.758)

**Approach.** Words are mapped to dense 100-dimensional **pre-trained GloVe vectors**, where semantic similarity becomes geometric proximity (`fire` sits near `flames`, far from `pancake`) — a representation bag-of-words cannot express, since one-hot features are mutually orthogonal. A bidirectional LSTM then reads the sequence **in order**, maintaining a running hidden state, so word order finally matters.

The LSTM's cell state acts as an additive memory pathway regulated by forget/input/output gates, which is what allows information and gradients to survive across many timesteps — the fix for the vanishing gradient problem that cripples plain RNNs on long sequences. Bidirectionality runs two LSTMs in opposite directions so every position sees both left and right context.

**Limitations surfaced.** Two, and they compound:

1. **GloVe embeddings are static.** The word `fire` receives one frozen vector regardless of sentence. The model reads context-blind tokens, so it cannot distinguish literal from figurative usage at the representation level — the exact distinction this task requires.
2. **Learning sequence structure from scratch on ~6,000 training examples.** The LSTM's weights begin essentially random and must learn language from a very small corpus.

### Rung 3 — DistilBERT, fine-tuned (F1 0.799)

**Approach.** A pretrained transformer, fine-tuned for 3 epochs at a learning rate of 2e-5.

**Self-attention** replaces recurrence: every token attends directly to every other token in a single step, with no sequential chain and no distance penalty. Each token emits a Query, Key, and Value; attention weights come from Query–Key dot products, and the token's new representation is a weighted sum of all Values in the sentence.

The consequence is **contextual embeddings**, which directly resolve Rung 2's first limitation: `fire` in *"forest fire near LA"* attends to `forest` and `near`, while `fire` in *"my inbox is on fire"* attends to `inbox` — producing **genuinely different vectors for the same word**.

**Pretraining** resolves the second limitation. DistilBERT was pretrained via masked language modeling on billions of words, so fine-tuning adapts an existing understanding of English rather than learning it from 6,000 tweets.

**Design choices worth noting:**
- **Learning rate 2e-5** (vs 1e-3 for the LSTM). Pretrained weights are already good; large updates would destroy that structure — catastrophic forgetting. Small updates nudge the representation toward the task while preserving what pretraining bought.
- **3 epochs, not 6.** With ~67M parameters against 6,000 examples, fine-tuning converges within a few passes and overfits after. Training loss fell to ~0.22 and flattened, consistent with fast convergence rather than memorization.
- **DistilBERT over BERT.** Knowledge distillation yields ~40% fewer parameters and ~60% faster inference while retaining roughly 97% of BERT's performance — the right trade-off for this scale of project.
- **Subword (WordPiece) tokenization** eliminates the out-of-vocabulary problem that word-level tokenization suffers on noisy tweet text: rare words, typos, and place names decompose into known fragments rather than collapsing to a single uninformative `<unk>` token.

---

## Analysis

### Why did bag-of-words tie the BiLSTM?

This was the most instructive result in the project.

A large fraction of this task is decided by **which words appear** — `earthquake`, `evacuate`, `debris` are close to decisive on their own. TF-IDF models exactly that, and does it near-optimally. The BiLSTM's additional capability is modeling word *order* and *relationships*, but on short tweets with strong keyword signal, order rarely determines the label. So its advantage had little to exploit, while it still paid the full cost of learning sequential structure from scratch over static, context-blind embeddings.

**The takeaway: model complexity only helps when the task actually contains the structure that complexity captures.** Otherwise you are paying for capability you cannot use.

### Why did DistilBERT win by only +4?

The two findings corroborate each other. If contextual understanding were the dominant missing ingredient, the transformer would have improved far more dramatically. Instead the gain was real but bounded — consistent with a task that is *mostly* lexical, where context is worth roughly four F1 points on top of keyword matching.

### Metric choice

The classes are imbalanced (~57% non-disaster / 43% disaster), which is visible in the results: DistilBERT's accuracy (0.826) sits above its F1 (0.799) precisely because accuracy is flattered by the majority class. F1 is reported as the headline metric because it balances precision against recall on the positive (disaster) class, which is the class of interest. Precision and recall came out well-balanced (0.792 / 0.807), indicating no pathological skew toward either false alarms or missed disasters.

### Error analysis

> *To be completed: examples where TF-IDF and BiLSTM both fail but DistilBERT succeeds, which should cluster around figurative and context-dependent language. Also worth including cases where all three models fail, which tend to be genuinely ambiguous or mislabeled.*

---

## What I would do differently

- **Cross-validation instead of a single split.** A 1,523-row validation set carries meaningful variance; differences of 1–2 F1 points are within noise. K-fold would make the comparison more robust.
- **Threshold tuning.** All models used the default 0.5 decision threshold. Sweeping the threshold to optimize F1 directly would likely add a few tenths.
- **Label noise.** This dataset is known to contain mislabeled examples, which caps achievable performance regardless of architecture.
- **Larger models.** RoBERTa or full BERT would likely add a point or two, at higher compute cost.
- **Longer-form data.** The tie between bag-of-words and the BiLSTM is partly an artifact of tweet length. On longer documents, where long-range dependencies genuinely matter, the gap between rungs should widen considerably.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install pandas numpy scikit-learn torch transformers
```

The competition CSVs and GloVe embeddings (`glove.6B.100d.txt`, ~331MB) are **gitignored due to size**. The notebook downloads GloVe automatically on first run; the tweet data is available from the [Kaggle competition](https://www.kaggle.com/competitions/nlp-getting-started).

## Repository structure

```
nlp-disaster-tweet/
├── README.md
├── notebook.ipynb          # all three models, end to end
└── results/
    └── comparison.csv      # metrics table
```

## Stack

Python · pandas · scikit-learn · PyTorch · HuggingFace Transformers · GloVe
