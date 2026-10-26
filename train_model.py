"""
train_model.py — Subtext AI
============================
This script loads the MELD dataset, preprocesses the text,
trains a TF-IDF + Logistic Regression emotion classifier,
evaluates it on the held-out test set, and saves the trained
artifacts for use by the Streamlit app.

HOW TO RUN:
    python3 train_model.py

OUTPUT:
    models/tfidf_vectorizer.pkl  — saved TF-IDF vectorizer
    models/emotion_model.pkl     — saved Logistic Regression model
    models/label_encoder.pkl     — saved label encoder
    models/evaluation_results.pkl — metrics and confusion matrix
"""

import os
import re
import ssl
import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns

# ── macOS SSL fix: download NLTK data before importing corpus ──
# macOS ships without updated CA certificates; we bypass SSL for NLTK only.
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, precision_recall_fscore_support
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0.  Download NLTK data (runs only once)
# ─────────────────────────────────────────────
print("Downloading NLTK resources...")
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
print("  NLTK resources ready.")

STOP_WORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()

# ─────────────────────────────────────────────
# 1.  Text preprocessing function
# ─────────────────────────────────────────────
def preprocess(text: str) -> str:
    """
    Clean and normalise a raw utterance.

    Steps:
      1. Lowercase everything
      2. Remove URLs, mentions, special characters
      3. Tokenise (split on whitespace)
      4. Remove stopwords (common words like 'the', 'is', 'a')
      5. Stem each word (e.g. 'running' → 'run')

    Why we do this:
      TF-IDF works better when the vocabulary is compact.
      Removing stopwords reduces noise; stemming groups related
      word forms so the model sees fewer, more meaningful tokens.
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Keep only letters and spaces (remove punctuation, numbers)
    text = re.sub(r"[^a-z\s]", "", text)

    # Tokenise
    tokens = text.split()

    # Remove stopwords and very short tokens, then stem
    tokens = [
        stemmer.stem(tok)
        for tok in tokens
        if tok not in STOP_WORDS and len(tok) > 1
    ]

    return " ".join(tokens)


# ─────────────────────────────────────────────
# 2.  Load the MELD splits
# ─────────────────────────────────────────────
print("\nLoading MELD dataset splits...")

DATA_DIR = "data"
train_df = pd.read_csv(os.path.join(DATA_DIR, "train_sent_emo.csv"))
dev_df   = pd.read_csv(os.path.join(DATA_DIR, "dev_sent_emo.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test_sent_emo.csv"))

print(f"  Train : {train_df.shape[0]:,} rows")
print(f"  Dev   : {dev_df.shape[0]:,} rows")
print(f"  Test  : {test_df.shape[0]:,} rows")

# Rename columns for consistency
# MELD CSV uses 'Utterance' and 'Emotion'
TEXT_COL    = "Utterance"
EMOTION_COL = "Emotion"

# ─────────────────────────────────────────────
# 3.  Drop rows with missing text or label
# ─────────────────────────────────────────────
for name, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
    before = len(df)
    df.dropna(subset=[TEXT_COL, EMOTION_COL], inplace=True)
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df.drop(df[df[TEXT_COL] == ""].index, inplace=True)
    print(f"  {name}: dropped {before - len(df)} invalid rows")

# ─────────────────────────────────────────────
# 4.  Apply preprocessing
# ─────────────────────────────────────────────
print("\nPreprocessing text...")
train_df["clean_text"] = train_df[TEXT_COL].apply(preprocess)
dev_df["clean_text"]   = dev_df[TEXT_COL].apply(preprocess)
test_df["clean_text"]  = test_df[TEXT_COL].apply(preprocess)

# Replace any empty strings after cleaning with a placeholder
for df in [train_df, dev_df, test_df]:
    df["clean_text"] = df["clean_text"].replace("", "unknown")

print("  Done.")

# ─────────────────────────────────────────────
# 5.  Encode labels
# ─────────────────────────────────────────────
le = LabelEncoder()
le.fit(train_df[EMOTION_COL])          # fit on training labels only

y_train = le.transform(train_df[EMOTION_COL])
y_dev   = le.transform(dev_df[EMOTION_COL])
y_test  = le.transform(test_df[EMOTION_COL])

print(f"\nEmotion classes: {list(le.classes_)}")

# ─────────────────────────────────────────────
# 6.  TF-IDF vectorization
# ─────────────────────────────────────────────
"""
TF-IDF stands for Term Frequency – Inverse Document Frequency.

- Term Frequency (TF): how often a word appears in ONE sentence.
- Inverse Document Frequency (IDF): penalises words that appear
  in MANY sentences (they are less informative).

The result is a numeric matrix where each row = one utterance,
each column = one word in the vocabulary, and the value = the
TF-IDF score of that word in that utterance.

Parameters used:
  max_features=20000  — keep only the 20,000 most important words
  ngram_range=(1,2)   — include single words AND two-word phrases
                        (e.g. "not happy" as a unit, not just 'not' and 'happy')
  sublinear_tf=True   — apply log scaling to term frequency to
                        reduce the impact of very frequent words
"""
print("\nFitting TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2              # ignore words that appear in fewer than 2 docs
)

X_train = vectorizer.fit_transform(train_df["clean_text"])
X_dev   = vectorizer.transform(dev_df["clean_text"])
X_test  = vectorizer.transform(test_df["clean_text"])

print(f"  Vocabulary size : {len(vectorizer.vocabulary_):,}")
print(f"  X_train shape   : {X_train.shape}")

# ─────────────────────────────────────────────
# 7.  Train Logistic Regression
# ─────────────────────────────────────────────
"""
Logistic Regression learns a set of weights for each TF-IDF
feature (word/phrase) per emotion class.

Parameters:
  max_iter=1000      — allow enough iterations to converge
  C=1.0              — regularisation strength (prevents overfitting)
  class_weight='balanced' — compensates for the fact that 'neutral'
                        has far more training examples than 'fear'.
                        Without this, the model would favour predicting
                        neutral all the time.
  solver='lbfgs'     — efficient solver for multiclass problems
  multi_class='multinomial' — one softmax across all 7 classes
"""
print("\nTraining Logistic Regression...")
clf = LogisticRegression(
    max_iter=1000,
    C=1.0,
    class_weight="balanced",
    solver="lbfgs",
    random_state=42
)
clf.fit(X_train, y_train)
print("  Training complete.")

# ─────────────────────────────────────────────
# 8.  Evaluate on dev set (model selection)
# ─────────────────────────────────────────────
y_dev_pred = clf.predict(X_dev)
dev_acc = accuracy_score(y_dev, y_dev_pred)
print(f"\nValidation accuracy : {dev_acc:.4f}")

# ─────────────────────────────────────────────
# 9.  Evaluate on the held-out test set
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("  FINAL EVALUATION ON TEST SET")
print("="*55)

y_test_pred = clf.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"Test Accuracy : {test_acc:.4f}\n")
print(classification_report(
    y_test, y_test_pred,
    target_names=le.classes_
))

# Per-class precision / recall / F1
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_test_pred, labels=list(range(len(le.classes_)))
)

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)

# ─────────────────────────────────────────────
# 10.  Save confusion matrix plot
# ─────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=le.classes_, yticklabels=le.classes_, ax=ax
)
ax.set_xlabel("Predicted Emotion", fontsize=12)
ax.set_ylabel("True Emotion", fontsize=12)
ax.set_title("Confusion Matrix — MELD Test Set", fontsize=14)
plt.tight_layout()
plt.savefig("models/confusion_matrix.png", dpi=150)
plt.close()
print("\nConfusion matrix saved → models/confusion_matrix.png")

# Emotion distribution plot
fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (split_name, df) in zip(axes, [
    ("Train", train_df), ("Validation", dev_df), ("Test", test_df)
]):
    counts = df[EMOTION_COL].value_counts()
    colors = sns.color_palette("Set2", len(counts))
    counts.plot(kind="bar", ax=ax, color=colors, edgecolor="black")
    ax.set_title(f"{split_name} Split — Emotion Distribution", fontsize=11)
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("models/emotion_distribution.png", dpi=150)
plt.close()
print("Emotion distribution plot saved → models/emotion_distribution.png")

# ─────────────────────────────────────────────
# 11.  Save model artifacts
# ─────────────────────────────────────────────
evaluation_results = {
    "dev_accuracy":   float(dev_acc),
    "test_accuracy":  float(test_acc),
    "classes":        list(le.classes_),
    "precision":      precision.tolist(),
    "recall":         recall.tolist(),
    "f1":             f1.tolist(),
    "support":        support.tolist(),
    "confusion_matrix": cm.tolist(),
    "train_size":     int(len(train_df)),
    "dev_size":       int(len(dev_df)),
    "test_size":      int(len(test_df)),
    "vocab_size":     int(len(vectorizer.vocabulary_)),
    # Label distribution
    "train_distribution": train_df[EMOTION_COL].value_counts().to_dict(),
    "dev_distribution":   dev_df[EMOTION_COL].value_counts().to_dict(),
    "test_distribution":  test_df[EMOTION_COL].value_counts().to_dict(),
}

with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("models/emotion_model.pkl", "wb") as f:
    pickle.dump(clf, f)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

with open("models/evaluation_results.pkl", "wb") as f:
    pickle.dump(evaluation_results, f)

print("\nModel artifacts saved:")
print("  models/tfidf_vectorizer.pkl")
print("  models/emotion_model.pkl")
print("  models/label_encoder.pkl")
print("  models/evaluation_results.pkl")
print("\n✓ Training complete!")
