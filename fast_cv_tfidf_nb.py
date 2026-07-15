import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

sys.stdout.reconfigure(encoding="utf-8")

df = pd.read_pickle("fast_cv_train_df.pkl")
y = df["label"].values
RANDOM_STATE = 42

# Combine context + prompt + response as the raw text signal
def build_text(row):
    ctx = row["context"] if isinstance(row["context"], str) and row["context"] not in ("[NULL]", "nan") else ""
    return f"{ctx} [SEP] {row['prompt_bn']} [SEP] {row['response_bn']}"

texts = df.apply(build_text, axis=1).values

def evaluate_tfidf_nb(model_cls, name, ngram_range=(1,2)):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    hallu, faithful, macro = [], [], []
    oof = np.zeros(len(y))
    for tr_idx, val_idx in skf.split(texts, y):
        vec = TfidfVectorizer(ngram_range=ngram_range, min_df=2, max_features=3000)
        X_tr = vec.fit_transform(texts[tr_idx])
        X_val = vec.transform(texts[val_idx])
        model = model_cls()
        model.fit(X_tr, y[tr_idx])
        probs = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = probs
        preds = (probs >= 0.5).astype(int)
        hallu.append(f1_score(y[val_idx], preds, pos_label=0))
        faithful.append(f1_score(y[val_idx], preds, pos_label=1))
        macro.append(f1_score(y[val_idx], preds, average="macro"))
    print(f"{name:25s} hallu_F1={np.mean(hallu):.4f}+/-{np.std(hallu):.4f}  "
          f"faithful_F1={np.mean(faithful):.4f}  macro_F1={np.mean(macro):.4f}")
    return oof

print(f"n={len(df)}")
print()
oof_mnb = evaluate_tfidf_nb(MultinomialNB, "TF-IDF + MultinomialNB")
oof_bnb = evaluate_tfidf_nb(BernoulliNB, "TF-IDF + BernoulliNB")
