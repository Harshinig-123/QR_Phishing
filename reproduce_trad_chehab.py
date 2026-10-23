"""
Reproduce the pixel-based baseline of Trad & Chehab (arXiv:2505.03451).

Data : https://github.com/fouadtrad/Detecting-Quishing-Attacks-with-Machine-Learning-Techniques-Through-QR-Code-Analysis
       unzip QuishingDataset.zip -> qr_codes_29.pickle, qr_codes_29_labels.pickle
Setup (paper, Sec. 3): 9,987 QR codes (69x69), 80/20 split, hyperparameters from Table 1,
       metrics at threshold 0.5, AUC from predicted probabilities.
       Feature selection (Sec. 4.3): keep pixels with non-zero importance from a
       tree model, retrain all models on that subset.

Usage:
    pip install numpy pandas scikit-learn lightgbm xgboost
    python reproduce_trad_chehab.py --data path/to/folder --seed 42 --out results_trad.csv
"""
import argparse
import pickle
import os

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def load_data(folder):
    with open(os.path.join(folder, "qr_codes_29.pickle"), "rb") as f:
        qr = pickle.load(f)
    with open(os.path.join(folder, "qr_codes_29_labels.pickle"), "rb") as f:
        labels = pickle.load(f)
    x = np.asarray(qr).reshape(len(qr), -1).astype(np.float32)  # 4761 pixel features
    y = np.asarray(labels).astype(int)
    return x, y


def make_models(seed):
    # Hyperparameters from Table 1 of the paper
    return {
        "Logistic Regression": LogisticRegression(C=0.1, solver="liblinear", random_state=seed),
        "Decision Tree": DecisionTreeClassifier(max_depth=3, min_samples_leaf=1, random_state=seed),
        "Random Forest": RandomForestClassifier(max_depth=20, n_estimators=100,
                                                random_state=seed, n_jobs=-1),
        "Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(learning_rate=0.2, n_estimators=150, random_state=seed,
                                 n_jobs=-1, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(learning_rate=0.1, n_estimators=200, random_state=seed,
                                   n_jobs=-1, verbose=-1),
    }


def evaluate(model, x_tr, y_tr, x_te, y_te):
    model.fit(x_tr, y_tr)
    prob = model.predict_proba(x_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred),
        "recall": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "auc": roc_auc_score(y_te, prob),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder holding the two pickle files")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results_trad.csv")
    args = ap.parse_args()

    x, y = load_data(args.data)
    print("data:", x.shape, "phishing share:", round(y.mean(), 3))
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=args.seed)

    rows = []

    # Experiment 1: all pixels
    for name, model in make_models(args.seed).items():
        r = evaluate(model, x_tr, y_tr, x_te, y_te)
        rows.append({"model": name, "features": "all 4761 pixels", **r})
        print(name, {k: round(v, 4) for k, v in r.items()})

    # Experiment 3: feature selection using importances from three tree models
    selectors = {
        "FS by Random Forest": RandomForestClassifier(max_depth=20, n_estimators=100,
                                                      random_state=args.seed, n_jobs=-1),
        "FS by XGBoost": XGBClassifier(learning_rate=0.2, n_estimators=150,
                                       random_state=args.seed, n_jobs=-1, eval_metric="logloss"),
        "FS by LightGBM": LGBMClassifier(learning_rate=0.1, n_estimators=200,
                                         random_state=args.seed, n_jobs=-1, verbose=-1),
    }
    for sel_name, sel in selectors.items():
        sel.fit(x_tr, y_tr)
        keep = np.where(sel.feature_importances_ > 0)[0]
        print(sel_name, "keeps", len(keep), "pixels")
        for name, model in make_models(args.seed).items():
            r = evaluate(model, x_tr[:, keep], y_tr, x_te[:, keep], y_te)
            rows.append({"model": name, "features": f"{sel_name} ({len(keep)} px)", **r})

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
