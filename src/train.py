"""
Trains a gradient-boosted classifier to predict `confirmed_fragmentation`
from episode-level claims features, and fits a SHAP explainer for
per-episode, per-feature explainability.

Label handling: `chart_audit_outcomes.csv` is joined onto the feature table
by `episode_id` ONLY here, at train time. It is never a column the feature
builder (episode_builder.py) produces, so there's no path for it to leak
into the feature set itself.
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision_score

from .episode_builder import build_episode_features, FEATURE_COLS

DATA_DIR = "data"
MODELS_DIR = "models"


def load_and_join():
    claims = pd.read_csv(f"{DATA_DIR}/claims.csv")
    audits = pd.read_csv(f"{DATA_DIR}/chart_audit_outcomes.csv")
    feat = build_episode_features(claims)
    feat = feat.merge(audits[["episode_id", "confirmed_fragmentation"]], on="episode_id", how="inner")
    return feat


def train():
    feat = load_and_join()
    X = feat[FEATURE_COLS].copy()
    X["has_reached_therapy"] = X["has_reached_therapy"].astype(int)
    y = feat["confirmed_fragmentation"].astype(int)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, feat.index, test_size=0.25, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
    )
    model.fit(X_train, y_train)

    proba_test = model.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)
    auc = roc_auc_score(y_test, proba_test)
    ap = average_precision_score(y_test, proba_test)
    prec = precision_score(y_test, pred_test, zero_division=0)
    rec = recall_score(y_test, pred_test, zero_division=0)
    print(f"Held-out test set -> ROC-AUC: {auc:.3f}  PR-AUC: {ap:.3f}  "
          f"Precision@0.5: {prec:.3f}  Recall@0.5: {rec:.3f}  (n_test={len(y_test)}, "
          f"positive rate={y_test.mean():.1%})")

    # score the FULL population for the dashboard (train+test), using the
    # trained model -- standard practice for a demo; a production system
    # would use proper out-of-fold scoring for anything shown as "the score"
    feat["fragmentation_proba"] = model.predict_proba(X)[:, 1]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump({"model": model, "explainer": explainer, "feature_cols": FEATURE_COLS}, f"{MODELS_DIR}/classifier.pkl")
    np.save(f"{MODELS_DIR}/shap_values.npy", shap_values)
    feat.to_csv(f"{MODELS_DIR}/episode_features_scored.csv", index=False)

    pd.Series({
        "roc_auc": auc, "pr_auc": ap, "precision_at_0.5": prec, "recall_at_0.5": rec,
        "n_test": int(len(y_test)), "test_positive_rate": float(y_test.mean()),
    }).to_json(f"{MODELS_DIR}/classifier_metrics.json")

    print(f"Scored {len(feat)} episodes. Artifacts saved to {MODELS_DIR}/")
    return feat, model, explainer, shap_values


if __name__ == "__main__":
    train()
