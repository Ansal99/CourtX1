import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
from xgboost import XGBClassifier

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "Final_dataset.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

CATEGORICAL = ["state", "case_type", "court_level",
               "evidence_strength", "opposite_lawyer_experience", "case_complexity"]
NUMERICAL   = ["case_duration_years", "has_documents", "num_witnesses",
               "legal_aid", "settlement_attempted", "bench_size", "num_cited_cases"]
FEATURES    = CATEGORICAL + NUMERICAL
TARGET      = "outcome"


def load_and_prepare(path):
    df = pd.read_csv(path)
    df = df[FEATURES + [TARGET]].copy()

    encoders = {}
    for col in CATEGORICAL:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df[FEATURES]
    y = df[TARGET]
    return X, y, encoders


def evaluate(name, model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model":     name,
        "accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred,    zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred,        zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
    }

    print(f"\n── {name} ──")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # ── Classification Report ────────────────────────────────────────────────
    print(f"\n  Classification Report ({name}):")
    print(classification_report(y_test, y_pred, target_names=["Loss (0)", "Win (1)"]))

    # ── Confusion Matrix — save as PNG ───────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    save_confusion_matrix(cm, name, y_test, y_pred)

    # ── Save confusion matrix values to metrics dict ─────────────────────────
    metrics["confusion_matrix"] = cm.tolist()   # [[TN, FP], [FN, TP]]

    return metrics


def save_confusion_matrix(cm, model_name, y_test, y_pred):
    """Save a styled confusion matrix PNG to models/."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Loss", "Predicted Win"],
        yticklabels=["Actual Loss",    "Actual Win"],
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )

    # Annotate TP / TN / FP / FN
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(
                j + 0.5, i + 0.72,
                labels[i][j],
                ha="center", va="center",
                fontsize=9, color="gray",
            )

    tn, fp, fn, tp = cm.ravel()
    acc  = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0

    ax.set_title(
        f"{model_name} — Confusion Matrix\n"
        f"Accuracy: {acc*100:.1f}%  |  Precision: {prec*100:.1f}%  |  Recall: {rec*100:.1f}%",
        fontsize=11, pad=12
    )
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("Actual Label",    fontsize=10)
    plt.tight_layout()

    fname = model_name.lower().replace(" ", "_") + "_confusion_matrix.png"
    fpath = os.path.join(MODEL_DIR, fname)
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Confusion matrix saved → models/{fname}")


def train():
    print("Loading data...")
    X, y, encoders = load_and_prepare(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # ── XGBoost ──────────────────────────────────────────────────────────────
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False)

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # ── Evaluate both (confusion matrices generated here) ────────────────────
    xgb_metrics = evaluate("XGBoost",      xgb, X_test, y_test)
    rf_metrics  = evaluate("RandomForest", rf,  X_test, y_test)

    all_metrics = [xgb_metrics, rf_metrics]
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    # ── SHAP explainer ────────────────────────────────────────────────────────
    print("\nBuilding SHAP explainer (this takes ~30s)...")
    explainer = shap.TreeExplainer(xgb)

    sample      = X_test.sample(min(2000, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(sample)
    mean_abs    = np.abs(shap_values).mean(axis=0)
    importance  = dict(zip(FEATURES, mean_abs.tolist()))

    with open(os.path.join(MODEL_DIR, "feature_importance.json"), "w") as f:
        json.dump(importance, f, indent=2)

    # ── Save all model files ──────────────────────────────────────────────────
    joblib.dump(xgb,       os.path.join(MODEL_DIR, "xgb_model.pkl"))
    joblib.dump(rf,        os.path.join(MODEL_DIR, "rf_model.pkl"))
    joblib.dump(explainer, os.path.join(MODEL_DIR, "shap_explainer.pkl"))
    joblib.dump(encoders,  os.path.join(MODEL_DIR, "encoders.pkl"))

    label_map = {col: list(enc.classes_) for col, enc in encoders.items()}
    label_map["FEATURES"] = FEATURES
    label_map["NUMERICAL"] = NUMERICAL
    with open(os.path.join(MODEL_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    print("\n✓ All files saved to /models/")
    print("  xgb_model.pkl | rf_model.pkl | shap_explainer.pkl")
    print("  encoders.pkl  | label_map.json | metrics.json | feature_importance.json")
    print("  xgboost_confusion_matrix.png | randomforest_confusion_matrix.png")


if __name__ == "__main__":
    train()