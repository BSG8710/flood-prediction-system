"""
train.py
---------
Training pipeline for the Flood Prediction System.

Workflow:
    1. Load dataset (data/flood_dataset.xlsx)
    2. Explore & visualize (saves charts to static/img/)
    3. Preprocess (missing values, outlier capping, scaling)
    4. Train multiple candidate models
    5. Compare models on Accuracy / Precision / Recall / F1 / Confusion Matrix
    6. Select the best model (ranked by F1-score, since the dataset is imbalanced)
    7. Save the best model, the scaler, and run metadata to models/

Run with:
    python train.py
"""

import json
import os
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use("Agg")  # headless backend, safe for servers/containers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# ----------------------------------------------------------------------
# Paths & constants
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "flood_dataset.xlsx")
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMG_DIR = os.path.join(BASE_DIR, "static", "img")

TARGET_COLUMN = "flood"

# The exact order of features the model is trained on. app.py must feed
# inputs to the model in this same order.
FEATURE_ORDER = [
    "Temp",
    "Humidity",
    "Cloud Cover",
    "ANNUAL",
    "Jan-Feb",
    "Mar-May",
    "Jun-Sep",
    "Oct-Dec",
    "avgjune",
    "sub",
]

RANDOM_STATE = 42


def load_dataset():
    """Step 1: Load the dataset from disk, with clear error handling."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{DATA_PATH}'. "
            "Make sure flood_dataset.xlsx is inside the data/ folder."
        )

    df = pd.read_excel(DATA_PATH)

    missing_cols = [c for c in FEATURE_ORDER + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

    print(f"[1/8] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def explore_and_visualize(df):
    """Step 2: Basic EDA. Saves a class-balance chart and a correlation heatmap."""
    os.makedirs(IMG_DIR, exist_ok=True)

    # Class balance
    plt.figure(figsize=(5, 4))
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    labels = ["No Flood (0)", "Flood (1)"]
    colors = ["#2e8b57", "#c0392b"]
    plt.bar(labels, counts.values, color=colors)
    plt.title("Flood Class Distribution")
    plt.ylabel("Number of Records")
    for i, v in enumerate(counts.values):
        plt.text(i, v + 0.5, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "class_distribution.png"), dpi=130)
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    corr = df[FEATURE_ORDER + [TARGET_COLUMN]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "correlation_heatmap.png"), dpi=130)
    plt.close()

    print("[2/8] EDA charts saved to static/img/")


def preprocess(df):
    """Step 3: Handle missing values, cap outliers (IQR method), split X/y."""
    df = df.copy()

    # Missing values -> median imputation per numeric column
    for col in FEATURE_ORDER:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Outlier capping using the IQR method (clip instead of drop, since the
    # dataset is small and every row of the minority "flood" class matters)
    for col in FEATURE_ORDER:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)

    df = df.dropna()

    X = df[FEATURE_ORDER]
    y = df[TARGET_COLUMN].astype(int)

    print(f"[3/8] Preprocessing complete. Features shape: {X.shape}")
    print(f"      Target distribution:\n{y.value_counts().to_string()}")
    return X, y


def build_models(scale_pos_weight):
    """Step 4: Define the candidate models.

    class_weight='balanced' (and scale_pos_weight for XGBoost) is used
    because the flood dataset is imbalanced (far more "no flood" rows).
    """
    return {
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, max_depth=5, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced",
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            max_depth=4,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        ),
    }


def evaluate_models(models, X_train, X_test, y_train, y_test):
    """Step 5: Train + compare every candidate model."""
    results = {}
    print("\n" + "=" * 60)
    print("MODEL COMPARISON ON TEST SET")
    print("=" * 60)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        cm = confusion_matrix(y_test, preds)

        results[name] = {
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm,
        }

        print(f"\n{name}:")
        print(f"  Accuracy : {acc * 100:.2f}%")
        print(f"  Precision: {prec * 100:.2f}%")
        print(f"  Recall   : {rec * 100:.2f}%")
        print(f"  F1-score : {f1 * 100:.2f}%")

    print("[4/8 & 5/8] Training and evaluation complete.")
    return results


def plot_comparison(results):
    """Bar chart comparing all models across metrics, saved for the UI."""
    names = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-score"]

    x = np.arange(len(names))
    width = 0.2

    plt.figure(figsize=(9, 5))
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results[n][metric] * 100 for n in names]
        plt.bar(x + i * width, values, width, label=label)

    plt.xticks(x + width * 1.5, names, rotation=15)
    plt.ylabel("Score (%)")
    plt.title("Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "model_comparison.png"), dpi=130)
    plt.close()


def plot_confusion_matrix(cm, model_name):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["No Flood", "Flood"]
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "best_model_confusion_matrix.png"), dpi=130)
    plt.close(fig)


def select_best_model(results):
    """Step 6: Pick the best model to deploy.

    Decision Tree often ties on F1-score, but gives all-or-nothing
    confidence (0% or 100%) because a single tree's leaves are frequently
    "pure" on a small dataset like this one. Random Forest averages the
    vote of 200 trees, so it reports smoother, more realistic probabilities
    (e.g. 26%, 63%, 91%) instead of only 0%/100%. Since the UI displays a
    live probability percentage, Random Forest is preferred whenever it's
    reasonably close in performance to the top F1-score.
    """
    best_f1 = max(results[n]["f1"] for n in results)
    rf_f1 = results["Random Forest"]["f1"]

    # Prefer Random Forest as long as it's within 5 points of the top score.
    if best_f1 - rf_f1 <= 0.05:
        best_name = "Random Forest"
    else:
        best_name = max(results, key=lambda n: results[n]["f1"])

    best = results[best_name]
    print("\n" + "=" * 60)
    print(f"BEST MODEL SELECTED: {best_name}")
    print(f"F1-score: {best['f1'] * 100:.2f}%  |  Accuracy: {best['accuracy'] * 100:.2f}%")
    print("=" * 60)
    return best_name, best


def train_and_evaluate():
    df = load_dataset()
    explore_and_visualize(df)
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / max(pos, 1)

    models = build_models(scale_pos_weight)
    results = evaluate_models(models, X_train_scaled, X_test_scaled, y_train, y_test)
    plot_comparison(results)

    best_name, best = select_best_model(results)
    plot_confusion_matrix(best["confusion_matrix"], best_name)

    print(f"\nDetailed Classification Report ({best_name}):")
    best_preds = best["model"].predict(X_test_scaled)
    print(classification_report(y_test, best_preds, target_names=["No Flood", "Flood"]))

    # Step 8: Persist model + scaler + metadata
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best["model"], os.path.join(MODELS_DIR, "flood_model.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))

    metadata = {
        "best_model_name": best_name,
        "feature_order": FEATURE_ORDER,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            name: {
                "accuracy": r["accuracy"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1"],
            }
            for name, r in results.items()
        },
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n[8/8] Model, scaler, and metadata saved to models/")
    return metadata


if __name__ == "__main__":
    train_and_evaluate()
