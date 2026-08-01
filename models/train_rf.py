# ============================================================
# Random Forest — Indian Rupee Counterfeit Detection
# Features: INR-specific banknote security characteristics
# ============================================================

import os, json, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SAVE_DIR = os.path.join(PROJECT_ROOT, "models", "saved")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "models", "results")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURES = ["intaglio_depth","security_thread_visible","watermark_clarity","color_shift_ink","microprint_score","uv_fluorescence","serial_number_font","denomination"]

def train():
    print("\n" + "━"*58)
    print("  Random Forest — Indian Rupee (INR) Counterfeit Detection")
    print("━"*58 + "\n")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    X_train, y_train = train_df[FEATURES].values, train_df["label"].values
    X_test, y_test = test_df[FEATURES].values, test_df["label"].values
    print("  Running GridSearchCV …")
    param_grid = {"n_estimators": [100, 200, 300], "max_depth": [5, 10, None]}
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print(f"  Best params: {grid.best_params_}")
    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    print(f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    joblib.dump(best, os.path.join(SAVE_DIR, "rf_model.pkl"))
    metrics = {"model":"Random Forest","currency":"INR","accuracy":round(acc,4),"precision":round(prec,4),"recall":round(rec,4),"f1_score":round(f1,4),"confusion_matrix":cm}
    with open(os.path.join(RESULTS_DIR, "rf_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    importances = dict(zip(FEATURES, [round(float(x), 4) for x in best.feature_importances_]))
    with open(os.path.join(RESULTS_DIR, "rf_feature_importance.json"), "w") as f:
        json.dump(importances, f, indent=2)
    print("  ✅ Random Forest training complete.\n")

if __name__ == "__main__":
    train()
