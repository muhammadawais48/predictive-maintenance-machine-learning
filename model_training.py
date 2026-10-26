"""Train and evaluate a baseline Random Forest failure classifier."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             ConfusionMatrixDisplay, PrecisionRecallDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from feature_engineering import extract_features

FEATURES = ["Type", "Air_temperature_K", "Process_temperature_K",
            "Rotational_speed_rpm", "Torque_Nm", "Tool_wear_min",
            "Temp_Difference", "Power_Proxy"]

def train_and_evaluate(db_path="predictive_maintenance.db", output_dir="outputs"):
    df = extract_features(db_path)
    X, y = df[FEATURES], df["Machine_failure"].astype(int)
    if y.nunique() != 2:
        raise ValueError("Training requires both failure and non-failure records")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    preprocess = ColumnTransformer([
        ("type", OneHotEncoder(handle_unknown="ignore"), ["Type"]),
        ("numeric", "passthrough", [f for f in FEATURES if f != "Type"])
    ])
    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42,
                                              class_weight="balanced", n_jobs=-1))
    ])
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "dataset_rows": int(len(df)), "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)), "total_failures": int(y.sum()),
        "test_failures": int(y_test.sum()),
        "accuracy": float(accuracy_score(y_test, predicted)),
        "failure_precision": float(precision_score(y_test, predicted, zero_division=0)),
        "failure_recall": float(recall_score(y_test, predicted, zero_division=0)),
        "failure_f1": float(f1_score(y_test, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "test_failure_rate": float(y_test.mean()),
        "confusion_matrix": confusion_matrix(y_test, predicted).tolist(),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output_dir / "classification_report.txt").write_text(
        classification_report(y_test, predicted, digits=3, zero_division=0)
    )
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    importance = pd.DataFrame({"feature": feature_names,
        "importance": model.named_steps["classifier"].feature_importances_})
    importance.sort_values("importance", ascending=False).to_csv(
        output_dir / "feature_importance.csv", index=False)
    ConfusionMatrixDisplay.from_predictions(y_test, predicted, display_labels=["No failure", "Failure"], cmap="Blues")
    plt.title("Confusion matrix — held-out test data")
    plt.tight_layout(); plt.savefig(output_dir / "confusion_matrix.png", dpi=150); plt.close()
    PrecisionRecallDisplay.from_predictions(y_test, probabilities, name="Random Forest")
    plt.title("Precision–recall curve — held-out test data")
    plt.tight_layout(); plt.savefig(output_dir / "precision_recall_curve.png", dpi=150); plt.close()
    print(json.dumps(metrics, indent=2))
    return metrics

if __name__ == "__main__":
    train_and_evaluate()
