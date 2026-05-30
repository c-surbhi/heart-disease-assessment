import os
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


def create_directories():
    directories = [
        "outputs/models",
        "outputs/plots",
        "outputs/reports"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def save_model(model, filepath):
    joblib.dump(model, filepath)


def load_model(filepath):
    return joblib.load(filepath)


def save_dataframe(df, filepath):
    df.to_csv(filepath, index=False)


def calculate_metrics(y_true, y_pred, y_proba):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC_AUC": roc_auc_score(y_true, y_proba),
        "PR_AUC": average_precision_score(y_true, y_proba)
    }

    return metrics


def print_metrics(model_name, metrics):
    print("\n" + "=" * 50)
    print(f"{model_name} Performance")
    print("=" * 50)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


def save_metrics(metrics, filepath):
    df = pd.DataFrame([metrics])
    df.to_csv(filepath, index=False)


def get_confusion_matrix(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    }


def calculate_profit(tp, tn, fp, fn):
    profit = (
        tp * 500 +
        tn * 0 -
        fp * 100 -
        fn * 1000
    )

    return profit