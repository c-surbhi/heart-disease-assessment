import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from utils import (
    create_directories,
    save_model,
    calculate_metrics,
    print_metrics,
    save_metrics
)


RANDOM_STATE = 42


def load_data():

    df = pd.read_csv("data/heart.csv")

    print("\nDataset Shape:")
    print(df.shape)

    print("\nMissing Values:")
    print(df.isnull().sum())

    return df


def split_data(df):

    X = df.drop("target", axis=1)

    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train):

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=500))
    ])

    param_grid = {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__solver": ["liblinear", "lbfgs"]
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("\nBest Logistic Regression Parameters")
    print(grid.best_params_)

    return grid.best_estimator_


def train_random_forest(X_train, y_train):

    model = RandomForestClassifier(
        random_state=RANDOM_STATE
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10]
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("\nBest Random Forest Parameters")
    print(grid.best_params_)

    return grid.best_estimator_


def evaluate_model(
    model,
    model_name,
    X_test,
    y_test
):

    y_pred = model.predict(X_test)

    y_proba = model.predict_proba(X_test)[:, 1]

    plot_roc_curve(
        y_test,
        y_proba,
        model_name
    )

    metrics = calculate_metrics(
        y_test,
        y_pred,
        y_proba
    )

    print_metrics(
        model_name,
        metrics
    )

    save_metrics(
        metrics,
        f"outputs/reports/{model_name}_metrics.csv"
    )

    return metrics

def plot_roc_curve(
    y_test,
    probabilities,
    model_name
):

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        label=model_name
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        f"ROC Curve - {model_name}"
    )

    plt.legend()

    plt.savefig(
        f"outputs/plots/{model_name}_roc.png",
        bbox_inches="tight"
    )

    plt.close()

def main():

    create_directories()

    df = load_data()

    X_train, X_test, y_train, y_test = split_data(df)

    print("\nTraining Logistic Regression...")

    logistic_model = train_logistic_regression(
        X_train,
        y_train
    )

    print("\nTraining Random Forest...")

    random_forest_model = train_random_forest(
        X_train,
        y_train
    )

    save_model(
        logistic_model,
        "outputs/models/logistic_regression.pkl"
    )

    save_model(
        random_forest_model,
        "outputs/models/random_forest.pkl"
    )

    print("\nEvaluating Logistic Regression")

    evaluate_model(
        logistic_model,
        "logistic_regression",
        X_test,
        y_test
    )

    print("\nEvaluating Random Forest")

    evaluate_model(
        random_forest_model,
        "random_forest",
        X_test,
        y_test
    )

    test_data = pd.concat(
        [
            X_test.reset_index(drop=True),
            y_test.reset_index(drop=True)
        ],
        axis=1
    )

    test_data.to_csv(
        "outputs/reports/test_dataset.csv",
        index=False
    )

    print("\nTraining Completed Successfully")


if __name__ == "__main__":
    main()