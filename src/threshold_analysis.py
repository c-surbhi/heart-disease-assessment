import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from utils import (
    load_model,
    calculate_profit
)


def evaluate_thresholds(
    y_true,
    probabilities
):

    results = []

    thresholds = np.arange(
        0.01,
        1.00,
        0.01
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions
        ).ravel()

        profit = calculate_profit(
            tp,
            tn,
            fp,
            fn
        )

        results.append(
            [
                threshold,
                tp,
                tn,
                fp,
                fn,
                profit
            ]
        )

    results_df = pd.DataFrame(
        results,
        columns=[
            "Threshold",
            "TP",
            "TN",
            "FP",
            "FN",
            "Profit"
        ]
    )

    return results_df


def plot_profit_curve(
    results_df,
    model_name
):

    plt.figure(figsize=(10, 6))

    plt.plot(
        results_df["Threshold"],
        results_df["Profit"]
    )

    plt.xlabel("Threshold")
    plt.ylabel("Profit")

    plt.title(
        f"Profit vs Threshold ({model_name})"
    )

    plt.grid(True)

    plt.savefig(
        f"outputs/plots/profit_curve_{model_name}.png",
        bbox_inches="tight"
    )

    plt.close()


def find_best_threshold(
    results_df
):

    best_row = results_df.loc[
        results_df["Profit"].idxmax()
    ]

    return best_row


def process_model(
    model,
    model_name,
    X_test,
    y_test
):

    probabilities = (
        model.predict_proba(X_test)[:, 1]
    )

    results_df = evaluate_thresholds(
        y_test,
        probabilities
    )

    results_df.to_csv(
        f"outputs/reports/{model_name}_thresholds.csv",
        index=False
    )

    plot_profit_curve(
        results_df,
        model_name
    )

    best_row = find_best_threshold(
        results_df
    )

    print("\n" + "=" * 60)
    print(model_name.upper())
    print("=" * 60)

    print(
        f"Best Threshold: "
        f"{best_row['Threshold']:.2f}"
    )

    print(
        f"Maximum Profit: "
        f"{best_row['Profit']:.0f}"
    )

    print(
        f"TP={int(best_row['TP'])} "
        f"TN={int(best_row['TN'])} "
        f"FP={int(best_row['FP'])} "
        f"FN={int(best_row['FN'])}"
    )

    return best_row


def main():

    test_df = pd.read_csv(
        "outputs/reports/test_dataset.csv"
    )

    X_test = test_df.drop(
        "target",
        axis=1
    )

    y_test = test_df["target"]

    logistic_model = load_model(
        "outputs/models/logistic_regression.pkl"
    )

    rf_model = load_model(
        "outputs/models/random_forest.pkl"
    )

    logistic_best = process_model(
        logistic_model,
        "logistic",
        X_test,
        y_test
    )

    rf_best = process_model(
        rf_model,
        "random_forest",
        X_test,
        y_test
    )

    summary = pd.DataFrame([
        {
            "Model": "Logistic Regression",
            "Threshold":
                logistic_best["Threshold"],
            "Profit":
                logistic_best["Profit"]
        },
        {
            "Model": "Random Forest",
            "Threshold":
                rf_best["Threshold"],
            "Profit":
                rf_best["Profit"]
        }
    ])

    summary.to_csv(
        "outputs/reports/profit_summary.csv",
        index=False
    )

    print("\nThreshold Optimization Completed")


if __name__ == "__main__":
    main()