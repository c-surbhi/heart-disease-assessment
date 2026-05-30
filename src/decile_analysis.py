import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import load_model


def create_decile_table(
    df,
    probability_column
):

    decile_df = df.copy()

    decile_df = decile_df.sort_values(
        by=probability_column,
        ascending=False
    ).reset_index(drop=True)

    decile_df["Decile"] = pd.qcut(
        decile_df.index,
        q=10,
        labels=False
    ) + 1

    summary = (
        decile_df
        .groupby("Decile")
        .agg(
            Records=("target", "count"),
            Actual_Positives=("target", "sum"),
            Avg_Probability=(probability_column, "mean")
        )
        .reset_index()
    )

    summary["Event_Rate"] = (
        summary["Actual_Positives"]
        / summary["Records"]
    )

    total_events = (
        summary["Actual_Positives"]
        .sum()
    )

    summary["Cumulative_Events"] = (
        summary["Actual_Positives"]
        .cumsum()
    )

    summary["Cumulative_Gain"] = (
        summary["Cumulative_Events"]
        / total_events
    ) * 100

    overall_event_rate = (
        decile_df["target"].sum()
        / len(decile_df)
    )

    summary["Lift"] = (
        summary["Event_Rate"]
        / overall_event_rate
    )

    return summary
    
    decile_df = df.copy()

    decile_df = decile_df.sort_values(
        by=probability_column,
        ascending=False
    )

    decile_df = decile_df.sort_values(
        by=probability_column,
        ascending=False
    ).reset_index(drop=True)

    decile_df["Decile"] = pd.qcut(
        decile_df.index,
        10,
        labels=False
    ) + 1

    summary = (
        decile_df
        .groupby("Decile")
        .agg(
            Records=("target", "count"),
            Actual_Positives=("target", "sum"),
            Avg_Probability=(probability_column, "mean")
        )
        .reset_index()
    )

    summary["Event_Rate"] = (
        summary["Actual_Positives"] /
        summary["Records"]
    )

    total_events = summary["Actual_Positives"].sum()

    summary["Cumulative_Events"] = (
        summary["Actual_Positives"].cumsum()
    )

    summary["Cumulative_Gain"] = (
        summary["Cumulative_Events"] /
        total_events
    ) * 100

    overall_event_rate = (
        decile_df["target"].sum() /
        len(decile_df)
    )

    summary["Lift"] = (
        summary["Event_Rate"] /
        overall_event_rate
    )

    return summary


def plot_cumulative_gain(
    logistic_table,
    rf_table
):

    plt.figure(figsize=(10, 6))

    plt.plot(
        logistic_table["Decile"],
        logistic_table["Cumulative_Gain"],
        marker="o",
        label="Logistic Regression"
    )

    plt.plot(
        rf_table["Decile"],
        rf_table["Cumulative_Gain"],
        marker="o",
        label="Random Forest"
    )

    plt.xlabel("Decile")
    plt.ylabel("Cumulative Gain (%)")
    plt.title("Cumulative Gain Chart")

    plt.legend()

    plt.grid(True)

    plt.savefig(
        "outputs/plots/cumulative_gain.png",
        bbox_inches="tight"
    )

    plt.close()


def plot_lift_chart(
    logistic_table,
    rf_table
):

    plt.figure(figsize=(10, 6))

    plt.plot(
        logistic_table["Decile"],
        logistic_table["Lift"],
        marker="o",
        label="Logistic Regression"
    )

    plt.plot(
        rf_table["Decile"],
        rf_table["Lift"],
        marker="o",
        label="Random Forest"
    )

    plt.xlabel("Decile")
    plt.ylabel("Lift")

    plt.title("Lift Chart")

    plt.legend()

    plt.grid(True)

    plt.savefig(
        "outputs/plots/lift_chart.png",
        bbox_inches="tight"
    )

    plt.close()


def main():

    test_df = pd.read_csv(
        "outputs/reports/test_dataset.csv"
    )

    X_test = test_df.drop(
        "target",
        axis=1
    )

    logistic_model = load_model(
        "outputs/models/logistic_regression.pkl"
    )

    rf_model = load_model(
        "outputs/models/random_forest.pkl"
    )

    logistic_prob = (
        logistic_model
        .predict_proba(X_test)[:, 1]
    )

    rf_prob = (
        rf_model
        .predict_proba(X_test)[:, 1]
    )

    logistic_df = test_df.copy()
    logistic_df["Probability"] = logistic_prob

    rf_df = test_df.copy()
    rf_df["Probability"] = rf_prob

    logistic_table = create_decile_table(
        logistic_df,
        "Probability"
    )

    rf_table = create_decile_table(
        rf_df,
        "Probability"
    )

    logistic_table.to_csv(
        "outputs/reports/logistic_deciles.csv",
        index=False
    )

    rf_table.to_csv(
        "outputs/reports/rf_deciles.csv",
        index=False
    )

    plot_cumulative_gain(
        logistic_table,
        rf_table
    )

    plot_lift_chart(
        logistic_table,
        rf_table
    )

    print("\nDecile Analysis Completed")

    print("\nLogistic Regression Deciles")
    print(logistic_table)

    print("\nRandom Forest Deciles")
    print(rf_table)


if __name__ == "__main__":
    main()