# Heart Disease Prediction

## Objective

Develop and compare two machine learning models for heart disease prediction using decile analysis and business-driven threshold optimization.

Models:

1. Logistic Regression
2. Random Forest

The project focuses on:

- Model evaluation
- Decile Analysis
- Lift Analysis
- Threshold Optimization
- Business Profit Maximization

---

## Dataset

Heart Disease Dataset

Target Variable:

- 0 = No Heart Disease
- 1 = Heart Disease

Features:

- age
- sex
- cp
- trestbps
- chol
- fbs
- restecg
- thalach
- exang
- oldpeak
- slope
- ca
- thal

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Project Execution

### Train Models

```bash
python src/train.py
```

Outputs:

```text
outputs/models/logistic_regression.pkl

outputs/models/random_forest.pkl
```

---

### Run Decile Analysis

```bash
python src/decile_analysis.py
```

Outputs:

```text
outputs/plots/cumulative_gain.png

outputs/plots/lift_chart.png
```

---

### Run Threshold Optimization

```bash
python src/threshold_analysis.py
```

Outputs:

```text
outputs/plots/profit_curve_logistic.png

outputs/plots/profit_curve_rf.png
```

---

### Generate Report

```bash
python src/report_generator.py
```

Output:

```text
outputs/reports/model_evaluation_report.docx
```

---

### Launch Streamlit App

```bash
streamlit run streamlit_app.py
```

---

## Business Cost Matrix

| Outcome | Value |
|----------|---------|
| TP | +500 |
| TN | 0 |
| FP | -100 |
| FN | -1000 |

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- PR AUC

---

## Decile Analysis Metrics

- Event Count
- Event Rate
- Average Probability
- Cumulative Gain
- Lift

---

## Threshold Optimization

Thresholds are evaluated from:

```python
0.01 -> 0.99
```

The threshold producing maximum business profit is selected.

---

## Streamlit Application

The Streamlit application allows interactive heart disease prediction using the trained Random Forest model.

Input patient details and obtain:

- Predicted Probability
- Predicted Class
- Risk Level
