# Archaeological Sample Classification

A reproducible **Python / scikit-learn** notebook demonstrating supervised classification of tabular geochemical-style data, exploratory PCA visualization, and model evaluation.

> **Data limitation:** The notebook uses a reproducible **synthetic dataset by default** (800 samples, 20 features, three classes). Its model scores are a software demonstration, **not evidence of performance on real archaeological samples**. To test real data, supply a labelled CSV.

## Methods

- Load a CSV or generate synthetic sample data (random seed 42).
- Keep numeric predictors, check class sizes and make a stratified 80/20 train/test split.
- Fit missing-value imputation and, where appropriate, scaling **inside scikit-learn Pipelines**, avoiding preprocessing leakage.
- Visualise the training set using two-component PCA.
- Compare K-Nearest Neighbors and Random Forest with stratified cross-validation using **macro F1** on training data only.
- Select the highest-scoring baseline using cross-validation, then evaluate it **once** on the held-out test set (classification report and confusion matrix).
- Inspect Random Forest impurity-based feature importances, noting their limitations.
- Save the fitted selected Pipeline as `best_model.joblib` (not committed).

## Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/Marzma/archaeology-ml-classification.git
cd archaeology-ml-classification
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
jupyter notebook archaeology-ml-classification.ipynb
```

In Jupyter, choose **Kernel → Restart & Run All**. No dataset download is required for the default synthetic example.

## Use your own CSV

Place a CSV in the project directory (default filename: `data.csv`). Each row represents a sample; include a class label column named `label` and numeric feature columns. Alternatively edit `CSV_PATH` and `LABEL_COL` in the notebook. Rows with missing labels are dropped; missing numeric predictor values are imputed within training folds. Non-numeric predictors are excluded.

**Important:** For samples clustered by site, context, individual or time period, a random train/test split can overestimate generalisation. Use group-aware validation for such datasets.

## Repository contents

- `archaeology-ml-classification.ipynb`: analysis and plots
- `requirements.txt`: Python dependencies
- `.gitignore`: ignores environments, local data and generated model artifacts

## Interpretation and limitations

This is a **baseline educational demonstration**, not a validated archaeological prediction system. The dataset is synthetic; model selection is limited to two fixed configurations. Random Forest impurity-based importance does not establish causality and may be biased for correlated variables. For real-world use, evaluate on independent samples and document data provenance, sampling structure, error patterns and generalisation limits.

## Skills demonstrated

Python, pandas, NumPy, scikit-learn, Jupyter, PCA, classification, cross-validation, leakage-aware preprocessing, model evaluation and reproducible documentation.
