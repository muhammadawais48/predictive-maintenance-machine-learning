# Predictive Maintenance for Industrial Machinery

**Python · SQLite · SQL feature engineering · scikit-learn · Random Forest**

A reproducible classification project that explores whether machine sensor measurements can help identify equipment failures. The workflow imports a CSV dataset into SQLite, calculates two additional features in SQL, trains a Random Forest classifier, and evaluates predictions on a held-out test set.

**Author:** Muhammad Awais

## Problem and approach

Unexpected machine failures can interrupt industrial operations. This project investigates a simple supervised-learning approach to identifying failure records using sensor measurements. The model is a learning exercise and **is not validated for deployment or live maintenance decisions**.

The supplied dataset contains **10,000 rows**, **589 machine-failure labels**, and **14 original columns**. The class imbalance makes failure-specific precision, recall and average precision important alongside accuracy.

## Dataset

The included `dataset.csv` has the column structure of the **AI4I 2020 Predictive Maintenance Dataset** (commonly distributed through the UCI Machine Learning Repository). The exact provenance and any redistribution terms of this uploaded CSV have **not been independently verified**. Before publishing or reusing it commercially, confirm its source and applicable license.

The original file contains sensor readings, product type, a binary `Machine failure` target and failure-type indicators. The model uses **`Machine failure` directly as its target**. Failure-type flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are **not included as model inputs** because they would leak outcome information.

`UDI` and `Product ID` are identifiers and are excluded from the model. The model uses product `Type`, temperature, speed, torque, tool wear and two engineered features:

- `Temp_Difference` = process temperature − air temperature
- `Power_Proxy` = rotational speed × torque (a relative proxy, **not** a calibrated power measurement)

The original version experimented with rolling averages ordered by `UDI`. Those were removed because row identifiers do not establish genuine measurement timestamps, and row-order features could create misleading patterns.

## Repository structure

```text
.
├── dataset.csv                 # Source CSV supplied with the project
├── database_setup.py           # CSV validation and SQLite import
├── feature_engineering.py      # SQL feature extraction
├── model_training.py           # Train, evaluate, and save figures
├── main.py                     # End-to-end entry point
├── requirements.txt            # Python dependencies
├── outputs/                    # Generated results and plots
├── tests/test_pipeline.py      # Data and pipeline checks
└── README.md
```

## Run locally

Requires **Python 3.10+**. From the repository directory:

```bash
python -m pip install -r requirements.txt
python main.py
```

The script creates a local `predictive_maintenance.db` database and refreshes the `outputs/` folder. The database is excluded from Git because it can be regenerated from `dataset.csv`.

Run the automated checks with:

```bash
python -m unittest discover -s tests -v
```

## Modeling choices

- **Train/test split:** 80/20, stratified by the failure label, `random_state=42`.
- **Preprocessing:** One-hot encoding for product type; numerical sensor measurements passed through unchanged.
- **Model:** Random Forest with 200 trees, balanced class weights and a fixed random seed.
- **Evaluation:** confusion matrix, precision, recall, F1, ROC-AUC and average precision on the held-out test set.
- **Threshold:** default probability threshold of 0.5. No tuning on the test set.

### Results from the included dataset

| Metric | Held-out test result |
|---|---:|
| Accuracy | 97.55% |
| Failure precision | 96.00% |
| Failure recall | 61.02% |
| Failure F1 | 74.61% |
| ROC-AUC | 0.9714 |
| Average precision | 0.7940 |

The confusion matrix is `[[1879, 3], [46, 72]]` (rows: actual non-failure/failure; columns: predicted non-failure/failure). This means **46 of 118 actual failures were missed** at the default threshold. Results are from a single stratified random split and should not be interpreted as field performance.

The complete measured results are saved in [`outputs/metrics.json`](outputs/metrics.json), with a classification report, feature importance table and figures in the same folder.

![Confusion matrix](outputs/confusion_matrix.png)

![Precision–recall curve](outputs/precision_recall_curve.png)

## Interpretation and limitations

Accuracy alone can be misleading because failure events are relatively rare. **Failure recall** indicates how many actual failures were detected; **precision** indicates how many flagged failures were correct. Average precision summarizes precision–recall performance across thresholds, and should be interpreted relative to the positive-class rate.

This is an exploratory portfolio project, not a production system. The dataset is static and does not demonstrate real-time prediction. A random split may be optimistic if observations share underlying machine or production-process patterns. Before operational use, evaluation would require independently collected data, a genuine time-based or machine-based split, calibration, and analysis of the cost of missed failures versus false alarms.

## Reproducibility

The CSV is included, all transformations are in source files, and random seeds are fixed. The generated results can be reproduced by running `python main.py`. Package versions may affect exact scores across environments.
