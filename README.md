# Fetal Health Multiclassification Using Machine Learning

**Course:** UE24CS352A - Machine Learning

**Team**
- Rujul Jain — PES1UG24CS388
- Sanjana S Aithal — PES1UG24CS421

## Problem
Classify fetal health into Normal, Suspect, and Pathological states using cardiotocography (CTG) measurements.

## Dataset
Official UCI Cardiotocography dataset:
https://archive.ics.uci.edu/dataset/193/cardiotocography

UCI reports 2,126 CTG records and 21 predictive features. The CTGs were classified by three expert obstetricians using consensus labels. This project uses the 3-class fetal-state target (NSP).

## Assigned reference
Stanford CS229 project:
https://cs229.stanford.edu/proj2021spr/report2/81954164.pdf

The reference removes duplicate records, standardizes the features, keeps 10 of the 21 features with SelectKBest and reports a best test accuracy of 95.58%. We follow the same pipeline but keep only **5 features**. Our numerical results come from our own execution.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python src/run_experiments.py       # train, tune and evaluate; writes results/
python src/generate_report.py       # two-page PDF write-up in report/
python src/generate_presentation.py # slide deck in presentation/
```

`data/fetal_health.csv` is included. If it is missing, `run_experiments.py` rebuilds it with `src/download_dataset.py` from a public copy of the raw UCI `CTG.xls` sheet, converting event counts to per-second rates as UCI publishes them. The full experiment takes about 4 minutes on a laptop.

The notebook `notebooks/Fetal_Health_Multiclassification.ipynb` runs the same pipeline step by step for the live demo.

## Pipeline
1. Remove the 13 exact duplicate records (2,126 to 2,113 records: 1,646 Normal, 292 Suspect, 175 Pathological).
2. Stratified 70/30 train-test split, seed 42 (1,479 train, 634 test).
3. StandardScaler, fitted on the training split only.
4. SelectKBest with the ANOVA F-test, k = 5, fitted on the training split only.
5. Ten classifiers, each tuned with a small grid search (5-fold stratified CV on the training split).
6. One evaluation on the test set.

Selected features: **DP** (prolonged decelerations), **ASTV** (abnormal short-term variability), **ALTV** (% time with abnormal long-term variability), **Mean** and **Mode** (heart-rate histogram).

## Models
Logistic Regression, Linear SVM, KNN, Decision Tree, Random Forest, Gradient Boosting, SVM, MLP, XGBoost and LightGBM.

## Results (test set, 5 features)

| Model | Accuracy | Macro F1 | Normal F1 | Suspect F1 | Pathological F1 |
|---|---|---|---|---|---|
| Random Forest | 93.38% | 0.882 | 0.962 | 0.783 | 0.902 |
| Gradient Boosting | 93.22% | 0.887 | 0.959 | 0.788 | 0.913 |
| LightGBM | 93.22% | 0.886 | 0.960 | 0.766 | 0.932 |
| XGBoost | 92.43% | 0.876 | 0.954 | 0.763 | 0.913 |
| Multi-layer Perceptron | 91.80% | 0.846 | 0.955 | 0.736 | 0.846 |
| K-Nearest Neighbors | 91.64% | 0.841 | 0.956 | 0.729 | 0.837 |
| Decision Tree | 91.17% | 0.847 | 0.947 | 0.722 | 0.872 |
| Support Vector Machine | 91.17% | 0.830 | 0.952 | 0.688 | 0.851 |
| Logistic Regression | 86.44% | 0.722 | 0.930 | 0.519 | 0.716 |
| Linear SVM | 84.86% | 0.671 | 0.924 | 0.467 | 0.622 |

The best 5-feature model (Random Forest, 93.38%) is 2.2 points below the reference paper's 95.58% with 10 features. Most of the loss is in the Suspect class.

PCA was also tried at the same size: with an untuned Random Forest, 5 PCA components give 89.75% against 93.53% for the 5 SelectKBest features (`results/pca_comparison.csv`).

## Outputs
- `results/model_results.csv`: metrics, CV accuracy and best parameters for every model
- `results/feature_scores.csv`, `results/feature_scores.png`: SelectKBest scores
- `results/pca_comparison.csv`: SelectKBest vs PCA
- `results/model_comparison.png`, `results/*_confusion_matrix.png`, `results/class_distribution.png`, `results/correlation_heatmap.png`

## Evaluation
Accuracy, macro precision, macro recall, macro F1, weighted F1, per-class F1 and confusion matrices are reported. Macro and per-class metrics are included because the dataset is class-imbalanced.

## Limitation
This is an academic ML prototype using a public benchmark dataset. It is not a clinically validated diagnostic system.
