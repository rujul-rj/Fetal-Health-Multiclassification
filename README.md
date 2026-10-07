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

The Stanford report is used as the methodological reference. Our numerical results must come from our own execution.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python src/run_experiments.py
python src/generate_report.py
```

The first command downloads the dataset from UCI, trains the models, evaluates them, and creates figures/results. The second creates the two-page PDF report from the actual experiment results.

## Models
Logistic Regression, Linear SVM, KNN, Decision Tree, Random Forest, Gradient Boosting, SVM, MLP, XGBoost, and LightGBM when available.

## Evaluation
Accuracy, macro precision, macro recall, macro F1, weighted F1, and confusion matrices are reported. Macro metrics are included because the dataset is class-imbalanced.

## Limitation
This is an academic ML prototype using a public benchmark dataset. It is not a clinically validated diagnostic system.
