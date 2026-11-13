# Presentation Content

The deck is generated from the results by `python src/generate_presentation.py`.

1. Title: Fetal Health Multiclassification, classifying CTG records with only 5 features
2. Problem statement: three-class CTG classification; reference paper uses 10 features (95.58%); our goal is 5
3. Dataset: UCI CTG, 2,126 records, 13 duplicates removed (2,113), class distribution
4. Methodology: dedup, stratified 70/30 split, train-only scaling and SelectKBest (k=5), grid-search tuning
5. Feature selection: DP, ASTV, ALTV, Mean, Mode with the F-score chart
6. Why SelectKBest and not PCA: 93.53% vs 89.75% with the same untuned Random Forest
7. Models: the ten classifiers
8. Results: test-set table with accuracy and per-class F1
9. Comparison with the paper: model comparison chart and best-model confusion matrix
10. Live demonstration: run_experiments.py and the notebook
11. Conclusions and limitations
