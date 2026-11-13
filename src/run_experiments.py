from pathlib import Path
import sys
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from download_dataset import download_dataset
from preprocessing import prepare, remove_duplicates, N_FEATURES
from models import get_models
from evaluation import evaluate
from feature_analysis import plot_feature_scores, compare_with_pca

warnings.filterwarnings("ignore")

DATA = ROOT / "data" / "fetal_health.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Best test accuracy in the reference paper (Stanford CS229, 10 features)
PAPER_ACCURACY = 0.9558


# Download dataset if it is not already present
if not DATA.exists():
    download_dataset()


# Load dataset
df = pd.read_csv(DATA)

print("Dataset shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())

df_clean = remove_duplicates(df)
print("Shape after removing duplicates:", df_clean.shape)
print(df_clean["NSP"].value_counts().sort_index())


# Class distribution
counts = df_clean["NSP"].value_counts().sort_index()

plt.figure(figsize=(6, 4))

sns.barplot(
    x=["Normal", "Suspect", "Pathological"],
    y=counts.values,
    color="black"
)

plt.xlabel("Fetal health class")
plt.ylabel("Number of records")
plt.title("Fetal Health Class Distribution (duplicates removed)")

plt.tight_layout()

plt.savefig(
    RESULTS / "class_distribution.png",
    dpi=180
)

plt.close()


# Correlation heatmap
plt.figure(figsize=(12, 9))

sns.heatmap(
    df_clean.corr(numeric_only=True),
    cmap="Greys",
    center=0
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    RESULTS / "correlation_heatmap.png",
    dpi=180
)

plt.close()


# Split, scale and select the top features (fitted on training data only)
X_train, X_test, y_train, y_test, scores = prepare(df)

selected = scores.loc[scores["Selected"], "Feature"].tolist()
print(f"\nSelected {N_FEATURES} features:", selected)
plot_feature_scores(scores)


# Tune and evaluate every model
cv = StratifiedKFold(5, shuffle=True, random_state=42)
rows = []

for name, (model, grid) in get_models().items():

    print("Training:", name)

    # Labels 1, 2, 3 are shifted to 0, 1, 2 for XGBoost and shifted back
    search = GridSearchCV(model, grid, cv=cv, scoring="accuracy", n_jobs=1)
    search.fit(X_train, y_train - 1)
    predictions = search.predict(X_test) + 1

    row = evaluate(
        name,
        search.best_estimator_,
        X_test,
        y_test,
        RESULTS,
        predictions=predictions
    )
    row["CV Accuracy"] = search.best_score_
    row["Best Parameters"] = str(search.best_params_)
    rows.append(row)


# Create final results table
results = pd.DataFrame(rows)

results = results.sort_values(
    "Accuracy",
    ascending=False
)

results.to_csv(
    RESULTS / "model_results.csv",
    index=False
)


# Model comparison plot
plt.figure(figsize=(9, 5))

plot_data = results.sort_values(
    "Accuracy"
)

plt.barh(
    plot_data["Model"],
    plot_data["Accuracy"],
    color="black"
)

plt.axvline(
    PAPER_ACCURACY,
    color="grey",
    linestyle="--",
    label=f"Reference paper, 10 features ({PAPER_ACCURACY:.2%})"
)

plt.xlim(0.8, 1.0)
plt.xlabel("Test accuracy")
plt.ylabel("Model")
plt.title(f"Model Comparison ({N_FEATURES} SelectKBest features)")
plt.legend(loc="lower right")

plt.tight_layout()

plt.savefig(
    RESULTS / "model_comparison.png",
    dpi=180
)

plt.close()


# SelectKBest vs PCA at the same number of inputs
compare_with_pca(selected)


# Print final results
print("\nFinal Model Results:")
print(results.drop(columns="Best Parameters").round(4).to_string(index=False))

best = results.iloc[0]
print(
    f"\nBest: {best['Model']} with accuracy {best['Accuracy']:.4f} "
    f"(reference paper: {PAPER_ACCURACY:.4f} with 10 features)"
)

print("\nResults saved to:", RESULTS)
