from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from download_dataset import download_dataset
from preprocessing import prepare
from models import get_models
from evaluation import evaluate


DATA = ROOT / "data" / "fetal_health.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


# Download dataset if it is not already present
if not DATA.exists():
    download_dataset()


# Load dataset
df = pd.read_csv(DATA)

print("Dataset shape:", df.shape)
print(df["NSP"].value_counts().sort_index())


# Class distribution
counts = df["NSP"].value_counts().sort_index()

plt.figure(figsize=(6, 4))

sns.barplot(
    x=["Normal", "Suspect", "Pathological"],
    y=counts.values,
    color="black"
)

plt.xlabel("Fetal health class")
plt.ylabel("Number of records")
plt.title("Fetal Health Class Distribution")

plt.tight_layout()

plt.savefig(
    RESULTS / "class_distribution.png",
    dpi=180
)

plt.close()


# Correlation heatmap
plt.figure(figsize=(12, 9))

sns.heatmap(
    df.corr(numeric_only=True),
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


# Prepare train/test data
X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = prepare(df)


# Models that require scaled features
scaled_models = {
    "Logistic Regression",
    "Linear SVM",
    "K-Nearest Neighbors",
    "Support Vector Machine",
    "Multi-layer Perceptron"
}


# Train and evaluate models
rows = []

for name, model in get_models().items():

    if name in scaled_models:
        Xtr = X_train_scaled
        Xte = X_test_scaled
    else:
        Xtr = X_train
        Xte = X_test

    print("Training:", name)

    if name == "XGBoost":
        # XGBoost requires class labels 0, 1, 2.
        # Original dataset labels are 1, 2, 3.
        y_train_xgb = y_train - 1

        model.fit(
            Xtr,
            y_train_xgb
        )

        # Convert XGBoost predictions back to
        # the original 1, 2, 3 labels.
        predictions = model.predict(Xte) + 1

        rows.append(
            evaluate(
                name,
                model,
                Xte,
                y_test,
                RESULTS,
                predictions=predictions
            )
        )

    else:
        model.fit(
            Xtr,
            y_train
        )

        rows.append(
            evaluate(
                name,
                model,
                Xte,
                y_test,
                RESULTS
            )
        )


# Create final results table
results = pd.DataFrame(rows)

results = results.sort_values(
    "Macro F1",
    ascending=False
)

results.to_csv(
    RESULTS / "model_results.csv",
    index=False
)


# Model comparison plot
plt.figure(figsize=(9, 5))

plot_data = results.sort_values(
    "Macro F1"
)

plt.barh(
    plot_data["Model"],
    plot_data["Macro F1"],
    color="black"
)

plt.xlabel("Macro F1")
plt.ylabel("Model")
plt.title("Model Comparison by Macro F1")

plt.tight_layout()

plt.savefig(
    RESULTS / "model_comparison.png",
    dpi=180
)

plt.close()


# Print final results
print("\nFinal Model Results:")
print(results.to_string(index=False))

print("\nResults saved to:")
print(RESULTS)