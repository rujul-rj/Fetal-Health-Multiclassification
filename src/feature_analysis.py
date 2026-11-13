import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from preprocessing import FEATURES, N_FEATURES, remove_duplicates


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fetal_health.csv"
RESULTS = ROOT / "results"

RESULTS.mkdir(exist_ok=True)


def plot_feature_scores(scores):
    # SelectKBest ANOVA F-scores, with the selected features in black
    scores = scores.sort_values("F Score")
    colors = ["black" if s else "lightgrey" for s in scores["Selected"]]

    plt.figure(figsize=(8, 6))
    plt.barh(scores["Feature"], scores["F Score"], color=colors)
    plt.xlabel("ANOVA F-score (training split)")
    plt.ylabel("Feature")
    plt.title(f"SelectKBest feature scores (top {N_FEATURES} in black)")
    plt.tight_layout()
    plt.savefig(RESULTS / "feature_scores.png", dpi=180)
    plt.close()

    scores.sort_values("F Score", ascending=False).to_csv(
        RESULTS / "feature_scores.csv", index=False
    )


def compare_with_pca(selected_features):
    # Compares SelectKBest with PCA at the same size (5 inputs)
    # using the same split, scaler and Random Forest.
    df = remove_duplicates(pd.read_csv(DATA))
    X, y = df[FEATURES], df["NSP"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    Z_train = pd.DataFrame(scaler.transform(X_train), columns=FEATURES)
    Z_test = pd.DataFrame(scaler.transform(X_test), columns=FEATURES)

    pca = PCA(n_components=N_FEATURES, random_state=42).fit(Z_train)
    inputs = {
        f"SelectKBest ({N_FEATURES} features)": (Z_train[selected_features], Z_test[selected_features]),
        f"PCA ({N_FEATURES} components)": (pca.transform(Z_train), pca.transform(Z_test)),
    }

    rows = []
    for name, (train, test) in inputs.items():
        model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=4)
        model.fit(train, y_train)
        predictions = model.predict(test)
        rows.append({
            "Reduction": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Macro F1": f1_score(y_test, predictions, average="macro"),
        })

    comparison = pd.DataFrame(rows)
    comparison["PCA variance explained"] = [None, pca.explained_variance_ratio_.sum()]
    comparison.to_csv(RESULTS / "pca_comparison.csv", index=False)
    print("\nSelectKBest vs PCA (Random Forest):")
    print(comparison.to_string(index=False))
    return comparison
