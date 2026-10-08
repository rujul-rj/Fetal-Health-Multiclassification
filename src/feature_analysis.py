import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from preprocessing import prepare, FEATURES


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fetal_health.csv"
RESULTS = ROOT / "results"

RESULTS.mkdir(exist_ok=True)


def analyze_feature_importance():
    # Load dataset
    df = pd.read_csv(DATA)

    # Use the same train/test split as the main project
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = prepare(df)

    # Random Forest does not require scaled features
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Calculate feature importance
    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    # Save feature importance values
    importance.to_csv(
        RESULTS / "feature_importance.csv",
        index=False
    )

    # Create feature importance plot
    plt.figure(figsize=(10, 7))

    plt.barh(
        importance["Feature"],
        importance["Importance"]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")

    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig(
        RESULTS / "feature_importance.png",
        dpi=180
    )

    plt.close()

    print("\nFeature Importance:")
    print(importance.to_string(index=False))

    print("\nSaved:")
    print(RESULTS / "feature_importance.csv")
    print(RESULTS / "feature_importance.png")


if __name__ == "__main__":
    analyze_feature_importance()
