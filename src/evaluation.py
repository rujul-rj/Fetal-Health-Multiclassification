import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def evaluate(
    name,
    model,
    X_test,
    y_test,
    results_dir,
    predictions=None
):
    if predictions is None:
        predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    macro_precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    class_f1 = f1_score(
        y_test,
        predictions,
        labels=[1, 2, 3],
        average=None,
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    # Confusion matrix using the original class labels
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[1, 2, 3]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Normal",
            "Suspect",
            "Pathological"
        ]
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    display.plot(
        ax=ax,
        cmap="Greys",
        colorbar=False
    )

    ax.set_title(f"{name} - Confusion Matrix")
    plt.tight_layout()

    safe_name = name.lower().replace(" ", "_").replace("-", "_")

    plt.savefig(
        results_dir / f"{safe_name}_confusion_matrix.png",
        dpi=180
    )

    plt.close()

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Normal F1": class_f1[0],
        "Suspect F1": class_f1[1],
        "Pathological F1": class_f1[2]
    }