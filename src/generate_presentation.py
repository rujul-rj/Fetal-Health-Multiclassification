from pathlib import Path
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "presentation" / "Fetal_Health_Multiclassification_Presentation.pptx"

PAPER_ACCURACY = 0.9558
BLACK = RGBColor(0, 0, 0)
GREY = RGBColor(0x55, 0x55, 0x55)
FONT = "Helvetica"


def add_text(slide, text, left, top, width, height, size=18, bold=False, color=BLACK):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    for i, line in enumerate(text if isinstance(text, list) else [text]):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.text = line
        p.space_after = Pt(8)
        for run in p.runs:
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.name = FONT
            run.font.color.rgb = color
    return box


def new_slide(prs, title):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_text(slide, title, 0.6, 0.35, 12, 0.8, size=30, bold=True)
    line = slide.shapes.add_connector(1, Inches(0.6), Inches(1.15), Inches(12.7), Inches(1.15))
    line.line.color.rgb = BLACK
    return slide


def bullets(slide, lines, left=0.6, top=1.45, width=12, size=20):
    return add_text(slide, ["•  " + l for l in lines], left, top, width, 5.5, size=size)


def add_table(slide, df, left, top, width):
    rows, cols = df.shape[0] + 1, df.shape[1]
    shape = slide.shapes.add_table(rows, cols, Inches(left), Inches(top), Inches(width), Inches(0.32 * rows))
    table = shape.table
    for j, col in enumerate(df.columns):
        table.cell(0, j).text = col
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        for j, v in enumerate(r):
            table.cell(i, j).text = str(v)
    for i in range(rows):
        for j in range(cols):
            cell = table.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0x22, 0x22, 0x22) if i == 0 else (
                RGBColor(0xEE, 0xEE, 0xEE) if i == 1 else RGBColor(0xFF, 0xFF, 0xFF))
            for p in cell.text_frame.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(13)
                    run.font.name = FONT
                    run.font.bold = i <= 1
                    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if i == 0 else BLACK


def make_presentation():
    df = pd.read_csv(RESULTS / "model_results.csv")
    scores = pd.read_csv(RESULTS / "feature_scores.csv")
    pca = pd.read_csv(RESULTS / "pca_comparison.csv")
    best = df.iloc[0]
    selected = scores.loc[scores["Selected"], "Feature"].tolist()
    kbest_acc, pca_acc = pca["Accuracy"].tolist()

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

    # 1. Title
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_text(slide, "Fetal Health Multiclassification", 0.8, 2.3, 11.7, 1, size=42, bold=True)
    add_text(slide, "Classifying CTG records with only 5 features", 0.8, 3.3, 11.7, 0.7, size=24, color=GREY)
    add_text(slide, ["Rujul Jain — PES1UG24CS388", "Sanjana S Aithal — PES1UG24CS421",
                     "UE24CS352A — Machine Learning"], 0.8, 4.6, 11.7, 1.6, size=18)

    # 2. Problem
    slide = new_slide(prs, "Problem Statement")
    bullets(slide, [
        "Cardiotocography (CTG) records fetal heart rate and uterine contractions.",
        "Task: classify each CTG record as Normal, Suspect or Pathological.",
        "Reference: Stanford CS229 report, SelectKBest with 10 features, 95.58% accuracy.",
        "Our goal: rebuild the same pipeline using only 5 features, half the measurements.",
    ])

    # 3. Dataset
    slide = new_slide(prs, "Dataset")
    bullets(slide, [
        "UCI Cardiotocography (ID 193): 2,126 records, 21 numeric features.",
        "Label NSP: consensus of three expert obstetricians.",
        "13 exact duplicate records removed, leaving 2,113 records.",
        "1,646 Normal, 292 Suspect, 175 Pathological (78% Normal).",
        "Imbalance means we report macro and per-class F1, not only accuracy.",
    ], width=6.4, size=18)
    slide.shapes.add_picture(str(RESULTS / "class_distribution.png"), Inches(7.2), Inches(1.6), width=Inches(5.6))

    # 4. Methodology
    slide = new_slide(prs, "Methodology")
    bullets(slide, [
        "1. Remove the 13 duplicate records.",
        "2. Stratified 70/30 train-test split (seed 42): 1,479 train, 634 test.",
        "3. StandardScaler fitted on the training split only.",
        "4. SelectKBest (ANOVA F-test), k = 5, fitted on the training split only.",
        "5. Ten classifiers, each tuned with a small grid search (5-fold CV on train).",
        "6. One final evaluation on the held-out test set.",
    ])

    # 5. Feature selection
    slide = new_slide(prs, "Feature Selection: the 5 Features")
    bullets(slide, [
        "DP: prolonged decelerations",
        "ASTV: % time with abnormal short-term variability",
        "ALTV: % time with abnormal long-term variability",
        "Mean: histogram mean of heart rate",
        "Mode: histogram mode of heart rate",
        f"Selected: {', '.join(selected)}",
    ], width=5.6, size=18)
    slide.shapes.add_picture(str(RESULTS / "feature_scores.png"), Inches(6.4), Inches(1.35), height=Inches(5.6))

    # 6. Why not PCA
    slide = new_slide(prs, "Why SelectKBest and not PCA?")
    bullets(slide, [
        "Same split, same untuned Random Forest (300 trees), 5 inputs each:",
        f"SelectKBest: {kbest_acc:.2%} accuracy.   PCA: {pca_acc:.2%} accuracy.",
        "PCA keeps the directions of most variance, not the most class information.",
        "PCA mixes all features together, so trees lose simple thresholds like ASTV > x.",
        "SelectKBest keeps real, interpretable clinical measurements.",
    ])

    # 7. Models
    slide = new_slide(prs, "Models")
    bullets(slide, [
        "Linear: Logistic Regression, Linear SVM",
        "Instance-based: K-Nearest Neighbors",
        "Kernel: Support Vector Machine (RBF)",
        "Neural network: Multi-layer Perceptron",
        "Trees: Decision Tree, Random Forest",
        "Boosting: Gradient Boosting, XGBoost, LightGBM",
    ])

    # 8. Results table
    slide = new_slide(prs, "Results (Test Set, 634 Records)")
    table = df[["Model", "Accuracy", "Macro F1", "Normal F1", "Suspect F1", "Pathological F1"]].copy()
    table["Accuracy"] = table["Accuracy"].map(lambda v: f"{v:.2%}")
    for c in ["Macro F1", "Normal F1", "Suspect F1", "Pathological F1"]:
        table[c] = table[c].map(lambda v: f"{v:.3f}")
    add_table(slide, table, 0.6, 1.45, 12.1)

    # 9. Comparison and confusion matrix
    slide = new_slide(prs, "Comparison with the Reference Paper")
    slide.shapes.add_picture(str(RESULTS / "model_comparison.png"), Inches(0.4), Inches(1.4), width=Inches(7.4))
    cm_name = best["Model"].lower().replace(" ", "_").replace("-", "_")
    slide.shapes.add_picture(str(RESULTS / f"{cm_name}_confusion_matrix.png"), Inches(8.0), Inches(1.4), width=Inches(5.0))
    add_text(slide, [
        f"Best: {best['Model']}, {best['Accuracy']:.2%} with 5 features",
        f"Paper: 95.58% with 10 features ({(PAPER_ACCURACY - best['Accuracy']) * 100:.1f} points higher)",
    ], 0.6, 5.7, 12, 1.2, size=18)

    # 10. Live demo
    slide = new_slide(prs, "Live Demonstration")
    bullets(slide, [
        "python src/run_experiments.py",
        "Load data and remove duplicates",
        "Split, scale and select the 5 features",
        "Tune and evaluate all ten models",
        "Show the results table, feature scores and confusion matrix",
        "Notebook: notebooks/Fetal_Health_Multiclassification.ipynb",
    ])

    # 11. Conclusion
    slide = new_slide(prs, "Conclusions and Limitations")
    bullets(slide, [
        f"5 features reach {best['Accuracy']:.2%} ({best['Model']}), with half the paper's inputs.",
        "This does not beat the paper's 95.58%, which uses 10 features.",
        f"Normal and Pathological stay strong; Suspect is hardest (F1 {best['Suspect F1']:.2f}).",
        "Removing duplicates and fitting on train only keep the evaluation honest.",
        "Single split: a 1–2 record difference (0.16% each) is noise.",
        "Academic prototype, not a clinically validated diagnostic system.",
    ], size=19)

    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    make_presentation()
