from pathlib import Path
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "report" / "Fetal_Health_Multiclassification_Report.pdf"

PAPER_ACCURACY = 0.9558

TITLE = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=14, leading=17, textColor=colors.black)
HEAD = ParagraphStyle("head", fontName="Helvetica-Bold", fontSize=10, leading=13, spaceBefore=6, spaceAfter=2, textColor=colors.black)
BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=8.7, leading=11.2, textColor=colors.black)
SMALL = ParagraphStyle("small", fontName="Helvetica", fontSize=7, leading=9, textColor=colors.black)


def make_report():
    path = RESULTS / "model_results.csv"
    if not path.exists():
        raise SystemExit("Run python src/run_experiments.py first.")
    df = pd.read_csv(path)
    scores = pd.read_csv(RESULTS / "feature_scores.csv")
    pca = pd.read_csv(RESULTS / "pca_comparison.csv")
    best = df.iloc[0]
    selected = scores.loc[scores["Selected"], "Feature"].tolist()
    gap = (PAPER_ACCURACY - best["Accuracy"]) * 100
    kbest_acc, pca_acc = pca["Accuracy"].tolist()

    story = [
        Paragraph("Fetal Health Multiclassification Using a Reduced Feature Set", TITLE),
        Paragraph("Rujul Jain — PES1UG24CS388 | Sanjana S Aithal — PES1UG24CS421 | UE24CS352A Machine Learning", SMALL),
        Spacer(1, 6),

        Paragraph("1. Problem Statement", HEAD),
        Paragraph(
            "Cardiotocography (CTG) records fetal heart rate and uterine contractions during pregnancy. "
            "The task is to classify each CTG record into one of three fetal-state classes: Normal, Suspect or "
            "Pathological. Our assigned reference, a Stanford CS229 report, selects 10 of the 21 CTG features with "
            "SelectKBest and reaches 95.58% test accuracy. Our goal was to rebuild the same pipeline with only "
            "<b>5 features</b>, so that a model needs half the measurements, and to see how much accuracy that costs.", BODY),

        Paragraph("2. Dataset", HEAD),
        Paragraph(
            "We use the UCI Cardiotocography dataset (ID 193): 2,126 CTG records with 21 numeric features and the "
            "3-class NSP label assigned by consensus of three expert obstetricians. The data contains <b>13 exact "
            "duplicate records</b>. We remove them, leaving 2,113 records (1,646 Normal, 292 Suspect, 175 "
            "Pathological), the same as the reference paper. Without this step a record could appear in both the "
            "training and the test set, which inflates test accuracy. The classes are imbalanced (78% Normal), so we "
            "report macro and per-class F1 alongside accuracy.", BODY),

        Paragraph("3. Approach", HEAD),
        Paragraph(
            "We follow the reference paper's pipeline: (1) remove duplicates; (2) stratified 70/30 train-test split "
            "(seed 42; 1,479 train, 634 test); (3) StandardScaler; (4) SelectKBest with the ANOVA F-test, keeping "
            f"k = 5 features; (5) train ten classifiers. The scaler and selector are fitted on the training split only, "
            "so the test set plays no part in choosing features. Each model is lightly tuned with a small grid search "
            "using 5-fold stratified cross-validation on the training split, and evaluated once on the test set. "
            f"The five features selected are: <b>{', '.join(selected)}</b> (prolonged decelerations, abnormal "
            "short-term variability, percentage of time with abnormal long-term variability, and the histogram mode "
            "and mean).", BODY),
        Paragraph(
            "We also considered PCA. With the same split and an untuned Random Forest, 5 PCA components give "
            f"{pca_acc:.2%} accuracy against {kbest_acc:.2%} for the 5 SelectKBest features. PCA builds components that "
            "keep the most variance, not the most class information, and it mixes features together, which removes "
            "the simple single-feature thresholds that tree models rely on. SelectKBest was therefore kept.", BODY),

        Paragraph("4. Implementation", HEAD),
        Paragraph(
            "The project is written in Python with pandas, scikit-learn, XGBoost and LightGBM. "
            "<font name='Courier'>download_dataset.py</font> builds the dataset, "
            "<font name='Courier'>preprocessing.py</font> handles duplicates, the split, scaling and SelectKBest, "
            "<font name='Courier'>models.py</font> defines the ten models and their grids, "
            "<font name='Courier'>evaluation.py</font> computes metrics and confusion matrices, and "
            "<font name='Courier'>run_experiments.py</font> runs the full pipeline and writes every table and "
            "figure in <font name='Courier'>results/</font>. A Jupyter notebook walks through the same steps for the "
            "live demonstration.", BODY),
        Spacer(1, 4),
        Image(str(RESULTS / "feature_scores.png"), width=10.5 * cm, height=7.9 * cm),
        PageBreak(),

        Paragraph("5. Results", HEAD),
        Paragraph(
            f"The best model is <b>{best['Model']}</b> with <b>{best['Accuracy']:.2%}</b> test accuracy "
            f"({round(best['Accuracy'] * 634)}/634 records) and macro F1 {best['Macro F1']:.3f}. "
            "Gradient Boosting and LightGBM follow closely. Tree ensembles clearly outperform the linear models, "
            "which cannot separate the Suspect class well using only five features.", BODY),
        Spacer(1, 4),
    ]

    header = ["Model", "Accuracy", "Macro F1", "Normal F1", "Suspect F1", "Path. F1", "CV acc."]
    rows = [header] + [
        [r["Model"], f"{r['Accuracy']:.4f}", f"{r['Macro F1']:.3f}", f"{r['Normal F1']:.3f}",
         f"{r['Suspect F1']:.3f}", f"{r['Pathological F1']:.3f}", f"{r['CV Accuracy']:.4f}"]
        for _, r in df.iterrows()
    ]
    table = Table(rows, colWidths=[4.6 * cm] + [2.0 * cm] * 6)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]))
    story += [table, Spacer(1, 6)]

    cm_name = best["Model"].lower().replace(" ", "_").replace("-", "_")
    figures = Table([[
        Image(str(RESULTS / "model_comparison.png"), width=10.2 * cm, height=5.7 * cm),
        Image(str(RESULTS / f"{cm_name}_confusion_matrix.png"), width=6.6 * cm, height=5.3 * cm),
    ]])
    story += [figures, Spacer(1, 4)]

    story += [
        Paragraph("6. Conclusions", HEAD),
        Paragraph(
            f"Using only 5 of the 21 CTG features, our best model reaches {best['Accuracy']:.2%} accuracy. That is "
            f"{gap:.1f} percentage points below the reference paper's 95.58%, which uses 10 features, so the "
            "5-feature model does not beat the paper. In exchange, it needs half the inputs. Normal and "
            "Pathological records are still recognised well (F1 about 0.96 and 0.90). Most of the loss is in the "
            f"Suspect class (F1 {best['Suspect F1']:.2f}), which sits between the other two and depends on features "
            "that were dropped, such as accelerations and short-term variability. Removing the 13 duplicate records "
            "makes the evaluation honest, and fitting the scaler and selector on the training data only prevents "
            "information leaking from the test set. SelectKBest outperformed PCA at the same size. A natural next "
            "step is to choose features with a model-based or class-weighted criterion aimed at the Suspect class.", BODY),
        Paragraph(
            "Limitations: results come from a single 70/30 split of a public benchmark, so a difference of one or two "
            "test records (0.16% each) between models is within noise. This is an academic prototype, not a clinically "
            "validated diagnostic system.", BODY),
        Spacer(1, 4),
        Paragraph(
            "Data: UCI Cardiotocography, DOI 10.24432/C51S4N. Reference: Ereyi and Huo, Multiclass Classification of "
            "Fetal Health using Cardiotocogram Data, Stanford CS229, Spring 2021. All numbers above, apart from the "
            "paper's 95.58%, come from our own runs.", SMALL),
    ]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.4 * cm, bottomMargin=1.2 * cm,
                            title="Fetal Health Multiclassification Report")
    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    make_report()
