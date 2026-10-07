from pathlib import Path
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "report" / "Fetal_Health_Multiclassification_Report.pdf"

def make_report():
    path = RESULTS / "model_results.csv"
    if not path.exists():
        raise SystemExit("Run python src/run_experiments.py first.")
    df = pd.read_csv(path)
    best = df.iloc[0]

    c = canvas.Canvas(str(OUT), pagesize=A4)
    W, H = A4
    x = 42
    y = H - 50
    c.setFillColorRGB(0,0,0)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(x, y, "Fetal Health Multiclassification Using Machine Learning")
    y -= 18
    c.setFont("Helvetica", 8.5)
    c.drawString(x, y, "Rujul Jain — PES1UG24CS388 | Sanjana S Aithal — PES1UG24CS421")
    y -= 30

    sections = [
        ("1. Problem Statement",
         "The project studies multiclass classification of fetal health from cardiotocography measurements. The target contains Normal, Suspect, and Pathological fetal-state classes."),
        ("2. Dataset",
         "The project uses the UCI Cardiotocography dataset containing 2,126 CTG records and 21 predictive features. The fetal-state target is NSP. The CTGs were assigned consensus labels by three expert obstetricians."),
        ("3. Approach",
         "The pipeline uses a stratified 70/30 train-test split. Standard scaling is fitted only on the training data for scale-sensitive models. Multiple supervised classifiers are trained and compared."),
        ("4. Implementation and Results",
         f"The experiment evaluates Logistic Regression, Linear SVM, KNN, Decision Tree, Random Forest, Gradient Boosting, SVM, MLP, XGBoost, and LightGBM when available. The highest observed macro F1 in this execution was {best['Macro F1']:.4f} for {best['Model']}, with accuracy {best['Accuracy']:.4f}."),
        ("5. Conclusion",
         "The experiment provides a reproducible three-class fetal-state classification pipeline. Macro metrics and confusion matrices are emphasized because the class distribution is imbalanced. The project is an academic prototype and is not a clinically validated diagnostic system.")
    ]

    for title, body in sections:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x, y, title)
        y -= 14
        c.setFont("Helvetica", 8.7)
        words = body.split()
        line = ""
        for word in words:
            trial = (line + " " + word).strip()
            if c.stringWidth(trial, "Helvetica", 8.7) < W - 84:
                line = trial
            else:
                c.drawString(x, y, line); y -= 11; line = word
        if line:
            c.drawString(x, y, line); y -= 11
        y -= 8

    c.setFont("Helvetica-Bold", 9)
    c.drawString(x, y, "Model comparison")
    y -= 13
    c.setFont("Helvetica", 7)
    headers = ["Model","Accuracy","Macro P","Macro R","Macro F1","Weighted F1"]
    xs = [x, 205, 255, 305, 355, 420]
    for xx,h in zip(xs,headers): c.drawString(xx,y,h)
    y -= 10
    for _,r in df.iterrows():
        vals=[r["Model"],f"{r['Accuracy']:.3f}",f"{r['Macro Precision']:.3f}",
              f"{r['Macro Recall']:.3f}",f"{r['Macro F1']:.3f}",f"{r['Weighted F1']:.3f}"]
        for xx,v in zip(xs,vals): c.drawString(xx,y,str(v)[:22])
        y -= 9

    c.setFont("Helvetica", 6.5)
    c.drawString(x, 28, "Dataset: UCI Cardiotocography, DOI 10.24432/C51S4N. Assigned reference: Stanford CS229 project 81954164.")
    c.save()
    print(OUT)

if __name__ == "__main__":
    make_report()
