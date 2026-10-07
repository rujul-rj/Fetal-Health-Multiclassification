from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "fetal_health.csv"

def download_dataset():
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=193)
    X = ds.data.features.copy()
    y = ds.data.targets.copy()
    target = "NSP" if "NSP" in y.columns else y.columns[-1]
    df = pd.concat([X, y[[target]]], axis=1)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} rows to {OUT}")
    return df

if __name__ == "__main__":
    download_dataset()
