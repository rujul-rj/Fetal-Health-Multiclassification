from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "fetal_health.csv"

# Public mirror of the raw UCI "CTG.xls" sheet (UCI dataset 193).
RAW_URL = "https://raw.githubusercontent.com/pycaret/datasets/main/data/common/CTG.csv"

FEATURES = [
    "LB","AC","FM","UC","DL","DS","DP","ASTV","MSTV","ALTV","MLTV",
    "Width","Min","Max","Nmax","Nzeros","Mode","Mean","Median","Variance","Tendency"
]

# These columns are raw event counts in the CTG sheet. UCI publishes them
# as events per second, so divide by the segment length (e - b).
COUNT_COLUMNS = ["AC","FM","UC","DL","DS","DP"]


def download_dataset():
    raw = pd.read_csv(RAW_URL)
    raw = raw.dropna(subset=["NSP", "LB", "b", "e"])
    duration = raw["e"] - raw["b"]
    for col in COUNT_COLUMNS:
        raw[col] = (raw[col] / duration).round(3)
    df = raw[FEATURES + ["NSP"]].astype(float)
    df["NSP"] = df["NSP"].astype(int)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} rows to {OUT}")
    return df


if __name__ == "__main__":
    download_dataset()
