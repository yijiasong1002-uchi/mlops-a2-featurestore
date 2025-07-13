from pathlib import Path
import numpy as np, pandas as pd
from datetime import datetime, timezone, timedelta

RAW_CSV = Path("athletes.csv")
OUT_DIR  = Path("athletes_feature_store/data") ; OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PQ   = OUT_DIR / "athletes_clean.parquet"

df = pd.read_csv(RAW_CSV)
print(f"Raw shape: {df.shape}")

# Remove not relevant columns
col = ["region","age","weight","height","howlong","gender","eat","background","experience","schedule","deadlift","candj","snatch","backsq"]
df = df.dropna(subset=col)

drop_cols = ["affiliate","team","name","fran","helen","grace","filthy50","fgonebad","run400","run5k","pullups","train"]
df = df.drop(columns=drop_cols, errors="ignore")

# Remove Outliers
df = df[df["weight"] < 1500]
df = df[(df["height"] > 48) & (df["height"] < 96)]
df = df[(df["age"] >= 18)]
df = df[df["gender"] != "--"]

df = df[
    ( (df["gender"] == "Male")   & (df["deadlift"] > 0) & (df["deadlift"] <= 1105) ) |
    ( (df["gender"] == "Female") & (df["deadlift"] > 0) & (df["deadlift"] <= 636)  )]

df = df[(df["candj"]  > 0) & (df["candj"]  <= 395)]
df = df[(df["snatch"] > 0) & (df["snatch"] <= 496)]
df = df[(df["backsq"] > 0) & (df["backsq"] <= 1069)]

# Clean Survey Data
decline_dict = {"Decline to answer|": np.nan}
df = df.replace(decline_dict)
df = df.dropna(subset=["background","experience","schedule","howlong","eat"])

num_cols = ["age","height","weight","candj","snatch","deadlift","backsq"]
for c in num_cols:
    if c in df.columns:
        df[c] = df[c].astype("float32")
        df[c].fillna(df[c].median(), inplace=True)

def extract_years(h):
    if pd.isna(h): return 0.5
    h = str(h)
    if "4+ years"  in h: return 4.0
    if "2-4 years" in h: return 3.0
    if "1-2 years" in h: return 1.5
    if "6-12 months" in h: return 0.75
    return 0.5
df["experience_years"] = df["howlong"].apply(extract_years)
height_m  = df["height"] * 0.0254
weight_kg = df["weight"] * 0.453592
df["bmi"] = weight_kg / (height_m**2)
df["strength_index"]  = df["deadlift"] / df["weight"]

# timestamp
now_utc = datetime.now(timezone.utc)
base    = now_utc - timedelta(days=365)
df["event_timestamp"] = [base + timedelta(seconds=i) for i in range(len(df))]
df["created"]         = df["event_timestamp"]

df["athlete_id"] = df["athlete_id"].astype("int64")

df.to_parquet(OUT_PQ, index=False)
print(f"Cleaned data → {OUT_PQ}  | shape={df.shape}")