import joblib
from sklearn.pipeline import Pipeline
import os
import pandas as pd
from pathlib import Path
import numpy as np

MODEL_PATH = "MODEL_PATH"
INPUT_DIR = "IN_PATH"
OUT_DIR = "OUT_PATH"

os.makedirs(OUT_DIR, exist_ok=True)

model = joblib.load(MODEL_PATH)

excel_files = []
for ext in ("*.xlsx", "*.xls"):
    excel_files.extend(Path(INPUT_DIR).glob(ext))

if not excel_files:
    raise FileNotFoundError(f"No Excel files found in: {INPUT_DIR}")

for file_path in excel_files:
    print(f"Processing: {file_path.name}")

    df = pd.read_excel(file_path)

    if df.shape[1] < 22:
        print(f"  Skipped: {file_path.name} (only {df.shape[1]} columns, need at least 22)")
        continue

    # check!
    X_target = df.iloc[:, 2:22].copy()
    X_target.columns = [f"X{i}" for i in range(1, 21)]

    X_target = X_target.replace(["-", "--", "NA", "N/A", "na", ""], np.nan)

    num_cols = [f"X{i}" for i in range(1, 21) if f"X{i}" != "X2"]

    for col in num_cols:
        X_target[col] = pd.to_numeric(X_target[col], errors="coerce")

    X_target["X2"] = X_target["X2"].astype(str).str.strip()
    X_target["X2"] = X_target["X2"].replace(["nan", "None", ""], np.nan)

    pred_xgb = model.predict(X_target)

    result_df = pd.DataFrame({
        "Index": df.iloc[:, 1],
        "Name": df.iloc[:, 0],
        "pred_barrier": pred_xgb
    })

    out_path = os.path.join(OUT_DIR, f"{file_path.stem}_pred.xlsx")
    result_df.to_excel(out_path, index=False)

    print(f"  Saved: {out_path}")

print("=" * 70)
print("All done.")
print("Saved to:", OUT_DIR)
print("=" * 70)