# ========================= DATA TABLE MATCHING RULES =========================
# The current public H2O2 table uses:
#   source_file, Name, Barrier_eV, X1, X2, ..., X20
# For prediction, Barrier_eV is an output column and must never be used as an
# input feature. The model input is exactly X1...X20. Read by column name,
# not by df.iloc positions, because future files may add or reorder metadata.
# If the input format changes, update the validation and explicit mapping
# below before running predictions.
# ============================================================================

import joblib
import os
import pandas as pd
from pathlib import Path
import numpy as np


TARGET_COL = "Barrier_eV"
FEATURE_COLS = [f"X{i}" for i in range(1, 21)]
CAT_COLS = ["X2"]
NUM_COLS = [col for col in FEATURE_COLS if col not in CAT_COLS]

MODEL_PATH = "MODEL_PATH"
INPUT_DIR = "IN_PATH"
OUT_DIR = "OUT_PATH"
# These are placeholders, not portable defaults. Before running prediction,
# confirm and update all three paths for the local model, input directory, and
# output directory on the current machine.

os.makedirs(OUT_DIR, exist_ok=True)

model = joblib.load(MODEL_PATH)


def _normalize_category(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    try:
        number = float(text)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return text


def normalize_features(frame):
    missing = [col for col in FEATURE_COLS if col not in frame.columns]
    if missing:
        raise ValueError(
            "Missing required descriptor columns: {}".format(", ".join(missing))
        )

    features = frame[FEATURE_COLS].copy()
    features = features.replace(["-", "--", "NA", "N/A", "na", ""], np.nan)
    for col in NUM_COLS:
        features[col] = pd.to_numeric(features[col], errors="coerce")
    features["X2"] = features["X2"].map(_normalize_category)
    return features


def validate_input_table(frame, path):
    # The current public prediction-compatible table has 22 columns when
    # Barrier_eV is absent, or 23 columns when it is present as a blank/known
    # output column. Both forms must still contain source_file, Name, X1-X20.
    required = {"source_file", "Name", *FEATURE_COLS}
    missing = sorted(required.difference(frame.columns))
    allowed = required.union({TARGET_COL})
    unexpected = [col for col in frame.columns if col not in allowed]
    print(f"{path}: {len(frame)} rows, {len(frame.columns)} columns")
    print(f"Columns: {list(frame.columns)}")
    if len(frame.columns) not in (22, 23) or missing or unexpected:
        raise ValueError(
            "Prediction table schema mismatch. Expected source_file, Name, "
            "X1-X20 and optional Barrier_eV; "
            f"missing={missing}, unexpected={unexpected}, "
            f"actual_count={len(frame.columns)}"
        )

excel_files = []
for ext in ("*.xlsx", "*.xls"):
    excel_files.extend(Path(INPUT_DIR).glob(ext))

if not excel_files:
    raise FileNotFoundError(f"No Excel files found in: {INPUT_DIR}")

for file_path in excel_files:
    print(f"Processing: {file_path.name}")

    df = pd.read_excel(file_path)

    validate_input_table(df, file_path.name)
    X_target = normalize_features(df)

    pred_xgb = model.predict(X_target)

    result_df = pd.DataFrame({
        "source_file": df["source_file"],
        "Name": df["Name"],
        TARGET_COL: pred_xgb
    })

    out_path = os.path.join(OUT_DIR, f"{file_path.stem}_pred.xlsx")
    result_df.to_excel(out_path, index=False)

    print(f"  Saved: {out_path}")

print("=" * 70)
print("All done.")
print("Saved to:", OUT_DIR)
print("=" * 70)
