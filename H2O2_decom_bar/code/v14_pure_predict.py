import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

DATA_PATH = "/root/workspace/h2o2_training/data/H2O2_merged_all.xlsx"
TARGET_DATA_PATH = "/root/workspace/h2o2_training/data/Target_data.xlsx"
OUT_DIR = "/root/workspace/h2o2_training/result/v14_pure_predict"
TARGET_COL = "Barrier_eV"
CAT_COLS = ["X2"]
RANDOM_STATE = 42

os.makedirs(OUT_DIR, exist_ok=True)

def evaluate(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred)
    }

print("=" * 70)
print("Pure prediction")
print("=" * 70)

df = pd.read_excel(DATA_PATH)
df_target = pd.read_excel(TARGET_DATA_PATH)

feature_cols = [c for c in df.columns if str(c).startswith("X")]
num_cols = [c for c in feature_cols if c not in CAT_COLS]

df_clean = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
X = df_clean[feature_cols].copy()
y = df_clean[TARGET_COL].astype(float).copy()

X_target = df_target[feature_cols].copy()
y_target_true = df_target[TARGET_COL].copy()

num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, CAT_COLS)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

best_params = {
    "n_estimators": 200,
    "max_depth": 2,
    "learning_rate": 0.02,
    "reg_alpha": 0.3,
    "reg_lambda": 1.5
}

# XGBoost 
model_xgb = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(
        **best_params,
        subsample=0.7, colsample_bytree=0.7,
        random_state=RANDOM_STATE, n_jobs=-1
    ))
])

# RandomForest
model_rf = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=300, max_depth=6,
        min_samples_split=10, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=-1
    ))
])

# GradientBoosting
model_gb = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=RANDOM_STATE
    ))
])

model_xgb.fit(X_train, y_train)
model_rf.fit(X_train, y_train)
model_gb.fit(X_train, y_train)

y_pred_xgb = model_xgb.predict(X_test)
y_pred_rf = model_rf.predict(X_test)
y_pred_gb = model_gb.predict(X_test)

m_xgb = evaluate(y_test, y_pred_xgb)
m_rf = evaluate(y_test, y_pred_rf)
m_gb = evaluate(y_test, y_pred_gb)

print(f"\nTest set:")
print(f"  XGBoost:        MAE={m_xgb['mae']:.4f}, R²={m_xgb['r2']:.4f}")
print(f"  RandomForest:   MAE={m_rf['mae']:.4f}, R²={m_rf['r2']:.4f}")
print(f"  GradientBoost:  MAE={m_gb['mae']:.4f}, R²={m_gb['r2']:.4f}")

y_pred_ensemble = (y_pred_xgb + y_pred_rf + y_pred_gb) / 3
m_ensemble = evaluate(y_test, y_pred_ensemble)
print(f"  Ensemble:       MAE={m_ensemble['mae']:.4f}, R²={m_ensemble['r2']:.4f}")

print("\n" + "=" * 70)
print("Target_data")
print("=" * 70)

pred_xgb = model_xgb.predict(X_target)
pred_rf = model_rf.predict(X_target)
pred_gb = model_gb.predict(X_target)

pred_ensemble = (pred_xgb + pred_rf + pred_gb) / 3

pred_std = np.std([pred_xgb, pred_rf, pred_gb], axis=0)

errors_xgb = np.abs(y_target_true - pred_xgb)
errors_rf = np.abs(y_target_true - pred_rf)
errors_gb = np.abs(y_target_true - pred_gb)
errors_ensemble = np.abs(y_target_true - pred_ensemble)

print(f"\nPrediction:")
print("-" * 90)
print(f"{'Sample':<35} {'True value':>8} {'XGB':>9} {'RF':>9} {'GB':>9} {'Ensemble':>9} {'Uncertainty':>9}")
print("-" * 90)

for i, name in enumerate(df_target['source_file']):
    short_name = name.split('/')[-1] if '/' in name else name
    print(f"{short_name[:33]:<35} {y_target_true.iloc[i]:>8.4f} {pred_xgb[i]:>9.4f} {pred_rf[i]:>9.4f} {pred_gb[i]:>9.4f} {pred_ensemble[i]:>9.4f} {pred_std[i]:>9.4f}")

print("-" * 90)

print(f"\nMAE:")
print(f"  XGBoost:        {errors_xgb.mean():.4f}")
print(f"  RandomForest:   {errors_rf.mean():.4f}")
print(f"  GradientBoost:  {errors_gb.mean():.4f}")
print(f"  Ensemble mean:       {errors_ensemble.mean():.4f}")

pred_result_df = pd.DataFrame({
    "sample": df_target['source_file'],
    "true_barrier": y_target_true,
    "pred_xgb": pred_xgb,
    "pred_rf": pred_rf,
    "pred_gb": pred_gb,
    "pred_ensemble": pred_ensemble,
    "uncertainty": pred_std,
    "error_xgb": errors_xgb,
    "error_rf": errors_rf,
    "error_gb": errors_gb,
    "error_ensemble": errors_ensemble
})

pred_result_df.to_csv(os.path.join(OUT_DIR, "target_predictions.csv"), index=False)
pred_result_df.to_excel(os.path.join(OUT_DIR, "target_predictions.xlsx"), index=False)

import joblib
joblib.dump(model_xgb, os.path.join(OUT_DIR, "model_xgb.joblib"))
joblib.dump(model_rf, os.path.join(OUT_DIR, "model_rf.joblib"))
joblib.dump(model_gb, os.path.join(OUT_DIR, "model_gb.joblib"))

result_summary = {
    "timestamp": datetime.now().isoformat(),
    "version": "v14_pure_predict",
    "test_set_performance": {
        "xgb": m_xgb,
        "rf": m_rf,
        "gb": m_gb,
        "ensemble": m_ensemble
    },
    "target_prediction_errors": {
        "xgb": errors_xgb.mean(),
        "rf": errors_rf.mean(),
        "gb": errors_gb.mean(),
        "ensemble": errors_ensemble.mean()
    },
    "predictions": pred_result_df.to_dict('records')
}

with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
    json.dump(result_summary, f, indent=2)

print("\n" + "=" * 70)
print("save to: " + OUT_DIR)
print("=" * 70)
