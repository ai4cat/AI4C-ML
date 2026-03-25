"""
Includes:
1. Hyperparameter optimization
2. Semi-supervised learning (pseudo-labeling)
3. Ensemble model training
4. Saving full pipelines (preprocessing + model)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

DATA_PATH = r"D:\H2O2_predict\data\H2O2_merged_all.xlsx"
TARGET_DATA_PATH = r"D:\H2O2_predict\data\Target_data.xlsx"
OUT_DIR = r"D:\H2O2_predict\data\finalresult"
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

# Load data
df = pd.read_excel(DATA_PATH)
df_target = pd.read_excel(TARGET_DATA_PATH)

feature_cols = [c for c in df.columns if str(c).startswith("X")]
num_cols = [c for c in feature_cols if c not in CAT_COLS]

df_clean = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
X = df_clean[feature_cols].copy()
y = df_clean[TARGET_COL].astype(float).copy()

print(f"Training data: {len(X)} samples")
print(f"Target data: {len(df_target)} samples")

# Preprocessing
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

# 1. Hyperparameter optimization
print("\n" + "=" * 70)
print("Strategy 1: Hyperparameter grid search")
print("=" * 70)

param_grid = {
    "model__n_estimators": [200, 300],
    "model__max_depth": [2, 3, 4],
    "model__learning_rate": [0.02, 0.03],
    "model__reg_alpha": [0.1, 0.3],
    "model__reg_lambda": [1.0, 1.5]
}

base_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

print("Running grid search...")
grid_search = GridSearchCV(
    base_pipeline,
    param_grid,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

y_pred_grid = grid_search.predict(X_test)
m_grid = evaluate(y_test, y_pred_grid)
print(f"Test MAE: {m_grid['mae']:.4f}, R²: {m_grid['r2']:.4f}")

# 2. Semi-supervised learning
print("\n" + "=" * 70)
print("Strategy 2: Semi-supervised learning (pseudo-labeling)")
print("=" * 70)

X_target = df_target[feature_cols].copy()

model_initial = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(
        **{k.replace("model__", ""): v for k, v in best_params.items()},
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

model_initial.fit(X_train, y_train)
pseudo_labels = model_initial.predict(X_target)

print(f"Pseudo-label predictions: {pseudo_labels}")

X_semi = pd.concat([X_train, X_target], ignore_index=True)
y_semi = pd.concat(
    [y_train.reset_index(drop=True), pd.Series(pseudo_labels)],
    ignore_index=True
)

print(f"Semi-supervised training set: {len(X_train)} -> {len(X_semi)}")

model_semi = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(
        **{k.replace("model__", ""): v for k, v in best_params.items()},
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

model_semi.fit(X_semi, y_semi)
y_pred_semi = model_semi.predict(X_test)
m_semi = evaluate(y_test, y_pred_semi)
print(f"Test MAE: {m_semi['mae']:.4f}, R²: {m_semi['r2']:.4f}")

# 3. Ensemble models
print("\n" + "=" * 70)
print("Strategy 3: Ensemble models")
print("=" * 70)

xgb_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(
        **{k.replace("model__", ""): v for k, v in best_params.items()},
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

gb_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=RANDOM_STATE
    ))
])

xgb_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)

m_xgb = evaluate(y_test, y_pred_xgb)
m_rf = evaluate(y_test, y_pred_rf)
m_gb = evaluate(y_test, y_pred_gb)

print(f"XGBoost: MAE={m_xgb['mae']:.4f}")
print(f"RandomForest: MAE={m_rf['mae']:.4f}")
print(f"GradientBoosting: MAE={m_gb['mae']:.4f}")

y_pred_ensemble = (y_pred_xgb + y_pred_rf + y_pred_gb) / 3
m_ensemble = evaluate(y_test, y_pred_ensemble)
print(f"Ensemble average: MAE={m_ensemble['mae']:.4f}, R²={m_ensemble['r2']:.4f}")

# =========================
# Save results summary
# =========================
result_summary = {
    "timestamp": datetime.now().isoformat(),
    "version": "version_final",
    "feature_columns": feature_cols,
    "categorical_columns": CAT_COLS,
    "numerical_columns": num_cols,
    "target_column": TARGET_COL,
    "hyperparameter_search": {
        "best_params": best_params,
        "test_r2": m_grid["r2"],
        "test_rmse": m_grid["rmse"],
        "test_mae": m_grid["mae"]
    },
    "semi_supervised": {
        "test_r2": m_semi["r2"],
        "test_rmse": m_semi["rmse"],
        "test_mae": m_semi["mae"]
    },
    "ensemble": {
        "xgb_r2": m_xgb["r2"],
        "xgb_rmse": m_xgb["rmse"],
        "xgb_mae": m_xgb["mae"],
        "rf_r2": m_rf["r2"],
        "rf_rmse": m_rf["rmse"],
        "rf_mae": m_rf["mae"],
        "gb_r2": m_gb["r2"],
        "gb_rmse": m_gb["rmse"],
        "gb_mae": m_gb["mae"],
        "ensemble_r2": m_ensemble["r2"],
        "ensemble_rmse": m_ensemble["rmse"],
        "ensemble_mae": m_ensemble["mae"]
    }
}

with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
    json.dump(result_summary, f, indent=2, ensure_ascii=False)

# Save full pipelines
joblib.dump(grid_search.best_estimator_, os.path.join(OUT_DIR, "best_xgb_gridsearc.joblib"))
joblib.dump(model_semi, os.path.join(OUT_DIR, "xgb_semi_supervised.joblib"))
joblib.dump(xgb_model, os.path.join(OUT_DIR, "xgb.joblib"))
joblib.dump(rf_model, os.path.join(OUT_DIR, "rf.joblib"))
joblib.dump(gb_model, os.path.join(OUT_DIR, "gb.joblib"))

print("\n" + "=" * 70)
print("All results and full pipelines have been saved to:")
print(OUT_DIR)
print("Saved pipeline files (including preprocessing + model):")
print("  - best_xgb_gridsearch.joblib")
print("  - xgb_semi_supervisede.joblib")
print("  - xgb.joblib")
print("  - rf.joblib")
print("  - gb.joblib")
print("=" * 70)
