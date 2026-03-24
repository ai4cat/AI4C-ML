"""
1. Hyperparameter optimization
2. Semi-supervised learning (pseudo-labeling)
3. Ensemble prediction with uncertainty estimation
"""

import os
import json
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

DATA_PATH = "INPUT_PATH"
TARGET_DATA_PATH = "TEST_DATA_PATH"
OUT_DIR = "OUT_PATH"

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

X_target = df_target[feature_cols].copy()

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

# 1. Hyperparameter optimization (XGBoost)
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

grid_search = GridSearchCV(
    base_pipeline,
    param_grid,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

y_pred_grid = grid_search.predict(X_test)
m_grid = evaluate(y_test, y_pred_grid)

# 2. Semi-supervised learning (pseudo-labeling)
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

X_semi = pd.concat([X_train, X_target], ignore_index=True)
y_semi = pd.concat(
    [y_train.reset_index(drop=True), pd.Series(pseudo_labels)],
    ignore_index=True
)

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

# 3. Ensemble models
xgb_model = model_initial

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

pred_xgb = xgb_model.predict(X_target)
pred_rf = rf_model.predict(X_target)
pred_gb = gb_model.predict(X_target)

pred_mean = (pred_xgb + pred_rf + pred_gb) / 3
pred_std = np.std([pred_xgb, pred_rf, pred_gb], axis=0)

# Save results
result_summary = {
    "timestamp": datetime.now().isoformat(),
    "best_params": best_params,
    "metrics": {
        "grid_mae": m_grid["mae"],
        "semi_mae": m_semi["mae"]
    },
    "predictions": {
        "ensemble": pred_mean.tolist(),
        "uncertainty": pred_std.tolist()
    }
}

with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
    json.dump(result_summary, f, indent=2)
