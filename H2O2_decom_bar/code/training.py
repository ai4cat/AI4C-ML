# ========================= DATA TABLE MATCHING RULES =========================
# The current public H2O2 training workbook has exactly 23 columns:
#   source_file, Name, Barrier_eV, X1, X2, ..., X20
#
#   - Barrier_eV is the supervised regression target.
#   - X1 ... X20 are the only model descriptors.
#   - source_file and Name are metadata used for provenance and overlap checks.
#
# Do not infer features with a positional slice such as df.iloc[:, 2:22].
# Another workbook may insert or remove metadata columns while keeping the
# same descriptor names. Always inspect the header and update the explicit
# column mapping below before training. The expected public column count is
# checked so a changed file format fails loudly instead of silently treating
# Barrier_eV or metadata as a descriptor.
# ============================================================================

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

from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
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
# The paths above are machine-specific examples. Before running this script,
# confirm the actual local paths for the labeled workbook, the unlabeled target
# workbook, and the output directory, then edit them for the current machine.
# Target_data.xlsx is in-house data and is not included in the public repository.
# It can be provided by the authors on request through the paper correspondence
# email. Before training, confirm that this in-house table uses the same Name
# identity column as the public labeled table so overlap checks remain valid.
TARGET_COL = "Barrier_eV"
FEATURE_COLS = [f"X{i}" for i in range(1, 21)]
CAT_COLS = ["X2"]
NUM_COLS = [col for col in FEATURE_COLS if col not in CAT_COLS]
PUBLIC_COLUMN_COUNT = 23
RANDOM_STATE = 42
IDENTITY_COL = "Name"

# Pseudo labels are accepted only when the independent initial models agree,
# the candidate is complete, and the prediction stays in the observed target
# domain. These are quality gates, not a formal uncertainty estimate.
PSEUDO_LABEL_SEEDS = (RANDOM_STATE, RANDOM_STATE + 1, RANDOM_STATE + 2)
MIN_PSEUDO_LABEL_DISAGREEMENT_EV = 0.05

os.makedirs(OUT_DIR, exist_ok=True)


def evaluate(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred)
    }


def _normalize_category(value):
    """Use one categorical representation for training, prediction, and checks."""
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
    """Return exactly X1...X20 in the types expected by the pipelines."""
    missing = [col for col in FEATURE_COLS if col not in frame.columns]
    if missing:
        raise ValueError(
            "Missing descriptor columns: {}".format(", ".join(missing))
        )

    features = frame[FEATURE_COLS].copy()
    # Keep common spreadsheet missing-value markers consistent with predict.py.
    features = features.replace(["-", "--", "NA", "N/A", "na", ""], np.nan)
    for col in NUM_COLS:
        features[col] = pd.to_numeric(features[col], errors="coerce")
    features["X2"] = features["X2"].map(_normalize_category)
    return features


def id_series(frame):
    """Return the Name identity used for grouping and overlap checks."""
    if IDENTITY_COL not in frame.columns:
        return None
    return normalized_id_series(frame, IDENTITY_COL)


def normalized_id_series(frame, column):
    """Return one explicitly selected identity column in canonical text form."""
    if column not in frame.columns:
        raise ValueError(
            f"Required identity column '{column}' is missing from the table."
        )
    values = frame[column].astype("string").str.strip()
    return values.mask(values.eq(""), pd.NA)


def build_group_labels(frame):
    """Keep repeated Name records in the same split and CV fold."""
    sample_ids = id_series(frame)
    if sample_ids is None:
        # Without an identifier, each row is treated as its own sample.
        return pd.Series(range(len(frame)), index=frame.index)

    # Per the current data definition, only repeated Name values are linked.
    # Identical X1...X20 values with different Names are allowed and are not
    # treated as label conflicts or forced into the same group.
    fallback_ids = sample_ids.copy()
    missing_mask = fallback_ids.isna()
    fallback_ids.loc[missing_mask] = [
        f"<ROW_{index}>" for index in fallback_ids.index[missing_mask]
    ]
    return pd.Series(pd.factorize(fallback_ids)[0], index=frame.index)


def validate_training_table(frame, path):
    """Validate the current public 23-column labeled-table contract."""
    expected = ["source_file", "Name", TARGET_COL] + FEATURE_COLS
    missing = [col for col in expected if col not in frame.columns]
    unexpected = [col for col in frame.columns if col not in expected]
    print(f"{path}: {len(frame)} rows, {len(frame.columns)} columns")
    print(f"Columns: {list(frame.columns)}")
    if len(frame.columns) != PUBLIC_COLUMN_COUNT or missing or unexpected:
        raise ValueError(
            "Training table schema mismatch. Expected the public 23-column "
            "schema {} ; missing={}, unexpected={}, actual_count={}".format(
                expected, missing, unexpected, len(frame.columns)
            )
        )
    ids = normalized_id_series(frame, IDENTITY_COL)
    if ids.isna().any():
        raise ValueError(
            f"{path} contains blank or missing {IDENTITY_COL} values. "
            "Every labeled row needs an identity for grouped splitting and "
            "overlap checks."
        )


def validate_target_table(frame, path):
    """Validate an unlabeled target table and its overlap-check identity."""
    print(f"{path}: {len(frame)} rows, {len(frame.columns)} columns")
    print(f"Columns: {list(frame.columns)}")
    missing = [col for col in FEATURE_COLS if col not in frame.columns]
    if missing:
        raise ValueError(
            "Target table is missing descriptor columns: {}".format(
                ", ".join(missing)
            )
        )
    if IDENTITY_COL not in frame.columns:
        raise ValueError(
            f"{path} must contain the shared identity column '{IDENTITY_COL}'. "
            "Target_data.xlsx is in-house data, but its Name column is required "
            "to verify overlap with the public labeled table."
        )
    ids = normalized_id_series(frame, IDENTITY_COL)
    if ids.isna().any():
        raise ValueError(
            f"{path} contains blank or missing {IDENTITY_COL} values. "
            "The target table cannot be checked for overlap until every row "
            "has a valid identity."
        )
    if TARGET_COL in frame.columns:
        raise ValueError(
            "Target_data.xlsx contains Barrier_eV. The semi-supervised target "
            "table must be unlabeled to avoid using hidden test labels."
        )


def assert_no_overlap(left, right, left_name, right_name):
    """Reject overlap using one shared, non-empty identity column.

    This check must fail loudly when the two tables cannot be compared. It must
    never silently skip the check because one table omitted its identity field
    or used a different identity-column name.
    """
    if IDENTITY_COL not in left.columns or IDENTITY_COL not in right.columns:
        raise ValueError(
            f"{left_name} and {right_name} must both contain the shared "
            f"identity column '{IDENTITY_COL}' before overlap can be checked."
        )

    left_ids = normalized_id_series(left, IDENTITY_COL)
    right_ids = normalized_id_series(right, IDENTITY_COL)
    if left_ids.isna().any() or right_ids.isna().any():
        raise ValueError(
            f"Blank or missing {IDENTITY_COL} values prevent overlap checking "
            f"between {left_name} and {right_name}."
        )

    id_overlap = set(left_ids).intersection(set(right_ids))
    if id_overlap:
        raise ValueError(
            f"{IDENTITY_COL} overlap detected between {left_name} and "
            f"{right_name}: {len(id_overlap)} IDs. Do not use these rows as "
            "unlabeled data."
        )


def build_preprocessor():
    """Create a fresh preprocessor for every saved pipeline."""
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([
        ("num", num_transformer, NUM_COLS),
        ("cat", cat_transformer, CAT_COLS)
    ])


def build_xgb_pipeline(model_params, random_state):
    params = {k.replace("model__", ""): v for k, v in model_params.items()}
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", XGBRegressor(
            **params,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=random_state,
            n_jobs=-1
        ))
    ])

# Load data and validate the actual table schemas before selecting columns.
df = pd.read_excel(DATA_PATH)
df_target = pd.read_excel(TARGET_DATA_PATH)
validate_training_table(df, DATA_PATH)
validate_target_table(df_target, TARGET_DATA_PATH)

df_clean = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
X = normalize_features(df_clean[FEATURE_COLS])
y = df_clean[TARGET_COL].astype(float).copy()
X_target = normalize_features(df_target[FEATURE_COLS])

print(f"Training data: {len(X)} samples")
print(f"Target data: {len(df_target)} samples")

# The old row-wise train_test_split could place the same Name in both sets.
# This happens before GridSearchCV; it is a holdout-split problem, not
# something created by cross-validation. The grouped holdout below keeps
# repeated Names in one partition, and the subsequent CV is grouped too.
all_sample_groups = build_group_labels(df_clean)
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, test_idx = next(splitter.split(X, y, groups=all_sample_groups))
X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()
y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()
df_train = df_clean.iloc[train_idx].copy()
df_test = df_clean.iloc[test_idx].copy()
assert_no_overlap(df_train, df_test, "training", "test")

# The unlabeled target table must not contain an existing Name from the
# labeled data. Repeated descriptor values with different Names are allowed by
# the current dataset definition and are not treated as overlap here.
assert_no_overlap(df_clean, df_target, "labeled", "target")

target_ids = normalized_id_series(df_target, IDENTITY_COL)
if target_ids.duplicated(keep=False).any():
    raise ValueError(
        "Target_data.xlsx contains duplicate Name values. Resolve the "
        "candidate identity before generating pseudo labels."
    )

train_groups = all_sample_groups.iloc[train_idx].to_numpy()

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
    ("preprocessor", build_preprocessor()),
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
    # GroupKFold prevents repeated Name records from crossing CV folds.
    cv=GroupKFold(n_splits=3),
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train, y_train, groups=train_groups)

best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

y_pred_grid = grid_search.predict(X_test)
m_grid = evaluate(y_test, y_pred_grid)
print(f"Test MAE: {m_grid['mae']:.4f}, R²: {m_grid['r2']:.4f}")

# 2. Semi-supervised learning
print("\n" + "=" * 70)
print("Strategy 2: Semi-supervised learning (pseudo-labeling)")
print("=" * 70)

# Generate pseudo labels with several independently seeded initial models.
# The spread between them is used as a disagreement gate; it is not claimed
# as a formal epistemic uncertainty estimate. The CV MAE supplies the scale
# for the acceptable disagreement and target-range checks.
cv_mae = float(-grid_search.best_score_)
pseudo_label_models = []
pseudo_prediction_matrix = []
for pseudo_seed in PSEUDO_LABEL_SEEDS:
    pseudo_model = build_xgb_pipeline(best_params, pseudo_seed)
    pseudo_model.fit(X_train, y_train)
    pseudo_label_models.append(pseudo_model)
    pseudo_prediction_matrix.append(pseudo_model.predict(X_target))

pseudo_prediction_matrix = np.vstack(pseudo_prediction_matrix)
pseudo_labels = pseudo_prediction_matrix.mean(axis=0)
pseudo_disagreement = pseudo_prediction_matrix.std(axis=0)
disagreement_limit = max(
    MIN_PSEUDO_LABEL_DISAGREEMENT_EV,
    2.0 * cv_mae,
)
target_range_margin = max(MIN_PSEUDO_LABEL_DISAGREEMENT_EV, 2.0 * cv_mae)
target_complete = X_target.notna().all(axis=1).to_numpy()
target_in_range = (
    (pseudo_labels >= float(y_train.min()) - target_range_margin)
    & (pseudo_labels <= float(y_train.max()) + target_range_margin)
)
pseudo_quality_mask = (
    target_complete
    & (pseudo_disagreement <= disagreement_limit)
    & target_in_range
)

print(
    "Pseudo-label quality control: "
    f"accepted {int(pseudo_quality_mask.sum())}/{len(X_target)} rows; "
    f"disagreement_limit={disagreement_limit:.6f} eV, "
    f"cv_mae={cv_mae:.6f} eV"
)
if not pseudo_quality_mask.any():
    raise ValueError(
        "No target rows passed pseudo-label quality control. Do not train the "
        "semi-supervised model until the target data or quality thresholds "
        "have been reviewed."
    )

X_target_selected = X_target.loc[pseudo_quality_mask].reset_index(drop=True)
selected_pseudo_labels = pseudo_labels[pseudo_quality_mask]

X_semi = pd.concat([X_train, X_target_selected], ignore_index=True)
y_semi = pd.concat(
    [y_train.reset_index(drop=True), pd.Series(selected_pseudo_labels)],
    ignore_index=True
)

print(
    f"Semi-supervised training set: {len(X_train)} -> {len(X_semi)} "
    f"({len(X_target_selected)} accepted target rows)"
)

model_semi = build_xgb_pipeline(best_params, RANDOM_STATE)

model_semi.fit(X_semi, y_semi)
y_pred_semi = model_semi.predict(X_test)
m_semi = evaluate(y_test, y_pred_semi)
print(f"Test MAE: {m_semi['mae']:.4f}, R²: {m_semi['r2']:.4f}")

# 3. Ensemble models
print("\n" + "=" * 70)
print("Strategy 3: Ensemble models")
print("=" * 70)

xgb_model = build_xgb_pipeline(best_params, RANDOM_STATE)

rf_model = Pipeline([
    ("preprocessor", build_preprocessor()),
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
    ("preprocessor", build_preprocessor()),
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
    "feature_columns": FEATURE_COLS,
    "categorical_columns": CAT_COLS,
    "numerical_columns": NUM_COLS,
    "target_column": TARGET_COL,
    "data_checks": {
        "public_column_count": PUBLIC_COLUMN_COUNT,
        "holdout_split": "GroupShuffleSplit by repeated Name groups",
        "cross_validation": "GroupKFold(n_splits=3) by repeated Name groups",
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target_rows": int(len(X_target)),
        "accepted_pseudo_label_rows": int(pseudo_quality_mask.sum()),
        "pseudo_label_cv_mae_eV": cv_mae,
        "pseudo_label_disagreement_limit_eV": disagreement_limit,
    },
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
