# ==============================
# IMPORTS
# ==============================

import pandas as pd
import numpy as np

# Sklearn utilities for ML pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# XGBoost regression model
from xgboost import XGBRegressor

# Address parsing tools
import re
import usaddress

# ==============================
# ADDRESS PARSING FUNCTION
# ==============================

def parse_us_address(addr: str) -> dict:
    """
    Parses a US address string and extracts:
    - City
    - State
    - Zip Code
    """

    addr = addr.strip()

    # Remove trailing country mention if present
    addr = re.sub(r",\s*États-Unis\s*$", "", addr, flags=re.IGNORECASE)
    addr = re.sub(r",\s*United States\s*$", "", addr, flags=re.IGNORECASE)

    # Use usaddress library to tag components
    tagged, _ = usaddress.tag(addr)

    city = tagged.get("PlaceName", "")
    state = tagged.get("StateName", "")
    zip_code = tagged.get("ZipCode", "")

    return {
        "City": city,
        "State": state,
        "Zip Code": zip_code
    }

# ==============================
# PREDICTION INTERVAL FUNCTION
# ==============================

def predict_interval_from_constraints(
    pipe,
    df_train: pd.DataFrame,
    feature_cols: list[str],
    constraints: dict,
    n_samples: int = 1000,
    alpha: float = 0.20,   # 80% confidence interval
    use_log: bool = True,
    random_state: int = 42
):
    """
    Returns:
        (low_price, median_price, high_price, n_rows_used)

    Idea:
    - Filter training data based on user constraints
    - Sample realistic rows
    - Predict a distribution
    - Compute quantiles
    """

    rng = np.random.default_rng(random_state)

    # 1️⃣ Filter rows matching constraints
    df_f = df_train.copy()

    for k, v in constraints.items():
        if k not in df_f.columns:
            continue

        if v is None or (isinstance(v, str) and v.strip() == ""):
            continue

        # Numeric exact match
        if isinstance(v, (int, float, np.integer, np.floating)):
            df_f = df_f[df_f[k].notna()]
            df_f = df_f[df_f[k] == v]
        else:
            df_f = df_f[df_f[k].astype(str) == str(v)]

    # If too few rows, fallback to full dataset
    if len(df_f) < 50:
        df_f = df_train.copy()

    # 2️⃣ Sample plausible houses
    take = min(n_samples, len(df_f))
    sampled = df_f.sample(
        n=take,
        replace=(take < n_samples),
        random_state=random_state
    )

    # 3️⃣ Build simulated input matrix
    X_sim = sampled[feature_cols].copy()

    # Force constraint values
    for k, v in constraints.items():
        if k in X_sim.columns and v is not None:
            X_sim[k] = v

    # 4️⃣ Predict distribution
    preds = pipe.predict(X_sim)

    # 5️⃣ Convert back from log if needed
    if use_log:
        preds = np.expm1(preds)

    # Compute interval
    low = float(np.quantile(preds, alpha/2))
    med = float(np.quantile(preds, 0.5))
    high = float(np.quantile(preds, 1 - alpha/2))

    return low, med, high, len(df_f)


# ==============================
# DATA LOADING
# ==============================

from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "American_Housing_Data_20231209.csv"

df = pd.read_csv(DATA_PATH)
df = df.copy()

# ==============================
# BASIC CLEANING
# ==============================

# Clean Zip Code formatting
if "Zip Code" in df.columns:
    df["Zip Code"] = (
        df["Zip Code"]
        .astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace(r"\.0$", "", regex=True)
    )

# Convert numeric columns
for col in ["Price", "Beds", "Baths", "Living Space", "Zip Code Population"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove incomplete rows
df = df.dropna(subset=["Price", "Beds", "Baths", "Living Space", "City", "State"])


# ==============================
# FEATURE ENGINEERING
# ==============================

target = "Price"

feature_cols = ["Beds", "Baths", "Living Space", "City", "State"]

# Optional features
if "Zip Code" in df.columns:
    feature_cols.append("Zip Code")
if "Zip Code Population" in df.columns:
    feature_cols.append("Zip Code Population")

X = df[feature_cols]
y = df[target]

# Use log transformation (recommended for prices)
use_log = True
if use_log:
    y = np.log1p(y)


# ==============================
# PREPROCESSING PIPELINE
# ==============================

num_cols = [c for c in feature_cols
            if c in ["Beds", "Baths", "Living Space", "Zip Code Population"]]

cat_cols = [c for c in feature_cols if c not in num_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)


# ==============================
# MODEL DEFINITION
# ==============================

model = XGBRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline(steps=[
    ("prep", preprocess),
    ("model", model)
])


# ==============================
# TRAIN / TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# ==============================
# TRAIN MODEL
# ==============================

pipe.fit(X_train, y_train)


# ==============================
# EVALUATION
# ==============================

pred = pipe.predict(X_test)

# Convert back from log scale
if use_log:
    y_test_d = np.expm1(y_test)
    pred_d = np.expm1(pred)
else:
    y_test_d = y_test
    pred_d = pred

mae = mean_absolute_error(y_test_d, pred_d)
rmse = np.sqrt(mean_squared_error(y_test_d, pred_d))
r2 = r2_score(y_test_d, pred_d)

print(f"MAE  : {mae:,.0f} $")
print(f"RMSE : {rmse:,.0f} $")
print(f"R²   : {r2:.3f}")


# ==============================
# EXAMPLE PREDICTIONS
# ==============================

# Parse address
addr = "100 W 2nd St, Boston, MA 02127, États-Unis"
info = parse_us_address(addr)

constraints = {
    "City": info["City"],
    "State": info["State"],
    "Zip Code": info["Zip Code"],
}

low, med, high, n_used = predict_interval_from_constraints(
    pipe, df, feature_cols, constraints
)

print(low, med, high, "based on", n_used, "matching rows")


# Stronger constraints -> narrower interval
constraints = {
    "City": "Portland",
    "State": "Oregon",
    "Beds": 4,
    "Baths": 3
}

low, med, high, n_used = predict_interval_from_constraints(
    pipe, df, feature_cols, constraints
)

print(low, med, high, "based on", n_used, "matching rows")


# ==============================
# SAVE MODEL
# ==============================

import joblib

MODEL_PATH = Path(__file__).resolve().parent.parent / "housing_pipe.joblib"

joblib.dump(
    {
        "pipe": pipe,
        "feature_cols": feature_cols,
        "use_log": use_log
    },
    MODEL_PATH
)

print("Saved model to:", MODEL_PATH)
