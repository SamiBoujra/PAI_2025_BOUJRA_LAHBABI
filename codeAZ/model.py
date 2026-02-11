import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
import numpy as np
import pandas as pd

import re
import usaddress

def parse_us_address(addr: str) -> dict:
    addr = addr.strip()
    # remove trailing country text if present
    addr = re.sub(r",\s*Ã‰tats-Unis\s*$", "", addr, flags=re.IGNORECASE)
    addr = re.sub(r",\s*United States\s*$", "", addr, flags=re.IGNORECASE)

    tagged, _ = usaddress.tag(addr)

    city = tagged.get("PlaceName", "")
    state = tagged.get("StateName", "")
    zip_code = tagged.get("ZipCode", "")

    return {
        "City": city,
        "State": state,
        "Zip Code": zip_code
    }

def predict_interval_from_constraints(
    pipe,
    df_train: pd.DataFrame,
    feature_cols: list[str],
    constraints: dict,
    n_samples: int = 1000,
    alpha: float = 0.20,   # 0.20 => 80% interval (10%-90%)
    use_log: bool = True,
    random_state: int = 42
):
    """
    Returns (p_low, p_med, p_high, n_used).
    constraints example: {"City":"Portland", "State":"Oregon", "Beds":4}
    """
    rng = np.random.default_rng(random_state)

    # 1) Filter training rows that match constraints (only for columns present)
    df_f = df_train.copy()
    for k, v in constraints.items():
        if k not in df_f.columns:
            continue
        if v is None or (isinstance(v, str) and v.strip() == ""):
            continue

        if isinstance(v, (int, float, np.integer, np.floating)):
            # numeric exact match by default; you can change to ranges if you want
            df_f = df_f[df_f[k].notna()]
            df_f = df_f[df_f[k] == v]
        else:
            df_f = df_f[df_f[k].astype(str) == str(v)]

    # If filtering got too strict, back off (use all training data)
    if len(df_f) < 50:
        df_f = df_train.copy()

    # 2) Sample plausible complete rows
    take = min(n_samples, len(df_f))
    sampled = df_f.sample(n=take, replace=(take < n_samples), random_state=random_state)

    # 3) Build X inputs; overwrite known constraints so they are fixed
    X_sim = sampled[feature_cols].copy()
    for k, v in constraints.items():
        if k in X_sim.columns and v is not None and not (isinstance(v, str) and v.strip() == ""):
            X_sim[k] = v

    # 4) Predict distribution
    preds = pipe.predict(X_sim)

    # 5) Undo log if needed
    if use_log:
        preds = np.expm1(preds)

    low = float(np.quantile(preds, alpha/2))
    med = float(np.quantile(preds, 0.5))
    high = float(np.quantile(preds, 1 - alpha/2))
    return low, med, high, len(df_f)


# 1) Charger
from pathlib import Path
DATA_PATH = Path(__file__).resolve().parent.parent / "American_Housing_Data_20231209.csv"
df = pd.read_csv(DATA_PATH)

# 2) Nettoyage minimal
df = df.copy()

# Nettoyer Zip Code si format "97 229"
if "Zip Code" in df.columns:
    df["Zip Code"] = (
        df["Zip Code"].astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace(r"\.0$", "", regex=True)
    )

# Convertir colonnes numÃ©riques possibles (adapte si besoin)
for col in ["Price", "Beds", "Baths", "Living Space", "Zip Code Population"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Garder lignes valides
df = df.dropna(subset=["Price", "Beds", "Baths", "Living Space", "City", "State"])

# 3) Features / Target
target = "Price"
feature_cols = ["Beds", "Baths", "Living Space", "City", "State"]

# Optionnel : ajouter Zip Code / population si prÃ©sents
if "Zip Code" in df.columns:
    feature_cols.append("Zip Code")
if "Zip Code Population" in df.columns:
    feature_cols.append("Zip Code Population")

X = df[feature_cols]
y = df[target]

# (Option trÃ¨s recommandÃ©e) prÃ©dire log(prix)
use_log = True
if use_log:
    y = np.log1p(y)

# 4) Colonnes numÃ©riques / catÃ©gorielles
num_cols = [c for c in feature_cols if c in ["Beds", "Baths", "Living Space", "Zip Code Population"]]
cat_cols = [c for c in feature_cols if c not in num_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# 5) ModÃ¨le XGBoost
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

# 6) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7) EntraÃ®ner
pipe.fit(X_train, y_train)

# 8) PrÃ©dire
pred = pipe.predict(X_test)

# Revenir en dollars si log
if use_log:
    y_test_d = np.expm1(y_test)
    pred_d = np.expm1(pred)
else:
    y_test_d = y_test
    pred_d = pred

# 9) Ã‰valuer
mae = mean_absolute_error(y_test_d, pred_d)
mse = mean_squared_error(y_test_d, pred_d)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_d, pred_d)

print(f"MAE  : {mae:,.0f} $")
print(f"RMSE : {rmse:,.0f} $")
print(f"RÂ²   : {r2:.3f}")

addr = "100 W 2nd St, Boston, MA 02127, Ã‰tats-Unis"
info = parse_us_address(addr)  # or parse_us_address(addr)

constraints = {
    "City": info["City"],
    "State": info["State"],
    "Zip Code": info["Zip Code"],

    # optional extra info (if the user provides later)
    # "Beds": 3,
    # "Baths": 2,
    # "Living Space": 1400,
}
print(constraints)
low, med, high, n_used = predict_interval_from_constraints(pipe, df, feature_cols, constraints)
print(low, med, high, "based on", n_used, "matching rows")

constraints = {"City": "Portland", "State": "Oregon", "Beds": 4, "Baths": 3}  # stronger -> narrower
low, med, high, n_used = predict_interval_from_constraints(pipe, df, feature_cols, constraints)
print(low, med, high, "based on", n_used, "matching rows")

import joblib

from pathlib import Path
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
