"""Rebuild the deployed LightGBM pipeline and report its performance.

Usage:
    python train_model.py            # evaluate only
    python train_model.py --save     # also overwrite melbourne_model.pkl

The pipeline matches the saved model: median imputation for numeric
features, most-frequent imputation + ordinal encoding for categorical
features, and LightGBM (500 trees) on log1p(Price).
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

DATA_PATH = "melb_data.csv"
FEATURES_PATH = "model_features.json"
MODEL_PATH = "melbourne_model.pkl"

NUMERIC = ["Distance", "Landsize", "Lattitude", "Longtitude",
           "Propertycount", "Bedroom2", "Bathroom", "Car"]
CATEGORICAL = ["Suburb", "Type", "Postcode", "CouncilArea"]


def make_pipeline():
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), CATEGORICAL),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LGBMRegressor(n_estimators=500, random_state=42, verbose=-1)),
    ])


def main():
    df = pd.read_csv(DATA_PATH)
    with open(FEATURES_PATH) as f:
        features = json.load(f)

    X = df[features].copy()
    X["Postcode"] = X["Postcode"].astype(object)
    y = np.log1p(df["Price"])  # model predicts log1p(price); expm1 converts back to AUD

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = make_pipeline().fit(X_train, y_train)

    pred = model.predict(X_test)
    true_aud, pred_aud = np.expm1(y_test), np.expm1(pred)
    mae = mean_absolute_error(true_aud, pred_aud)

    cv = cross_val_score(make_pipeline(), X, y, cv=KFold(5, shuffle=True, random_state=42), scoring="r2")

    print(f"Test R2 (log price):      {r2_score(y_test, pred):.3f}")
    print(f"Test R2 (AUD):            {r2_score(true_aud, pred_aud):.3f}")
    print(f"Test MAE (AUD):           {mae:,.0f}  ({mae / df['Price'].median():.1%} of median price)")
    print(f"5-fold CV R2 (log price): {cv.mean():.3f}")

    if "--save" in sys.argv:
        joblib.dump(model, MODEL_PATH)
        print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
