"""Train, evaluate, and save the project models."""

import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _load_target(path):
    """Load a one-column target CSV as a Series."""
    target = pd.read_csv(path)
    if target.shape[1] != 1:
        raise ValueError(f"Target file must contain exactly one column: {path}")
    return target.iloc[:, 0]


def validate_splits(X_train, X_test, y_train, y_test):
    """Validate feature and target contracts before model fitting."""
    if X_train.empty or X_test.empty:
        raise ValueError("Training and test feature sets must not be empty.")
    if X_train.columns.has_duplicates or X_test.columns.has_duplicates:
        raise ValueError("Feature columns must be unique.")
    if not X_train.columns.equals(X_test.columns):
        missing = sorted(set(X_train.columns) - set(X_test.columns))
        unexpected = sorted(set(X_test.columns) - set(X_train.columns))
        raise ValueError(
            "Train/test feature schemas differ. "
            f"Missing from test: {missing}; unexpected in test: {unexpected}."
        )
    if len(X_train) != len(y_train):
        raise ValueError("Training features and target have different row counts.")
    if len(X_test) != len(y_test):
        raise ValueError("Test features and target have different row counts.")
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Feature sets must not contain missing values.")
    if y_train.isna().any() or y_test.isna().any():
        raise ValueError("Targets must not contain missing values.")


def load_data(config):
    """Load and validate processed train/test artifacts."""
    X_train = pd.read_csv(config.X_TRAIN, low_memory=False)
    X_test = pd.read_csv(config.X_TEST, low_memory=False)
    y_train = _load_target(config.Y_TRAIN)
    y_test = _load_target(config.Y_TEST)
    validate_splits(X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test


def _sample_aligned(X, y, requested_size, random_state):
    """Sample aligned feature and target rows without exceeding available data."""
    sample_size = min(requested_size, len(X))
    sampled_X = X.sample(n=sample_size, random_state=random_state)
    sampled_y = y.loc[sampled_X.index]
    return sampled_X, sampled_y


def sample_data(X_train, y_train, X_test, y_test, config):
    """Create deterministic, size-safe samples for quick training."""
    X_tr, y_tr = _sample_aligned(X_train, y_train, config.SAMPLE_SIZE_TRAIN, config.RANDOM_STATE)
    X_te, y_te = _sample_aligned(X_test, y_test, config.SAMPLE_SIZE_TEST, config.RANDOM_STATE)
    return X_tr, X_te, y_tr, y_te


def train_lr(X_train, y_train):
    """Train Linear Regression."""
    print("Training Linear Regression...")
    start = time.time()
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"   Time: {time.time() - start:.2f}s")
    return model


def train_rf(X_train, y_train, config):
    """Train Random Forest."""
    print("Training Random Forest...")
    start = time.time()
    model = RandomForestRegressor(**config.RF_PARAMS)
    model.fit(X_train, y_train)
    print(f"   Time: {time.time() - start:.2f}s")
    return model


def train_xgb(X_train, y_train, config):
    """Train XGBoost."""
    print("Training XGBoost...")
    start = time.time()
    model = xgb.XGBRegressor(**config.XGB_PARAMS)
    model.fit(X_train, y_train, verbose=False)
    print(f"   Time: {time.time() - start:.2f}s")
    return model


def evaluate(model, X_test, y_test, name):
    """Calculate model error metrics."""
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"   {name}: RMSE=€{rmse:,.0f}, MAE=€{mae:,.0f}")
    return rmse, mae, y_pred


def save_model(model, path):
    """Save a model to disk."""
    joblib.dump(model, path)
    size = path.stat().st_size / 1024 / 1024
    print(f"   Saved: {path.name} ({size:.2f} MB)")


def run_training(config, *, sample=False):
    """Execute the training stage, optionally using bounded samples."""
    X_train, X_test, y_train, y_test = load_data(config)

    if sample:
        X_train, X_test, y_train, y_test = sample_data(X_train, y_train, X_test, y_test, config)
        print(f"Sample mode: {len(X_train):,} train rows, {len(X_test):,} test rows")
    else:
        print(f"Full mode: {len(X_train):,} train rows, {len(X_test):,} test rows")

    results = {}

    lr = train_lr(X_train, y_train)
    rmse, _, _ = evaluate(lr, X_test, y_test, "Linear Regression")
    results["Linear Regression"] = rmse
    save_model(lr, config.LR_MODEL)

    rf = train_rf(X_train, y_train, config)
    rmse, _, _ = evaluate(rf, X_test, y_test, "Random Forest")
    results["Random Forest"] = rmse
    save_model(rf, config.RF_MODEL)

    xgb_model = train_xgb(X_train, y_train, config)
    rmse, _, _ = evaluate(xgb_model, X_test, y_test, "XGBoost")
    results["XGBoost"] = rmse
    save_model(xgb_model, config.XGB_MODEL)

    print("\n" + "=" * 50)
    print("RANKINGS (by RMSE)")
    print("=" * 50)
    for index, (name, rmse) in enumerate(sorted(results.items(), key=lambda item: item[1]), 1):
        print(f"{index}. {name}: €{rmse:,.0f}")

    return results
