"""
train.py
Train and save all models.
"""

import pandas as pd
import numpy as np
import joblib
import time

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error


def load_data(config):
    """Load processed data."""
    X_train = pd.read_csv(config.X_TRAIN)
    X_test = pd.read_csv(config.X_TEST)
    y_train = pd.read_csv(config.Y_TRAIN).squeeze()
    y_test = pd.read_csv(config.Y_TEST).squeeze()
    return X_train, X_test, y_train, y_test


def sample_data(X_train, y_train, X_test, y_test, config):
    """Create samples for fast training."""
    X_tr = X_train.sample(n=config.SAMPLE_SIZE_TRAIN, random_state=config.RANDOM_STATE)
    y_tr = y_train.loc[X_tr.index]
    
    X_te = X_test.sample(n=config.SAMPLE_SIZE_TEST, random_state=config.RANDOM_STATE)
    y_te = y_test.loc[X_te.index]
    
    return X_tr, X_te, y_tr, y_te


def train_lr(X_train, y_train):
    """Train Linear Regression."""
    print("Training Linear Regression...")
    start = time.time()
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"   Time: {time.time()-start:.2f}s")
    return model


def train_rf(X_train, y_train, config):
    """Train Random Forest."""
    print("Training Random Forest...")
    start = time.time()
    model = RandomForestRegressor(**config.RF_PARAMS)
    model.fit(X_train, y_train)
    print(f"   Time: {time.time()-start:.2f}s")
    return model


def train_xgb(X_train, y_train, config):
    """Train XGBoost."""
    print("Training XGBoost...")
    start = time.time()
    model = xgb.XGBRegressor(**config.XGB_PARAMS)
    model.fit(X_train, y_train, verbose=False)
    print(f"   Time: {time.time()-start:.2f}s")
    return model


def evaluate(model, X_test, y_test, name):
    """Calculate metrics."""
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"   {name}: RMSE=€{rmse:,.0f}, MAE=€{mae:,.0f}")
    return rmse, mae, y_pred


def save_model(model, path):
    """Save model to disk."""
    joblib.dump(model, path)
    size = path.stat().st_size / 1024 / 1024
    print(f"   Saved: {path.name} ({size:.2f} MB)")


def run_training(config):
    """Execute full training stage."""
    X_train, X_test, y_train, y_test = load_data(config)
    
    # Sample for speed
    X_tr, X_te, y_tr, y_te = sample_data(X_train, y_train, X_test, y_test, config)
    
    # Train all models
    results = {}
    
    # Linear Regression
    lr = train_lr(X_tr, y_tr)
    rmse, mae, _ = evaluate(lr, X_te, y_te, "Linear Regression")
    results['Linear Regression'] = rmse
    save_model(lr, config.LR_MODEL)
    
    # Random Forest
    rf = train_rf(X_tr, y_tr, config)
    rmse, mae, _ = evaluate(rf, X_te, y_te, "Random Forest")
    results['Random Forest'] = rmse
    save_model(rf, config.RF_MODEL)
    
    # XGBoost
    xgb_model = train_xgb(X_tr, y_tr, config)
    rmse, mae, y_pred = evaluate(xgb_model, X_te, y_te, "XGBoost")
    results['XGBoost'] = rmse
    save_model(xgb_model, config.XGB_MODEL)
    
    # Rank results
    print("\n" + "="*50)
    print("RANKINGS (by RMSE)")
    print("="*50)
    for i, (name, rmse) in enumerate(sorted(results.items(), key=lambda x: x[1]), 1):
        print(f"{i}. {name}: €{rmse:,.0f}")