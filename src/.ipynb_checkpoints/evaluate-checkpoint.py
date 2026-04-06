"""
evaluate.py
Evaluate models and generate visualizations.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_model_and_data(config):
    """Load best model and test data."""
    model = joblib.load(config.XGB_MODEL)
    X_test = pd.read_csv(config.X_TEST)
    y_test = pd.read_csv(config.Y_TEST).squeeze()
    return model, X_test, y_test


def calculate_metrics(y_true, y_pred):
    """Calculate all metrics."""
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }


def plot_predictions(y_true, y_pred, config):
    """Create prediction scatter plot."""
    n = min(5000, len(y_true))
    y_t = y_true.iloc[:n] if hasattr(y_true, 'iloc') else y_true[:n]
    y_p = y_pred[:n]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(y_t, y_p, alpha=0.5, s=20, color='#3498db')
    
    max_val = max(y_t.max(), y_p.max())
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect')
    
    ax.set_xlabel('Actual Sales (€)')
    ax.set_ylabel('Predicted Sales (€)')
    ax.set_title('XGBoost: Predicted vs Actual')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    path = config.VIZ_DIR / 'prediction_vs_actual.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {path.name}")


def plot_residuals(y_true, y_pred, config):
    """Create residual plots."""
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram
    ax1.hist(residuals, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--')
    ax1.set_title('Residual Distribution')
    ax1.set_xlabel('Error (€)')
    
    # Scatter
    ax2.scatter(y_pred, residuals, alpha=0.5, s=20, color='#3498db')
    ax2.axhline(y=0, color='red', linestyle='--')
    ax2.set_title('Residuals vs Predicted')
    ax2.set_xlabel('Predicted (€)')
    ax2.set_ylabel('Residual (€)')
    
    path = config.VIZ_DIR / 'residuals.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {path.name}")


def run_evaluation(config):
    """Execute full evaluation stage."""
    model, X_test, y_test = load_model_and_data(config)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    print("\n" + "="*50)
    print("XGBOOST METRICS")
    print("="*50)
    for name, val in metrics.items():
        if name in ['RMSE', 'MAE']:
            print(f"{name}: €{val:,.2f}")
        elif name == 'MAPE':
            print(f"{name}: {val:.2f}%")
        else:
            print(f"{name}: {val:.4f}")
    
    # Visualizations
    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50)
    plot_predictions(y_test, y_pred, config)
    plot_residuals(y_test, y_pred, config)