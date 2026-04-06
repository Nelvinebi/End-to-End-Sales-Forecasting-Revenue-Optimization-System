"""
feature_engineering.py
Create time-based features, encode categoricals, and split train/test.
"""

import pandas as pd
import numpy as np


def create_time_features(df):
    """
    Extract date components for seasonality modeling.
    
    Creates: Year, Month, Day, WeekOfYear, DayOfWeek, IsWeekend, IsPromo
    """
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    df['IsPromo'] = df['Promo']
    
    return df


def encode_categoricals(df):
    """
    One-hot encode categorical variables for ML models.
    
    Encodes: StoreType, Assortment, StateHoliday
    """
    categorical_cols = ['StoreType', 'Assortment', 'StateHoliday']
    
    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True  # Avoid dummy variable trap
    )
    
    # Convert boolean columns to integers (required for XGBoost)
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    
    return df


def remove_leakage_columns(df):
    """
    Remove columns not available at prediction time.
    
    Removes: Customers (unknown future), Open (always 1), PromoInterval (text)
    """
    leakage_cols = ['Customers', 'Open', 'PromoInterval']
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])
    
    return df


def time_based_split(df, config):
    """
    Split data chronologically to prevent data leakage.
    
    Train: Before 2015-01-01
    Test:  2015-01-01 and after
    """
    split_date = config.SPLIT_DATE
    
    train_df = df[df['Date'] < split_date].copy()
    test_df = df[df['Date'] >= split_date].copy()
    
    print(f"Train period: {train_df['Date'].min()} to {train_df['Date'].max()}")
    print(f"Test period:  {test_df['Date'].min()} to {test_df['Date'].max()}")
    print(f"   Train rows: {len(train_df):,}")
    print(f"   Test rows:  {len(test_df):,}")
    
    return train_df, test_df


def prepare_xy(train_df, test_df):
    """
    Create feature matrices (X) and target vectors (y).
    
    Drops: Sales (target), Date (extracted features already)
    """
    drop_cols = ['Sales', 'Date']
    
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df['Sales']
    
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df['Sales']
    
    return X_train, X_test, y_train, y_test


def save_splits(X_train, X_test, y_train, y_test, config):
    """Save train/test splits to disk for training stage."""
    X_train.to_csv(config.X_TRAIN, index=False)
    X_test.to_csv(config.X_TEST, index=False)
    y_train.to_csv(config.Y_TRAIN, index=False)
    y_test.to_csv(config.Y_TEST, index=False)
    
    print(f"\n✅ Saved train/test splits:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test:  {X_test.shape}")
    print(f"   Features: {len(X_train.columns)}")


def run_feature_engineering(config):
    """
    Execute full feature engineering pipeline.
    
    Called by main.py with config object.
    """
    print(f"\nLoading: {config.CLEANED_DATA.name}")
    df = pd.read_csv(config.CLEANED_DATA)
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"   Initial shape: {df.shape}")
    
    # Apply transformations
    df = create_time_features(df)
    df = encode_categoricals(df)
    df = remove_leakage_columns(df)
    
    print(f"   After engineering: {df.shape}")
    
    # Split and save
    train_df, test_df = time_based_split(df, config)
    X_train, X_test, y_train, y_test = prepare_xy(train_df, test_df)
    save_splits(X_train, X_test, y_train, y_test, config)
    
    print("\n✅ Feature engineering complete")