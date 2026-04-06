"""
data_preprocessing.py
Load raw data, clean, and save processed datasets.
"""

import pandas as pd
import numpy as np


def load_raw_data(config):
    """Load train.csv and store.csv."""
    config.check_raw_data()
    
    print(f"Loading: {config.RAW_TRAIN.name}")
    train = pd.read_csv(config.RAW_TRAIN)
    
    print(f"Loading: {config.RAW_STORE.name}")
    store = pd.read_csv(config.RAW_STORE)
    
    print(f"   Train: {train.shape}")
    print(f"   Store: {store.shape}")
    
    return train, store


def merge_data(train, store):
    """Merge sales with store metadata."""
    df = train.merge(store, on='Store', how='left')
    print(f"✅ Merged: {df.shape}")
    return df


def clean_data(df):
    """Clean and prepare data."""
    # Remove closed stores
    df = df[df['Open'] == 1].copy()
    print(f"   After removing closed: {len(df):,} rows")
    
    # Convert date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Handle missing values
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(
        df['CompetitionDistance'].median()
    )
    df = df.fillna(0)
    
    return df


def save_processed_data(df, config):
    """Save cleaned data."""
    df.to_csv(config.CLEANED_DATA, index=False)
    print(f"✅ Saved: {config.CLEANED_DATA}")


def run_preprocessing(config):
    """Execute full preprocessing stage."""
    train, store = load_raw_data(config)
    df = merge_data(train, store)
    df = clean_data(df)
    save_processed_data(df, config)