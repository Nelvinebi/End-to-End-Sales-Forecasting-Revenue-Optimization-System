"""
config.py
Central configuration for all pipeline stages.
"""

import os
from pathlib import Path


class Config:
    """Configuration management."""
    
    def __init__(self):
        # Base paths
        self.ROOT_DIR = Path(__file__).parent.parent
        self.DATA_DIR = self.ROOT_DIR / 'data'
        self.MODELS_DIR = self.ROOT_DIR / 'models'
        self.VIZ_DIR = self.ROOT_DIR / 'visualization'
        self.NOTEBOOKS_DIR = self.ROOT_DIR / 'notebooks'
        
        # Subdirectories
        self.RAW_DATA = self.DATA_DIR / 'raw'
        self.PROCESSED_DATA = self.DATA_DIR / 'processed'
        
        # Ensure directories exist
        for dir_path in [self.MODELS_DIR, self.VIZ_DIR, self.PROCESSED_DATA]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Data files
        self.RAW_TRAIN = self.RAW_DATA / 'train.csv'
        self.RAW_STORE = self.RAW_DATA / 'store.csv'
        self.CLEANED_DATA = self.PROCESSED_DATA / 'cleaned_data.csv'
        self.X_TRAIN = self.PROCESSED_DATA / 'X_train.csv'
        self.X_TEST = self.PROCESSED_DATA / 'X_test.csv'
        self.Y_TRAIN = self.PROCESSED_DATA / 'y_train.csv'
        self.Y_TEST = self.PROCESSED_DATA / 'y_test.csv'
        
        # Model files
        self.XGB_MODEL = self.MODELS_DIR / 'xgboost_sales_model.pkl'
        self.RF_MODEL = self.MODELS_DIR / 'random_forest_model.pkl'
        self.LR_MODEL = self.MODELS_DIR / 'linear_regression_model.pkl'
        
        # Training config
        self.SAMPLE_SIZE_TRAIN = 100000
        self.SAMPLE_SIZE_TEST = 50000
        self.SPLIT_DATE = '2015-01-01'
        self.RANDOM_STATE = 42
        
        # Model hyperparameters
        self.XGB_PARAMS = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.RANDOM_STATE,
            'n_jobs': -1,
            'objective': 'reg:squarederror'
        }
        
        self.RF_PARAMS = {
            'n_estimators': 50,
            'max_depth': 8,
            'min_samples_split': 100,
            'random_state': self.RANDOM_STATE,
            'n_jobs': -1
        }
    
    def check_raw_data(self):
        """Verify raw data exists."""
        if not self.RAW_TRAIN.exists():
            raise FileNotFoundError(f"Missing: {self.RAW_TRAIN}\nDownload from Kaggle first!")
        if not self.RAW_STORE.exists():
            raise FileNotFoundError(f"Missing: {self.RAW_STORE}\nDownload from Kaggle first!")
        print(f"✅ Raw data verified")