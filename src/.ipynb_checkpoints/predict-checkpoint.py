"""
predict.py
Make predictions on new data.
"""

import pandas as pd
import numpy as np
import joblib

from config import Config


def create_sample_input():
    """Create example input."""
    return {
        'Store': 1,
        'DayOfWeek': 5,
        'Promo': 1,
        'SchoolHoliday': 0,
        'CompetitionDistance': 500,
        'CompetitionOpenSinceMonth': 6,
        'CompetitionOpenSinceYear': 2010,
        'Promo2': 0,
        'Promo2SinceWeek': 0,
        'Promo2SinceYear': 0,
        'Year': 2024,
        'Month': 12,
        'Day': 20,
        'WeekOfYear': 51,
        'IsWeekend': 1,
        'IsPromo': 1,
        'StoreType_b': 0,
        'StoreType_c': 0,
        'StoreType_d': 0,
        'Assortment_b': 0,
        'Assortment_c': 0,
        'StateHoliday_0': 1,
        'StateHoliday_a': 0,
        'StateHoliday_b': 0,
        'StateHoliday_c': 0
    }


def prepare_input(data_dict):
    """Convert dict to DataFrame."""
    required = [
        'Store', 'DayOfWeek', 'Promo', 'SchoolHoliday',
        'CompetitionDistance', 'CompetitionOpenSinceMonth',
        'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
        'Promo2SinceYear', 'Year', 'Month', 'Day', 'WeekOfYear',
        'IsWeekend', 'IsPromo', 'StoreType_b', 'StoreType_c',
        'StoreType_d', 'Assortment_b', 'Assortment_c',
        'StateHoliday_0', 'StateHoliday_a', 'StateHoliday_b', 'StateHoliday_c'
    ]
    
    df = pd.DataFrame([data_dict])
    
    # Add missing with 0
    for col in required:
        if col not in df.columns:
            df[col] = 0
    
    return df[required]


def run_prediction(config=None):
    """Execute prediction stage."""
    if config is None:
        config = Config()
    
    print("="*50)
    print("PREDICTION")
    print("="*50)
    
    # Load model
    model = joblib.load(config.XGB_MODEL)
    print(f"✅ Loaded: {config.XGB_MODEL.name}")
    
    # Create input
    data = create_sample_input()
    input_df = prepare_input(data)
    
    # Predict
    pred = model.predict(input_df)[0]
    
    print(f"\n📊 Input: Store {data['Store']}, "
          f"{['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][data['DayOfWeek']]}, "
          f"Promo={'Yes' if data['Promo'] else 'No'}")
    print(f"\n🎯 Predicted Sales: €{pred:,.2f}")
    
    return pred


if __name__ == "__main__":
    run_prediction()