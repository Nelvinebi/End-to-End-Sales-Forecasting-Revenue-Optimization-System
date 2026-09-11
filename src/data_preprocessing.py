"""
Load raw Rossmann data, validate its contract, clean it, and save it.
"""

import pandas as pd

TRAIN_COLUMNS = {
    "Store",
    "DayOfWeek",
    "Date",
    "Sales",
    "Customers",
    "Open",
    "Promo",
    "StateHoliday",
    "SchoolHoliday",
}

STORE_COLUMNS = {
    "Store",
    "StoreType",
    "Assortment",
    "CompetitionDistance",
    "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear",
    "Promo2",
    "Promo2SinceWeek",
    "Promo2SinceYear",
    "PromoInterval",
}

NULLABLE_STORE_NUMERIC_COLUMNS = [
    "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear",
    "Promo2SinceWeek",
    "Promo2SinceYear",
]


def validate_schema(df, required_columns, dataset_name):
    """Validate required columns and basic dataframe integrity."""
    if df.empty:
        raise ValueError(f"{dataset_name} dataset is empty.")

    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValueError(f"{dataset_name} contains duplicate columns: {duplicate_columns}")

    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"{dataset_name} is missing required columns: {missing_columns}")


def load_raw_data(config):
    """Load and validate train.csv and store.csv."""
    config.check_raw_data()

    print(f"Loading: {config.RAW_TRAIN.name}")
    train = pd.read_csv(config.RAW_TRAIN, low_memory=False)

    print(f"Loading: {config.RAW_STORE.name}")
    store = pd.read_csv(config.RAW_STORE, low_memory=False)

    validate_schema(train, TRAIN_COLUMNS, "Training")
    validate_schema(store, STORE_COLUMNS, "Store")

    if train[list(TRAIN_COLUMNS)].isna().any().any():
        raise ValueError("Training data contains nulls in required columns.")

    required_store_values = ["Store", "StoreType", "Assortment", "Promo2"]
    if store[required_store_values].isna().any().any():
        raise ValueError("Store data contains nulls in required identifier fields.")

    print(f"   Train: {train.shape}")
    print(f"   Store: {store.shape}")

    return train, store


def merge_data(train, store):
    """Merge sales with store metadata using a many-to-one contract."""
    validate_schema(train, TRAIN_COLUMNS, "Training")
    validate_schema(store, STORE_COLUMNS, "Store")

    if train["Store"].isna().any():
        raise ValueError("Training data contains null Store identifiers.")

    if store["Store"].isna().any():
        raise ValueError("Store data contains null Store identifiers.")

    duplicate_stores = store.loc[store["Store"].duplicated(), "Store"].unique()
    if len(duplicate_stores):
        raise ValueError(
            f"Store data contains duplicate Store identifiers: {duplicate_stores[:10].tolist()}"
        )

    merged = train.merge(
        store,
        on="Store",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    unmatched_stores = merged.loc[merged["_merge"] != "both", "Store"].drop_duplicates().tolist()
    if unmatched_stores:
        raise ValueError(
            "Training rows have no matching store metadata for Store values: "
            f"{unmatched_stores[:10]}"
        )

    merged = merged.drop(columns="_merge")
    print(f"Merged: {merged.shape}")
    return merged


def clean_data(df):
    """Validate values, retain model-eligible rows, and impute metadata."""
    validate_schema(df, TRAIN_COLUMNS | STORE_COLUMNS, "Merged")

    parsed_dates = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    invalid_date_count = int(parsed_dates.isna().sum())
    if invalid_date_count:
        raise ValueError(f"Merged data contains {invalid_date_count:,} invalid Date values.")

    open_values = set(df["Open"].dropna().unique())
    invalid_open_values = sorted(open_values - {0, 1})
    if invalid_open_values:
        raise ValueError(f"Open must contain only 0 or 1; found: {invalid_open_values}")

    sales = pd.to_numeric(df["Sales"], errors="coerce")
    invalid_sales_count = int(sales.isna().sum())
    if invalid_sales_count:
        raise ValueError(f"Merged data contains {invalid_sales_count:,} invalid Sales values.")

    df = df.copy()
    df["Date"] = parsed_dates
    df["Sales"] = sales

    initial_rows = len(df)
    df = df.loc[(df["Open"] == 1) & (df["Sales"] > 0)].copy()
    removed_rows = initial_rows - len(df)
    print(
        f"   Retained open stores with positive sales: {len(df):,} rows ({removed_rows:,} removed)"
    )

    if df.empty:
        raise ValueError("No model-eligible rows remain after cleaning.")

    competition_median = df["CompetitionDistance"].median()
    if pd.isna(competition_median):
        competition_median = 0
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(competition_median)

    df[NULLABLE_STORE_NUMERIC_COLUMNS] = df[NULLABLE_STORE_NUMERIC_COLUMNS].fillna(0)
    df["PromoInterval"] = df["PromoInterval"].fillna("None")

    remaining_null_columns = df.columns[df.isna().any()].tolist()
    if remaining_null_columns:
        raise ValueError(f"Unhandled null values remain in columns: {remaining_null_columns}")

    return df.sort_values("Date").reset_index(drop=True)


def save_processed_data(df, config):
    """Save cleaned data."""
    df.to_csv(config.CLEANED_DATA, index=False)
    print(f"Saved: {config.CLEANED_DATA}")


def run_preprocessing(config):
    """Execute the full preprocessing stage."""
    train, store = load_raw_data(config)
    df = merge_data(train, store)
    df = clean_data(df)
    save_processed_data(df, config)
