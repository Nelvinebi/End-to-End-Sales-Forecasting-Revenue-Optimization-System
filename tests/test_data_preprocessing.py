import pandas as pd
import pytest

from data_preprocessing import (
    TRAIN_COLUMNS,
    clean_data,
    merge_data,
    validate_schema,
)


def make_train():
    return pd.DataFrame(
        {
            "Store": [1, 1, 2, 2],
            "DayOfWeek": [5, 4, 3, 6],
            "Date": [
                "2020-01-03",
                "2020-01-02",
                "2020-01-01",
                "2020-01-04",
            ],
            "Sales": [100, 0, 200, 150],
            "Customers": [10, 0, 20, 15],
            "Open": [1, 1, 0, 1],
            "Promo": [0, 0, 1, 1],
            "StateHoliday": ["0", "0", "0", "0"],
            "SchoolHoliday": [0, 0, 0, 0],
        }
    )


def make_store():
    return pd.DataFrame(
        {
            "Store": [1, 2],
            "StoreType": ["a", "b"],
            "Assortment": ["a", "b"],
            "CompetitionDistance": [None, 50.0],
            "CompetitionOpenSinceMonth": [None, 1.0],
            "CompetitionOpenSinceYear": [None, 2019.0],
            "Promo2": [0, 1],
            "Promo2SinceWeek": [None, 1.0],
            "Promo2SinceYear": [None, 2019.0],
            "PromoInterval": [None, "Jan,Apr,Jul,Oct"],
        }
    )


def test_validate_schema_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        validate_schema(
            pd.DataFrame({"Store": [1]}),
            TRAIN_COLUMNS,
            "Training",
        )


def test_validate_schema_rejects_empty_dataframe():
    empty_train = pd.DataFrame(columns=sorted(TRAIN_COLUMNS))

    with pytest.raises(ValueError, match="dataset is empty"):
        validate_schema(empty_train, TRAIN_COLUMNS, "Training")


def test_merge_data_preserves_rows_and_adds_store_metadata():
    train = make_train()
    merged = merge_data(train, make_store())

    assert len(merged) == len(train)
    assert "_merge" not in merged.columns
    assert merged["StoreType"].tolist() == ["a", "a", "b", "b"]


def test_merge_data_rejects_duplicate_store_identifiers():
    duplicated_store = pd.concat(
        [make_store(), make_store().iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate Store identifiers"):
        merge_data(make_train(), duplicated_store)


def test_merge_data_rejects_unmatched_store_identifiers():
    train = make_train()
    train.loc[0, "Store"] = 999

    with pytest.raises(ValueError, match="no matching store metadata"):
        merge_data(train, make_store())


def test_clean_data_filters_sorts_and_imputes():
    merged = merge_data(make_train(), make_store())
    cleaned = clean_data(merged)

    assert cleaned["Sales"].tolist() == [100, 150]
    assert cleaned["Open"].eq(1).all()
    assert cleaned["Sales"].gt(0).all()
    assert cleaned["Date"].is_monotonic_increasing
    assert cleaned.isna().sum().sum() == 0
    assert cleaned["CompetitionDistance"].tolist() == [50.0, 50.0]
    assert cleaned.loc[0, "PromoInterval"] == "None"


def test_clean_data_rejects_invalid_dates():
    merged = merge_data(make_train(), make_store())
    merged.loc[0, "Date"] = "not-a-date"

    with pytest.raises(ValueError, match="invalid Date values"):
        clean_data(merged)


def test_clean_data_rejects_invalid_open_values():
    merged = merge_data(make_train(), make_store())
    merged.loc[0, "Open"] = 2

    with pytest.raises(ValueError, match="Open must contain only 0 or 1"):
        clean_data(merged)


def test_clean_data_rejects_dataset_without_eligible_rows():
    merged = merge_data(make_train(), make_store())
    merged["Open"] = 0

    with pytest.raises(ValueError, match="No model-eligible rows"):
        clean_data(merged)
