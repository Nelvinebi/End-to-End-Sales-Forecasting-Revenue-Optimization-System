from types import SimpleNamespace

import pandas as pd
import pytest

from train import sample_data, validate_splits


def make_splits():
    X_train = pd.DataFrame({"feature": range(5)})
    y_train = pd.Series(range(10, 15), name="Sales")
    X_test = pd.DataFrame({"feature": range(3)})
    y_test = pd.Series(range(20, 23), name="Sales")
    return X_train, X_test, y_train, y_test


def test_sample_data_is_size_safe_and_aligned():
    X_train, X_test, y_train, y_test = make_splits()
    config = SimpleNamespace(SAMPLE_SIZE_TRAIN=100, SAMPLE_SIZE_TEST=100, RANDOM_STATE=42)

    X_tr, X_te, y_tr, y_te = sample_data(X_train, y_train, X_test, y_test, config)

    assert len(X_tr) == len(X_train)
    assert len(X_te) == len(X_test)
    assert y_tr.index.equals(X_tr.index)
    assert y_te.index.equals(X_te.index)
    assert y_tr.tolist() == y_train.loc[X_tr.index].tolist()
    assert y_te.tolist() == y_test.loc[X_te.index].tolist()


def test_validate_splits_rejects_schema_mismatch():
    X_train, X_test, y_train, y_test = make_splits()
    X_test = X_test.rename(columns={"feature": "different_feature"})

    with pytest.raises(ValueError, match="schemas differ"):
        validate_splits(X_train, X_test, y_train, y_test)


def test_validate_splits_rejects_target_length_mismatch():
    X_train, X_test, y_train, y_test = make_splits()

    with pytest.raises(ValueError, match="Training features and target"):
        validate_splits(X_train, X_test, y_train.iloc[:-1], y_test)


def test_validate_splits_rejects_missing_values():
    X_train, X_test, y_train, y_test = make_splits()
    X_train.loc[0, "feature"] = None

    with pytest.raises(ValueError, match="must not contain missing values"):
        validate_splits(X_train, X_test, y_train, y_test)
