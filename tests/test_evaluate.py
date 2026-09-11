import numpy as np
import pytest

from evaluate import calculate_metrics


def test_calculate_metrics_excludes_zero_targets_from_mape():
    metrics = calculate_metrics(
        np.array([0.0, 100.0, 200.0]),
        np.array([50.0, 90.0, 220.0]),
    )

    assert metrics["MAPE"] == pytest.approx(10.0)
    assert np.isfinite(metrics["RMSE"])
    assert np.isfinite(metrics["MAE"])
    assert np.isfinite(metrics["R2"])


def test_calculate_metrics_returns_nan_mape_when_all_targets_are_zero():
    metrics = calculate_metrics(
        np.array([0.0, 0.0]),
        np.array([10.0, 20.0]),
    )

    assert np.isnan(metrics["MAPE"])


def test_calculate_metrics_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="matching shapes"):
        calculate_metrics(
            np.array([100.0, 200.0]),
            np.array([100.0]),
        )
