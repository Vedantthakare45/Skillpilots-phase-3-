import csv
import math

import numpy as np
import pytest

from regression_eval import (
    evaluate_regression,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

Y_TRUE = [3.0, -0.5, 2.0, 7.0]
Y_PRED = [2.5, 0.0, 2.0, 8.0]


def test_mae():
    assert mean_absolute_error(Y_TRUE, Y_PRED) == pytest.approx(0.5)


def test_mse():
    assert mean_squared_error(Y_TRUE, Y_PRED) == pytest.approx(0.375)


def test_rmse():
    assert root_mean_squared_error(Y_TRUE, Y_PRED) == pytest.approx(math.sqrt(0.375))


def test_r2():
    assert r2_score(Y_TRUE, Y_PRED) == pytest.approx(0.9486081370449679)


def test_perfect_prediction():
    r = evaluate_regression(Y_TRUE, Y_TRUE)
    assert r["MAE"] == 0 and r["MSE"] == 0 and r["RMSE"] == 0
    assert r["R2"] == pytest.approx(1.0)


def test_r2_constant_target():
    assert r2_score([2, 2, 2], [2, 2, 2]) == 1.0
    assert r2_score([2, 2, 2], [1, 2, 3]) == 0.0


def test_accepts_numpy_and_lists():
    a = evaluate_regression(np.array(Y_TRUE), np.array(Y_PRED))
    b = evaluate_regression(Y_TRUE, Y_PRED)
    assert a == b


def test_evaluate_returns_all_keys():
    assert set(evaluate_regression(Y_TRUE, Y_PRED)) == {"MAE", "MSE", "RMSE", "R2"}


def test_shape_mismatch():
    with pytest.raises(ValueError):
        evaluate_regression([1, 2, 3], [1, 2])


def test_empty_input():
    with pytest.raises(ValueError):
        evaluate_regression([], [])


def test_nan_input():
    with pytest.raises(ValueError):
        evaluate_regression([1, float("nan")], [1, 2])


def test_csv_logging(tmp_path):
    log = tmp_path / "logs" / "results.csv"
    evaluate_regression(Y_TRUE, Y_PRED, log_path=str(log), model_name="m1")
    evaluate_regression(Y_TRUE, Y_TRUE, log_path=str(log), model_name="m2")
    with open(log, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2  # header written only once
    assert rows[0]["model_name"] == "m1"
    assert float(rows[0]["MAE"]) == pytest.approx(0.5)
    assert rows[1]["model_name"] == "m2"
    assert float(rows[1]["R2"]) == pytest.approx(1.0)


def test_no_log_file_when_not_requested(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    evaluate_regression(Y_TRUE, Y_PRED)
    assert list(tmp_path.iterdir()) == []
