import json

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

import train
from cv_training import cross_validate_model, log_results_to_json


@pytest.fixture
def data():
    return make_regression(n_samples=100, n_features=4, noise=5.0, random_state=0)


def test_returns_expected_keys(data):
    X, y = data
    r = cross_validate_model(LinearRegression(), X, y, n_folds=5)
    assert set(r) == {"scoring", "n_folds", "scores", "mean", "std"}
    assert len(r["scores"]) == 5


def test_mean_and_std_match_scores(data):
    X, y = data
    r = cross_validate_model(LinearRegression(), X, y, n_folds=4)
    assert r["mean"] == pytest.approx(np.mean(r["scores"]))
    assert r["std"] == pytest.approx(np.std(r["scores"]))


def test_number_of_folds_respected(data):
    X, y = data
    assert len(cross_validate_model(LinearRegression(), X, y, n_folds=10)["scores"]) == 10


def test_reproducible(data):
    X, y = data
    a = cross_validate_model(LinearRegression(), X, y)
    b = cross_validate_model(LinearRegression(), X, y)
    assert a == b


def test_custom_scoring(data):
    X, y = data
    r = cross_validate_model(LinearRegression(), X, y, scoring="neg_mean_squared_error")
    assert r["mean"] < 0


def test_invalid_folds(data):
    X, y = data
    with pytest.raises(ValueError):
        cross_validate_model(LinearRegression(), X, y, n_folds=1)
    with pytest.raises(ValueError):
        cross_validate_model(LinearRegression(), X, y, n_folds=1000)


def test_json_logging_appends(tmp_path):
    log = tmp_path / "out" / "log.json"
    r = {"scoring": "r2", "n_folds": 3, "scores": [0.1, 0.2, 0.3], "mean": 0.2, "std": 0.08}
    log_results_to_json(r, str(log), model_name="a")
    log_results_to_json(r, str(log), model_name="b", extra={"data": "x"})
    runs = json.loads(log.read_text())
    assert [x["model"] for x in runs] == ["a", "b"]
    assert runs[1]["data"] == "x"
    assert runs[0]["mean"] == 0.2


def test_json_logging_rejects_corrupt_file(tmp_path):
    log = tmp_path / "bad.json"
    log.write_text("not json")
    with pytest.raises(ValueError):
        log_results_to_json({}, str(log))


def test_cli_end_to_end(tmp_path, capsys):
    log = tmp_path / "r.json"
    train.main(["--folds", "3", "--model", "ridge", "--output", str(log)])
    out = capsys.readouterr().out
    assert "Mean:" in out
    runs = json.loads(log.read_text())
    assert runs[0]["n_folds"] == 3 and runs[0]["model"] == "ridge"


def test_cli_rejects_bad_folds():
    with pytest.raises(SystemExit):
        train.parse_args(["--folds", "1"])


def test_cli_csv_input(tmp_path):
    import pandas as pd

    X, y = make_regression(n_samples=60, n_features=3, noise=1.0, random_state=1)
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["price"] = y
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    train.main(["--data", str(csv), "--target", "price", "--folds", "3", "--no-log"])
