import json

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVC

import tune
from tuning import load_model, tune_estimator


@pytest.fixture
def reg_data():
    return make_regression(n_samples=120, n_features=5, noise=5.0, random_state=0)


def test_returns_best_estimator_score_params(reg_data):
    X, y = reg_data
    best, score, params = tune_estimator(Ridge(), {"alpha": [0.01, 1.0, 100.0]}, X, y, n_jobs=1)
    assert isinstance(best, Ridge)
    assert isinstance(score, float)
    assert params["alpha"] in [0.01, 1.0, 100.0]
    assert best.alpha == params["alpha"]


def test_best_estimator_is_fitted(reg_data):
    X, y = reg_data
    best, _, _ = tune_estimator(Ridge(), {"alpha": [0.1, 1.0]}, X, y, n_jobs=1)
    assert best.predict(X).shape == y.shape


def test_uses_five_fold_cv_by_default(reg_data, monkeypatch):
    import inspect

    import tuning.tuner as tuner_module

    assert inspect.signature(tune_estimator).parameters["cv"].default == 5

    captured = {}
    real = tuner_module.GridSearchCV

    def factory(*args, **kwargs):
        captured.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(tuner_module, "GridSearchCV", factory)
    X, y = reg_data
    tune_estimator(Ridge(), {"alpha": [1.0]}, X, y, n_jobs=1)
    assert captured["cv"] == 5


def test_picks_best_alpha_deterministically():
    # Noise-free linear data: the smallest regularisation must win.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = X @ np.array([1.0, 2.0, 3.0])
    _, _, params = tune_estimator(Ridge(), {"alpha": [0.001, 1000.0]}, X, y, n_jobs=1)
    assert params["alpha"] == 0.001


def test_classifier_with_accuracy():
    X, y = make_classification(n_samples=150, n_features=6, random_state=0)
    _, score, _ = tune_estimator(SVC(), {"C": [0.1, 1, 10]}, X, y, scoring="accuracy", n_jobs=1)
    assert 0.5 < score <= 1.0


def test_model_saved_and_loadable(reg_data, tmp_path):
    X, y = reg_data
    path = tmp_path / "m" / "model.joblib"
    best, _, _ = tune_estimator(
        Ridge(), {"alpha": [0.1, 1.0]}, X, y, n_jobs=1, save_path=str(path)
    )
    assert path.exists()
    loaded = load_model(str(path))
    np.testing.assert_allclose(loaded.predict(X), best.predict(X))


def test_empty_grid_rejected(reg_data):
    X, y = reg_data
    with pytest.raises(ValueError):
        tune_estimator(Ridge(), {}, X, y)


def test_non_list_values_rejected(reg_data):
    X, y = reg_data
    with pytest.raises(ValueError):
        tune_estimator(Ridge(), {"alpha": 1.0}, X, y)
    with pytest.raises(ValueError):
        tune_estimator(Ridge(), {"alpha": []}, X, y)


# ---- CLI ----

def test_resolve_short_name_and_dotted_path():
    assert isinstance(tune.resolve_estimator("ridge"), Ridge)
    assert isinstance(tune.resolve_estimator("sklearn.ensemble.RandomForestRegressor"), RandomForestRegressor)


def test_resolve_rejects_unknown_and_non_sklearn():
    with pytest.raises(SystemExit):
        tune.resolve_estimator("nonsense")
    with pytest.raises(SystemExit):
        tune.resolve_estimator("os.system")


def test_load_params_errors(tmp_path):
    with pytest.raises(SystemExit):
        tune.load_params(str(tmp_path / "missing.json"))
    bad = tmp_path / "bad.json"
    bad.write_text("{oops")
    with pytest.raises(SystemExit):
        tune.load_params(str(bad))
    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    with pytest.raises(SystemExit):
        tune.load_params(str(empty))


def test_cli_end_to_end(tmp_path, capsys):
    params = tmp_path / "p.json"
    params.write_text(json.dumps({"alpha": [0.1, 1.0, 10.0]}))
    out = tmp_path / "best.joblib"
    tune.main(["--estimator", "ridge", "--params", str(params), "--output", str(out), "--n-jobs", "1"])
    text = capsys.readouterr().out
    assert "Best params" in text
    assert out.exists()


def test_cli_classifier_uses_classification_data(tmp_path, capsys):
    params = tmp_path / "p.json"
    params.write_text(json.dumps({"C": [0.1, 1]}))
    out = tmp_path / "svc.joblib"
    tune.main(["--estimator", "svc", "--params", str(params), "--scoring", "accuracy",
               "--output", str(out), "--n-jobs", "1"])
    assert out.exists()
