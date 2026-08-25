from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

import acs_state_game


class _FixedProbabilityModel:
    def predict_proba(self, x):
        positive = np.full(x.shape[0], 0.4, dtype=float)
        return np.column_stack((1.0 - positive, positive))


def _make_game(monkeypatch, *, utility_cache_mode: str):
    fit_calls = []

    def fake_fit_utility_model(x_train, y_train, **kwargs):
        fit_calls.append((x_train.copy(), y_train.copy(), dict(kwargs)))
        return _FixedProbabilityModel(), "test_model"

    monkeypatch.setattr(acs_state_game, "fit_utility_model", fake_fit_utility_model)
    utility_cache = {}
    game = acs_state_game.ACSStateCoalitionGame(
        states=["AA", "BB"],
        encoded_train={
            "AA": sparse.csr_matrix([[0.0], [1.0]]),
            "BB": sparse.csr_matrix([[1.0], [0.0]]),
        },
        train_labels={
            "AA": np.array([0, 1]),
            "BB": np.array([1, 0]),
        },
        encoded_eval=sparse.csr_matrix([[0.0], [1.0]]),
        eval_y=np.array([0, 1]),
        fixed_lambda=1.0,
        solver="liblinear",
        max_iter=100,
        seed=2026,
        utility_cache=utility_cache,
        utility_cache_mode=utility_cache_mode,
    )
    return game, utility_cache, fit_calls


def test_memoize_mode_reuses_repeated_coalition(monkeypatch):
    game, utility_cache, fit_calls = _make_game(
        monkeypatch,
        utility_cache_mode="memoize",
    )
    coalition = np.array([True, False])

    first = game.evaluate(coalition)
    second = game.evaluate(coalition)

    assert first == second
    assert len(fit_calls) == 1
    assert len(utility_cache) == 1


def test_recompute_mode_refits_but_retains_unique_coalition_count(monkeypatch):
    game, utility_cache, fit_calls = _make_game(
        monkeypatch,
        utility_cache_mode="recompute",
    )
    coalition = np.array([True, False])

    first = game.evaluate(coalition)
    second = game.evaluate(coalition)

    assert first == second
    assert len(fit_calls) == 2
    assert len(utility_cache) == 1


def test_unknown_utility_cache_mode_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="Unknown utility cache mode"):
        _make_game(monkeypatch, utility_cache_mode="unknown")
