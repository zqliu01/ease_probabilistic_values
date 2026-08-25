"""Coalition game for ACSIncome state-source valuation."""

from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import acs_model

UTILITY_MODEL_CHOICES = ("logistic", "xgboost", "gbm")
UTILITY_CACHE_MODE_CHOICES = ("memoize", "recompute")
_XGB_CLASSIFIER = None


def normalize_utility_model(utility_model: str) -> str:
    """Normalize user-facing utility model aliases."""

    if utility_model == "gbm":
        return "xgboost"
    if utility_model in {"logistic", "xgboost"}:
        return utility_model
    raise ValueError(
        f"Unknown utility model {utility_model!r}; choose from {UTILITY_MODEL_CHOICES}."
    )


def normalize_utility_cache_mode(utility_cache_mode: str) -> str:
    """Normalize the policy used for repeated coalition evaluations."""

    mode = str(utility_cache_mode).strip().lower()
    if mode in UTILITY_CACHE_MODE_CHOICES:
        return mode
    raise ValueError(
        f"Unknown utility cache mode {utility_cache_mode!r}; "
        f"choose from {UTILITY_CACHE_MODE_CHOICES}."
    )


def require_xgb_classifier():
    """Load XGBoost only when requested so logistic runs remain independent of it."""

    global _XGB_CLASSIFIER
    if _XGB_CLASSIFIER is not None:
        return _XGB_CLASSIFIER
    try:
        from xgboost import XGBClassifier
    except Exception as exc:  # pragma: no cover - depends on optional native runtime.
        raise RuntimeError(
            "XGBoost utility requested, but xgboost could not be imported. "
            "Install the optional dependency and ensure its native runtime is available."
        ) from exc
    _XGB_CLASSIFIER = XGBClassifier
    return _XGB_CLASSIFIER


def fit_utility_model(
    x_train: sparse.csr_matrix,
    y_train: np.ndarray,
    *,
    utility_model: str,
    fixed_lambda: float = 1.0,
    solver: str = "liblinear",
    max_iter: int = 5000,
    seed: int = 2026,
    xgb_n_estimators: int = 5,
    xgb_max_depth: int = 5,
    xgb_learning_rate: float = 0.1,
    xgb_tree_method: str = "hist",
    xgb_n_jobs: int = 1,
    xgb_subsample: float = 1.0,
    xgb_colsample_bytree: float = 1.0,
):
    """Fit the requested coalition utility model and return ``(model, mode)``."""

    utility_model = normalize_utility_model(utility_model)
    if utility_model == "logistic":
        if fixed_lambda <= 0.0:
            raise ValueError("fixed_lambda must be positive for logistic utility.")
        model = LogisticRegression(
            C=1.0 / fixed_lambda,
            solver=solver,
            max_iter=int(max_iter),
            random_state=int(seed),
        )
        mode = "ridge_fixed"
    else:
        xgb_classifier = require_xgb_classifier()
        model = xgb_classifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method=xgb_tree_method,
            n_estimators=int(xgb_n_estimators),
            max_depth=int(xgb_max_depth),
            learning_rate=float(xgb_learning_rate),
            n_jobs=int(xgb_n_jobs),
            subsample=float(xgb_subsample),
            colsample_bytree=float(xgb_colsample_bytree),
            random_state=int(seed),
            verbosity=0,
        )
        mode = "xgb_binary_logistic"
    model.fit(x_train, y_train)
    return model, mode


class ACSStateCoalitionGame:
    """Coalition utility for fixed-size ACS state training sources."""

    def __init__(
        self,
        *,
        states,
        encoded_train,
        train_labels,
        encoded_eval,
        eval_y,
        fixed_lambda,
        solver,
        max_iter,
        seed,
        utility_cache,
        utility_cache_mode="memoize",
        utility_model="logistic",
        xgb_n_estimators=5,
        xgb_max_depth=5,
        xgb_learning_rate=0.1,
        xgb_tree_method="hist",
        xgb_n_jobs=1,
        xgb_subsample=1.0,
        xgb_colsample_bytree=1.0,
    ):
        self.states = list(states)
        self.encoded_train = encoded_train
        self.train_labels = train_labels
        self.encoded_eval = encoded_eval
        self.eval_y = np.asarray(eval_y, dtype=int)
        self.fixed_lambda = float(fixed_lambda)
        self.solver = solver
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self.utility_cache = utility_cache
        self.utility_cache_mode = normalize_utility_cache_mode(utility_cache_mode)
        self.utility_model = normalize_utility_model(utility_model)
        self.xgb_n_estimators = int(xgb_n_estimators)
        self.xgb_max_depth = int(xgb_max_depth)
        self.xgb_learning_rate = float(xgb_learning_rate)
        self.xgb_tree_method = xgb_tree_method
        self.xgb_n_jobs = int(xgb_n_jobs)
        self.xgb_subsample = float(xgb_subsample)
        self.xgb_colsample_bytree = float(xgb_colsample_bytree)

    def _mask_from_subset(self, subset) -> int:
        subset = np.asarray(subset, dtype=bool)[: len(self.states)]
        mask = 0
        for idx, selected in enumerate(subset):
            if selected:
                mask |= 1 << idx
        return mask

    def evaluate(self, subset) -> float:
        mask = self._mask_from_subset(subset)
        cached = self.utility_cache.get(mask)
        if self.utility_cache_mode == "memoize" and cached is not None:
            return float(cached["utility"])

        start = time.perf_counter()
        if mask == 0:
            utility, loss = acs_model.constant_utility(
                self.eval_y,
                float(np.mean(self.eval_y)),
            )
            result = {
                "mask": mask,
                "n_states": 0,
                "n_train": 0,
                "utility": float(utility),
                "log_loss": float(loss),
                "mode": "constant_eval_prior",
                "elapsed_sec": time.perf_counter() - start,
            }
            self.utility_cache[mask] = result
            return float(utility)

        selected_states = [
            state for idx, state in enumerate(self.states) if mask & (1 << idx)
        ]
        x_train = sparse.vstack(
            [self.encoded_train[state] for state in selected_states],
            format="csr",
        )
        y_train = np.concatenate([self.train_labels[state] for state in selected_states])
        if len(np.unique(y_train)) < 2:
            utility, loss = acs_model.constant_utility(
                self.eval_y,
                float(np.mean(y_train)),
            )
            mode = "constant_train_prior"
        else:
            model, mode = fit_utility_model(
                x_train,
                y_train,
                utility_model=self.utility_model,
                fixed_lambda=self.fixed_lambda,
                solver=self.solver,
                max_iter=self.max_iter,
                seed=self.seed,
                xgb_n_estimators=self.xgb_n_estimators,
                xgb_max_depth=self.xgb_max_depth,
                xgb_learning_rate=self.xgb_learning_rate,
                xgb_tree_method=self.xgb_tree_method,
                xgb_n_jobs=self.xgb_n_jobs,
                xgb_subsample=self.xgb_subsample,
                xgb_colsample_bytree=self.xgb_colsample_bytree,
            )
            scores = model.predict_proba(self.encoded_eval)[:, 1]
            loss = float(log_loss(self.eval_y, scores, labels=[0, 1]))
            utility = -loss

        result = {
            "mask": mask,
            "n_states": len(selected_states),
            "n_train": int(len(y_train)),
            "utility": float(utility),
            "log_loss": float(loss),
            "mode": mode,
            "elapsed_sec": time.perf_counter() - start,
        }
        self.utility_cache[mask] = result
        return float(utility)
