"""Game adapters and reference-value helpers for the real-world benchmark."""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    CIFAR10_PRECOMPUTED_DIR,
    DATASET_SPECS,
    DEFAULT_INSTANCE_COUNT,
    PACKAGE_ROOT,
    RANDOM_STATE,
    REGRESSION_MSR_ROOT,
    get_dataset,
)


if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if REGRESSION_MSR_ROOT.exists() and str(REGRESSION_MSR_ROOT) not in sys.path:
    sys.path.insert(0, str(REGRESSION_MSR_ROOT))


_RUNTIME_CACHE: dict[tuple[Any, ...], "GameRuntime"] = {}
_TABULAR_DATA_CACHE: dict[str, tuple[Any, Any]] = {}
_TABULAR_MODEL_CACHE: dict[tuple[str, int], tuple[Any, np.ndarray, list[np.ndarray]]] = {}
_SHAPIQ_PRECOMPUTED_CACHE: dict[tuple[str, int, int], list[Any]] = {}


@dataclass
class GameRuntime:
    dataset_name: str
    instance_id: int
    n_players: int
    game: Any
    reference_source: str
    model: Any | None = None
    baseline: np.ndarray | None = None
    explicand: np.ndarray | None = None

    def evaluate(self, coalition: np.ndarray) -> float:
        return evaluate_game_object(self.game, normalize_coalition(coalition, self.n_players))

    def value_array(self) -> np.ndarray | None:
        if isinstance(self.game, ArrayValueGame):
            return self.game.values
        return None


class ArrayValueGame:
    """Game backed by an array indexed by little-endian coalition bitmask."""

    def __init__(self, values: np.ndarray):
        values = np.asarray(values, dtype=np.float64)
        n_float = np.log2(len(values))
        n = int(round(float(n_float)))
        if 2**n != len(values):
            raise ValueError(f"Expected 2^n values, got length {len(values)}.")
        self.values = values
        self.n_players = n

    def evaluate(self, coalition: np.ndarray) -> float:
        return float(self.values[coalition_to_index(coalition)])


class BaselineTreeGame:
    """Baseline-imputation tree game v(S)=f(x_S, baseline_-S)."""

    def __init__(self, model: Any, baseline: np.ndarray, explicand: np.ndarray):
        self.model = model
        self.baseline = np.asarray(baseline, dtype=np.float64).reshape(-1)
        self.explicand = np.asarray(explicand, dtype=np.float64).reshape(-1)
        if self.baseline.shape != self.explicand.shape:
            raise ValueError("Baseline and explicand must have the same shape.")
        self.n_players = int(self.baseline.shape[0])

    def evaluate(self, coalition: np.ndarray) -> float:
        mask = normalize_coalition(coalition, self.n_players)
        x = self.baseline.copy()
        x[mask] = self.explicand[mask]
        pred = self.model.predict(x.reshape(1, -1))
        arr = np.asarray(pred, dtype=np.float64).reshape(-1)
        return float(arr[0])


class BenchmarkGame:
    """Constructor passed to EaseSHAP estimators."""

    def __init__(
        self,
        *,
        dataset_name: str,
        instance_id: int,
        random_state: int = RANDOM_STATE,
        tabular_game_mode: str = "baseline_tree",
    ):
        self.runtime = load_runtime(
            dataset_name=dataset_name,
            instance_id=instance_id,
            random_state=random_state,
            tabular_game_mode=tabular_game_mode,
        )

    def evaluate(self, coalition: np.ndarray) -> float:
        return self.runtime.evaluate(coalition)


def normalize_coalition(coalition: np.ndarray, n_players: int) -> np.ndarray:
    """Return an n-player mask, ignoring the dummy player used by GELS-style estimators."""

    mask = np.asarray(coalition, dtype=bool).reshape(-1)
    n_players = int(n_players)
    if len(mask) == n_players:
        return mask
    if len(mask) == n_players + 1:
        return mask[:n_players]
    raise ValueError(
        f"Coalition length {len(mask)} does not match n_players={n_players} "
        f"or n_players + 1={n_players + 1}."
    )


def coalition_to_index(coalition: np.ndarray) -> int:
    mask = np.asarray(coalition, dtype=bool).reshape(-1)
    out = 0
    for idx, active in enumerate(mask):
        if active:
            out |= 1 << idx
    return int(out)


def index_to_coalition(index: int, n_players: int) -> np.ndarray:
    return np.array([(int(index) >> bit) & 1 for bit in range(int(n_players))], dtype=bool)


def evaluate_game_object(game: Any, coalition: np.ndarray) -> float:
    mask = np.asarray(coalition, dtype=bool).reshape(1, -1)

    if hasattr(game, "evaluate"):
        return float(game.evaluate(mask.reshape(-1)))

    if callable(game):
        try:
            value = game(mask)
        except TypeError:
            value = game(mask.reshape(-1))
        return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])

    if hasattr(game, "value_function"):
        value = game.value_function(mask)
        return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])

    raise TypeError(f"Do not know how to evaluate game object of type {type(game)!r}.")


def semivalue_coefficients(num_player: int, semivalue: str, semivalue_param: Any) -> np.ndarray:
    n = int(num_player)
    if semivalue == "shapley":
        return np.array([1.0 / (n * float(_comb(n - 1, s))) for s in range(n)], dtype=np.float64)
    if semivalue == "weighted_banzhaf":
        p = float(semivalue_param)
        return np.array([p**s * (1.0 - p) ** (n - 1 - s) for s in range(n)], dtype=np.float64)
    if semivalue == "beta_shapley":
        from scipy import special

        alpha, beta = semivalue_param
        weights = np.ones(n, dtype=np.float64)
        tmp_range = np.arange(1, n, dtype=np.float64)
        weights *= np.divide(tmp_range, tmp_range + (alpha + beta - 1)).prod()
        for s in range(n):
            cur = weights[s]
            tmp_range = np.arange(1, s + 1, dtype=np.float64)
            cur *= np.divide(tmp_range + (beta - 1), tmp_range).prod()
            tmp_range = np.arange(1, n - s, dtype=np.float64)
            cur *= np.divide((alpha - 1) + tmp_range, tmp_range).prod()
            weights[s] = cur / float(special.comb(n - 1, s, exact=False))
        return weights
    raise NotImplementedError(f"Unknown semivalue {semivalue!r}.")


def _comb(n: int, k: int) -> int:
    import math

    return math.comb(int(n), int(k))


def exact_semivalue_from_values(
    values: np.ndarray,
    *,
    semivalue: str,
    semivalue_param: Any,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    n = int(round(float(np.log2(len(values)))))
    if 2**n != len(values):
        raise ValueError(f"Expected value array length 2^n, got {len(values)}.")

    coeff = semivalue_coefficients(n, semivalue, semivalue_param)
    phi = np.zeros(n, dtype=np.float64)
    for mask in range(1 << n):
        s = int(mask.bit_count())
        if s >= n:
            continue
        base_value = values[mask]
        for player in range(n):
            bit = 1 << player
            if mask & bit:
                continue
            phi[player] += coeff[s] * (values[mask | bit] - base_value)
    return phi


def enumerate_values(runtime: GameRuntime) -> np.ndarray:
    cached = runtime.value_array()
    if cached is not None:
        return np.asarray(cached, dtype=np.float64)
    n = runtime.n_players
    values = np.empty(1 << n, dtype=np.float64)
    for mask in range(1 << n):
        values[mask] = runtime.evaluate(index_to_coalition(mask, n))
    return values


def _extract_interaction_values(obj: Any, n_players: int) -> np.ndarray:
    if hasattr(obj, "get_n_order_values"):
        values = obj.get_n_order_values(1)
    elif hasattr(obj, "values"):
        values = obj.values
    else:
        values = obj
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if len(arr) == n_players + 1:
        arr = arr[1:]
    if len(arr) != n_players:
        raise ValueError(f"Expected {n_players} values, got shape {arr.shape}.")
    return arr.astype(np.float64, copy=False)


def load_runtime(
    *,
    dataset_name: str,
    instance_id: int,
    random_state: int = RANDOM_STATE,
    tabular_game_mode: str = "baseline_tree",
) -> GameRuntime:
    key = (dataset_name, int(instance_id), int(random_state), tabular_game_mode)
    if key not in _RUNTIME_CACHE:
        dataset = get_dataset(dataset_name)
        kind = dataset["kind"]
        if kind == "cifar10_precomputed":
            runtime = _load_cifar10_runtime(dataset, int(instance_id))
        elif kind == "shapiq_precomputed":
            runtime = _load_shapiq_precomputed_runtime(dataset, int(instance_id))
        elif kind == "tabular_pathdependent":
            if tabular_game_mode == "baseline_tree":
                runtime = _load_regressionmsr_tabular_runtime(dataset, int(instance_id), int(random_state))
            elif tabular_game_mode == "pathdependent":
                runtime = _load_shapiq_tabular_pathdependent_runtime(dataset, int(instance_id), int(random_state))
            else:
                raise ValueError(
                    "`tabular_game_mode` must be 'baseline_tree' or 'pathdependent', "
                    f"got {tabular_game_mode!r}."
                )
        else:
            raise ValueError(f"Unsupported dataset kind {kind!r}.")
        _RUNTIME_CACHE[key] = runtime
    return _RUNTIME_CACHE[key]


def _load_cifar10_runtime(dataset: dict[str, Any], instance_id: int) -> GameRuntime:
    path = Path(dataset.get("precomputed_dir", CIFAR10_PRECOMPUTED_DIR)) / f"instance_{instance_id:03d}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing CIFAR10 precomputed values: {path}. "
            "Run prepare_cifar10.py on a GPU first."
        )
    data = np.load(path, allow_pickle=False)
    raw_values = np.asarray(data["values"], dtype=np.float64).reshape(-1)
    if "coalitions" in data.files:
        coalitions = np.asarray(data["coalitions"], dtype=bool)
        n_players = int(data["n_players"]) if "n_players" in data.files else coalitions.shape[1]
        values = np.empty(1 << n_players, dtype=np.float64)
        for coalition, value in zip(coalitions, raw_values):
            values[coalition_to_index(coalition)] = value
    else:
        values = raw_values
    game = ArrayValueGame(values)
    return GameRuntime(
        dataset_name=dataset["name"],
        instance_id=instance_id,
        n_players=game.n_players,
        game=game,
        reference_source="cifar10_precomputed_enumeration",
    )


def _load_shapiq_precomputed_runtime(dataset: dict[str, Any], instance_id: int) -> GameRuntime:
    try:
        from shapiq.benchmark import load_games_from_configuration
    except ImportError as exc:
        raise ImportError(
            "Loading shapiq precomputed games requires shapiq with benchmark support."
        ) from exc

    cache_key = (dataset["game_identifier"], int(dataset["config_id"]), int(dataset["n_instances"]))
    if cache_key not in _SHAPIQ_PRECOMPUTED_CACHE:
        games = load_games_from_configuration(
            game_class=dataset["game_class"],
            n_player_id=int(dataset["n_player_id"]),
            config_id=int(dataset["config_id"]),
            n_games=int(dataset["n_instances"]),
        )
        _SHAPIQ_PRECOMPUTED_CACHE[cache_key] = list(games)
    games = _SHAPIQ_PRECOMPUTED_CACHE[cache_key]
    if not games:
        cache_dir = _shapiq_precomputed_cache_dir(dataset)
        raise FileNotFoundError(
            "No shapiq precomputed games were loaded for "
            f"dataset={dataset['name']!r}, game_class={dataset['game_class']!r}, "
            f"n_player_id={int(dataset['n_player_id'])}, config_id={int(dataset['config_id'])}. "
            "For the PolySHAP ViT4by4 benchmark, seed the shapiq cache with "
            "`python prepare_polyshap_vit4by4.py` before running the CPU array. "
            f"Expected cache directory: {cache_dir}."
        )
    if instance_id >= len(games):
        raise IndexError(
            f"Requested instance {instance_id}, but only {len(games)} precomputed "
            f"games were loaded for dataset={dataset['name']!r}."
        )
    game = games[int(instance_id)]
    n_players = int(getattr(game, "n_players", dataset["n_players"]))
    return GameRuntime(
        dataset_name=dataset["name"],
        instance_id=instance_id,
        n_players=n_players,
        game=game,
        reference_source="shapiq_precomputed_enumeration",
    )


def _shapiq_precomputed_cache_dir(dataset: dict[str, Any]) -> str:
    try:
        from shapiq.benchmark.precompute import SHAPIQ_DATA_DIR
    except Exception:
        return "<shapiq benchmark cache unavailable>"
    game_name = dataset.get("precomputed_game_name", dataset["game_class"])
    return str(Path(SHAPIQ_DATA_DIR) / str(game_name) / str(int(dataset["n_players"])))


def _load_regressionmsr_tabular_runtime(
    dataset: dict[str, Any],
    instance_id: int,
    random_state: int,
) -> GameRuntime:
    try:
        import sklearn.ensemble
    except ImportError as exc:
        raise ImportError(
            "Baseline tree tabular games require scikit-learn."
        ) from exc

    dataset_name = _regressionmsr_dataset_name(dataset["name"])
    if dataset_name not in _TABULAR_DATA_CACHE:
        _TABULAR_DATA_CACHE[dataset_name] = _load_tabular_dataset(dataset_name)
    X, y = _TABULAR_DATA_CACHE[dataset_name]
    n_players = int(X.shape[1])

    model_key = (dataset_name, int(random_state))
    if model_key not in _TABULAR_MODEL_CACHE:
        model = sklearn.ensemble.RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=int(random_state),
            max_features=n_players,
        )
        model.fit(X.values, np.asarray(y, dtype=np.float64).reshape(-1))
        baseline, explicands = _load_tabular_inputs(X, num_runs=DEFAULT_INSTANCE_COUNT)
        _TABULAR_MODEL_CACHE[model_key] = (model, np.asarray(baseline, dtype=np.float64), explicands)

    model, baseline, explicands = _TABULAR_MODEL_CACHE[model_key]
    if instance_id >= len(explicands):
        raise IndexError(f"Instance {instance_id} is not available for {dataset_name}.")
    explicand = np.asarray(explicands[instance_id], dtype=np.float64).reshape(1, -1)
    game = BaselineTreeGame(model=model, baseline=baseline, explicand=explicand)
    return GameRuntime(
        dataset_name=dataset["name"],
        instance_id=instance_id,
        n_players=n_players,
        game=game,
        reference_source="regressionmsr_treeprob_baseline",
        model=model,
        baseline=baseline,
        explicand=explicand,
    )


def _load_shapiq_tabular_pathdependent_runtime(
    dataset: dict[str, Any],
    instance_id: int,
    random_state: int,
) -> GameRuntime:
    cls = _import_shapiq_tabular_class(dataset["class_name"])
    tree_game_cls = _import_tree_shapiq_xai_class()
    setup_game = cls(model_name="random_forest", imputer="baseline", random_state=int(random_state))
    x_explain = np.asarray(setup_game.setup.x_test)[int(instance_id), :]
    tree_game = tree_game_cls(x_explain, setup_game.setup.model, verbose=False)
    n_players = int(getattr(tree_game, "n_players", dataset["n_players"]))
    return GameRuntime(
        dataset_name=dataset["name"],
        instance_id=instance_id,
        n_players=n_players,
        game=tree_game,
        reference_source="shapiq_pathdependent_treeshap",
        model=getattr(setup_game.setup, "model", None),
        baseline=None,
        explicand=x_explain.reshape(1, -1),
    )


def _regressionmsr_dataset_name(dataset_name: str) -> str:
    mapping = {
        "breast_cancer": "Breast Cancer",
        "nhanesi": "NHANES",
        "communities_crime": "Communities",
    }
    if dataset_name not in mapping:
        raise KeyError(f"No regressionMSR dataset mapping for {dataset_name!r}.")
    return mapping[dataset_name]


def _load_tabular_dataset(dataset_name: str):
    import pandas as pd

    if dataset_name == "Breast Cancer":
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    elif dataset_name == "NHANES":
        try:
            import shap
        except ImportError as exc:
            raise ImportError(
                "NHANES requires shap.datasets.nhanesi. Install shap before running this dataset."
            ) from exc
        X, y = shap.datasets.nhanesi()
        y = pd.Series(np.asarray(y, dtype=np.float64).reshape(-1), name="target")
    elif dataset_name == "Communities":
        try:
            import shap
        except ImportError as exc:
            raise ImportError(
                "CommunitiesAndCrime requires shap.datasets.communitiesandcrime. "
                "Install shap before running this dataset."
            ) from exc
        X, y = shap.datasets.communitiesandcrime()
        y = pd.Series(np.asarray(y, dtype=np.float64).reshape(-1), name="target")
    else:
        raise KeyError(f"Unsupported tabular dataset {dataset_name!r}.")

    X = X.fillna(X.mean())
    return X, y


def _load_tabular_inputs(X: Any, num_runs: int):
    baseline = X.mean().to_numpy(dtype=np.float64, copy=True).reshape(1, -1)
    explicands = []
    for run_idx in range(int(num_runs)):
        seed = run_idx * int(num_runs)
        np.random.seed(seed)
        explicand_idx = int(np.random.choice(X.shape[0]))
        explicand = X.iloc[explicand_idx].to_numpy(dtype=np.float64, copy=True).reshape(1, -1)
        for col_idx in range(explicand.shape[1]):
            while baseline[0, col_idx] == explicand[0, col_idx]:
                explicand_idx = int(np.random.choice(X.shape[0]))
                explicand[0, col_idx] = X.iloc[explicand_idx, col_idx]
        explicands.append(explicand)
    return baseline, explicands


def _import_shapiq_tabular_class(class_name: str):
    candidates = [
        "shapiq.games.benchmark.local_xai.benchmark_tabular",
        "shapiq_games.benchmark.local_xai.benchmark_tabular",
        "shapiq_games.benchmark.local_xai",
    ]
    errors = []
    for module_name in candidates:
        try:
            module = __import__(module_name, fromlist=[class_name])
            return getattr(module, class_name)
        except Exception as exc:  # pragma: no cover - depends on installed shapiq version
            errors.append(f"{module_name}: {exc}")
    raise ImportError(f"Could not import {class_name}. Tried: {'; '.join(errors)}")


def _import_tree_shapiq_xai_class():
    candidates = [
        ("shapiq.games.benchmark.treeshapiq_xai", "TreeSHAPIQXAI"),
        ("shapiq_games.benchmark.treeshapiq_xai", "TreeSHAPIQXAI"),
    ]
    errors = []
    for module_name, class_name in candidates:
        try:
            module = __import__(module_name, fromlist=[class_name])
            return getattr(module, class_name)
        except Exception as exc:  # pragma: no cover - depends on installed shapiq version
            errors.append(f"{module_name}: {exc}")
    raise ImportError(f"Could not import TreeSHAPIQXAI. Tried: {'; '.join(errors)}")


def semivalue_to_treeprob_weighting(semivalue: str, semivalue_param: Any) -> str:
    if semivalue == "shapley":
        return "shapley"
    if semivalue == "weighted_banzhaf":
        p = float(semivalue_param)
        return "banzhaf" if abs(p - 0.5) < 1e-12 else f"weighted_banzhaf_{p:g}"
    if semivalue == "beta_shapley":
        alpha, beta = semivalue_param

        def fmt(value: Any) -> str:
            value = float(value)
            return str(int(value)) if value == int(value) else f"{value:g}"

        return f"beta_shapley_{fmt(alpha)}_{fmt(beta)}"
    raise NotImplementedError(f"Unknown semivalue {semivalue!r}.")


def compute_reference_values(
    *,
    dataset_name: str,
    instance_id: int,
    semivalue: str,
    semivalue_param: Any,
    random_state: int = RANDOM_STATE,
    tabular_game_mode: str = "baseline_tree",
) -> tuple[np.ndarray, dict[str, Any]]:
    runtime = load_runtime(
        dataset_name=dataset_name,
        instance_id=instance_id,
        random_state=random_state,
        tabular_game_mode=tabular_game_mode,
    )
    dataset = get_dataset(dataset_name)

    if runtime.n_players <= 20 or dataset["kind"] in {"cifar10_precomputed", "shapiq_precomputed"}:
        values = enumerate_values(runtime)
        phi = exact_semivalue_from_values(
            values,
            semivalue=semivalue,
            semivalue_param=semivalue_param,
        )
        return phi, {
            "reference_source": runtime.reference_source,
            "reference_mode": "full_enumeration",
            "num_coalitions": int(len(values)),
        }

    if dataset["kind"] == "tabular_pathdependent" and tabular_game_mode == "baseline_tree":
        if runtime.model is None or runtime.baseline is None or runtime.explicand is None:
            raise RuntimeError("Treeprob reference requires model, baseline, and explicand.")
        weighting = semivalue_to_treeprob_weighting(semivalue, semivalue_param)
        phi = _tree_prob_sklearn_random_forest(
            baseline=runtime.baseline,
            explicand=runtime.explicand,
            model=runtime.model,
            semivalue=semivalue,
            semivalue_param=semivalue_param,
        )
        return phi, {
            "reference_source": runtime.reference_source,
            "reference_mode": "sklearn_treeprob",
            "treeprob_weighting": weighting,
        }

    if dataset["kind"] == "tabular_pathdependent" and tabular_game_mode == "pathdependent":
        if semivalue != "shapley":
            raise NotImplementedError(
                "Path-dependent non-Shapley references are not implemented. "
                "Use --tabular-game-mode baseline_tree for all-semivalue runs."
            )
        exact = runtime.game.exact_values(index="SV", order=1)
        phi = _extract_interaction_values(exact, runtime.n_players)
        return phi, {
            "reference_source": runtime.reference_source,
            "reference_mode": "treeshapiq_exact_sv",
        }

    raise NotImplementedError(
        f"No reference implementation for dataset={dataset_name!r}, semivalue={semivalue!r}."
    )


def available_datasets() -> list[str]:
    return [spec["name"] for spec in DATASET_SPECS]


def _tree_prob_sklearn_random_forest(
    *,
    baseline: np.ndarray,
    explicand: np.ndarray,
    model: Any,
    semivalue: str,
    semivalue_param: Any,
) -> np.ndarray:
    baseline = np.asarray(baseline, dtype=np.float64)
    explicand = np.asarray(explicand, dtype=np.float64)
    if baseline.ndim == 1:
        baseline = baseline.reshape(1, -1)
    if explicand.ndim == 1:
        explicand = explicand.reshape(1, -1)
    if explicand.shape[0] != 1:
        raise ValueError("The local benchmark expects one explicand at a time.")

    n_features = int(explicand.shape[1])
    p = semivalue_coefficients(n_features, semivalue, semivalue_param)
    phi = np.zeros(n_features, dtype=np.float64)
    x = explicand[0]

    estimators = getattr(model, "estimators_", None)
    if estimators is None:
        raise TypeError("Expected a fitted sklearn RandomForestRegressor with estimators_.")

    for estimator in estimators:
        tree = estimator.tree_
        tree_phi = np.zeros(n_features, dtype=np.float64)
        values = np.asarray(tree.value[:, 0, 0], dtype=np.float64)
        for c in baseline:
            xlist = np.zeros(n_features, dtype=np.int64)
            clist = np.zeros(n_features, dtype=np.int64)
            _tree_prob_recurse_rf(
                node_index=0,
                s_path=0,
                n_path=0,
                xlist=xlist,
                clist=clist,
                x=x,
                c=np.asarray(c, dtype=np.float64),
                children_left=tree.children_left,
                children_right=tree.children_right,
                features=tree.feature,
                thresholds=tree.threshold,
                values=values,
                phi=tree_phi,
                p=p,
            )
        tree_phi /= float(len(baseline))
        phi += tree_phi

    phi /= float(len(estimators))
    return phi


def _tree_prob_path_weights(n: int, s_path: int, n_path: int, p: np.ndarray) -> tuple[float, float]:
    from math import comb

    free_count = n - n_path - s_path
    pos_w = 0.0
    neg_w = 0.0
    for coalition_size in range(s_path, n - n_path + 1):
        k = coalition_size - s_path
        coeff = comb(free_count, k)
        pos_w += p[coalition_size - 1] * coeff
        if n_path > 0 and coalition_size < n:
            neg_w += p[coalition_size] * coeff
    return pos_w, neg_w


def _tree_prob_recurse_rf(
    *,
    node_index: int,
    s_path: int,
    n_path: int,
    xlist: np.ndarray,
    clist: np.ndarray,
    x: np.ndarray,
    c: np.ndarray,
    children_left: np.ndarray,
    children_right: np.ndarray,
    features: np.ndarray,
    thresholds: np.ndarray,
    values: np.ndarray,
    phi: np.ndarray,
    p: np.ndarray,
) -> tuple[float, float]:
    if children_left[node_index] == -1 and children_right[node_index] == -1:
        pos_weight, neg_weight = _tree_prob_path_weights(len(x), s_path, n_path, p)
        value = values[node_index]
        neg = -neg_weight * value if not np.isnan(neg_weight) else 0.0
        return pos_weight * value, neg

    feature = int(features[node_index])
    threshold = thresholds[node_index]
    left = int(children_left[node_index])
    right = int(children_right[node_index])

    if xlist[feature] > 0:
        child = right if x[feature] > threshold else left
        return _tree_prob_recurse_rf(
            node_index=child,
            s_path=s_path,
            n_path=n_path,
            xlist=xlist,
            clist=clist,
            x=x,
            c=c,
            children_left=children_left,
            children_right=children_right,
            features=features,
            thresholds=thresholds,
            values=values,
            phi=phi,
            p=p,
        )

    if clist[feature] > 0:
        child = right if c[feature] > threshold else left
        return _tree_prob_recurse_rf(
            node_index=child,
            s_path=s_path,
            n_path=n_path,
            xlist=xlist,
            clist=clist,
            x=x,
            c=c,
            children_left=children_left,
            children_right=children_right,
            features=features,
            thresholds=thresholds,
            values=values,
            phi=phi,
            p=p,
        )

    x_right = x[feature] > threshold
    c_right = c[feature] > threshold
    if x_right == c_right:
        child = right if x_right else left
        return _tree_prob_recurse_rf(
            node_index=child,
            s_path=s_path,
            n_path=n_path,
            xlist=xlist,
            clist=clist,
            x=x,
            c=c,
            children_left=children_left,
            children_right=children_right,
            features=features,
            thresholds=thresholds,
            values=values,
            phi=phi,
            p=p,
        )

    if x_right and not c_right:
        xlist[feature] += 1
        pos_x, neg_x = _tree_prob_recurse_rf(
            node_index=right,
            s_path=s_path + 1,
            n_path=n_path,
            xlist=xlist,
            clist=clist,
            x=x,
            c=c,
            children_left=children_left,
            children_right=children_right,
            features=features,
            thresholds=thresholds,
            values=values,
            phi=phi,
            p=p,
        )
        xlist[feature] -= 1

        clist[feature] += 1
        pos_c, neg_c = _tree_prob_recurse_rf(
            node_index=left,
            s_path=s_path,
            n_path=n_path + 1,
            xlist=xlist,
            clist=clist,
            x=x,
            c=c,
            children_left=children_left,
            children_right=children_right,
            features=features,
            thresholds=thresholds,
            values=values,
            phi=phi,
            p=p,
        )
        clist[feature] -= 1
        phi[feature] += pos_x + neg_c
        return pos_x + pos_c, neg_x + neg_c

    xlist[feature] += 1
    pos_x, neg_x = _tree_prob_recurse_rf(
        node_index=left,
        s_path=s_path + 1,
        n_path=n_path,
        xlist=xlist,
        clist=clist,
        x=x,
        c=c,
        children_left=children_left,
        children_right=children_right,
        features=features,
        thresholds=thresholds,
        values=values,
        phi=phi,
        p=p,
    )
    xlist[feature] -= 1

    clist[feature] += 1
    pos_c, neg_c = _tree_prob_recurse_rf(
        node_index=right,
        s_path=s_path,
        n_path=n_path + 1,
        xlist=xlist,
        clist=clist,
        x=x,
        c=c,
        children_left=children_left,
        children_right=children_right,
        features=features,
        thresholds=thresholds,
        values=values,
        phi=phi,
        p=p,
    )
    clist[feature] -= 1
    phi[feature] += pos_x + neg_c
    return pos_x + pos_c, neg_x + neg_c
