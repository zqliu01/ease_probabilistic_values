"""Configuration for the real-world semivalue benchmark."""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
PAPER_DIR = EXPERIMENTS_DIR.parent
WORKSPACE_DIR = PAPER_DIR.parent
PACKAGE_ROOT = PAPER_DIR / "EaseSHAP"
REGRESSION_MSR_ROOT = WORKSPACE_DIR / "reference" / "codes" / "regressionMSR"

OUT = SCRIPT_DIR / "results"
CONFIGS_DIR = OUT / "configs"
GROUNDTRUTH_DIR = OUT / "groundtruth"
FIGURES_DIR = SCRIPT_DIR / "figures"
DATA_DIR = SCRIPT_DIR / "data"
CIFAR10_PRECOMPUTED_DIR = DATA_DIR / "cifar10_precomputed"
POLYSHAP_ARCHIVE_URL = "https://github.com/FFmgll/PolySHAP/archive/refs/heads/main.zip"
POLYSHAP_CACHE_DIR = DATA_DIR / "polyshap_cache"
POLYSHAP_VIT4BY4_GAME_NAME = "ImageClassifier_Game"
POLYSHAP_VIT4BY4_N_PLAYERS = 16

RANDOM_STATE = 40
BASE_SEED = 2026
DEFAULT_N_RUNS = 1
DEFAULT_INSTANCE_COUNT = 30
MAX_BUDGET_PER_PLAYER = 200
CHECKPOINT_INTERVAL_PER_PLAYER = 20
NUM_CHECKPOINTS = 10
EASE_SWITCH_FRACTION = 0.2


DATASET_SPECS: list[dict[str, Any]] = [
    {
        "name": "vit4by4",
        "title": "ViT4by4Patches",
        "domain": "image",
        "kind": "shapiq_precomputed",
        "n_players": 16,
        "game_identifier": "ViT4by4Patches",
        "game_class": "ImageClassifierLocalXAI",
        "n_player_id": 2,
        "config_id": 1,
        "precomputed_game_name": POLYSHAP_VIT4BY4_GAME_NAME,
        "n_instances": DEFAULT_INSTANCE_COUNT,
    },
    {
        "name": "cifar10",
        "title": "CIFAR10",
        "domain": "image",
        "kind": "cifar10_precomputed",
        "n_players": 16,
        "precomputed_dir": CIFAR10_PRECOMPUTED_DIR,
        "n_instances": DEFAULT_INSTANCE_COUNT,
    },
    {
        "name": "breast_cancer",
        "title": "BreastCancer",
        "domain": "tabular",
        "kind": "tabular_pathdependent",
        "n_players": 30,
        "class_name": "BreastCancer",
        "n_instances": DEFAULT_INSTANCE_COUNT,
    },
    {
        "name": "nhanesi",
        "title": "NHANESI",
        "domain": "tabular",
        "kind": "tabular_pathdependent",
        "n_players": 79,
        "class_name": "NHANESI",
        "n_instances": DEFAULT_INSTANCE_COUNT,
    },
    {
        "name": "communities_crime",
        "title": "CommunitiesAndCrime",
        "domain": "tabular",
        "kind": "tabular_pathdependent",
        "n_players": 101,
        "class_name": "CommunitiesAndCrime",
        "n_instances": DEFAULT_INSTANCE_COUNT,
    },
]


SEMIVALUE_SPECS: list[dict[str, Any]] = [
    {
        "name": "shapley",
        "title": "Shapley",
        "semivalue": "shapley",
        "semivalue_param": None,
    },
    {
        "name": "beta1_4",
        "title": "Beta(1, 4)",
        "semivalue": "beta_shapley",
        "semivalue_param": (1, 4),
    },
    {
        "name": "beta4_1",
        "title": "Beta(4, 1)",
        "semivalue": "beta_shapley",
        "semivalue_param": (4, 1),
    },
    {
        "name": "wb0p25",
        "title": "WeightedBanzhaf(0.25)",
        "semivalue": "weighted_banzhaf",
        "semivalue_param": 0.25,
    },
    {
        "name": "wb0p5",
        "title": "WeightedBanzhaf(0.5)",
        "semivalue": "weighted_banzhaf",
        "semivalue_param": 0.5,
    },
    {
        "name": "wb0p75",
        "title": "WeightedBanzhaf(0.75)",
        "semivalue": "weighted_banzhaf",
        "semivalue_param": 0.75,
    },
]


EASESHAP_COMMON_KWARGS: dict[str, Any] = {
    "exact_boundary_handling": True,
    "boundary_policy": "fixed",
    "use_complement_sampling": True,
    "surrogate_ridge_lambda": 0.01,
    "surrogate_ridge_schedule": "times_m",
    "num_folds": 2,
    "surrogate_readout_mode": "crossfit",
}

EASESHAP_SIZE_PLAYER_RIDGE_KWARGS: dict[str, Any] = {
    "surrogate_ridge_lambda": 1.0,
    "surrogate_ridge_schedule": "fixed",
    "surrogate_ridge_scaling": "size_trace",
}


METHOD_SPECS: list[dict[str, Any]] = [
    {
        "name": "EaseSHAP_interaction_nonlinear",
        "backend": "EaseSHAP",
        "support": "all",
        "estimator_kwargs": {
            **EASESHAP_COMMON_KWARGS,
            "boundary_order": 1,
            "pilot_design_updates": 1,
            "surrogate_basis": 1,
            "include_nonlinear_size_terms": True,
        },
    },
    {
        "name": "EaseSHAP_size_player",
        "backend": "EaseSHAP",
        "support": "all",
        "estimator_kwargs": {
            **EASESHAP_COMMON_KWARGS,
            **EASESHAP_SIZE_PLAYER_RIDGE_KWARGS,
            "boundary_order": 1,
            "surrogate_basis": "size_player",
            "include_nonlinear_size_terms": False,
            "surrogate_stats_backend": "exact_conditional",
            "surrogate_solver_mode": "size_player",
            "surrogate_r_correction_alpha": 1.0,
            "surrogate_u_correction_alpha": 1.0,
            "surrogate_correction_solver_mode": "matrix_free",
            "surrogate_correction_max_iter": 10,
        },
    },
    {"name": "OFA_fixed", "backend": "OFA_fixed", "support": "all", "estimator_kwargs": {}},
    {"name": "OFA_baseline", "backend": "OFA_baseline", "support": "all", "estimator_kwargs": {}},
    {"name": "sampling_lift", "backend": "sampling_lift", "support": "all", "estimator_kwargs": {}},
    {"name": "SHAP_IQ", "backend": "SHAP_IQ", "support": "all", "estimator_kwargs": {}},
    {"name": "GELS", "backend": "GELS", "support": "all", "estimator_kwargs": {}},
    {"name": "improved_AME", "backend": "improved_AME", "support": "all", "estimator_kwargs": {}},
    {"name": "kernelSHAP", "backend": "kernelSHAP", "support": "shapley", "estimator_kwargs": {}},
    {
        "name": "LeverageSHAP",
        "backend": "LeverageSHAP",
        "support": "shapley",
        "estimator_kwargs": {"sampling_with_replacement": True},
        "label": "LeverageSHAP",
    },
    {
        "name": "LeverageSHAP_paired_border",
        "backend": "LeverageSHAP_border",
        "support": "shapley",
        "estimator_kwargs": {},
        "label": "LeverageSHAP (border trick)",
    },
    {"name": "permutation", "backend": "permutation", "support": "shapley", "estimator_kwargs": {}},
    {"name": "complement", "backend": "complement", "support": "shapley", "estimator_kwargs": {}},
    {"name": "group_testing", "backend": "group_testing", "support": "shapley", "estimator_kwargs": {}},
    {"name": "WSL", "backend": "WSL", "support": "non_shapley", "estimator_kwargs": {}},
    {
        "name": "weighted_permutation",
        "backend": "weighted_permutation",
        "support": "non_shapley",
        "estimator_kwargs": {},
    },
    {"name": "OFA_optimal", "backend": "OFA_optimal", "support": "non_shapley", "estimator_kwargs": {}},
    {
        "name": "WGELS_shapley",
        "backend": "WGELS_shapley",
        "support": "non_shapley",
        "estimator_kwargs": {},
    },
    {"name": "AME", "backend": "AME", "support": "ame", "estimator_kwargs": {}},
    {
        "name": "RegressionMSR_unbiased",
        "backend": "RegressionMSR_unbiased",
        "support": "all",
        "estimator_kwargs": {
            "sampling_with_replacement": True,
            "paired_sampling": None,
            "num_folds": 2,
        },
    },
    {
        "name": "RegressionMSR_unbiased_no_replacement",
        "backend": "RegressionMSR_unbiased",
        "support": "all",
        "estimator_kwargs": {
            "sampling_with_replacement": False,
            "paired_sampling": None,
            "num_folds": 2,
        },
        "label": "RegressionMSR (no replacement)",
    },
    {
        "name": "PolySHAP_regression_optional",
        "backend": "PolySHAP_regression",
        "support": "shapley",
        "optional": True,
        "estimator_kwargs": {"max_order": 2},
    },
]


METHOD_LABELS = {
    method["name"]: method.get("label", method["name"])
    for method in METHOD_SPECS
}
METHOD_LABELS.update(
    {
        "EaseSHAP_interaction_nonlinear": "EASE-FO",
        "EaseSHAP_size_player": "EASE-SP",
        "OFA_fixed": "OFA",
        "OFA_baseline": "OFA baseline",
        "sampling_lift": "Sampling lift",
        "SHAP_IQ": "SHAP-IQ",
        "GELS": "GELS",
        "improved_AME": "Improved AME",
        "kernelSHAP": "kernelSHAP",
        "permutation": "Permutation",
        "complement": "Complement",
        "group_testing": "Group testing",
        "WSL": "WSL",
        "weighted_permutation": "Weighted permutation",
        "OFA_optimal": "OFA optimal",
        "WGELS_shapley": "WGELS",
        "AME": "AME",
        "RegressionMSR_unbiased": "RegressionMSR",
        "RegressionMSR_unbiased_no_replacement": "RegressionMSR (no repl.)",
        "LeverageSHAP_paired_border": "LeverageSHAP (border)",
        "PolySHAP_regression_optional": "PolySHAP-2ADD",
    }
)
METHOD_ORDER = {method["name"]: idx for idx, method in enumerate(METHOD_SPECS)}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def fraction_label(value: float) -> str:
    value = float(value)
    rounded = round(value, 2)
    if abs(value - rounded) < 1e-12:
        text = f"{rounded:.2f}"
    else:
        text = format(Decimal(str(value)).normalize(), "f")
        if "." not in text:
            text = f"{text}.0"
    return text.replace(".", "p")


_PILOT_METHOD_RE = re.compile(r"^(?P<base>.+)_pilot(?P<pilot>[0-9]+p[0-9]+)$")


def parse_fraction_list(value: str | list[str] | None) -> list[float] | None:
    if value is None:
        return None
    values = value if isinstance(value, list) else [value]
    by_label: dict[str, float] = {}
    for item in values:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                fraction = float(part)
                by_label.setdefault(fraction_label(fraction), fraction)
    return [
        by_label[label]
        for label in sorted(by_label, key=lambda current: float(current.replace("p", ".")))
    ] or None


def pilot_method_name(base_method_name: str, switch_fraction: float) -> str:
    return f"{base_method_name}_pilot{fraction_label(switch_fraction)}"


def method_base_name(method_name: str) -> str:
    if method_name in METHOD_LABELS:
        return method_name
    match = _PILOT_METHOD_RE.match(method_name)
    if match and match.group("base") in METHOD_LABELS:
        return match.group("base")
    return method_name


def method_pilot_fraction_label(method_name: str) -> str | None:
    match = _PILOT_METHOD_RE.match(method_name)
    if not match:
        return None
    return match.group("pilot").replace("p", ".")


def method_display_label(method_name: str) -> str:
    base = method_base_name(method_name)
    label = METHOD_LABELS.get(base, method_name)
    pilot = method_pilot_fraction_label(method_name)
    if pilot is not None and base != method_name:
        return f"{label} p={pilot}"
    return label


def method_sort_key(method_name: str) -> tuple[int, float, str]:
    base = method_base_name(method_name)
    base_order = METHOD_ORDER.get(base, len(METHOD_ORDER))
    pilot = method_pilot_fraction_label(method_name)
    pilot_order = float(pilot) if pilot is not None else -1.0
    return (base_order, pilot_order, method_name)


def method_is_easeshap(method_name: str) -> bool:
    base = method_base_name(method_name)
    try:
        return get_method(base)["backend"] == "EaseSHAP"
    except KeyError:
        return False


def benchmark_config_label(
    *,
    budget_per_player: int,
    ease_switch_fraction: float = EASE_SWITCH_FRACTION,
    ease_switch_fractions: list[float] | None = None,
    ease_fo_pilot_design_updates: int = 1,
    num_checkpoints: int = NUM_CHECKPOINTS,
    tabular_game_mode: str = "baseline_tree",
) -> str:
    fractions = (
        sorted(float(fraction) for fraction in ease_switch_fractions)
        if ease_switch_fractions
        else None
    )
    pilot_label = (
        "pilots" + "_".join(fraction_label(fraction) for fraction in fractions)
        if fractions
        else f"pilot{fraction_label(float(ease_switch_fraction))}"
    )
    label = f"m{int(budget_per_player):03d}n_{pilot_label}"
    if int(ease_fo_pilot_design_updates) != 1:
        label = f"{label}_fo_updates{int(ease_fo_pilot_design_updates)}"
    if int(num_checkpoints) != NUM_CHECKPOINTS:
        label = f"{label}_ckpt{int(num_checkpoints):02d}"
    if tabular_game_mode != "baseline_tree":
        label = f"{label}_{safe_name(tabular_game_mode)}"
    return label


def config_results_dir(
    *,
    config_name: str | None = None,
    budget_per_player: int = MAX_BUDGET_PER_PLAYER,
    ease_switch_fraction: float = EASE_SWITCH_FRACTION,
    ease_switch_fractions: list[float] | None = None,
    ease_fo_pilot_design_updates: int = 1,
    num_checkpoints: int = NUM_CHECKPOINTS,
    tabular_game_mode: str = "baseline_tree",
) -> Path:
    name = config_name or benchmark_config_label(
        budget_per_player=budget_per_player,
        ease_switch_fraction=ease_switch_fraction,
        ease_switch_fractions=ease_switch_fractions,
        ease_fo_pilot_design_updates=ease_fo_pilot_design_updates,
        num_checkpoints=num_checkpoints,
        tabular_game_mode=tabular_game_mode,
    )
    return CONFIGS_DIR / safe_name(name)


def config_raw_dir(config_dir: Path) -> Path:
    return Path(config_dir) / "raw"


def config_plots_dir(config_dir: Path) -> Path:
    return config_figures_dir(config_dir)


def config_figures_dir(config_dir: Path) -> Path:
    return FIGURES_DIR / safe_name(Path(config_dir).name)


def add_results_config_args(parser: Any, *, include_tabular_game_mode: bool = True) -> None:
    parser.add_argument(
        "--budget-per-player",
        type=int,
        default=MAX_BUDGET_PER_PLAYER,
        help="Total utility-evaluation budget per player. Default 200 gives m=200n.",
    )
    parser.add_argument(
        "--ease-switch-fraction",
        type=float,
        default=EASE_SWITCH_FRACTION,
        help=(
            "Fraction of the total budget used before EASE switches from pilot "
            "sampling to its adapted design. Default 0.2 gives pilot0p20."
        ),
    )
    parser.add_argument(
        "--ease-fo-pilot-design-updates",
        type=int,
        default=1,
        help="Number of fixed-pilot design updates for EASE-FO only. Default: 1.",
    )
    parser.add_argument(
        "--num-checkpoints",
        type=int,
        default=NUM_CHECKPOINTS,
        help="Number of equal-interval checkpoints. Default 10.",
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help=(
            "Optional results config name under results/configs/. By default this "
            "is derived from the budget, EASE pilot fraction, and any non-default "
            "EASE-FO update count."
        ),
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Optional explicit results config directory. Overrides --config-name.",
    )
    if include_tabular_game_mode:
        parser.add_argument(
            "--tabular-game-mode",
            choices=["baseline_tree", "pathdependent"],
            default="baseline_tree",
        )


def add_ease_switch_fractions_arg(parser: Any) -> None:
    parser.add_argument(
        "--ease-switch-fractions",
        action="append",
        default=None,
        help=(
            "Comma-separated EaseSHAP pilot fractions. When set, EaseSHAP methods "
            "are expanded into per-pilot variants while non-EaseSHAP methods run once."
        ),
    )


def validate_results_config_args(args: Any) -> None:
    if hasattr(args, "ease_switch_fractions"):
        args.ease_switch_fractions = parse_fraction_list(args.ease_switch_fractions)
    if args.budget_per_player <= 0:
        raise ValueError("--budget-per-player must be positive.")
    if not (0.0 <= float(args.ease_switch_fraction) <= 1.0):
        raise ValueError("--ease-switch-fraction must lie in [0, 1].")
    if int(args.ease_fo_pilot_design_updates) < 1:
        raise ValueError("--ease-fo-pilot-design-updates must be at least 1.")
    if getattr(args, "ease_switch_fractions", None):
        bad = [
            fraction
            for fraction in args.ease_switch_fractions
            if not (0.0 <= float(fraction) <= 1.0)
        ]
        if bad:
            raise ValueError(f"--ease-switch-fractions must lie in [0, 1], got {bad}.")
        if len(args.ease_switch_fractions) == 1:
            args.ease_switch_fraction = float(args.ease_switch_fractions[0])
            args.ease_switch_fractions = None
    if args.num_checkpoints <= 0:
        raise ValueError("--num-checkpoints must be positive.")
    if int(args.budget_per_player) % int(args.num_checkpoints) != 0:
        raise ValueError(
            "--budget-per-player must be divisible by --num-checkpoints "
            "for equal-interval checkpoints."
        )


def resolve_config_dir_from_args(args: Any) -> Path:
    if args.config_dir is not None:
        return Path(args.config_dir)
    return config_results_dir(
        config_name=args.config_name,
        budget_per_player=args.budget_per_player,
        ease_switch_fraction=args.ease_switch_fraction,
        ease_switch_fractions=getattr(args, "ease_switch_fractions", None),
        ease_fo_pilot_design_updates=args.ease_fo_pilot_design_updates,
        num_checkpoints=args.num_checkpoints,
        tabular_game_mode=getattr(args, "tabular_game_mode", "baseline_tree"),
    )


def get_dataset(name: str) -> dict[str, Any]:
    for spec in DATASET_SPECS:
        if spec["name"] == name or spec["title"] == name:
            return spec
    raise KeyError(f"Unknown dataset {name!r}.")


def get_semivalue(name: str) -> dict[str, Any]:
    for spec in SEMIVALUE_SPECS:
        if spec["name"] == name:
            return spec
    raise KeyError(f"Unknown semivalue {name!r}.")


def get_method(name: str) -> dict[str, Any]:
    for spec in METHOD_SPECS:
        if spec["name"] == name:
            return spec
    raise KeyError(f"Unknown method {name!r}.")


def param_text(param: Any) -> str:
    if param is None:
        return ""
    return json.dumps(param)


def is_symmetric_semivalue(semivalue: str, semivalue_param: Any) -> bool:
    if semivalue == "shapley":
        return True
    if semivalue == "weighted_banzhaf":
        return abs(float(semivalue_param) - 0.5) < 1e-12
    if semivalue == "beta_shapley":
        alpha, beta = semivalue_param
        return abs(float(alpha) - float(beta)) < 1e-12
    return False


def is_compatible(method: dict[str, Any], semivalue_spec: dict[str, Any]) -> bool:
    support = method.get("support", "all")
    semivalue = semivalue_spec["semivalue"]
    semivalue_param = semivalue_spec["semivalue_param"]
    if support == "all":
        return True
    if support == "shapley":
        return semivalue == "shapley"
    if support == "non_shapley":
        return semivalue != "shapley"
    if support == "symmetric":
        return is_symmetric_semivalue(semivalue, semivalue_param)
    if support == "ame":
        if semivalue == "weighted_banzhaf":
            return True
        if semivalue == "beta_shapley":
            alpha, beta = semivalue_param
            return float(alpha) > 1.0 and float(beta) > 1.0
        return False
    raise ValueError(f"Unknown support code {support!r}.")


def compatible_methods(
    semivalue_spec: dict[str, Any],
    *,
    include_optional: bool = False,
) -> list[dict[str, Any]]:
    methods = []
    for method in METHOD_SPECS:
        if method.get("optional", False) and not include_optional:
            continue
        if is_compatible(method, semivalue_spec):
            methods.append(method)
    return methods


def max_budget(n_players: int, budget_per_player: int = MAX_BUDGET_PER_PLAYER) -> int:
    return int(n_players) * int(budget_per_player)


def checkpoint_interval_nue_avg(
    budget_per_player: int = MAX_BUDGET_PER_PLAYER,
    num_checkpoints: int = NUM_CHECKPOINTS,
) -> int:
    budget_per_player = int(budget_per_player)
    num_checkpoints = int(num_checkpoints)
    if budget_per_player <= 0:
        raise ValueError("budget_per_player must be positive.")
    if num_checkpoints <= 0:
        raise ValueError("num_checkpoints must be positive.")
    if budget_per_player % num_checkpoints != 0:
        raise ValueError(
            "budget_per_player must be divisible by num_checkpoints for equal-interval checkpoints."
        )
    return budget_per_player // num_checkpoints


def equal_interval_checkpoints(
    n_players: int,
    budget_per_player: int = MAX_BUDGET_PER_PLAYER,
    num_checkpoints: int = NUM_CHECKPOINTS,
) -> list[int]:
    track_nue_avg = checkpoint_interval_nue_avg(
        budget_per_player=budget_per_player,
        num_checkpoints=num_checkpoints,
    )
    return [int(n_players) * track_nue_avg * idx for idx in range(1, num_checkpoints + 1)]


def actual_budget_grid(
    n_players: int,
    budget_per_player: int = MAX_BUDGET_PER_PLAYER,
    num_checkpoints: int = NUM_CHECKPOINTS,
) -> tuple[int, list[int]]:
    nue_avg = benchmark_nue_avg(n_players, budget_per_player=budget_per_player)
    track_nue_avg = checkpoint_interval_nue_avg(
        budget_per_player=budget_per_player,
        num_checkpoints=num_checkpoints,
    )
    budgets = [int(n_players) * step for step in range(track_nue_avg, nue_avg + 1, track_nue_avg)]
    return nue_avg, budgets


def benchmark_nue_avg(
    n_players: int,
    budget_per_player: int = MAX_BUDGET_PER_PLAYER,
) -> int:
    del n_players
    return int(budget_per_player)


def benchmark_track_nue_avg(
    budget_per_player: int = MAX_BUDGET_PER_PLAYER,
    num_checkpoints: int = NUM_CHECKPOINTS,
) -> int:
    """Tracking stride used by the real-world benchmark."""

    return checkpoint_interval_nue_avg(
        budget_per_player=budget_per_player,
        num_checkpoints=num_checkpoints,
    )


def benchmark_actual_budgets(
    n_players: int,
    budget_per_player: int = MAX_BUDGET_PER_PLAYER,
    num_checkpoints: int = NUM_CHECKPOINTS,
) -> tuple[int, int, list[int]]:
    nue_avg = benchmark_nue_avg(n_players, budget_per_player=budget_per_player)
    track_nue_avg = benchmark_track_nue_avg(
        budget_per_player=budget_per_player,
        num_checkpoints=num_checkpoints,
    )
    budgets = [
        int(n_players) * step
        for step in range(track_nue_avg, nue_avg + 1, track_nue_avg)
    ]
    return nue_avg, track_nue_avg, budgets


def nearest_actual_checkpoint_indices(
    requested_checkpoints: list[int],
    actual_budgets: list[int],
) -> list[int]:
    if not actual_budgets:
        return []
    indices = []
    for requested in requested_checkpoints:
        best = min(
            range(len(actual_budgets)),
            key=lambda idx: (abs(actual_budgets[idx] - requested), actual_budgets[idx]),
        )
        indices.append(int(best))
    return indices


def boundary_eval_count_order1(n_players: int) -> int:
    n = int(n_players)
    if n <= 0:
        return 0
    if n == 1:
        return 2
    return 2 * n + 2


def ease_pilot_nue_for_switch(n_players: int, switch_budget: int) -> int:
    random_budget = max(0, int(switch_budget) - boundary_eval_count_order1(n_players))
    return max(0, int(round(random_budget / float(n_players))))


def ease_switch_budget(
    n_players: int,
    budget_per_player: int = MAX_BUDGET_PER_PLAYER,
    switch_fraction: float = EASE_SWITCH_FRACTION,
) -> int:
    return int(round(max_budget(n_players, budget_per_player=budget_per_player) * float(switch_fraction)))


def result_path(
    dataset_name: str,
    semivalue_name: str,
    method_name: str,
    instance_id: int,
    run_idx: int,
    *,
    raw_dir: Path,
) -> Path:
    base = Path(raw_dir)
    return (
        base
        / safe_name(dataset_name)
        / safe_name(semivalue_name)
        / safe_name(method_name)
        / f"instance_{int(instance_id):03d}_run{int(run_idx):02d}.pkl"
    )


def reference_path(
    dataset_name: str,
    semivalue_name: str,
    instance_id: int,
    *,
    groundtruth_dir: Path | None = None,
) -> Path:
    base = GROUNDTRUTH_DIR if groundtruth_dir is None else Path(groundtruth_dir)
    return (
        base
        / safe_name(dataset_name)
        / safe_name(semivalue_name)
        / f"instance_{int(instance_id):03d}.npz"
    )


def iter_task_specs(
    *,
    datasets: list[str] | None = None,
    semivalues: list[str] | None = None,
    methods: list[str] | None = None,
    instance_ids: list[int] | None = None,
    n_runs: int = DEFAULT_N_RUNS,
    include_optional: bool = False,
    ease_switch_fractions: list[float] | None = None,
    exclude_methods: set[str] | None = None,
) -> list[dict[str, Any]]:
    dataset_specs = [get_dataset(name) for name in datasets] if datasets else DATASET_SPECS
    semivalue_specs = [get_semivalue(name) for name in semivalues] if semivalues else SEMIVALUE_SPECS
    method_specs = [get_method(name) for name in methods] if methods else None
    excluded = set(exclude_methods or set())

    tasks: list[dict[str, Any]] = []
    for dataset in dataset_specs:
        ids = instance_ids if instance_ids is not None else list(range(int(dataset["n_instances"])))
        for semivalue in semivalue_specs:
            candidates = method_specs if method_specs is not None else compatible_methods(
                semivalue,
                include_optional=include_optional,
            )
            for method in candidates:
                if method.get("optional", False) and not include_optional:
                    continue
                if method["name"] in excluded:
                    continue
                if not is_compatible(method, semivalue):
                    continue
                method_variants: list[tuple[str, float | None]]
                if method["backend"] == "EaseSHAP" and ease_switch_fractions:
                    method_variants = [
                        (pilot_method_name(method["name"], fraction), float(fraction))
                        for fraction in ease_switch_fractions
                    ]
                else:
                    method_variants = [(method["name"], None)]
                for instance_id in ids:
                    for run_idx in range(int(n_runs)):
                        for method_name, switch_fraction in method_variants:
                            task = {
                                "dataset": dataset["name"],
                                "semivalue": semivalue["name"],
                                "method": method_name,
                                "instance_id": int(instance_id),
                                "run_idx": int(run_idx),
                                "seed": BASE_SEED + int(run_idx) * 137,
                            }
                            if method_name != method["name"]:
                                task["base_method"] = method["name"]
                            if switch_fraction is not None:
                                task["ease_switch_fraction"] = switch_fraction
                            tasks.append(task)
    return tasks
