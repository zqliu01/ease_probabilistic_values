import math
import sys
from pathlib import Path

import numpy as np

from easeshap.utilityFuncs import gameSOU, gameSOUPaper, gameSOUStructuredGaussian


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FAITHFUL_BENCH = PROJECT_ROOT / "reference" / "Faithful_GSV-main" / "exp1_benchmark"
if str(FAITHFUL_BENCH) not in sys.path:
    sys.path.append(str(FAITHFUL_BENCH))

from utils import generate_sou_game, sv_sou_true  # noqa: E402


def _sets_to_bool_matrix(sets, n):
    mat = np.zeros((len(sets), n), dtype=bool)
    for row, cur in enumerate(sets):
        if cur:
            mat[row, list(cur)] = True
    return mat


def test_game_sou_seed_is_reproducible_across_fresh_paths(tmp_path):
    kwargs = dict(num_player=9, num_unanimity=17, seed=123)
    game_a = gameSOU(path=str(tmp_path / "sou_a"), **kwargs)
    game_b = gameSOU(path=str(tmp_path / "sou_b"), **kwargs)

    np.testing.assert_array_equal(game_a.coalitions, game_b.coalitions)
    np.testing.assert_allclose(game_a.weights, game_b.weights)


def test_game_sou_paper_matches_faithful_benchmark_generator(tmp_path):
    n = 8
    d = n ** 2
    seed = 42
    old_sets, old_alpha = generate_sou_game(n, d, seed=seed)
    old_coalitions = _sets_to_bool_matrix(old_sets, n)

    game = gameSOUPaper(num_player=n, d=d, seed=seed, path=str(tmp_path / "paper_sou"))

    np.testing.assert_array_equal(game.coalitions, old_coalitions)
    np.testing.assert_allclose(game.alpha, old_alpha)

    old_sv = sv_sou_true(n, old_sets, old_alpha)
    np.testing.assert_allclose(game.get_semivalue(semivalue="shapley"), old_sv)


def test_structured_gaussian_sou_has_expected_low_and_high_order_terms(tmp_path):
    n = 7
    num_high = 19
    game = gameSOUStructuredGaussian(
        num_player=n,
        alpha=0.4,
        num_high_order=num_high,
        seed=123,
        path=str(tmp_path / "structured_sou"),
    )

    num_low = n + n * (n - 1) // 2
    assert game.coalitions.shape == (num_low + num_high, n)
    assert game.weights.shape == (num_low + num_high,)
    assert game.sigma2 == num_low + num_high

    low_sizes = game.sizes[game.component == 0]
    high_sizes = game.sizes[game.component == 1]
    assert np.count_nonzero(low_sizes == 1) == n
    assert np.count_nonzero(low_sizes == 2) == n * (n - 1) // 2
    assert np.all(high_sizes >= 3)
    assert np.all(high_sizes <= n - 1)


def test_structured_gaussian_sou_seed_is_reproducible_across_fresh_paths(tmp_path):
    kwargs = dict(num_player=8, alpha=0.7, num_high_order=23, sigma2=2 * 8 ** 2, seed=456)
    game_a = gameSOUStructuredGaussian(path=str(tmp_path / "structured_a"), **kwargs)
    game_b = gameSOUStructuredGaussian(path=str(tmp_path / "structured_b"), **kwargs)

    np.testing.assert_array_equal(game_a.coalitions, game_b.coalitions)
    np.testing.assert_allclose(game_a.weights, game_b.weights)


def test_structured_gaussian_sou_shapley_matches_marginal_definition(tmp_path):
    n = 4
    game = gameSOUStructuredGaussian(
        num_player=n,
        alpha=0.5,
        num_high_order=3,
        seed=789,
        path=str(tmp_path / "structured_shapley"),
    )

    shapley = np.zeros(n)
    for player in range(n):
        for mask in range(1 << (n - 1)):
            subset = np.zeros(n, dtype=bool)
            other_players = [j for j in range(n) if j != player]
            for bit, other in enumerate(other_players):
                subset[other] = bool(mask & (1 << bit))
            size = int(subset.sum())
            coeff = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            with_player = subset.copy()
            with_player[player] = True
            shapley[player] += coeff * (game.evaluate(with_player) - game.evaluate(subset))

    np.testing.assert_allclose(game.get_semivalue(semivalue="shapley"), shapley)
