import math

import numpy as np

from easeshap.utilityFuncs import (
    gameSOU,
    gameSOUPaper,
    gameSOUStructuredGaussian,
    gameSOUStructuredGaussianBitset,
    generate_sou_game,
    sv_sou_true,
)


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


def test_sv_sou_true_uses_actual_term_count_normalization():
    got = sv_sou_true(
        3,
        [set([0, 1]), set([1])],
        np.array([3.0, 4.0]),
    )
    np.testing.assert_allclose(got, np.array([0.75, 2.75, 0.0]))


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


def test_structured_gaussian_sou_max_high_order_size_is_backward_compatible(tmp_path):
    kwargs = dict(num_player=10, alpha=0.6, num_high_order=37, seed=2468)
    legacy = gameSOUStructuredGaussian(path=str(tmp_path / "legacy"), **kwargs)
    oversized = gameSOUStructuredGaussian(
        path=str(tmp_path / "oversized"),
        max_high_order_size=kwargs["num_player"] + 5,
        **kwargs,
    )

    assert legacy.max_high_order_size is None
    assert oversized.max_high_order_size is None
    assert legacy.effective_max_high_order_size == kwargs["num_player"] - 1
    assert oversized.effective_max_high_order_size == kwargs["num_player"] - 1
    np.testing.assert_array_equal(legacy.coalitions, oversized.coalitions)
    np.testing.assert_allclose(legacy.weights, oversized.weights)


def test_structured_gaussian_sou_caps_high_order_sizes_and_separates_cache(tmp_path):
    path = tmp_path / "shared_cache"
    kwargs = dict(num_player=12, alpha=0.5, num_high_order=53, seed=1357, path=str(path))
    legacy = gameSOUStructuredGaussian(**kwargs)
    capped = gameSOUStructuredGaussian(max_high_order_size=5, **kwargs)

    capped_sizes = capped.sizes[capped.component == 1]
    assert capped.max_high_order_size == 5
    assert capped.effective_max_high_order_size == 5
    assert np.all(capped_sizes >= 3)
    assert np.all(capped_sizes <= 5)
    assert np.any(legacy.sizes[legacy.component == 1] > 5)
    assert len(list(path.glob("*.npz"))) == 2


def test_structured_gaussian_bitset_matches_reference_evaluator(tmp_path):
    kwargs = dict(
        num_player=12,
        alpha=0.55,
        num_high_order=41,
        sigma2=190.0,
        seed=654,
    )
    path = str(tmp_path / "structured_bitset")
    reference = gameSOUStructuredGaussian(path=path, **kwargs)
    fast = gameSOUStructuredGaussianBitset(path=path, **kwargs)

    np.testing.assert_array_equal(fast.coalitions, reference.coalitions)
    np.testing.assert_allclose(fast.weights, reference.weights)
    for semivalue, param in [
        ("shapley", None),
        ("weighted_banzhaf", 0.5),
        ("beta_shapley", (2, 3)),
    ]:
        np.testing.assert_allclose(
            fast.get_semivalue(semivalue=semivalue, semivalue_param=param),
            reference.get_semivalue(semivalue=semivalue, semivalue_param=param),
            rtol=1e-14,
            atol=1e-14,
        )

    rng = np.random.default_rng(321)
    checks = [
        np.zeros(kwargs["num_player"], dtype=bool),
        np.ones(kwargs["num_player"], dtype=bool),
    ]
    checks.extend(rng.random((200, kwargs["num_player"])) < 0.5)
    for subset in checks:
        np.testing.assert_allclose(fast.evaluate(subset), reference.evaluate(subset), rtol=1e-12, atol=1e-12)


def test_structured_gaussian_bitset_accepts_python_bool_inputs(tmp_path):
    kwargs = dict(
        num_player=6,
        alpha=0.45,
        num_high_order=13,
        sigma2=80.0,
        seed=432,
    )
    path = str(tmp_path / "structured_bitset_python_bool")
    reference = gameSOUStructuredGaussian(path=path, **kwargs)
    fast = gameSOUStructuredGaussianBitset(path=path, **kwargs)

    subset = [True, False, True, False, False, True]
    np_subset = np.asarray(subset, dtype=bool)
    np.testing.assert_allclose(fast.evaluate(subset), reference.evaluate(np_subset), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        fast.evaluate([]),
        reference.evaluate(np.zeros(kwargs["num_player"], dtype=bool)),
        rtol=1e-12,
        atol=1e-12,
    )


def test_structured_gaussian_bitset_handles_multiword_masks(tmp_path):
    n = 70
    kwargs = dict(
        num_player=n,
        alpha=0.35,
        num_high_order=83,
        sigma2=400.0,
        seed=987,
        allow_duplicate_high_order=False,
    )
    path = str(tmp_path / "structured_bitset_multiword")
    reference = gameSOUStructuredGaussian(path=path, **kwargs)
    fast = gameSOUStructuredGaussianBitset(path=path, **kwargs)

    rng = np.random.default_rng(777)
    for subset in rng.random((100, n)) < 0.35:
        np.testing.assert_allclose(fast.evaluate(subset), reference.evaluate(subset), rtol=1e-12, atol=1e-12)


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
