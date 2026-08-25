"""Baseline estimator implementations and legacy comparison methods."""

import itertools
import math
import multiprocessing as mp
import sys

import numpy as np
from .base import estimatorTemplate
from scipy import special
from scipy.linalg import lstsq


class exact_value(estimatorTemplate):
    def __init__(self, **kwargs):
        super(exact_value, self).__init__(**kwargs)
        self.values = np.zeros(self.num_player, dtype=np.float64)
        self.num_sample = 2 ** (self.num_player - 1)
        self.batch_size = -(-self.nue_per_proc // (2 * self.num_player))
        self.nue_per_proc_run = self.batch_size * 2 * self.num_player

    def sampling(self):
        count = 0
        samples = np.empty((self.batch_size, self.num_player-1), dtype=bool)
        for subset in itertools.product([True, False], repeat=self.num_player-1):
            samples[count] = subset
            count += 1
            if count == self.batch_size:
                yield samples.copy()
                count = 0
        if count:
            yield samples[:count]

    def run(self, samples):
        weights = np.empty(self.num_player, dtype=np.float64)
        for i in range(self.num_player):
            if self.semivalue == "shapley":
                weights[i] = special.beta(self.num_player - i, i + 1)
            elif self.semivalue == "weighted_banzhaf":
                weights[i] = (self.semivalue_param ** i) * ((1 - self.semivalue_param) ** (self.num_player - 1 - i))
            elif self.semivalue == "beta_shapley":
                weights[i] = 1
                alpha, beta = self.semivalue_param
                for k in range(1, i+1):
                    weights[i] *= (beta+k-1) / (alpha+beta+k-1)
                for k in range(i+1, self.num_player):
                    weights[i] *= (alpha+k-i-1) / (alpha+beta+k-1)
            else:
                raise NotImplementedError(f"Check {self.semivalue}")

        game = self.game_func(**self.game_args)
        fragment = np.zeros(self.num_player)
        right_index = np.zeros(self.num_player, dtype=bool)
        left_index = np.ones_like(right_index)
        for sample in samples:
            weight = weights[sample.sum()]
            right_index[:self.num_player - 1] = sample
            left_index[:self.num_player - 1] = sample
            fragment[-1] += weight * (game.evaluate(left_index) - game.evaluate(right_index))
            for player in range(self.num_player - 1):
                right_index[-1], right_index[player] = right_index[player], right_index[-1]
                left_index[-1], left_index[player] = left_index[player], left_index[-1]
                fragment[player] += weight * (game.evaluate(left_index) - game.evaluate(right_index))
                right_index[-1], right_index[player] = right_index[player], right_index[-1]
                left_index[-1], left_index[player] = left_index[player], left_index[-1]
        return fragment

    def aggregate(self, fragment):
        self.values += fragment

    def finalize(self):
        return self.values, self.values[None, :]


class sampling_lift(estimatorTemplate):
    def __init__(self, **kwargs):
        super(sampling_lift, self).__init__(**kwargs)
        self.interval_track = self.nue_track_avg // 2
        self.num_sample = self.nue_avg // 2
        self.batch_size = -(-self.nue_per_proc // (2 * self.num_player))
        self.nue_per_proc_run = self.batch_size * 2 * self.num_player

        self.results_aggregate = dict(estimates=np.zeros(self.num_player, dtype=np.float64), count=0)
        self.buffer = np.empty((self.buffer_size, self.num_player), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player - 1), dtype=bool)

    def _init_indiv(self):
        assert self.nue_track_avg % 2 == 0
        assert self.nue_avg % 2 == 0

    def _generator(self):
        if self.semivalue == "weighted_banzhaf":
            t = self.semivalue_param
        elif self.semivalue == "shapley":
            t = np.random.rand()
        elif self.semivalue == "beta_shapley":
            t = np.random.beta(self.semivalue_param[1], self.semivalue_param[0])
        else:
            raise NotImplementedError
        return np.random.binomial(1, t, size=self.num_player - 1).astype(bool)

    def run(self, samples):
        game = self.game_func(**self.game_args)
        results_collect = np.zeros((len(samples), self.num_player), dtype=np.float64)
        subset = np.zeros(self.num_player, dtype=bool)
        for i, sample in enumerate(samples):
            results = results_collect[i]
            subset[:self.num_player-1] = sample
            results[-1] -= game.evaluate(subset)
            subset[-1] = 1
            results[-1] += game.evaluate(subset)
            for player in range(self.num_player - 1):
                subset[-1], subset[player] = subset[player], subset[-1]
                results[player] += game.evaluate(subset)
                subset[player] = 0
                results[player] -= game.evaluate(subset)
                subset[player] = 1
                subset[-1], subset[player] = subset[player], subset[-1]
            subset[-1] = 0
        return results_collect

    def _process(self, inputs):
        num_pre = self.results_aggregate["count"]
        num_cur = len(inputs) + num_pre
        self.results_aggregate["estimates"] *= num_pre / num_cur
        self.results_aggregate["estimates"] += inputs.sum(axis=0) / num_cur
        self.results_aggregate["count"] = num_cur

    def _estimate(self):
        return self.results_aggregate["estimates"]


class sampling_lift_paired(sampling_lift):
    def __init__(self, **kwargs):
        super(sampling_lift_paired, self).__init__(**kwargs)
        self.lock_switch = False

    def _init_indiv(self):
        assert self.nue_track_avg % 2 == 0
        assert self.nue_avg % 2 == 0
        if self.semivalue == "weighted_banzhaf":
            assert self.semivalue_param == 0.5
        if self.semivalue == "beta_shapley":
            assert self.semivalue_param[0] == self.semivalue_param[1]


class WSL(sampling_lift):
    def __init__(self, **kwargs):
        super(WSL, self).__init__(**kwargs)
        self.weights = self.distribution_cardinality() * self.num_player

    def _init_indiv(self):
        assert self.nue_track_avg % 2 == 0
        assert self.nue_avg % 2 == 0
        assert self.semivalue != "shapley"  # for the Shapley, sampling_lift = WSL

    def _generator(self):
        t = np.random.rand()
        return np.random.binomial(1, t, size=self.num_player - 1).astype(bool)

    def run(self, samples):
        results_collect = super(WSL, self).run(samples)
        scalars = self.weights[samples.sum(axis=1)]
        return scalars[:, None] * results_collect


class WSL_paired(WSL):
    def __init__(self, **kwargs):
        super(WSL_paired, self).__init__(**kwargs)
        self.lock_switch = False


class WSL_banzhaf(WSL):
    def __init__(self, **kwargs):
        super(WSL, self).__init__(**kwargs)
        tmp = 2 ** (self.num_player - 1)
        vs = self.distribution_cardinality()
        self.weights = np.array([tmp / special.binom(self.num_player - 1, s) * vs[s] for s in range(self.num_player)])

    def _init_indiv(self):
        assert self.nue_track_avg % 2 == 0
        assert self.nue_avg % 2 == 0
        assert not (self.semivalue == "weighted_banzhaf" and self.semivalue_param == 0.5)

    def _generator(self):
        return np.random.binomial(1, 0.5, size=self.num_player - 1).astype(bool)


class WSL_banzhaf_paired(WSL_banzhaf):
    def __init__(self, **kwargs):
        super(WSL_banzhaf_paired, self).__init__(**kwargs)
        self.lock_switch = False


class permutation(sampling_lift):
    # the evaluation of U(0) is not counted for the total budget of utility evaluations.
    def __init__(self, **kwargs):
        super(sampling_lift, self).__init__(**kwargs)
        self.num_sample = self.nue_avg
        self.interval_track = self.nue_track_avg
        self.batch_size = -(-self.nue_per_proc // self.num_player)
        self.nue_per_proc_run = self.batch_size * self.num_player

        self.results_aggregate = dict(estimates=np.zeros(self.num_player, dtype=np.float64), count=0)
        self.buffer = np.empty((self.buffer_size, self.num_player), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player), dtype=np.int64)

    def _init_indiv(self):
        assert self.semivalue == "shapley"

    def _generator(self):
        return np.random.permutation(self.num_player)

    def run(self, samples):
        game = self.game_func(**self.game_args)
        results_collect = np.zeros((len(samples), self.num_player), dtype=np.float64)
        subset = np.zeros(self.num_player, dtype=bool)
        empty_value = game.evaluate(subset)
        for i, sample in enumerate(samples):
            results = results_collect[i]
            pre_value = empty_value
            for j in range(self.num_player):
                player = sample[j]
                results[player] -= pre_value
                subset[player] = True
                cur_value = game.evaluate(subset)
                results[player] += cur_value
                pre_value = cur_value
            subset.fill(False)
        return results_collect


class permutation_paired(permutation):
    def __init__(self, **kwargs):
        super(permutation_paired, self).__init__(**kwargs)
        self.takeInverse = False
        self.pi_pre = None

    def _generator(self):
        if self.takeInverse:
            self.takeInverse = False
            return np.argsort(self.pi_pre)
        else:
            self.takeInverse = True
            self.pi_pre = np.random.permutation(self.num_player)
            return self.pi_pre


class weighted_permutation(permutation):
    def __init__(self, **kwargs):
        super(weighted_permutation, self).__init__(**kwargs)
        self.weights = self.distribution_cardinality() * self.num_player

    def _init_indiv(self):
        assert self.semivalue != "shapley"

    def run(self, samples):
        game = self.game_func(**self.game_args)
        results_collect = np.zeros((len(samples), self.num_player), dtype=np.float64)
        subset = np.zeros(self.num_player, dtype=bool)
        empty_value = game.evaluate(subset)
        for i, sample in enumerate(samples):
            results = results_collect[i]
            pre_value = empty_value
            for j in range(self.num_player):
                player = sample[j]
                results[player] -= pre_value
                subset[player] = True
                cur_value = game.evaluate(subset)
                results[player] += cur_value
                results[player] *= self.weights[j]
                pre_value = cur_value
            subset.fill(False)
        return results_collect


class weighted_permutation_paired(weighted_permutation, permutation_paired):
    def __init__(self, **kwargs):
        super(weighted_permutation_paired, self).__init__(**kwargs)
        self.takeInverse = False
        self.pi_pre = None

    def _generator(self):
        return permutation_paired._generator(self)


class MSR(estimatorTemplate):
    def __init__(self, **kwargs):
        super(MSR, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        self.results_aggregate = np.zeros((4, self.num_player), dtype=np.float64)
        self.buffer = np.empty((self.buffer_size, self.num_player + 1), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player), dtype=bool)

    def _init_indiv(self):
        assert self.semivalue == "weighted_banzhaf"
        assert 0 < self.semivalue_param and self.semivalue_param < 1

    def _generator(self):
        return np.random.binomial(1, self.semivalue_param, size=self.num_player).astype(bool)

    def run(self, samples):
        game = self.game_func(**self.game_args)
        results_collect = np.empty((len(samples), self.num_player + 1), dtype=np.float64)
        results_collect[:, :self.num_player] = samples
        for i, sample in enumerate(samples):
            results_collect[i, -1] = game.evaluate(sample)
        return results_collect

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player]
        ues = inputs[:, [-1]]
        self.results_aggregate[0] += (ues * subsets).sum(axis=0)
        self.results_aggregate[1] += subsets.sum(axis=0)
        subsets = 1 - subsets
        self.results_aggregate[2] += (ues * subsets).sum(axis=0)
        self.results_aggregate[3] += subsets.sum(axis=0)

    def _estimate(self):
        counts = self.results_aggregate[1].copy()
        counts[counts == 0] = -1
        left = np.divide(self.results_aggregate[0], counts)
        counts = self.results_aggregate[3].copy()
        counts[counts == 0] = -1
        right = np.divide(self.results_aggregate[2], counts)
        return left - right


class MSR_paired(MSR):
    def __init__(self, **kwargs):
        super(MSR_paired, self).__init__(**kwargs)
        self.lock_switch = False

    def _init_indiv(self):
        assert self.semivalue == "weighted_banzhaf"
        assert self.semivalue_param == 0.5

class improved_AME(estimatorTemplate):
    def __init__(self, **kwargs):
        super(improved_AME, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        self.samples = np.empty((self.batch_size, self.num_player + 1), dtype=np.float64)
        self.buffer = np.empty((self.buffer_size, self.num_player + 2), dtype=np.float64)
        self.results_aggregate = dict(estimates=np.zeros(self.num_player, dtype=np.float64), count=0)


    def _generator(self):
        sample = np.empty(self.num_player + 1, dtype=np.float64)
        if self.semivalue == "weighted_banzhaf":
            t = self.semivalue_param
        elif self.semivalue == "shapley":
            t = np.random.rand()
        elif self.semivalue == "beta_shapley":
            t = np.random.beta(self.semivalue_param[1], self.semivalue_param[0])
        else:
            raise NotImplementedError
        sample[-1] = t
        sample[:-1] = np.random.binomial(1, t, size=self.num_player)
        return sample

    def run(self, samples):
        game = self.game_func(**self.game_args)
        results_collect = np.empty((len(samples), self.num_player + 2), dtype=np.float64)
        results_collect[:, :-1] = samples
        for i, sample in enumerate(samples):
            results_collect[i, -1] = game.evaluate(sample[:-1].astype(bool))
        return results_collect

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player]
        ues = inputs[:, [-1]]
        weights = inputs[:, [-2]]
        inv_weights = np.zeros_like(weights)
        np.divide(1.0, weights, out=inv_weights, where=weights > 0)

        num_pre = self.results_aggregate["count"]
        num_cur = len(inputs) + num_pre
        self.results_aggregate["estimates"] *= num_pre / num_cur

        self.results_aggregate["estimates"] += (ues * inv_weights * subsets).sum(axis=0) / num_cur
        subsets = 1 - subsets
        weights = 1 - weights
        inv_weights = np.zeros_like(weights)
        np.divide(1.0, weights, out=inv_weights, where=weights > 0)
        self.results_aggregate["estimates"] -= (ues * inv_weights * subsets).sum(axis=0) / num_cur
        self.results_aggregate["count"] = num_cur


    def _estimate(self):
        return self.results_aggregate["estimates"]




class weighted_MSR(MSR):
    def __init__(self, **kwargs):
        super(weighted_MSR, self).__init__(**kwargs)
        self.weights = self.distribution_cardinality()
        self.scalar = 2**(self.num_player - 1)

    def _init_indiv(self):
        assert not (self.semivalue == "weighted_banzhaf" and self.semivalue_param == 0.5)

    def _generator(self):
        return np.random.binomial(1, 0.5, size=self.num_player).astype(bool)

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player]
        ues = inputs[:, [-1]]
        sizes = subsets.sum(axis=1).astype(np.int64)
        weights = np.array([self.scalar / special.binom(self.num_player - 1, s - 1) * self.weights[s - 1] if s > 0 else 1 for s in sizes])
        self.results_aggregate[0] += (ues * subsets * weights[:, None]).sum(axis=0)
        self.results_aggregate[1] += subsets.sum(axis=0)
        subsets = 1 - subsets
        weights = np.array([self.scalar / special.binom(self.num_player - 1, s) * self.weights[s] if s < self.num_player else 1 for s in sizes])
        self.results_aggregate[2] += (ues * subsets * weights[:, None]).sum(axis=0)
        self.results_aggregate[3] += subsets.sum(axis=0)


class weighted_MSR_paired(weighted_MSR):
    def __init__(self, **kwargs):
        super(weighted_MSR_paired, self).__init__(**kwargs)
        self.lock_switch = False


class kernelSHAP(MSR):
    @staticmethod
    def calculate_constants(game_func, game_args, num_player):
        game = game_func(**game_args)
        subset = np.zeros(num_player, dtype=bool)
        v_empty = game.evaluate(subset)
        subset.fill(True)
        v_full = game.evaluate(subset)
        return v_empty, v_full

    def __init__(self, **kwargs):
        super(MSR, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        self.results_aggregate = dict(mat_A=np.zeros((self.num_player, self.num_player), dtype=np.float64),
                                      vec_b=np.zeros(self.num_player, dtype=np.float64),
                                      count=0)
        self.buffer = np.empty((self.buffer_size, self.num_player + 1), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player), dtype=bool)

        with mp.Pool(1) as pool:
            self.constants = pool.apply(self.calculate_constants, (self.game_func, self.game_args, self.num_player))

    def _init_indiv(self):
        assert self.semivalue == "shapley"

        tmp = np.arange(1, self.num_player, dtype=np.float64)
        weights = 1 / np.multiply(tmp, tmp[::-1])
        self.weights = weights / weights.sum()
        self.s_range = np.arange(1, self.num_player)
        self.pos_range = np.arange(self.num_player)

    def _generator(self):
        s = np.random.choice(self.s_range, p=self.weights)
        pos = np.random.choice(self.pos_range, size=s, replace=False)
        subset = np.zeros(self.num_player, dtype=bool)
        subset[pos] = True
        return subset

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player]
        ues = inputs[:, [-1]]
        A_tmp = subsets.T @ subsets
        b_tmp = subsets * (ues - self.constants[0])

        num_pre = self.results_aggregate["count"]
        num_cur = len(b_tmp) + num_pre
        self.results_aggregate["mat_A"] *= num_pre / num_cur
        self.results_aggregate["mat_A"] += A_tmp / num_cur
        self.results_aggregate["vec_b"] *= num_pre / num_cur
        self.results_aggregate["vec_b"] += b_tmp.sum(axis=0) / num_cur
        self.results_aggregate["count"] = num_cur

    def _estimate(self):
        A_inv = np.linalg.pinv(self.results_aggregate["mat_A"])
        vec_b = self.results_aggregate["vec_b"]
        vec_1 = np.ones(len(vec_b))
        v_empty, v_full = self.constants
        tmp = vec_b - (np.dot(vec_1, np.dot(A_inv, vec_b)) - v_full + v_empty) / np.dot(vec_1, np.dot(A_inv, vec_1))
        return np.dot(A_inv, tmp)


class kernelSHAP_paired(kernelSHAP):
    def __init__(self, **kwargs):
        super(kernelSHAP_paired, self).__init__(**kwargs)
        self.lock_switch = False


def _ith_combination(num_items, size, index):
    """Return the lexicographic `index`-th subset of a given size."""
    combination = []
    start = 0
    remaining = size
    index = int(index)
    for _ in range(size):
        for item in range(start, num_items):
            count = math.comb(num_items - item - 1, remaining - 1)
            if index < count:
                combination.append(item)
                start = item + 1
                remaining -= 1
                break
            index -= count
    return combination


def _sample_unique_indices(rng, population_size, num_samples):
    """Sample unique integer indices from range(population_size) without materializing it."""
    population_size = int(population_size)
    num_samples = int(num_samples)
    if num_samples < 0 or num_samples > population_size:
        raise ValueError("Cannot sample more unique indices than the population size.")
    selected = set()
    for j in range(population_size - num_samples, population_size):
        candidate = int(rng.integers(0, j + 1))
        selected.add(j if candidate in selected else candidate)
    return selected


def _combination_generator(rng, num_items, size, num_samples):
    """Sample combinations without materializing the full combination space."""
    num_combos = math.comb(num_items, size)
    try:
        indices = _sample_unique_indices(rng, num_combos, num_samples)
        for index in indices:
            yield _ith_combination(num_items, size, index)
    except (OverflowError, ValueError):
        for _ in range(num_samples):
            yield rng.choice(num_items, size, replace=False)


class LeverageSHAP(kernelSHAP):
    """LeverageSHAP estimator of Musco and Witter (ICLR 2025)."""

    def __init__(self, *, sampling_with_replacement=False, **kwargs):
        if not isinstance(sampling_with_replacement, (bool, np.bool_)):
            raise ValueError(
                "`sampling_with_replacement` must be a bool, "
                f"got {sampling_with_replacement!r}."
            )
        self.sampling_with_replacement = bool(sampling_with_replacement)
        super(LeverageSHAP, self).__init__(**kwargs)
        total_budget = max(0, self.nue_avg * self.num_player - 2)
        self.num_sample = int(total_budget // 2) * 2
        self.results_aggregate = dict(mat_A=np.zeros((self.num_player, self.num_player), dtype=np.float64),
                                      vec_b=np.zeros(self.num_player, dtype=np.float64),
                                      count=0)
        self.buffer = np.empty((self.buffer_size, self.num_player + 2), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player + 1), dtype=np.float64)

    def _init_indiv(self):
        assert self.semivalue == "shapley"
        if self.num_player < 2:
            raise ValueError("LeverageSHAP requires at least two players")
        if self.num_sample <= 0:
            raise ValueError("LeverageSHAP requires a budget larger than two utility evaluations")
        if not self.sampling_with_replacement:
            self._bernoulli_constant = self._find_bernoulli_constant()

    def _find_bernoulli_constant(self):
        max_samples = min(self.num_sample, 2 ** self.num_player - 2)

        def expected_samples(constant):
            return np.sum([
                min(special.binom(self.num_player, size), 2.0 * constant)
                for size in range(1, self.num_player)
            ])

        left = 1.0
        right = special.binom(self.num_player, self.num_player // 2) * self.num_player ** 2
        constant = 1.0
        for _ in range(128):
            current = expected_samples(constant)
            if round(current) == max_samples:
                break
            if current < max_samples:
                left = constant
            else:
                right = constant
            constant = (left + right) / 2.0
        return round(constant)

    def sampling(self):
        self._init_indiv()
        self._rng = np.random.Generator(np.random.PCG64(self.estimator_seed))

        sample_iter = (
            self._sample_with_replacement()
            if self.sampling_with_replacement
            else self._sample_without_replacement()
        )
        count = 0
        for row in sample_iter:
            self.samples[count] = row
            count += 1
            if count == self.batch_size:
                yield self.samples.copy()
                count = 0
        if count:
            yield self.samples[:count].copy()

    def _sample_without_replacement(self):
        sample_counts = []
        for size in range(1, self.num_player):
            binom_count = special.binom(self.num_player, size)
            probability = min(1.0, 2.0 * self._bernoulli_constant / binom_count)
            try:
                count = self._rng.binomial(int(binom_count), probability)
            except OverflowError:
                count = int(probability * binom_count)
            if size == self.num_player // 2:
                if self.num_player % 2 == 0:
                    sample_counts.append(count // 2)
                else:
                    sample_counts.append(count)
                break
            sample_counts.append(count)

        for idx, count in enumerate(sample_counts):
            size = idx + 1
            binom_count = special.binom(self.num_player, size)
            probability = min(1.0, 2.0 * self._bernoulli_constant / binom_count)
            weight = 1.0 / (probability * binom_count * (self.num_player - size) * size)
            if self.num_player % 2 == 0 and size == self.num_player // 2:
                combinations = _combination_generator(self._rng, self.num_player - 1, size - 1, count)
                for indices in combinations:
                    yield from self._paired_rows(list(indices) + [self.num_player - 1], weight)
            else:
                combinations = _combination_generator(self._rng, self.num_player, size, count)
                for indices in combinations:
                    yield from self._paired_rows(indices, weight)

    def _sample_with_replacement(self):
        valid_sizes = np.arange(1, self.num_player)
        num_pairs = self.num_sample // 2
        sampled_sizes = self._rng.choice(valid_sizes, size=num_pairs)
        for size in sampled_sizes:
            indices = self._rng.choice(self.num_player, size=int(size), replace=False)
            weight = 1.0 / (float(size) * float(self.num_player - size))
            yield from self._paired_rows(indices, weight)

    def _paired_rows(self, indices, weight):
        mask = np.zeros(self.num_player, dtype=bool)
        mask[np.asarray(indices, dtype=np.int64)] = True
        yield self._row_from_mask(mask, weight)
        yield self._row_from_mask(~mask, weight)

    def _row_from_mask(self, mask, weight):
        row = np.empty(self.num_player + 1, dtype=np.float64)
        row[:self.num_player] = mask
        row[-1] = weight
        return row

    def run(self, samples):
        game = self.game_func(**self.game_args)
        results_collect = np.empty((len(samples), self.num_player + 2), dtype=np.float64)
        results_collect[:, :self.num_player + 1] = samples
        for i, sample in enumerate(samples):
            results_collect[i, -1] = game.evaluate(sample[:self.num_player].astype(bool))
        return results_collect

    def aggregate(self, results_collect):
        self.buffer[self.pos_buffer:self.pos_buffer + len(results_collect)] = results_collect
        self.pos_buffer += len(results_collect)
        num_collect = self.pos_buffer // self.interval_track
        if num_collect:
            for i in range(num_collect):
                self._process(self.buffer[i*self.interval_track:(i+1)*self.interval_track])
                if self.pos_traj < len(self.values_traj):
                    self.values_traj[self.pos_traj] = self._estimate()
                    self.pos_traj += 1
            num_left = self.pos_buffer - (i + 1) * self.interval_track
            self.buffer[:num_left] = self.buffer[(i + 1) * self.interval_track:self.pos_buffer]
            self.pos_buffer = num_left

    def finalize(self):
        if self.pos_buffer:
            self._process(self.buffer[:self.pos_buffer])
            self.pos_buffer = 0
        values_final = self._estimate()
        if self.values_traj.size:
            if self.pos_traj < len(self.values_traj):
                self.values_traj[self.pos_traj:] = values_final
            else:
                self.values_traj[-1] = values_final
        return values_final, self.values_traj

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player]
        weights = inputs[:, -2]
        ues = inputs[:, -1] - self.constants[0]

        sizes = subsets.sum(axis=1)
        centered = subsets - sizes[:, None] / self.num_player
        residual = ues - ((self.constants[1] - self.constants[0]) / self.num_player) * sizes

        A_tmp = centered.T @ (weights[:, None] * centered)
        b_tmp = centered.T @ (weights * residual)

        num_pre = self.results_aggregate["count"]
        num_cur = len(inputs) + num_pre
        self.results_aggregate["mat_A"] *= num_pre / num_cur
        self.results_aggregate["mat_A"] += A_tmp / num_cur
        self.results_aggregate["vec_b"] *= num_pre / num_cur
        self.results_aggregate["vec_b"] += b_tmp / num_cur
        self.results_aggregate["count"] = num_cur

    def _estimate(self):
        if self.results_aggregate["count"] == 0:
            raise RuntimeError("LeverageSHAP did not collect any sampled coalitions")
        tmp = np.linalg.lstsq(
            self.results_aggregate["mat_A"],
            self.results_aggregate["vec_b"],
            rcond=None,
        )[0]
        tmp -= tmp.mean()
        return tmp + (self.constants[1] - self.constants[0]) / self.num_player


LeverageSHAP_original = LeverageSHAP


class LeverageSHAP_border(LeverageSHAP):
    """
    LeverageSHAP with the PolySHAP/shapiq border trick sampler.

    This keeps the first-order LeverageSHAP WLS estimator from ``LeverageSHAP``
    and uses the PolySHAP border-trick configuration with
    ``sampling_weights=np.ones(n + 1)``: empty/full are handled as
    constants, low/high saturated interior sizes are enumerated exactly, and
    the remaining interior budget is sampled without replacement with
    complement pairing.
    """

    def __init__(self, *, sampling_with_replacement=False, **kwargs):
        if sampling_with_replacement:
            raise ValueError("LeverageSHAP_border uses border-trick sampling without replacement.")
        super(LeverageSHAP_border, self).__init__(
            sampling_with_replacement=False,
            **kwargs,
        )

    def _init_indiv(self):
        assert self.semivalue == "shapley"
        if self.num_player < 2:
            raise ValueError("LeverageSHAP requires at least two players")
        if self.num_sample <= 0:
            raise ValueError("LeverageSHAP requires a budget larger than two utility evaluations")

    def _sample_without_replacement(self):
        yield from self._sample_border_trick()

    def _polyshap_leverage_kernel_weight(self, size):
        return 1.0 / (
            (self.num_player - 1)
            * float(special.comb(self.num_player - 2, int(size) - 1, exact=False))
        )

    def _sample_border_trick(self):
        n = self.num_player
        max_interior = max(0, 2 ** n - 2)
        remaining_budget = int(min(self.num_sample, max_interior))
        if remaining_budget <= 0:
            return

        sizes_to_sample = list(range(1, n))
        # Matches the PolySHAP experiment override for LeverageSHAP
        # (sampling_weights=np.ones(n + 1)), not shapiq's default sampler.
        adjusted_weights = np.ones(len(sizes_to_sample), dtype=np.float64)
        adjusted_weights /= adjusted_weights.sum()
        sizes_to_compute = []

        while remaining_budget > 0 and sizes_to_sample:
            binom_counts = np.array(
                [float(special.comb(n, size, exact=False)) for size in sizes_to_sample],
                dtype=np.float64,
            )
            expected_counts = remaining_budget * adjusted_weights
            move_mask = expected_counts >= binom_counts
            if not move_mask.any():
                break

            moved_sizes = [
                size for size, should_move in zip(sizes_to_sample, move_mask)
                if should_move
            ]
            sizes_to_compute.extend(moved_sizes)
            remaining_budget -= sum(math.comb(n, size) for size in moved_sizes)
            remaining_budget = max(0, remaining_budget)

            keep_mask = ~move_mask
            sizes_to_sample = [
                size for size, keep in zip(sizes_to_sample, keep_mask)
                if keep
            ]
            if sizes_to_sample:
                adjusted_weights = adjusted_weights[keep_mask]
                adjusted_weights /= adjusted_weights.sum()

        for size in sorted(sizes_to_compute, key=lambda value: -abs(n / 2 - value)):
            weight = self._polyshap_leverage_kernel_weight(size)
            for combo in itertools.combinations(range(n), size):
                yield self._row_from_indices(combo, weight)

        if remaining_budget <= 0 or not sizes_to_sample:
            return

        sampled_coalitions = []
        sampled_set = set()
        size_prob = {
            int(size): float(prob)
            for size, prob in zip(sizes_to_sample, adjusted_weights)
        }

        def add_coalition(coalition):
            coalition = tuple(sorted(int(idx) for idx in coalition))
            if coalition in sampled_set:
                return False
            sampled_set.add(coalition)
            sampled_coalitions.append(coalition)
            return True

        sizes_arr = np.asarray(sizes_to_sample, dtype=np.int64)
        stalled_draws = 0
        while remaining_budget > 0:
            size = int(self._rng.choice(sizes_arr, p=adjusted_weights))
            coalition = tuple(sorted(self._rng.choice(n, size=size, replace=False).tolist()))
            if add_coalition(coalition):
                remaining_budget -= 1
                stalled_draws = 0
            else:
                stalled_draws += 1

            if remaining_budget > 0 and (n - size) in size_prob:
                coalition_set = set(coalition)
                complement = tuple(idx for idx in range(n) if idx not in coalition_set)
                if add_coalition(complement):
                    remaining_budget -= 1
                    stalled_draws = 0
                else:
                    stalled_draws += 1

            if stalled_draws > max(1000, 10 * len(sampled_coalitions), 10 * remaining_budget):
                for fallback_size in sizes_to_sample:
                    for fallback in itertools.combinations(range(n), fallback_size):
                        if add_coalition(fallback):
                            remaining_budget -= 1
                            if remaining_budget == 0:
                                break
                    if remaining_budget == 0:
                        break
                stalled_draws = 0

        n_sampled = len(sampled_coalitions)
        for coalition in sampled_coalitions:
            size = len(coalition)
            binom_count = float(special.comb(n, size, exact=False))
            sampling_adjustment = binom_count / (size_prob[size] * n_sampled)
            weight = self._polyshap_leverage_kernel_weight(size) * sampling_adjustment
            yield self._row_from_indices(coalition, weight)

    def _row_from_indices(self, indices, weight):
        mask = np.zeros(self.num_player, dtype=bool)
        mask[np.asarray(indices, dtype=np.int64)] = True
        return self._row_from_mask(mask, weight)


class leverage(kernelSHAP):
    def __init__(self, **kwargs):
        super(leverage, self).__init__(**kwargs)
        self.results_aggregate = dict(mat_A=np.zeros((self.num_player, self.num_player), dtype=np.float64),
                                      vec_b=np.zeros(self.num_player, dtype=np.float64),
                                      count=0)


    def _init_indiv(self):
        super(leverage, self)._init_indiv()
        self.weights = np.sqrt(self.weights * self.num_player)


    def _generator(self):
        s = np.random.choice(self.s_range)
        pos = np.random.choice(self.pos_range, size=s, replace=False)
        subset = np.zeros(self.num_player, dtype=bool)
        subset[pos] = True
        return subset

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player]
        ues = inputs[:, -1] - self.constants[0]

        sizes = subsets.sum(axis=1)
        vec_tmp = sizes / self.num_player
        subsets -= vec_tmp[:, None]
        weights_tmp = self.weights[sizes.astype(np.int64) - 1]
        subsets = weights_tmp[:, None] * subsets
        A_tmp = subsets.T @ subsets
        b_tmp = ues - ((self.constants[1] - self.constants[0]) / self.num_player) * sizes
        b_tmp = np.multiply(weights_tmp, b_tmp)
        b_tmp = np.dot(b_tmp, subsets)

        num_pre = self.results_aggregate["count"]
        num_cur = len(inputs) + num_pre
        self.results_aggregate["mat_A"] *= num_pre / num_cur
        self.results_aggregate["mat_A"] += A_tmp / num_cur
        self.results_aggregate["vec_b"] *= num_pre / num_cur
        self.results_aggregate["vec_b"] += b_tmp / num_cur
        self.results_aggregate["count"] = num_cur

    def _estimate(self):
        # A_inv = np.linalg.pinv(self.results_aggregate["mat_A"])
        tmp, _, _, _ = lstsq(self.results_aggregate["mat_A"], self.results_aggregate["vec_b"], lapack_driver='gelsy',
                    check_finite=False)
        return tmp + (self.constants[1] - self.constants[0]) / self.num_player

class leverage_paired(leverage):
    def __init__(self, **kwargs):
        super(leverage_paired, self).__init__(**kwargs)
        self.lock_switch = False


class modified_leverage(kernelSHAP):
    def __init__(self, **kwargs):
        super(modified_leverage, self).__init__(**kwargs)
        self.results_aggregate = dict(vec_b=np.zeros(self.num_player, dtype=np.float64),
                                      count=0)


    def _init_indiv(self):
        assert self.semivalue == "shapley"

        tmp = np.arange(1, self.num_player, dtype=np.float64)
        self.weights = (self.num_player - 1) / np.multiply(tmp, tmp[::-1])
        self.s_range = np.arange(1, self.num_player)
        self.pos_range = np.arange(self.num_player)


    def _generator(self):
        s = np.random.choice(self.s_range)
        pos = np.random.choice(self.pos_range, size=s, replace=False)
        subset = np.zeros(self.num_player, dtype=bool)
        subset[pos] = True
        return subset

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player]
        ues = inputs[:, -1] - self.constants[0]

        sizes = subsets.sum(axis=1)
        vec_tmp = sizes / self.num_player
        subsets -= vec_tmp[:, None]
        b_tmp = ues - ((self.constants[1] - self.constants[0]) / self.num_player) * sizes
        b_tmp = np.multiply(self.weights[sizes.astype(np.int64) - 1], b_tmp)
        b_tmp = np.dot(b_tmp, subsets)

        num_pre = self.results_aggregate["count"]
        num_cur = len(inputs) + num_pre
        self.results_aggregate["vec_b"] *= num_pre / num_cur
        self.results_aggregate["vec_b"] += b_tmp / num_cur
        self.results_aggregate["count"] = num_cur

    def _estimate(self):
        vec = self.results_aggregate["vec_b"]
        tmp = self.num_player * vec - vec.sum()
        return tmp + (self.constants[1] - self.constants[0]) / self.num_player


class modified_leverage_paired(modified_leverage):
    def __init__(self, **kwargs):
        super(modified_leverage_paired, self).__init__(**kwargs)
        self.lock_switch = False


class test_leverage(kernelSHAP):
    def __init__(self, **kwargs):
        super(test_leverage, self).__init__(**kwargs)
        self.results_aggregate = dict(vec_b=np.zeros(self.num_player, dtype=np.float64),
                                      count=0)


    def _init_indiv(self):
        assert self.semivalue == "shapley"

        tmp = np.arange(1, self.num_player, dtype=np.float64)
        tmp = 1 / np.multiply(tmp, tmp[::-1])
        tmp_sum = tmp.sum()
        self.weights = tmp_sum * np.ones(self.num_player - 1, dtype=np.float64)
        self.p = tmp / tmp_sum
        self.s_range = np.arange(1, self.num_player)
        self.pos_range = np.arange(self.num_player)


    def _generator(self):
        s = np.random.choice(self.s_range, p=self.p)
        pos = np.random.choice(self.pos_range, size=s, replace=False)
        subset = np.zeros(self.num_player, dtype=bool)
        subset[pos] = True
        return subset

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player]
        ues = inputs[:, -1] - self.constants[0]

        sizes = subsets.sum(axis=1)
        vec_tmp = sizes / self.num_player
        subsets -= vec_tmp[:, None]
        b_tmp = ues - ((self.constants[1] - self.constants[0]) / self.num_player) * sizes
        b_tmp = np.multiply(self.weights[sizes.astype(np.int64) - 1], b_tmp)
        b_tmp = np.dot(b_tmp, subsets)

        num_pre = self.results_aggregate["count"]
        num_cur = len(inputs) + num_pre
        self.results_aggregate["vec_b"] *= num_pre / num_cur
        self.results_aggregate["vec_b"] += b_tmp / num_cur
        self.results_aggregate["count"] = num_cur

    def _estimate(self):
        vec = self.results_aggregate["vec_b"]
        tmp = self.num_player * vec - vec.sum()
        return tmp + (self.constants[1] - self.constants[0]) / self.num_player

class test_leverage_paired(test_leverage):
    def __init__(self, **kwargs):
        super(test_leverage_paired, self).__init__(**kwargs)
        self.lock_switch = False


class unbiased_kernelSHAP(kernelSHAP):
    def __init__(self, **kwargs):
        super(MSR, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        self.results_aggregate = dict(estimates=np.zeros(self.num_player, dtype=np.float64), count=0)
        self.buffer = np.empty((self.buffer_size, self.num_player + 1), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player), dtype=bool)

        with mp.Pool(1) as pool:
            self.constants = pool.apply(self.calculate_constants, (self.game_func, self.game_args, self.num_player))
        self.scalar = 2 * np.reciprocal(np.arange(1, self.num_player, dtype=np.float64)).sum()

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player]
        ues = inputs[:, [-1]]
        tmp = (subsets - subsets.sum(axis=1, keepdims=True) / self.num_player) * (ues - self.constants[0])

        num_pre = self.results_aggregate["count"]
        num_cur = len(tmp) + num_pre
        self.results_aggregate["estimates"] *= num_pre / num_cur
        self.results_aggregate["estimates"] += tmp.sum(axis=0) / num_cur
        self.results_aggregate["count"] = num_cur

    def _estimate(self):
        return (self.constants[1] - self.constants[0]) / self.num_player + \
               self.results_aggregate["estimates"] * self.scalar


class unbiased_kernelSHAP_paired(unbiased_kernelSHAP, kernelSHAP_paired):
    def __init__(self, **kwargs):
        super(unbiased_kernelSHAP_paired, self).__init__(**kwargs)
        self.lock_switch = False


class ARM(MSR):
    def __init__(self, **kwargs):
        super(MSR, self).__init__(**kwargs)
        self.num_sample = (self.nue_avg * self.num_player) // 2
        self.interval_track = (self.nue_track_avg * self.num_player) // 2
        self.batch_size = -(-self.nue_per_proc // 2)
        self.nue_per_proc_run = self.batch_size * 2

        self.results_aggregate = np.zeros((4, self.num_player), dtype=np.float64)
        self.buffer = np.empty((self.buffer_size, 2, self.num_player + 1), dtype=np.float64)
        self.samples = np.empty((self.batch_size, 2, self.num_player), dtype=bool)

    def _init_indiv(self):
        assert (self.nue_avg * self.num_player) % 2 == 0
        assert (self.nue_track_avg * self.num_player) % 2 == 0

        weight = self.distribution_cardinality()
        weight_left = np.divide(weight, np.arange(1, self.num_player + 1))
        self.weight_left = weight_left / weight_left.sum()
        weight_right = np.divide(weight, np.arange(self.num_player, 0, -1))
        self.weight_right = weight_right / weight_right.sum()

        self.s_range_left = np.arange(1, self.num_player + 1)
        self.s_range_right = np.arange(self.num_player)
        self.pos_range = np.arange(self.num_player)

    def _generator(self):
        subset = np.zeros((2, self.num_player), dtype=bool)
        s = np.random.choice(self.s_range_left, p=self.weight_left)
        pos_left = np.random.choice(self.pos_range, size=s, replace=False)
        s = np.random.choice(self.s_range_right, p=self.weight_right)
        pos_right = np.random.choice(self.pos_range, size=s, replace=False)
        subset[0, pos_left] = True
        subset[1, pos_right] = True
        return subset

    def run(self, samples):
        game = self.game_func(**self.game_args)
        results_collect = np.empty((len(samples), 2, self.num_player + 1), dtype=np.float64)
        results_collect[:, :, :self.num_player] = samples
        for i, sample in enumerate(samples):
            results_collect[i, 0, -1] = game.evaluate(sample[0])
            results_collect[i, 1, -1] = game.evaluate(sample[1])
        return results_collect

    def _process(self, inputs):
        subsets = inputs[:, 0, :self.num_player]
        ues = inputs[:, 0, [-1]]
        self.results_aggregate[0] += (ues * subsets).sum(axis=0)
        self.results_aggregate[1] += subsets.sum(axis=0)
        subsets = 1 - inputs[:, 1, :self.num_player]
        ues = inputs[:, 1, [-1]]
        self.results_aggregate[2] += (ues * subsets).sum(axis=0)
        self.results_aggregate[3] += subsets.sum(axis=0)


class ARM_shapley(ARM):
    def __init__(self, **kwargs):
        super(ARM_shapley, self).__init__(**kwargs)

    def _init_indiv(self):
        assert self.semivalue != "shapley"
        assert (self.nue_avg * self.num_player) % 2 == 0
        assert (self.nue_track_avg * self.num_player) % 2 == 0

        weight_left = self.num_player / np.arange(1, self.num_player + 1)
        self.weight_left = weight_left / weight_left.sum()
        self.weight_right = self.weight_left[::-1]

        self.s_range_left = np.arange(1, self.num_player + 1)
        self.s_range_right = np.arange(self.num_player)
        self.pos_range = np.arange(self.num_player)

        self.weights = self.distribution_cardinality() * self.num_player

    def run(self, samples):
        game = self.game_func(**self.game_args)
        results_collect = np.empty((len(samples), 2, self.num_player + 1), dtype=np.float64)
        results_collect[:, :, :self.num_player] = samples
        for i, sample in enumerate(samples):
            subset = sample[0]
            results_collect[i, 0, -1] = game.evaluate(subset) * self.weights[subset.sum() - 1]
            subset = sample[1]
            results_collect[i, 1, -1] = game.evaluate(subset) * self.weights[subset.sum()]
        return results_collect


class ARM_banzhaf(ARM_shapley):
    def __init__(self, **kwargs):
        super(ARM_shapley, self).__init__(**kwargs)
        tmp = 2 ** (self.num_player - 1)
        vs = self.distribution_cardinality()
        self.weights = np.array([tmp / special.binom(self.num_player - 1, s) * vs[s] for s in range(self.num_player)])

    def _init_indiv(self):
        # assert not (self.semivalue == "weighted_banzhaf" and self.semivalue_param == 0.5)
        assert (self.nue_avg * self.num_player) % 2 == 0
        assert (self.nue_track_avg * self.num_player) % 2 == 0

        weights = np.ones(self.num_player, dtype=np.float64)
        for k in range(self.num_player):
            for i in range(k):
                weights[k] *= (self.num_player - 1 - i) / (i + 1) * 0.5**2
            weights[k] *= 0.5 ** (self.num_player - 1 - 2 * k)

        weight_left = np.divide(weights, np.arange(1, self.num_player + 1))
        self.weight_left = weight_left / weight_left.sum()
        weight_right = np.divide(weights, np.arange(self.num_player, 0, -1))
        self.weight_right = weight_right / weight_right.sum()

        self.s_range_left = np.arange(1, self.num_player + 1)
        self.s_range_right = np.arange(self.num_player)
        self.pos_range = np.arange(self.num_player)

    def run(self, samples):
        return super(ARM_banzhaf, self).run(samples)


class complement(estimatorTemplate):
    def __init__(self, **kwargs):
        super(complement, self).__init__(**kwargs)
        self.num_sample = (self.nue_avg * self.num_player) // 2
        self.interval_track = (self.nue_track_avg * self.num_player) // 2
        self.batch_size = -(-self.nue_per_proc // 2)
        self.nue_per_proc_run = self.batch_size * 2

        self.results_aggregate = np.zeros((2, self.num_player, self.num_player), dtype=np.float64)
        self.buffer = np.empty((self.buffer_size, self.num_player + 1), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player), dtype=bool)

    def _init_indiv(self):
        assert self.semivalue == "shapley"
        assert (self.nue_avg * self.num_player) % 2 == 0
        assert (self.nue_track_avg * self.num_player) % 2 == 0

        self.s_range = np.arange(1, self.num_player + 1)

    def _generator(self):
        subset = np.zeros(self.num_player, dtype=bool)
        s = np.random.choice(self.s_range)
        pi = np.random.permutation(self.num_player)
        subset[pi[:s]] = True
        return subset

    def run(self, samples):
        game = self.game_func(**self.game_args)
        results_collect = np.zeros((len(samples), self.num_player + 1), dtype=np.float64)
        results_collect[:, :self.num_player] = samples
        for i, sample in enumerate(samples):
            results_collect[i, -1] += game.evaluate(sample)
            results_collect[i, -1] -= game.evaluate(~sample)
        return results_collect

    def _process(self, inputs):
        for take in inputs:
            subset = take[:self.num_player].astype(bool)
            subset_c = ~subset
            v = take[-1]
            subset_size = subset.sum()
            self.results_aggregate[0, subset, subset_size - 1] += v
            self.results_aggregate[0, subset_c, self.num_player - subset_size - 1] -= v
            self.results_aggregate[1, subset, subset_size - 1] += 1
            self.results_aggregate[1, subset_c, self.num_player - subset_size - 1] += 1

    def _estimate(self):
        # what in the below seems to fail occasionally, it returns nan for some entry while it should be a real number.
        # tmp = np.divide(self.results_aggregate[0], self.results_aggregate[1], where=self.results_aggregate[1] != 0)
        # return tmp.mean(axis=1)
        counts = self.results_aggregate[1].copy()
        counts[counts == 0] = -1
        return np.mean(np.divide(self.results_aggregate[0], counts), axis=1)


class AME(estimatorTemplate):
    def __init__(self, **kwargs):
        super(AME, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        self.results_aggregate = dict(mat_A=np.zeros((self.num_player, self.num_player), dtype=np.float64),
                                      vec_b=np.zeros(self.num_player, dtype=np.float64))
        self.buffer = np.empty((self.buffer_size, self.num_player + 2), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player + 1), dtype=np.float64)

    def _init_indiv(self):
        if self.semivalue == "weighted_banzhaf":
            assert 0 < self.semivalue_param and self.semivalue_param < 1
            self.variance = 1 / self.semivalue_param / (1 - self.semivalue_param)
        elif self.semivalue == "beta_shapley":
            assert 1 < self.semivalue_param[0] and 1 < self.semivalue_param[1]
            alpha, beta = self.semivalue_param
            ab = alpha + beta
            self.variance = (ab - 1) * (ab - 2) / (alpha - 1) / (beta - 1)
        else:
            raise NotImplementedError

    def _generator(self):
        sample = np.empty(self.num_player + 1, dtype=np.float64)
        if self.semivalue == "weighted_banzhaf":
            prob = self.semivalue_param
        elif self.semivalue == "beta_shapley":
            prob = np.random.beta(self.semivalue_param[1], self.semivalue_param[0])
        else:
            raise NotImplementedError
        sample[:-1] = np.random.binomial(1, prob, size=self.num_player)
        sample[-1] = prob
        return sample

    def run(self, samples):
        game = self.game_func(**self.game_args)
        results_collect = np.zeros((len(samples), self.num_player + 2), dtype=np.float64)
        results_collect[:, :-1] = samples
        for i, sample in enumerate(samples):
            subset = sample[:self.num_player].astype(bool)
            results_collect[i, -1] = game.evaluate(subset)
        return results_collect

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player]
        ps = inputs[:, [-2]]
        ues = inputs[:, [-1]]
        tmp = subsets * (1 / ps) - (1 - subsets) * (1 / (1 - ps))
        self.results_aggregate["mat_A"] += tmp.T @ tmp
        self.results_aggregate["vec_b"] += (ues * tmp).sum(axis=0)

    def _estimate(self):
        return self.variance * (np.linalg.pinv(self.results_aggregate["mat_A"]) @ self.results_aggregate["vec_b"])


class AME_paired(AME):
    def __init__(self, **kwargs):
        super(AME_paired, self).__init__(**kwargs)
        self.lock_switch = False

    def _init_indiv(self):
        super(AME_paired, self)._init_indiv()
        if self.semivalue == "weighted_banzhaf":
            assert self.semivalue_param == 0.5
        if self.semivalue == "beta_shapley":
            assert self.semivalue_param[0] == self.semivalue_param[1]


class group_testing(sampling_lift):
    def __init__(self, **kwargs):
        super(sampling_lift, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        self.results_aggregate = dict(estimates=np.zeros(self.num_player, dtype=np.float64), count=0)
        self.buffer = np.empty((self.buffer_size, self.num_player), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player + 1), dtype=bool)

    def _init_indiv(self):
        assert self.semivalue == "shapley"

        tmp = 1 / np.arange(1, self.num_player + 1, dtype=np.float64)
        weights = tmp + tmp[::-1]
        self.const = weights.sum()
        self.weights = weights / self.const
        self.s_range = np.arange(1, self.num_player + 1)
        self.pos_range = np.arange(self.num_player + 1)

    def _generator(self):
        subset = np.zeros(self.num_player + 1, dtype=bool)
        s = np.random.choice(self.s_range, p=self.weights)
        pos = np.random.choice(self.pos_range, size=s, replace=False)
        subset[pos] = True
        return subset

    def run(self, samples):
        game = self.game_func(**self.game_args)
        results_collect = np.zeros((len(samples), self.num_player), dtype=np.float64)
        for i, sample in enumerate(samples):
            tmp = sample * game.evaluate(sample)
            results_collect[i] = tmp[:self.num_player] - tmp[-1]
        return results_collect * self.const


class group_testing_paired(group_testing):
    def __init__(self, **kwargs):
        super(group_testing_paired, self).__init__(**kwargs)
        self.lock_switch = False


class GELS_ranking(kernelSHAP):
    def __init__(self, **kwargs):
        super(MSR, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        self.results_aggregate = np.zeros((2, self.num_player), dtype=np.float64)
        self.buffer = np.empty((self.buffer_size, self.num_player + 1), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player), dtype=bool)

    def _init_indiv(self):
        self.num_player -= 1
        weights = self.distribution_cardinality()
        self.num_player += 1
        tmp = np.arange(1, self.num_player, dtype=np.float64)
        tmp = np.multiply(tmp / self.num_player, (self.num_player - tmp) / (self.num_player - 1))
        tmp = np.reciprocal(tmp)
        weights = np.multiply(weights, tmp)
        self.weights = weights / weights.sum()
        self.s_range = np.arange(1, self.num_player)
        self.pos_range = np.arange(self.num_player)

    def _generator(self):
        return super(GELS_ranking, self)._generator()

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player]
        ues = inputs[:, [-1]]
        self.results_aggregate[0] += (ues * subsets).sum(axis=0)
        self.results_aggregate[1] += subsets.sum(axis=0)

    def _estimate(self):
        counts = self.results_aggregate[1].copy()
        counts[counts == 0] = -1
        return np.divide(self.results_aggregate[0], counts)


class GELS_ranking_paired(GELS_ranking):
    def __init__(self, **kwargs):
        super(GELS_ranking_paired, self).__init__(**kwargs)
        self.lock_switch = False

    def _init_indiv(self):
        super(GELS_ranking_paired, self)._init_indiv()
        if self.semivalue == "weighted_banzhaf":
            assert self.semivalue_param == 0.5
        if self.semivalue == "beta_shapley":
            assert self.semivalue_param[0] == self.semivalue_param[1]


class GELS(GELS_ranking):
    def __init__(self, **kwargs):
        super(MSR, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        weights = self.distribution_cardinality()
        self.scalar = (np.divide(weights, np.arange(self.num_player, 0, -1)) * self.num_player).sum()
        self.num_player += 1

        self.results_aggregate = np.zeros((2, self.num_player), dtype=np.float64)
        self.buffer = np.empty((self.buffer_size, self.num_player + 1), dtype=np.float64)
        self.samples = np.zeros((self.batch_size, self.num_player), dtype=bool)

    def _estimate(self):
        estimates = super(GELS, self)._estimate() * self.scalar
        return estimates[:-1] - estimates[-1]


class GELS_paired(GELS, GELS_ranking_paired):
    # For the Shapley value, this estimator is equal to group_testing_paired
    def __init__(self, **kwargs):
        super(GELS_paired, self).__init__(**kwargs)
        self.lock_switch = False

    def _init_indiv(self):
        GELS_ranking_paired._init_indiv(self)


class WGELS_shapley(GELS):
    def __init__(self, **kwargs):
        super(MSR, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        self.scalar = (1 / np.arange(self.num_player, 0, -1)).sum()
        self.reweights = self.distribution_cardinality() * self.num_player
        self.num_player += 1

        self.results_aggregate = np.zeros((2, self.num_player), dtype=np.float64)
        self.buffer = np.empty((self.buffer_size, self.num_player + 1), dtype=np.float64)
        self.samples = np.zeros((self.batch_size, self.num_player), dtype=bool)


    def _init_indiv(self):
        assert self.semivalue != "shapley"
        tmp = np.arange(1, self.num_player, dtype=np.float64)
        tmp = np.multiply(tmp / self.num_player, (self.num_player - tmp) / (self.num_player - 1))
        tmp = np.reciprocal(tmp)
        weights = tmp / (self.num_player - 1)
        self.weights = weights / weights.sum()
        self.s_range = np.arange(1, self.num_player)
        self.pos_range = np.arange(self.num_player)

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player]
        ues = inputs[:, [-1]]
        last_player = subsets[:, -1].astype(bool)
        sizes = subsets[:, :-1].sum(axis=1).astype(np.int64)
        weights = np.array([self.reweights[s] if pre else self.reweights[s-1] for (pre, s) in zip(last_player, sizes)])
        self.results_aggregate[0] += (ues * subsets * weights[:, None]).sum(axis=0)
        self.results_aggregate[1] += subsets.sum(axis=0)


class WGELS_shapley_paired(WGELS_shapley):
    def __init__(self, **kwargs):
        super(WGELS_shapley_paired, self).__init__(**kwargs)
        self.lock_switch = False


class WGELS_banzhaf(WGELS_shapley):
    def __init__(self, **kwargs):
        super(MSR, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        self.banzhaf_weights = np.ones(self.num_player, dtype=np.float64)
        for k in range(self.num_player):
            for i in range(k):
                self.banzhaf_weights[k] *= (self.num_player - 1 - i) / (i + 1) * 0.5 ** 2
            self.banzhaf_weights[k] *= 0.5 ** (self.num_player - 1 - 2 * k)
        self.scalar = (np.divide(self.banzhaf_weights, np.arange(self.num_player, 0, -1)) * self.num_player).sum()

        weights = self.distribution_cardinality()
        tmp = 2**(self.num_player - 1)
        self.reweights = np.array([tmp / special.binom(self.num_player - 1, s) * weights[s] for s in range(self.num_player)])
        self.num_player += 1
        self.results_aggregate = np.zeros((2, self.num_player), dtype=np.float64)
        self.buffer = np.empty((self.buffer_size, self.num_player + 1), dtype=np.float64)
        self.samples = np.zeros((self.batch_size, self.num_player), dtype=bool)

    def _init_indiv(self):
        assert not (self.semivalue == "weighted_banzhaf" and self.semivalue_param == 0.5)
        tmp = np.arange(1, self.num_player, dtype=np.float64)
        tmp = np.multiply(tmp / self.num_player, (self.num_player - tmp) / (self.num_player - 1))
        tmp = np.reciprocal(tmp)
        weights = np.multiply(self.banzhaf_weights, tmp)
        self.weights = weights / weights.sum()
        self.s_range = np.arange(1, self.num_player)
        self.pos_range = np.arange(self.num_player)


class WGELS_banzhaf_paired(WGELS_banzhaf):
    def __init__(self, **kwargs):
        super(WGELS_banzhaf_paired, self).__init__(**kwargs)
        self.lock_switch = False


class GELS_shapley(GELS_ranking):
    def __init__(self, **kwargs):
        super(GELS_shapley, self).__init__(**kwargs)
        with mp.Pool(1) as pool:
            self.constants = pool.apply(self.calculate_constants, (self.game_func, self.game_args, self.num_player))

    def _init_indiv(self):
        assert self.semivalue == "shapley"

        tmp = 1 / np.arange(1, self.num_player, dtype=np.float64)
        weights = np.multiply(tmp, tmp[::-1])
        self.weights = weights / weights.sum()
        self.scalar = tmp.sum()
        self.s_range = np.arange(1, self.num_player)
        self.pos_range = np.arange(self.num_player)

    def _estimate(self):
        estimates = super(GELS_shapley, self)._estimate() * self.scalar
        offset = (self.constants[1] - self.constants[0] - estimates.sum()) / self.num_player
        return estimates + offset


class GELS_shapley_paired(GELS_shapley):
    # This estimator is equal to unbiased_kernelSHAP_paired
    def __init__(self, **kwargs):
        super(GELS_shapley_paired, self).__init__(**kwargs)
        self.lock_switch = False


class simSHAP(kernelSHAP):
    def __init__(self, **kwargs):
        super(MSR, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        self.results_aggregate = dict(estimates=np.zeros(self.num_player, dtype=np.float64), count=0)
        self.buffer = np.empty((self.buffer_size, self.num_player + 1), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player), dtype=bool)

        with mp.Pool(1) as pool:
            self.constants = pool.apply(self.calculate_constants, (self.game_func, self.game_args, self.num_player))

    def _init_indiv(self):
        assert self.semivalue == "shapley"

        tmp = np.arange(1, self.num_player, dtype=np.float64)
        weights = 1 / np.multiply(tmp, tmp[::-1])
        self.gamma = weights.sum()
        self.weights = weights / self.gamma
        self.s_range = np.arange(1, self.num_player)
        self.pos_range = np.arange(self.num_player)

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player]
        ues = inputs[:, [-1]]
        sizes = subsets.sum(axis=1, keepdims=True)

        tmp = ((self.num_player - sizes) * subsets - sizes * (1 - subsets)) * ues
        num_pre = self.results_aggregate["count"]
        num_cur = num_pre + ues.shape[0]
        self.results_aggregate["estimates"] *= num_pre / num_cur
        self.results_aggregate["estimates"] += tmp.sum(axis=0) / num_cur
        self.results_aggregate["count"] = num_cur

    def _estimate(self):
        return self.results_aggregate["estimates"] * self.gamma \
               + (self.constants[1] - self.constants[0]) / self.num_player


class simSHAP_paired(simSHAP):
    def __init__(self, **kwargs):
        super(simSHAP_paired, self).__init__(**kwargs)
        self.lock_switch = False


class OFA(MSR):
    @staticmethod
    def calculate_constants(game_func, game_args, num_player):
        game = game_func(**game_args)
        subset = np.zeros(num_player, dtype=bool)
        v_empty = game.evaluate(subset)
        v_singleton = np.empty(num_player, dtype=np.float64)
        for i in range(num_player):
            subset[i] = True
            v_singleton[i] = game.evaluate(subset)
            subset[i] = False

        subset.fill(True)
        v_full = game.evaluate(subset)
        v_remove = np.empty(num_player, dtype=np.float64)
        for i in range(num_player):
            subset[i] = False
            v_remove[i] = game.evaluate(subset)
            subset[i] = True

        return v_empty, v_full, v_singleton, v_remove

    def __init__(self, **kwargs):
        super(MSR, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        self.results_aggregate = dict(estimates=np.zeros((self.num_player, self.num_player - 3, 2), dtype=np.float64),
                                      counts=np.zeros((self.num_player, self.num_player - 3, 2), dtype=np.int64))
        self.buffer = np.empty((self.buffer_size, self.num_player + 1), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player), dtype=bool)

        with mp.Pool(1) as pool:
            self.constants = pool.apply(self.calculate_constants, (self.game_func, self.game_args, self.num_player))
        self.weights = self.distribution_cardinality()

        self.p_sampling = None

    def _init_indiv(self):
        self.s_range = np.arange(2, self.num_player - 1)
        self.pos_range = np.arange(self.num_player)


    def _generator(self):
        subset = np.zeros(self.num_player, dtype=bool)
        if self.p_sampling is None:
            s = np.random.choice(self.s_range)
        else:
            s = np.random.choice(self.s_range, p=self.p_sampling)
        pos = np.random.choice(self.pos_range, size=s, replace=False)
        subset[pos] = True
        return subset

    def _process(self, inputs):
        for take in inputs:
            subset = take[:self.num_player].astype(bool)
            subset_c = ~subset
            v = take[-1]
            idx = subset.sum() - 2
            counts_pre = self.results_aggregate["counts"][subset, idx, 0]
            counts_cur = counts_pre + 1
            self.results_aggregate["estimates"][subset, idx, 0] *= counts_pre / counts_cur
            self.results_aggregate["estimates"][subset, idx, 0] += v / counts_cur
            self.results_aggregate["counts"][subset, idx, 0] += 1

            counts_pre = self.results_aggregate["counts"][subset_c, idx, 1]
            counts_cur = counts_pre + 1
            self.results_aggregate["estimates"][subset_c, idx, 1] *= counts_pre / counts_cur
            self.results_aggregate["estimates"][subset_c, idx, 1] += v / counts_cur
            self.results_aggregate["counts"][subset_c, idx, 1] += 1

    def _estimate(self):
        tmp = (self.results_aggregate["estimates"][:, :, 0] * self.weights[None, 1:self.num_player - 2]).sum(axis=1)
        tmp += self.constants[1] * self.weights[-1]
        tmp += self.constants[2] * self.weights[0]
        tmp += (self.constants[3].sum() - self.constants[3]) * self.weights[-2] / (self.num_player - 1)

        tmp -= (self.results_aggregate["estimates"][:, :, 1] * self.weights[None, 2:self.num_player - 1]).sum(axis=1)
        tmp -= self.constants[0] * self.weights[0]
        tmp -= self.constants[3] * self.weights[-1]
        tmp -= (self.constants[2].sum() - self.constants[2]) * self.weights[1] / (self.num_player - 1)
        return tmp


class OFA_optimal(OFA):
    def __init__(self, **kwargs):
        super(OFA_optimal, self).__init__(**kwargs)
        assert self.semivalue != "shapley"
        tmp = self.num_player / np.arange(2, self.num_player - 1)
        tmp = np.sqrt(tmp + tmp[::-1])
        self.p_sampling = tmp / tmp.sum()


class OFA_optimal_paired(OFA_optimal):
    def __init__(self, **kwargs):
        super(OFA_optimal_paired, self).__init__(**kwargs)
        self.lock_switch = False


class OFA_fixed(OFA):
    def __init__(self, **kwargs):
        super(OFA_fixed, self).__init__(**kwargs)
        weights = self.distribution_cardinality()
        tmp = weights[1:self.num_player - 2]**2 / np.arange(2, self.num_player - 1)
        tmp += weights[2:self.num_player - 1]**2 / np.arange(self.num_player - 2, 1, -1)
        tmp = tmp**0.5
        self.p_sampling = tmp / tmp.sum()


class OFA_fixed_paired(OFA_fixed):
    def __init__(self, **kwargs):
        super(OFA_fixed_paired, self).__init__(**kwargs)
        self.lock_switch = False

        if self.semivalue == "weighted_banzhaf":
            assert self.semivalue_param == 0.5
        if self.semivalue == "beta_shapley":
            assert self.semivalue_param[0] == self.semivalue_param[1]


class OFA_baseline(estimatorTemplate):
    def __init__(self, **kwargs):
        super(OFA_baseline, self).__init__(**kwargs)
        assert self.nue_avg % 2 == 0
        self.num_sample = self.nue_avg // 2
        assert self.nue_track_avg % 2 == 0
        self.interval_track = self.nue_track_avg // 2
        self.batch_size = -(-self.nue_per_proc // (self.num_player * 2))
        self.nue_per_proc_run = self.batch_size * self.num_player * 2

        self.results_aggregate = np.zeros((self.num_player, self.num_player), dtype=np.float64)
        self.count = np.zeros(self.num_player, dtype=np.int64)
        self.buffer = np.empty((self.buffer_size, self.num_player + 1), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player), dtype=np.int64)

        self.weights = self.distribution_cardinality()

    def _init_indiv(self):
        self.current_player = 0
        self.index = np.ones(self.num_player, dtype=bool)
        self.players = np.arange(self.num_player)


    def _generator(self):
        subset = np.empty(self.num_player, dtype=np.int64)
        subset[0] = self.current_player
        self.index[self.current_player] = False
        pi = np.random.permutation(self.num_player - 1)
        subset[1:] = self.players[self.index][pi]
        self.index[self.current_player] = True
        self.current_player = (self.current_player + 1) % self.num_player
        return subset


    def run(self, samples):
        game = self.game_func(**self.game_args)
        results_collect = np.zeros((len(samples), self.num_player + 1), dtype=np.float64)
        results_collect[:, 0] = samples[:, 0]
        subset = np.zeros(self.num_player, dtype=bool)
        for i, sample in enumerate(samples):
            current_player = sample[0]
            perm = sample[1:]
            results_collect[i, 1] -= game.evaluate(subset)
            subset[current_player] = True
            results_collect[i, 1] += game.evaluate(subset)
            for j, player in enumerate(perm):
                subset[player] = True
                results_collect[i, j + 2] += game.evaluate(subset)
                subset[current_player] = False
                results_collect[i, j + 2] -= game.evaluate(subset)
                subset[current_player] = True
            subset.fill(False)
        return results_collect


    def _process(self, inputs):
        for take in inputs:
            current_player = int(take[0])
            count_cur = self.count[current_player]
            self.results_aggregate[current_player] *= count_cur / (count_cur + 1)
            self.results_aggregate[current_player] += take[1:] / (count_cur + 1)
            self.count[current_player] += 1


    def _estimate(self):
        return (self.results_aggregate * self.weights[None, :]).sum(axis=1)


class OFA_baseline_paired(OFA_baseline):
    def __init__(self, **kwargs):
        super(OFA_baseline_paired, self).__init__(**kwargs)
        self.pi_pre = None
        self.take_inverse = False

    def _generator(self):
        subset = np.empty(self.num_player, dtype=np.int64)
        if self.take_inverse:
            subset[0] = self.current_player
            subset[1:] = self.players[self.index][np.argsort(self.pi_pre)]
            self.index[self.current_player] = True
            self.current_player = (self.current_player + 1) % self.num_player
            self.take_inverse = False
        else:
            subset[0] = self.current_player
            self.index[self.current_player] = False
            pi = np.random.permutation(self.num_player - 1)
            subset[1:] = self.players[self.index][pi]
            self.pi_pre = pi
            self.take_inverse = True
        return subset


class SHAP_IQ(kernelSHAP):
    def __init__(self, **kwargs):
        super(MSR, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        self.results_aggregate = dict(estimates=np.zeros(self.num_player, dtype=np.float64), count=0)
        self.buffer = np.empty((self.buffer_size, self.num_player + 1), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player), dtype=bool)

        with mp.Pool(1) as pool:
            constants = pool.apply(self.calculate_constants, (self.game_func, self.game_args, self.num_player))
        self.scalar = 2 * np.reciprocal(np.arange(1, self.num_player, dtype=np.float64)).sum()

        weights = self.distribution_cardinality()
        self.constant = (constants[1] - constants[0]) * weights[-1]
        self.empty = constants[0]
        tmp = np.arange(self.num_player - 1, -1, -1)
        self.weights_p = tmp * weights
        self.weights_n = tmp[::-1] * weights

    def _init_indiv(self):
        tmp = np.arange(1, self.num_player, dtype=np.float64)
        weights = 1 / np.multiply(tmp, tmp[::-1])
        self.weights = weights / weights.sum()
        self.s_range = np.arange(1, self.num_player)
        self.pos_range = np.arange(self.num_player)

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player].astype(bool)
        ues = inputs[:, [-1]]
        sizes = subsets.sum(axis=1)
        tmp = subsets * self.weights_p[sizes - 1][:, None]
        subsets = ~subsets
        tmp -= subsets * self.weights_n[sizes][:, None]
        tmp = tmp * (ues - self.empty)

        num_pre = self.results_aggregate["count"]
        num_cur = len(tmp) + num_pre
        self.results_aggregate["estimates"] *= num_pre / num_cur
        self.results_aggregate["estimates"] += tmp.sum(axis=0) / num_cur
        self.results_aggregate["count"] = num_cur

    def _estimate(self):
        return self.constant + \
               self.results_aggregate["estimates"] * self.scalar


class SHAP_IQ_paired(SHAP_IQ):
    def __init__(self, **kwargs):
        super(SHAP_IQ_paired, self).__init__(**kwargs)
        self.lock_switch = False


# =============================================================================
# RegressionMSR / TreeMSR baselines
# Witter, Liu, and Musco (2025), arXiv:2506.11849
# =============================================================================

# ---------------------------------------------------------------------------
# RegressionMSR surrogate-fitting helpers
# ---------------------------------------------------------------------------

def _msr_leverage_shap(X, y, prob_sampled, v0, v1, p):
    """
    Closed-form Leverage SHAP linear fit.

    Parameters
    ----------
    X           : (N, n) float64 coalition indicator matrix (interior sizes only)
    y           : (N,)   utility values
    prob_sampled: (N,)   sampling probabilities (sample_dist[size - adjust])
    v0          : scalar v(empty set)
    v1          : scalar v(grand coalition)
    p           : (n,)   semivalue weight vector  p[k] = dist_card[k] / C(n-1,k)

    Returns
    -------
    reg_phi : (n,) linear surrogate semivalue estimates
    """
    n     = X.shape[1]
    sizes = X.sum(axis=1).astype(int)          # (N,), entries in 1..n-1
    Sv    = y - v0                              # (N,)

    sum_weighting = -p[sizes] * (n - sizes) + p[sizes - 1] * sizes   # (N,)
    sum_phi       = float((prob_sampled * sum_weighting) @ Sv)        # scalar
    sum_phi      += (v1 - v0) * p[n - 1] * n  # p[-1] = p[n-1]

    Sb           = Sv - sizes * sum_phi / n    # (N,)
    Proj         = np.eye(n) - 1.0 / n        # (n, n)
    reg_weighting= (p[sizes - 1] + p[sizes]) / prob_sampled           # (N,)

    # P @ X.T @ diag(w) @ Sb  and  P @ X.T @ diag(w) @ X @ P
    wX      = X * reg_weighting[:, np.newaxis]                        # (N, n)
    PZSSb   = Proj @ (wX.T @ Sb)                                      # (n,)
    PZSSZP  = Proj @ (wX.T @ X) @ Proj                                # (n, n)
    # Match regressionMSR reference exactly (uses np.linalg.lstsq).
    sol     = np.linalg.lstsq(PZSSZP, PZSSb, rcond=None)[0]
    return sol + sum_phi / n                                           # (n,)


def _msr_ith_combination(pool, r, index):
    """Return the `index`-th size-`r` combination of `pool` (0-based lexicographic)."""
    n = len(pool)
    combo = []
    k = r
    start = 0
    for _ in range(r):
        for j in range(start, n):
            count = math.comb(n - j - 1, k - 1)
            if index < count:
                combo.append(pool[j])
                k -= 1
                start = j + 1
                break
            index -= count
    return tuple(combo)


def _msr_combination_generator(gen, n, s, num_samples):
    """
    Mirror regressionMSR.estimators.est_utils.combination_generator.
    Samples combinations without materializing all of them.
    """
    num_combos = math.comb(n, s)
    try:
        indices = gen.choice(num_combos, num_samples, replace=False)
        for i in indices:
            yield _msr_ith_combination(range(n), s, int(i))
    except OverflowError:
        for _ in range(num_samples):
            yield gen.choice(n, s, replace=False)


def _is_symmetric_semivalue(semivalue, semivalue_param):
    if semivalue == "shapley":
        return True
    if semivalue == "weighted_banzhaf":
        return abs(float(semivalue_param) - 0.5) <= 1e-12
    if semivalue == "beta_shapley":
        alpha, beta = semivalue_param
        return abs(float(alpha) - float(beta)) <= 1e-12
    return False


# ---------------------------------------------------------------------------
# Shared base class
# ---------------------------------------------------------------------------

class _MSRBase(estimatorTemplate):
    """
    Shared base for RegressionMSR and TreeMSR.

    Implements the sampling distribution, k-fold cross-validation, and MSR
    correction exactly as in UniversalMSR.explain() from regMSR.py.
    Subclasses implement _setup_case() and _fit_surrogate().
    """

    def __init__(
        self,
        *,
        sampling_with_replacement=None,
        paired_sampling=None,
        use_special_surrogates=True,
        num_folds=10,
        **kwargs,
    ):
        if sampling_with_replacement is None:
            self.sampling_with_replacement = None
        elif isinstance(sampling_with_replacement, (bool, np.bool_)):
            self.sampling_with_replacement = bool(sampling_with_replacement)
        else:
            raise ValueError(
                "`sampling_with_replacement` must be None, True, or False, "
                f"got {sampling_with_replacement!r}."
            )

        if paired_sampling is None:
            self.paired_sampling = None
        elif isinstance(paired_sampling, (bool, np.bool_)):
            self.paired_sampling = bool(paired_sampling)
        else:
            raise ValueError(
                "`paired_sampling` must be None, True, or False, "
                f"got {paired_sampling!r}."
            )

        if isinstance(use_special_surrogates, (bool, np.bool_)):
            self.use_special_surrogates = bool(use_special_surrogates)
        else:
            raise ValueError(
                "`use_special_surrogates` must be bool, "
                f"got {use_special_surrogates!r}."
            )

        if not isinstance(num_folds, (int, np.integer)):
            raise ValueError(f"`num_folds` must be an integer >= 2, got {num_folds!r}.")
        self._kfold = int(num_folds)
        if self._kfold < 2:
            raise ValueError(f"`num_folds` must be >= 2, got {self._kfold!r}.")

        super(_MSRBase, self).__init__(**kwargs)
        n = self.num_player

        # ── semivalue weight vector p[k] (length n) ──────────────────────────
        # p[k] = dist_card[k] / C(n-1, k)  for k = 0 .. n-1
        # Matches the semantics of get_p() in regressionMSR.utils.p_generator.
        dist_card = self.distribution_cardinality()   # (n,)
        p = np.zeros(n, dtype=np.float64)
        for k in range(n):
            cn1k = float(special.comb(n - 1, k, exact=False))
            if cn1k > 0.0:
                p[k] = dist_card[k] / cn1k
        self._p = p                                   # (n,)

        # ── subclass-specific flags (set in _setup_case) ──────────────────────
        self._is_leverage_shap = False
        self._is_banzhaf       = False
        self._setup_case()

        # ── pair_sampling: automatic iff weights are symmetric ───────────────
        symmetric_weights = _is_symmetric_semivalue(self.semivalue, self.semivalue_param)
        if self.paired_sampling is None:
            self._pair_sampling = symmetric_weights
        elif self.paired_sampling:
            if not symmetric_weights:
                raise ValueError(
                    "`paired_sampling=True` requires symmetric semivalue weights. "
                    "Use `paired_sampling=None` for automatic behavior or "
                    "`paired_sampling=False` to disable pairing."
                )
            self._pair_sampling = True
        else:
            self._pair_sampling = False
        if self._pair_sampling:
            self.lock_switch = False

        # ── sampling distribution (exact match of UniversalMSR.__init__) ─────
        self._build_sample_dist()

        # ── budget ────────────────────────────────────────────────────────────
        self.num_sample       = self.nue_avg       * n
        self.interval_track   = self.nue_track_avg * n
        self.batch_size       = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        # Match UniversalMSR.explain() budget rounding to k-fold CV.
        if self._pair_sampling:
            self.num_sample = (self.num_sample // (2 * self._kfold)) * (2 * self._kfold)
        else:
            self.num_sample = (self.num_sample // self._kfold) * self._kfold

        # ── boundary values (Leverage SHAP) ────────────────────────────────────
        self._v0 = None        # v(empty set)
        self._v1 = None        # v(grand coalition)

        # ── accumulators ──────────────────────────────────────────────────────
        self._X_chunks = []
        self._y_chunks = []
        self._prob_chunks = []
        self._density_chunks = []
        self._current_estimate = np.zeros(n, dtype=np.float64)
        self._rng = np.random.Generator(np.random.PCG64(self.estimator_seed))
        self._sample_matrix = None
        self._sample_prob = None
        self._sample_correction_density = None

        self.buffer  = np.empty((self.buffer_size, n + 3), dtype=np.float64)
        self.samples = np.empty((self.batch_size,  n),     dtype=bool)

    # ── subclass hooks ────────────────────────────────────────────────────────

    def _setup_case(self):
        """Set self._is_leverage_shap / self._is_banzhaf based on semivalue."""
        raise NotImplementedError

    def _fit_surrogate(self, X_train, y_train, X_test, prob_train):
        """
        Fit surrogate on (X_train, y_train), return (reg_phi, reg_pred_test).
        reg_phi      : (n,) semivalue estimates from the surrogate.
        reg_pred_test: (|test|,) surrogate predictions on X_test.
        """
        raise NotImplementedError

    # ── sampling distribution ─────────────────────────────────────────────────

    def _build_sample_dist(self):
        """
        Build _sample_dist, _valid_sizes, _adjust, and _q_cumsum.
        Exactly mirrors UniversalMSR.__init__ from regMSR.py.
        """
        n = self.num_player
        p = self._p

        if self._is_leverage_shap:
            # Leverage SHAP: valid sizes 1 .. n-1
            s_arr = np.arange(1, n)
            sd    = (p[s_arr] + p[s_arr - 1]) * s_arr * (n - s_arr)
            self._valid_sizes = s_arr
            self._adjust      = 1
        elif self._is_banzhaf:
            # Kernel Banzhaf: uniform over all sizes 0 .. n
            sd                = np.ones(n + 1, dtype=np.float64)
            self._valid_sizes = np.arange(n + 1)
            self._adjust      = 0
        else:
            # General: sqrt(p[s-1]^2 * s + p[s]^2 * (n-s))  for s = 0..n
            sd = []
            for s in range(n + 1):
                prob = 0.0
                if s > 0: prob += p[s - 1] ** 2 * s
                if s < n: prob += p[s]     ** 2 * (n - s)
                sd.append(np.sqrt(prob))
            sd                = np.array(sd, dtype=np.float64)
            self._valid_sizes = np.arange(n + 1)
            self._adjust      = 0

        sd_total             = sd.sum()
        self._sample_dist    = sd / sd_total if sd_total > 0 else sd

        # Sampling probabilities over valid_sizes:  q ∝ sample_dist * C(n, s)
        binoms  = np.array([float(special.comb(n, s, exact=False))
                            for s in self._valid_sizes])
        q_raw   = self._sample_dist * binoms
        q_sum   = q_raw.sum()
        q_norm  = q_raw / q_sum if q_sum > 0 else np.ones(len(q_raw)) / len(q_raw)
        self._q_cumsum = np.cumsum(q_norm)
        # Normalizer for subset density D(S) when sampling with replacement:
        # D(S) = sample_dist[|S|-adjust] / subset_density_norm.
        self._subset_density_norm = float(q_sum)
        self._sample_prob_density_norm = float(q_sum)
        self._last_sampling_with_replacement = True

    # ── easeshap interface ─────────────────────────────────────────────────────

    def _init_indiv(self):
        # Leverage SHAP uses model predictions at the all-zero baseline and
        # all-one explicand rather than sampled boundary rows.
        if self._is_leverage_shap and (self._v0 is None or self._v1 is None):
            game = self.game_func(**self.game_args)
            n = self.num_player
            self._v0 = float(game.evaluate(np.zeros(n, dtype=bool)))
            self._v1 = float(game.evaluate(np.ones(n, dtype=bool)))

    def _add_sample(self, X, prob_arr, row_idx, indices, prob, density_arr=None, density=None):
        if density is None:
            density = prob
        if not self._pair_sampling:
            X[row_idx, indices] = True
            prob_arr[row_idx] = prob
            if density_arr is not None:
                density_arr[row_idx] = density
        else:
            X[2 * row_idx, indices] = True
            prob_arr[2 * row_idx] = prob
            if density_arr is not None:
                density_arr[2 * row_idx] = density
            comp = np.setdiff1d(np.arange(self.num_player), np.asarray(indices, dtype=int))
            X[2 * row_idx + 1, comp] = True
            prob_arr[2 * row_idx + 1] = prob
            if density_arr is not None:
                density_arr[2 * row_idx + 1] = density

    def _find_bernoulli_constant(self, target_samples):
        # Exact mirror of UniversalMSR.find_bernoulli_constant
        n = self.num_player
        C = 1.0 / (2 ** n)
        m = int(min(target_samples, 2 ** n - 2 * self._adjust))
        L = 1.0 / (100 ** n)
        R = float(100 ** n)
        binoms = np.array([float(special.comb(n, s, exact=False)) for s in self._valid_sizes])
        sample_dist_C = np.minimum(self._sample_dist * C, np.ones_like(self._sample_dist))
        expected = float(np.sum(sample_dist_C * binoms))
        while round(expected) != m and (R - L) > 1e-6:
            if expected < m:
                L = C
            else:
                R = C
            C = (L + R) / 2.0
            sample_dist_C = np.minimum(self._sample_dist * C, np.ones_like(self._sample_dist))
            expected = float(np.sum(sample_dist_C * binoms))
        return C

    def _sample_with_replacement(self, target_samples, *, update_density_norm=True):
        n = self.num_player
        replacement_norm = float(self._subset_density_norm)
        if update_density_norm:
            self._sample_prob_density_norm = replacement_norm
        # Paired mode writes two rows per draw. If target is odd, allocate one
        # extra row and truncate at the end to avoid leaving an unfilled zero-
        # probability row.
        work_target = target_samples
        if self._pair_sampling and (work_target % 2 == 1):
            work_target += 1

        X = np.zeros((work_target, n), dtype=bool)
        probs = np.zeros(work_target, dtype=np.float64)
        densities = np.zeros(work_target, dtype=np.float64)

        size_binom = np.array([float(special.comb(n, s, exact=False)) for s in self._valid_sizes])
        sample_dist_size = self._sample_dist * size_binom
        sample_dist_size = sample_dist_size / sample_dist_size.sum()
        num_draws = work_target // 2 if self._pair_sampling else work_target
        sampled_sizes = self._rng.choice(self._valid_sizes, size=num_draws, p=sample_dist_size)

        for idx, s in enumerate(sampled_sizes):
            s = int(s)
            indices = self._rng.choice(n, size=s, replace=False)
            prob = self._sample_dist[s - self._adjust]
            density = prob / replacement_norm
            self._add_sample(X, probs, idx, indices, prob, densities, density)
        return X[:target_samples], probs[:target_samples], densities[:target_samples]

    def _sample_without_replacement(self, target_samples):
        n = self.num_player
        C = self._find_bernoulli_constant(target_samples)
        sample_dist_C = np.minimum(self._sample_dist * C, np.ones_like(self._sample_dist))

        size_binom = np.array([float(special.comb(n, s, exact=False)) for s in self._valid_sizes])
        no_replacement_norm = float(np.sum(sample_dist_C * size_binom))
        self._sample_prob_density_norm = no_replacement_norm
        m_total = int(np.sum([round(p * b) for p, b in zip(sample_dist_C, size_binom)]))
        if self._pair_sampling:
            m_total = (m_total // 2) * 2

        X = np.zeros((m_total, n), dtype=bool)
        probs = np.zeros(m_total, dtype=np.float64)
        densities = np.zeros(m_total, dtype=np.float64)
        idx = 0

        for s, prob in zip(self._valid_sizes, sample_dist_C):
            s = int(s)
            m_s = int(round(prob * float(special.comb(n, s, exact=False))))
            if m_s <= 0:
                continue
            density = prob / no_replacement_norm

            if self._pair_sampling and s == n // 2 and n % 2 == 0:
                # Avoid duplicate complements when n even and s=n/2.
                gen = _msr_combination_generator(self._rng, n - 1, s - 1, m_s // 2)
                for combo in gen:
                    self._add_sample(X, probs, idx, list(combo) + [n - 1], prob, densities, density)
                    idx += 1
                break

            gen = _msr_combination_generator(self._rng, n, s, m_s)
            for combo in gen:
                self._add_sample(X, probs, idx, combo, prob, densities, density)
                idx += 1

            if self._pair_sampling and s == n // 2:
                break

        if idx < m_total:
            # In paired mode idx counts pair-draws, i.e., 2 rows each.
            used_rows = 2 * idx if self._pair_sampling else idx
            X = X[:used_rows]
            probs = probs[:used_rows]
            densities = densities[:used_rows]

        # Keep easeshap trajectory shape stable: force exact target length.
        cur = len(X)
        if cur > target_samples:
            X = X[:target_samples]
            probs = probs[:target_samples]
            densities = densities[:target_samples]
        elif cur < target_samples:
            add_X, add_p, add_d = self._sample_with_replacement(
                target_samples - cur,
                update_density_norm=False,
            )
            X = np.vstack([X, add_X]) if cur else add_X
            probs = np.concatenate([probs, add_p]) if cur else add_p
            densities = np.concatenate([densities, add_d]) if cur else add_d
        return X, probs, densities

    def sampling(self):
        """Generate sampled coalition matrix following UniversalMSR sampling rules."""
        self._init_indiv()
        if self.num_sample <= 0:
            return

        use_with_replacement = (
            self.num_player >= 50
            if self.sampling_with_replacement is None
            else self.sampling_with_replacement
        )
        self._last_sampling_with_replacement = bool(use_with_replacement)
        if use_with_replacement:
            (
                self._sample_matrix,
                self._sample_prob,
                self._sample_correction_density,
            ) = self._sample_with_replacement(self.num_sample)
        else:
            (
                self._sample_matrix,
                self._sample_prob,
                self._sample_correction_density,
            ) = self._sample_without_replacement(self.num_sample)

        N = len(self._sample_matrix)
        for start in range(0, N, self.batch_size):
            end = min(start + self.batch_size, N)
            batch = np.empty((end - start, self.num_player + 2), dtype=np.float64)
            batch[:, :self.num_player] = self._sample_matrix[start:end].astype(np.float64)
            batch[:, self.num_player] = self._sample_prob[start:end]
            batch[:, self.num_player + 1] = self._sample_correction_density[start:end]
            yield batch

    def _generator(self):
        raise NotImplementedError("_generator is unused in _MSRBase; use sampling().")

    def run(self, samples):
        """Evaluate game for each coalition in the batch."""
        game    = self.game_func(**self.game_args)
        n       = self.num_player
        X = samples[:, :n].astype(bool)
        prob = samples[:, n]
        density = samples[:, n + 1]
        results = np.empty((len(samples), n + 3), dtype=np.float64)
        results[:, :n] = X.astype(np.float64)
        for i in range(len(samples)):
            results[i, n] = game.evaluate(X[i])
            results[i, n + 1] = prob[i]
            results[i, n + 2] = density[i]
        return results

    def _process(self, inputs):
        """Accumulate sampled (X, y, prob) chunks."""
        n     = self.num_player
        X_raw = inputs[:, :n].astype(np.float64)
        y_raw = inputs[:, n].copy()
        p_raw = inputs[:, n + 1].copy()
        d_raw = inputs[:, n + 2].copy()
        self._X_chunks.append(X_raw)
        self._y_chunks.append(y_raw)
        self._prob_chunks.append(p_raw)
        self._density_chunks.append(d_raw)

    def _estimate(self):
        """Run k-fold regression and return semivalue estimates."""
        return self._run_kfold(k=self._kfold)

    def _run_kfold(self, k=10):
        """
        k-fold cross-validation with MSR correction.
        Exact mirror of the fold loop in UniversalMSR.explain() from regMSR.py.
        """
        n = self.num_player
        if not self._X_chunks:
            return np.zeros(n)

        X = np.vstack(self._X_chunks)          # (N, n)
        y = np.concatenate(self._y_chunks)     # (N,)
        prob_sampled = np.concatenate(self._prob_chunks)  # (N,)
        N = len(y)
        if N < max(n + 1, k):
            return np.zeros(n)

        sizes = X.sum(axis=1).astype(int)      # (N,)

        # ── shuffle, keeping pairs together when pair_sampling ─────────────
        if self._pair_sampling and N % 2 == 0:
            N_pairs  = N // 2
            pair_ids = self._rng.permutation(N_pairs)
            shuffled = np.reshape(
                np.column_stack((2 * pair_ids, 2 * pair_ids + 1)), -1
            )
        else:
            shuffled = self._rng.permutation(N)

        N_use     = (N // k) * k               # trim to multiple of k
        shuffled  = shuffled[:N_use]
        fold_size = N_use // k

        phi = np.zeros(n, dtype=np.float64)

        for fold in range(k):
            test_idx  = shuffled[fold * fold_size : (fold + 1) * fold_size]
            train_idx = np.concatenate([
                shuffled[:fold * fold_size],
                shuffled[(fold + 1) * fold_size : N_use]
            ])
            if len(train_idx) < n + 1:
                continue

            X_train  = X[train_idx]
            y_train  = y[train_idx]
            X_test   = X[test_idx]
            y_test   = y[test_idx]
            sz_test  = sizes[test_idx]
            pr_test  = prob_sampled[test_idx]
            pr_train = prob_sampled[train_idx]

            # Do not silently skip fold failures. If surrogate fitting fails,
            # propagate the exception.
            reg_phi, reg_pred = self._fit_surrogate(
                X_train, y_train, X_test, pr_train
            )

            residuals = y_test - reg_pred      # (|test|,)

            # ── MSR correction (per player) ───────────────────────────────
            for idx in range(n):
                in_mask  = (X_test[:, idx] == 1)
                out_mask = ~in_mask

                mean_with    = 0.0
                mean_without = 0.0

                if in_mask.any():
                    sz_in     = sz_test[in_mask]
                    mean_with = (
                        residuals[in_mask] * self._p[sz_in - 1]
                        / pr_test[in_mask]
                    ).mean()

                if out_mask.any():
                    sz_out       = sz_test[out_mask]
                    mean_without = (
                        residuals[out_mask] * self._p[sz_out]
                        / pr_test[out_mask]
                    ).mean()

                phi[idx] += (reg_phi[idx] + mean_with - mean_without) / k

                # Efficiency correction for Leverage SHAP (mirrors regMSR.py)
                # sum(reg_phi) ≈ v1-v0 by construction, so this term ≈ 0.
                if (self._is_leverage_shap
                        and self._v0 is not None
                        and self._v1 is not None):
                    phi[idx] += (self._v1 - reg_phi.sum()) / n - self._v0 / n

        return phi


# ---------------------------------------------------------------------------
# RegressionMSR  (LinearMSR with Leverage SHAP / Kernel Banzhaf / OLS)
# ---------------------------------------------------------------------------

class RegressionMSR(_MSRBase):
    """
    LinearMSR of Witter, Liu, and Musco (2025), arXiv:2506.11849.

    Fits a linear surrogate f(S) to sampled (coalition, utility) pairs and
    applies the MSR residual correction with configurable k-fold cross-validation.
    The implementation uses three special cases:

      • Shapley / Beta(1,1) → Leverage SHAP  (closed-form projected WLS)
      • Banzhaf (p=0.5)     → Kernel Banzhaf (centred OLS, no intercept)
      • All other semivalues → sklearn LinearRegression (with intercept)

    Budget: nue_avg * num_player utility evaluations.
    """

    def _setup_case(self):
        if not self.use_special_surrogates:
            return
        if self._is_shapley_like():
            self._is_leverage_shap = True
        elif self._is_banzhaf_half():
            self._is_banzhaf = True

    def _is_shapley_like(self):
        if self.semivalue == 'shapley':
            return True
        if self.semivalue == 'beta_shapley':
            a, b = self.semivalue_param
            return abs(float(a) - 1.0) < 1e-9 and abs(float(b) - 1.0) < 1e-9
        return False

    def _is_banzhaf_half(self):
        if self.semivalue == 'weighted_banzhaf':
            return abs(float(self.semivalue_param) - 0.5) < 1e-9
        return False

    def _fit_surrogate(self, X_train, y_train, X_test, prob_train):
        if self._is_leverage_shap:
            reg_phi  = _msr_leverage_shap(
                X_train, y_train, prob_train,
                self._v0, self._v1, self._p
            )
            reg_pred = X_test @ reg_phi
        elif self._is_banzhaf:
            # Kernel Banzhaf: lstsq on centred design matrix (matches kernel_banzhaf)
            # Match reference kernel_banzhaf implementation exactly.
            reg_phi = np.linalg.lstsq(X_train - 0.5, y_train, rcond=None)[0]
            reg_pred          = X_test @ reg_phi
        else:
            import sklearn.linear_model
            lm       = sklearn.linear_model.LinearRegression()
            lm.fit(X_train, y_train)
            reg_phi  = lm.coef_
            reg_pred = lm.predict(X_test)
        return reg_phi, reg_pred


# ---------------------------------------------------------------------------
# PolySHAP  (Fumagalli, Witter, and Musco 2026, arxiv:2601.18608)
# ---------------------------------------------------------------------------

class PolySHAP_regression(estimatorTemplate):
    """
    PolySHAP of Fumagalli, Witter, and Musco (2026), implemented through the
    EaseSHAP estimator interface.

    This fits a constrained KernelSHAP-weighted polynomial surrogate over
    monomial coalition indicators 1{T subseteq S}, for 1 <= |T| <= max_order,
    then returns the Shapley values of the fitted polynomial. The default
    max_order=2 matches second-order unpaired PolySHAP; max_order=1 reduces to
    KernelSHAP.

    Budget    : nue_avg * num_player total evaluations.
    Semivalues: Shapley only.
    """

    def __init__(self, *, max_order=2, **kwargs):
        self.max_order = int(max_order)
        if self.max_order < 1:
            raise ValueError(f"`max_order` must be >= 1, got {max_order!r}.")

        super(PolySHAP_regression, self).__init__(**kwargs)
        n = self.num_player
        if self.max_order > n:
            raise ValueError(f"`max_order` must be <= num_player={n}, got {self.max_order}.")

        # Budget: same pattern as MSR / RegressionMSR
        self.num_sample       = self.nue_avg       * n
        self.interval_track   = self.nue_track_avg  * n
        self.batch_size       = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        # KernelSHAP weights mu(S), indexed by coalition size.
        self._kernel_w = np.zeros(n + 1)
        for s in range(1, n):
            cn2s1 = float(special.comb(n - 2, s - 1, exact=False))
            if cn2s1 > 0:
                self._kernel_w[s] = 1.0 / cn2s1

        # ── sampling distribution over sizes (matches default Approximator weights) ─
        # interior size weight ∝ 1/(s*(n-s)); boundaries handled explicitly.
        q_unnorm = np.zeros(n + 1, dtype=np.float64)
        for s in range(1, n):
            q_unnorm[s] = 1.0 / (s * (n - s))
        q_unnorm[0] = 0.0
        q_unnorm[n] = 0.0
        q_total = q_unnorm.sum()
        if q_total > 0:
            self._q_size = q_unnorm / q_total
        else:
            # Fallback: uniform (should not occur for reasonable n)
            self._q_size          = np.zeros(n + 1)
            self._q_size[1:n]     = 1.0 / (n - 1)
        self._q_cumsum = np.cumsum(self._q_size)

        # Explicit PolySHAP monomial features and their Shapley readout.
        self._interaction_blocks = []
        self._feature_dim = 0
        zeta_cols = []
        for degree in range(1, self.max_order + 1):
            m = math.comb(n, degree)
            if m == 0:
                continue
            flat = np.fromiter(
                itertools.chain.from_iterable(itertools.combinations(range(n), degree)),
                dtype=np.int64,
                count=m * degree,
            )
            combos = flat.reshape(m, degree)
            start = self._feature_dim
            self._interaction_blocks.append((degree, combos, start))
            self._feature_dim += m

            zeta_block = np.zeros((n, m), dtype=np.float64)
            cols = np.arange(m, dtype=np.int64)
            for pos in range(degree):
                zeta_block[combos[:, pos], cols] = 1.0 / float(degree)
            zeta_cols.append(zeta_block)
        self._zeta_poly = np.concatenate(zeta_cols, axis=1)

        # ── boundary values (v(empty) and v(N), set during first _process) ────
        self._v_empty = None
        self._v_full  = None

        # ── accumulators ──────────────────────────────────────────────────────
        self._X_chunks = []
        self._y_chunks = []
        self._current_estimate = np.zeros(n, dtype=np.float64)

        self.buffer  = np.empty((self.buffer_size, n + 1), dtype=np.float64)
        self.samples = np.empty((self.batch_size,  n),     dtype=bool)

    def _init_indiv(self):
        assert self.semivalue == "shapley", (
            "PolySHAP_regression uses KernelSHAP weights defined for Shapley "
            "values only.  For other semivalues use RegressionMSR."
        )

    def sampling(self):
        """Override base sampling to prepend boundary coalitions first."""
        self._init_indiv()
        np.random.seed(self.estimator_seed)

        # ── Step 1: evaluate empty set and grand coalition ───────────────────
        n = self.num_player
        boundary = np.zeros((2, n), dtype=bool)
        boundary[1, :] = True                   # grand coalition
        yield boundary

        # ── Step 2: regular random batches for the remaining budget ───────────
        remaining = self.num_sample - 2
        count = 0
        pending_antithetic = None
        for _ in range(remaining):
            if not self.switch or pending_antithetic is None:
                sample = self._generator()
                self.samples[count] = sample
                pending_antithetic = np.asarray(sample).copy()
                self.switch = True
            else:
                self.samples[count] = 1 - pending_antithetic
                pending_antithetic = None
                self.switch = False
            count += 1
            if count == self.batch_size:
                yield self.samples.copy()
                count = 0
        if count:
            yield self.samples[:count]

    def _generator(self):
        n = self.num_player
        u = np.random.rand()
        s = int(np.searchsorted(self._q_cumsum, u))
        s = max(1, min(s, n - 1))
        coalition = np.zeros(n, dtype=bool)
        coalition[np.random.choice(n, size=s, replace=False)] = True
        return coalition

    def run(self, samples):
        game = self.game_func(**self.game_args)
        n    = self.num_player
        results = np.empty((len(samples), n + 1), dtype=np.float64)
        results[:, :n] = samples.astype(np.float64)
        for i, coalition in enumerate(samples):
            results[i, n] = game.evaluate(coalition)
        return results

    def _process(self, inputs):
        n     = self.num_player
        X     = inputs[:, :n].astype(bool)
        y_raw = inputs[:, n].copy()
        sizes = X.sum(axis=1).astype(int)

        # Separate boundary coalitions from interior ones
        empty_mask    = (sizes == 0)
        full_mask     = (sizes == n)
        interior_mask = ~empty_mask & ~full_mask

        if empty_mask.any():
            self._v_empty = float(y_raw[empty_mask][-1])
        if full_mask.any():
            self._v_full  = float(y_raw[full_mask][-1])
        if interior_mask.any():
            self._X_chunks.append(X[interior_mask].astype(np.float64))
            self._y_chunks.append(y_raw[interior_mask])

        if (self._v_empty is not None
                and self._v_full  is not None
                and self._X_chunks):
            self._current_estimate = self._run_polyshap_regression()

    def _estimate(self):
        return self._current_estimate.copy()

    def _build_polyshap_feature_block(self, X):
        X = np.asarray(X, dtype=np.float64)
        Z = np.empty((len(X), self._feature_dim), dtype=np.float64)
        for degree, combos, start in self._interaction_blocks:
            end = start + len(combos)
            if degree == 1:
                Z[:, start:end] = X[:, combos[:, 0]]
            else:
                Z[:, start:end] = np.prod(X[:, combos], axis=2)
        return Z

    def _run_polyshap_regression(self):
        """Solve the constrained PolySHAP WLS problem from Algorithm 1."""
        n = self.num_player
        if not self._X_chunks:
            return np.zeros(n)

        X  = np.vstack(self._X_chunks)                        # (N, n)
        y_centered = np.concatenate(self._y_chunks) - self._v_empty  # (N,)
        N  = len(y_centered)
        if N < self._feature_dim:
            return np.zeros(n)

        sizes = X.sum(axis=1).astype(int)
        full_set_value = float(self._v_full) - float(self._v_empty)

        # sampling_adjustment_weights = 1 / (prob(coalition) * N)
        # with prob(coalition) = q_size[s] / C(n,s), since sampling is:
        #   choose size s ~ q_size, then choose coalition uniformly among C(n,s).
        comb_ns = np.array([float(special.comb(n, s, exact=False)) for s in sizes])
        prob = self._q_size[sizes] / comb_ns
        sampling_adjustment = 1.0 / (prob * N)

        sampling_normalization = np.sqrt(self._kernel_w[sizes] * sampling_adjustment)
        Z = self._build_polyshap_feature_block(X)
        Z_tilde = sampling_normalization[:, np.newaxis] * Z
        y_tilde = y_centered * sampling_normalization

        d = self._feature_dim
        projection = np.eye(d) - 1.0 / d
        lhs = Z_tilde @ projection
        rhs = y_tilde - full_set_value / d * np.sum(Z_tilde, axis=1)

        try:
            lstsq_solution = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            coeff = projection @ lstsq_solution + full_set_value / d
            phi = self._zeta_poly @ coeff
        except Exception:
            return np.zeros(n)

        return phi


class PolySHAP_regression_paired(PolySHAP_regression):
    """
    PolySHAP with paired (complement) sampling.

    Setting lock_switch = False activates the antithetic coupling mechanism
    inherited from estimatorTemplate: every sampled coalition S is immediately
    followed by its complement S^c.
    """

    def __init__(self, **kwargs):
        super(PolySHAP_regression_paired, self).__init__(**kwargs)
        self.lock_switch = False    # enable antithetic (complement) sampling


# ---------------------------------------------------------------------------
# TreeMSR  (XGBoost surrogate + TreeSHAP / TreeProb)
# ---------------------------------------------------------------------------

class TreeMSR(_MSRBase):
    """
    TreeMSR of Witter, Liu, and Musco (2025), arXiv:2506.11849.

    Replaces the linear surrogate with XGBoost using its default parameters.
    Surrogate semivalues are extracted via tree traversal:
      • Shapley      → shap.TreeExplainer
      • Other svs    → dynamic programming over tree paths with general
                       semivalue weights p[k]

    MSR correction and configurable k-fold CV are inherited from _MSRBase.

    Budget   : nue_avg * num_player utility evaluations.
    Semivalues: all (Shapley via shap; others via regressionMSR.exact.treeprob).
    """

    def _setup_case(self):
        # TreeMSR always uses the general sampling distribution (never Leverage SHAP)
        self._is_leverage_shap = False
        self._is_banzhaf       = False

    def _fit_surrogate(self, X_train, y_train, X_test, prob_train):
        n = self.num_player

        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("TreeMSR requires xgboost.  pip install xgboost")

        # Use XGBoost's default regressor parameters.
        model = xgb.XGBRegressor()
        model.fit(X_train, y_train)
        reg_pred = model.predict(X_test)

        baseline  = np.zeros((1, n), dtype=np.float64)
        explicand = np.ones( (1, n), dtype=np.float64)

        if self.semivalue == 'shapley':
            # TreeSHAP via the shap library.
            try:
                import shap, warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    explainer = shap.TreeExplainer(model, data=baseline)
                    reg_phi   = explainer.shap_values(explicand)[0]   # (n,)
            except ImportError:
                raise ImportError("TreeMSR + Shapley requires shap.  pip install shap")
        else:
            # TreeProb: exact semivalues via DP over tree paths (treeprob.py)
            try:
                import os as _os
                import importlib
                _rmsr_root = _os.path.abspath(_os.path.join(
                    _os.path.dirname(__file__), '..', '..', '..', 'regressionMSR'
                ))
                if _rmsr_root not in sys.path:
                    sys.path.insert(0, _rmsr_root)
                tree_prob = importlib.import_module(
                    "regressionMSR.exact.treeprob"
                ).tree_prob
                weighting_str = self._semivalue_to_weighting_str()
                reg_phi = tree_prob(
                    baseline, [explicand], model, weighting=weighting_str
                )[0].squeeze()                                        # (n,)
            except (ImportError, ModuleNotFoundError):
                # Fallback: OLS linear surrogate phi
                N_tr   = len(y_train)
                X_aug  = np.hstack([X_train, np.ones((N_tr, 1))])
                coef, _, _, _ = lstsq(X_aug, y_train, check_finite=False)
                reg_phi = coef[:n]

        return reg_phi, reg_pred

    def _semivalue_to_weighting_str(self):
        """Convert easeshap semivalue spec to regressionMSR weighting string."""
        if self.semivalue == 'shapley':
            return 'shapley'
        if self.semivalue == 'weighted_banzhaf':
            p = float(self.semivalue_param)
            return 'banzhaf' if abs(p - 0.5) < 1e-9 else f'weighted_banzhaf_{p}'
        if self.semivalue == 'beta_shapley':
            a, b = self.semivalue_param
            def _fmt(v):
                fv = float(v)
                return str(int(fv)) if fv == int(fv) else str(fv)
            return f'beta_shapley_{_fmt(a)}_{_fmt(b)}'
        return 'shapley'


class _MSRUnbiasedPaperMixin:
    """
    Implements the Algorithm-1 residual correction from Witter et al. (2025):

      phi_i <- phi_i(f) + (1/|S_l|) * sum_{S in S_l}
               (v(S)-f(S)) * ( p_{|S|-1} 1[i in S] - p_{|S|} 1[i notin S] ) / D(S)

    This mixin changes only the residual-correction formula. Sampling,
    surrogate fitting, K-fold splitting, and special RegressionMSR branches are
    inherited from the parent class.
    """

    def _subset_density_by_size(self, sizes):
        """
        Return D(S) for each sampled coalition size in `sizes`.
        The density is induced by the parent RegressionMSR sampling law:
            D(S) = q_{|S|} / Z,
            Z    = sum_t q_t * C(n, t),
        where q_t == self._sample_dist[t - self._adjust].
        """
        idx = sizes - self._adjust
        out = np.zeros(len(sizes), dtype=np.float64)
        valid = (idx >= 0) & (idx < len(self._sample_dist))
        if np.any(valid):
            out[valid] = self._sample_dist[idx[valid]]

        norm = float(getattr(self, "_subset_density_norm", 0.0))
        if norm <= 0.0:
            raise ValueError("Invalid subset density normalizer in unbiased MSR.")
        out /= norm
        return out

    def _correction_density(self, sizes, prob_sampled, correction_density=None):
        """
        Return the density used in the paper-style residual correction.

        New sampled data carries this density per row so mixed no-replacement
        and fallback replacement rows can be corrected consistently. The
        size/probability fallback is kept for direct calls in older tests or
        ad-hoc estimator use. With replacement, that fallback is the draw
        density D(S). With the no-replacement sampler,
        saturated coalition sizes have inclusion probability one, so using
        only the replacement-law size density would over-weight them. In that
        case, normalize the sampler's returned per-coalition probabilities.
        """
        if correction_density is not None:
            out = np.asarray(correction_density, dtype=np.float64)
            if np.any(out <= 0):
                raise ValueError("Encountered zero correction density in unbiased MSR fold.")
            return out

        if getattr(self, "_last_sampling_with_replacement", True):
            return self._subset_density_by_size(sizes)

        norm = float(getattr(self, "_sample_prob_density_norm", 0.0))
        if norm <= 0.0:
            raise ValueError("Invalid sample probability normalizer in unbiased MSR.")
        out = np.asarray(prob_sampled, dtype=np.float64) / norm
        if np.any(out <= 0):
            raise ValueError("Encountered zero sample probability in unbiased MSR fold.")
        return out

    def _run_kfold(self, k=10):
        n = self.num_player
        if not self._X_chunks:
            return np.zeros(n)

        X = np.vstack(self._X_chunks)
        y = np.concatenate(self._y_chunks)
        prob_sampled = np.concatenate(self._prob_chunks)
        correction_density = np.concatenate(self._density_chunks)
        N = len(y)
        if N < max(n + 1, k):
            return np.zeros(n)

        sizes = X.sum(axis=1).astype(int)

        if self._pair_sampling and N % 2 == 0:
            N_pairs = N // 2
            pair_ids = self._rng.permutation(N_pairs)
            shuffled = np.reshape(
                np.column_stack((2 * pair_ids, 2 * pair_ids + 1)), -1
            )
        else:
            shuffled = self._rng.permutation(N)

        N_use = (N // k) * k
        shuffled = shuffled[:N_use]
        fold_size = N_use // k

        phi = np.zeros(n, dtype=np.float64)

        for fold in range(k):
            test_idx = shuffled[fold * fold_size : (fold + 1) * fold_size]
            train_idx = np.concatenate([
                shuffled[:fold * fold_size],
                shuffled[(fold + 1) * fold_size : N_use],
            ])
            if len(train_idx) < n + 1:
                continue

            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]
            sz_test = sizes[test_idx]
            pr_train = prob_sampled[train_idx]
            pr_test = prob_sampled[test_idx]
            density_test = correction_density[test_idx]

            reg_phi, reg_pred = self._fit_surrogate(X_train, y_train, X_test, pr_train)
            residuals = y_test - reg_pred
            d_test = self._correction_density(sz_test, pr_test, density_test)
            if np.any(d_test <= 0):
                raise ValueError("Encountered zero subset density D(S) in unbiased MSR fold.")

            # Safe size-indexed semivalue weights for terms p_{|S|-1} and p_{|S|}.
            p_prev = np.zeros(len(sz_test), dtype=np.float64)
            prev_mask = sz_test > 0
            if np.any(prev_mask):
                p_prev[prev_mask] = self._p[sz_test[prev_mask] - 1]

            p_cur = np.zeros(len(sz_test), dtype=np.float64)
            cur_mask = sz_test < n
            if np.any(cur_mask):
                p_cur[cur_mask] = self._p[sz_test[cur_mask]]

            for idx in range(n):
                contained = X_test[:, idx]  # {0,1}
                term = p_prev * contained - p_cur * (1.0 - contained)
                correction = np.mean(residuals * term / d_test)
                phi[idx] += (reg_phi[idx] + correction) / k

        return phi


class RegressionMSR_unbiased(_MSRUnbiasedPaperMixin, RegressionMSR):
    """
    Paper-style unbiased RegressionMSR variant.
    Same sampling and surrogate behavior as RegressionMSR, with only the
    residual correction replaced by the Algorithm-1 unbiased correction.
    """


__all__ = [
    "exact_value",
    "sampling_lift",
    "sampling_lift_paired",
    "WSL",
    "WSL_paired",
    "WSL_banzhaf",
    "WSL_banzhaf_paired",
    "permutation",
    "permutation_paired",
    "weighted_permutation",
    "weighted_permutation_paired",
    "MSR",
    "MSR_paired",
    "improved_AME",
    "weighted_MSR",
    "weighted_MSR_paired",
    "kernelSHAP",
    "kernelSHAP_paired",
    "LeverageSHAP",
    "LeverageSHAP_original",
    "LeverageSHAP_border",
    "leverage",
    "leverage_paired",
    "modified_leverage",
    "modified_leverage_paired",
    "test_leverage",
    "test_leverage_paired",
    "unbiased_kernelSHAP",
    "unbiased_kernelSHAP_paired",
    "ARM",
    "ARM_shapley",
    "ARM_banzhaf",
    "complement",
    "AME",
    "AME_paired",
    "group_testing",
    "group_testing_paired",
    "GELS_ranking",
    "GELS_ranking_paired",
    "GELS",
    "GELS_paired",
    "WGELS_shapley",
    "WGELS_shapley_paired",
    "WGELS_banzhaf",
    "WGELS_banzhaf_paired",
    "GELS_shapley",
    "GELS_shapley_paired",
    "simSHAP",
    "simSHAP_paired",
    "OFA",
    "OFA_optimal",
    "OFA_optimal_paired",
    "OFA_fixed",
    "OFA_fixed_paired",
    "OFA_baseline",
    "OFA_baseline_paired",
    "SHAP_IQ",
    "SHAP_IQ_paired",
    "RegressionMSR",
    "PolySHAP_regression",
    "PolySHAP_regression_paired",
    "TreeMSR",
    "RegressionMSR_unbiased",
]
