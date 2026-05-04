import numpy as np
import itertools
import math
from scipy import special
import sys
import multiprocessing as mp
from tqdm import tqdm
from .utils import vd_tqdm
from scipy.linalg import lstsq

# -----------------------------------------------------------------------------
# runEstimator: orchestration for sampling, batching, and aggregation
#
# 1) Budget / batch relationships (key definitions and flow)
#    - nue_avg: total utility-evaluation budget per player (average). This is the
#      global budget that drives the overall number of samples to generate.
#      Each estimator converts nue_avg into num_sample based on its cost per
#      sample (e.g., sampling_lift uses ~2*num_player evals per sample,
#      permutation uses ~num_player, MSR uses ~1, etc.).
#
#    - nue_per_proc: target utility-evaluation budget per batch per process.
#      Each estimator converts this into batch_size by dividing by its
#      per-sample evaluation cost. This is why you often see division by
#      num_player (or 2*num_player): the cost per sample scales with the
#      number of players, so batch_size is set so a batch stays within the
#      desired evaluation budget.
#
#    - nue_per_proc_run: the realized utility-evaluation count per batch
#      after rounding batch_size to an integer. It is computed in the estimator
#      and printed at runtime. This makes the *actual* per-batch cost explicit.
#
#    - Number of batches: determined by num_sample and batch_size:
#          num_batches = ceil(num_sample / batch_size)
#      Total evaluations are therefore approximately
#          num_batches * nue_per_proc_run ≈ nue_avg * num_player
#      (up to rounding). Processes only affect how many batches run in parallel;
#      they do not change the total budget.
#
# 2) Sampling / run / aggregate / finalize (data flow)
#    - sampling(): yields arrays of samples of size batch_size (except possibly
#      the final smaller batch). Each sample encodes a coalition/permutation/etc.
#      The exact sample shape depends on the estimator (e.g., boolean subsets,
#      permutations, paired samples), but the outer dimension is the batch.
#
#    - run(samples): takes one batch of samples and returns a batch of results
#      (e.g., marginal contributions or sufficient statistics) with the same
#      leading dimension as the input batch.
#
#    - aggregate(results): appends batch results into a buffer, processes full
#      chunks of size interval_track, and records intermediate estimates into
#      values_traj. This is how tracking is interleaved with computation.
#
#    - finalize(): processes any leftover buffered results and returns the final
#      estimate plus the full trajectory of intermediate estimates.
# -----------------------------------------------------------------------------
class runEstimator:
    def __init__(self, *, estimator, n_process, semivalue, semivalue_param, game_func, game_args, num_player, nue_avg,
                 nue_per_proc, nue_track_avg, estimator_seed=2026, file_prog=None, **kwargs_estimator):
        self.estimator = estimator
        self.n_process = n_process - 1 # one process is used for aggregating results
        self.file_prog = file_prog
        self.semivalue = semivalue
        self.semivalue_param = semivalue_param
        self.game_func = game_func
        self.game_args = game_args
        self.estimator_seed = estimator_seed
        self.num_player = num_player

        # the number of utility evaluations used to do estimation on average (divided by the number of players)
        self.nue_avg = nue_avg

        # the number of utility evaluations each process will run in one batch.
        self.nue_per_proc = nue_per_proc

        # record the estimates of all players after using nue_track_avg, 2*nue_track_avg, ..., utility evaluations on average.
        self.nue_track_avg = nue_track_avg

        self.kwargs_estimator = kwargs_estimator

        self.estimator_run = None

    def run(self):
        estimator_args = dict(
            semivalue=self.semivalue,
            semivalue_param=self.semivalue_param,
            game_func=self.game_func,
            game_args=self.game_args,
            num_player=self.num_player,
            nue_avg=self.nue_avg,
            nue_per_proc=self.nue_per_proc,
            nue_track_avg=self.nue_track_avg,
            estimator_seed=self.estimator_seed
        )
        estimator = getattr(sys.modules[__name__], self.estimator)(**estimator_args, **self.kwargs_estimator)
        print(f"The number of utility evalutions each process runs in one batch is {estimator.nue_per_proc_run}")
        requires_serial_feedback = bool(getattr(estimator, "_requires_serial_feedback", False))
        num_batches_hint = int(getattr(estimator, "_num_batches_hint", -(-estimator.num_sample // estimator.batch_size)))
        if self.n_process > 1 and not requires_serial_feedback:
            with mp.Pool(self.n_process) as pool:
                process = pool.imap(estimator.run, estimator.sampling())
                for chunk in vd_tqdm(process, total=num_batches_hint,
                                  miniters=self.n_process, maxinterval=float('inf'), file_prog=self.file_prog):
                    estimator.aggregate(chunk)
        else:
            for samples in tqdm(estimator.sampling(), total=num_batches_hint):
                estimator.aggregate(estimator.run(samples))
        self.estimator_run = estimator
        return estimator.finalize()


class estimatorTemplate:
    def __init__(self, *, semivalue, semivalue_param, game_func, game_args, num_player, nue_avg, nue_per_proc, nue_track_avg,
                 estimator_seed, **kwargs_estimator):
        """
        Base class template for semivalue estimators.
        
        Parameters organized by category:
        
        GAME SPECIFICATION:
        -------------------
        game_func : callable
            The game/utility function constructor. Should return an object with an evaluate() method
            that computes the utility value for any coalition (subset of players).
            
        game_args : dict
            Arguments to pass to game_func during instantiation. Used as game_func(**game_args).
            
        num_player : int
            Total number of players in the game. Determines the size of the coalition space (2^num_player).
        
        SEMIVALUE SPECIFICATION:
        ------------------------
        semivalue : str
            Type of semivalue to compute. Supported values:
            - "shapley": Shapley value (uniform distribution over coalition sizes)
            - "weighted_banzhaf": Weighted Banzhaf value with parameter p
            - "beta_shapley": Beta-Shapley value with alpha, beta parameters
            
        semivalue_param : float or tuple
            Parameter(s) for the semivalue:
            - For "shapley": Not used (can be None)
            - For "weighted_banzhaf": p value in (0, 1) controlling the Bernoulli distribution
            - For "beta_shapley": Tuple (alpha, beta) for the beta distribution
        
        SAMPLING BUDGET & TRACKING:
        ---------------------------
        nue_avg : int
            Total utility evaluation budget divided by num_player (i.e., budget per player on average).
            This is the primary parameter controlling estimation accuracy vs. computational cost.
            
        nue_per_proc : int
            Number of utility evaluations allocated to each process in one batch.
            Controls the granularity of parallel processing and how often results are aggregated.
            The actual value used (nue_per_proc_run) may differ due to rounding in subclasses.
            
        nue_track_avg : int
            Tracking interval for recording intermediate estimates. An estimate is saved to values_traj
            after every nue_track_avg utility evaluations (on average, per player).
            Used to monitor convergence and create trajectories of estimate quality over time.
        
        OTHER PARAMETERS:
        -----------------
        estimator_seed : int
            Random seed for reproducibility of sampling. Ensures deterministic behavior across runs.
            
        kwargs_estimator : dict
            Additional keyword arguments specific to particular estimator subclasses.
            Allows for extensibility without modifying the base class signature.
        """
        
        # Store game specification
        self.game_func = game_func
        self.game_args = game_args
        self.num_player = num_player
        
        # Store semivalue specification
        self.semivalue = semivalue
        self.semivalue_param = semivalue_param
        
        # Store sampling budget and tracking parameters
        self.nue_avg = nue_avg
        self.nue_per_proc = nue_per_proc
        self.nue_track_avg = nue_track_avg
        
        # Store other parameters
        self.estimator_seed = estimator_seed
        self.kwargs_estimator = kwargs_estimator

        # Initialize trajectory storage for intermediate estimates
        num_traj = self.nue_avg // self.nue_track_avg
        self.values_traj = np.empty((num_traj, self.num_player), dtype=np.float64)
        self.pos_traj = 0  # Current position in values_traj
        
        # Initialize buffer and sampling attributes (subclasses will define specifics)
        self.buffer = None  # Temporary storage for results; size = interval_track + batch_size - 1
        self.interval_track = None  # Number of samples between trajectory recordings (subclass-specific)
        self.batch_size = None  # Number of samples per batch (subclass-specific)
        self.num_sample = None  # Total number of samples to generate (subclass-specific)
        self.nue_per_proc_run = None  # Actual utility evaluations per process run after rounding (subclass-specific)
        self.pos_buffer = 0  # Current fill position in buffer
        self.samples = None  # Array to hold current batch of samples (subclass-specific)

        # Coupling/antithetic sampling control
        self.lock_switch = True  # Whether to allow switching (unlocked by paired estimators)
        self.switch_state = False  # Current state of antithetic coupling

    @property
    def switch(self):
        return self.switch_state

    @switch.setter
    def switch(self, state):
        if not self.lock_switch:
            self.switch_state = state

    @property
    def buffer_size(self):
        return self.interval_track + self.batch_size - 1

    def run(self):
        pass

    def _init_indiv(self):
        pass

    def sampling(self):
        self._init_indiv()
        np.random.seed(self.estimator_seed)

        count = 0
        for _ in range(self.num_sample):
            if not self.switch:
                self.samples[count] = self._generator()
                self.switch = True
            else:
                self.samples[count] = 1 - self.samples[count - 1]
                self.switch = False
            count += 1
            if count == self.batch_size:
                yield self.samples.copy()
                count = 0
        if count:
            yield self.samples[:count]

    def _generator(self):
        pass

    def aggregate(self, results_collect):
        self.buffer[self.pos_buffer:self.pos_buffer + len(results_collect)] = results_collect
        self.pos_buffer += len(results_collect)
        num_collect = self.pos_buffer // self.interval_track
        if num_collect:
            for i in range(num_collect):
                self._process(self.buffer[i*self.interval_track:(i+1)*self.interval_track])
                self.values_traj[self.pos_traj] = self._estimate()
                self.pos_traj += 1
            num_left = self.pos_buffer - (i + 1) * self.interval_track
            self.buffer[:num_left] = self.buffer[(i + 1) * self.interval_track:self.pos_buffer]
            self.pos_buffer = num_left

    def finalize(self):
        if self.pos_buffer:
            self._process(self.buffer[:self.pos_buffer])
            values_final = self._estimate()
        else:
            values_final = self.values_traj[-1]
        # Ensure trajectory tail is always valid. Some estimators may round
        # num_sample down (e.g., for k-fold divisibility), leaving the final
        # tracking slot(s) unfilled if total samples are just below the last
        # checkpoint. Fill any remaining slots with the final estimate.
        if self.values_traj.size:
            if self.pos_traj < len(self.values_traj):
                self.values_traj[self.pos_traj:] = values_final
            else:
                self.values_traj[-1] = values_final
        return values_final, self.values_traj

    def _process(self, inputs):
        pass

    def _estimate(self):
        pass

    def distribution_cardinality(self):
        if self.semivalue == "shapley":
            weights = np.full(self.num_player, 1. / self.num_player, dtype=np.float64)
        elif self.semivalue == "weighted_banzhaf":
            weights = np.ones(self.num_player, dtype=np.float64)
            for k in range(self.num_player):
                for i in range(k):
                    weights[k] *= (self.num_player - 1 - i) / (i + 1) * self.semivalue_param * (1 - self.semivalue_param)
                weights[k] *= (1 - self.semivalue_param) ** (self.num_player - 1 - 2 * k)
        elif self.semivalue == "beta_shapley":
            alpha, beta = self.semivalue_param
            weights = np.ones(self.num_player, dtype=np.float64)
            tmp_range = np.arange(1, self.num_player, dtype=np.float64)
            weights *= np.divide(tmp_range, tmp_range + (alpha + beta - 1)).prod()
            for s in range(self.num_player):
                r_cur = weights[s]
                tmp_range = np.arange(1, s + 1, dtype=np.float64)
                r_cur *= np.divide(tmp_range + (beta - 1), tmp_range).prod()
                tmp_range = np.arange(1, self.num_player - s, dtype=np.float64)
                r_cur *= np.divide((alpha - 1) + tmp_range, tmp_range).prod()
                weights[s] = r_cur
        else:
            raise NotImplementedError(f"Check {self.semivalue}")
        return weights


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

        num_pre = self.results_aggregate["count"]
        num_cur = len(inputs) + num_pre
        self.results_aggregate["estimates"] *= num_pre / num_cur

        self.results_aggregate["estimates"] += (ues * np.reciprocal(weights) * subsets).sum(axis=0) / num_cur
        subsets = 1 - subsets
        weights = 1 - weights
        self.results_aggregate["estimates"] -= (ues * np.reciprocal(weights) * subsets).sum(axis=0) / num_cur
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
        # Note what in the above is equal to
        # pos = np.random.choice(np.arange(self.num_player), size=s, replace=False)
        # subset[pos] = True
        # But we stay loyal to the original paper
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
# External baseline estimators — RegressionMSR / TreeMSR
# Kolpaczki et al. (2025), arxiv:2506.11849
# =============================================================================

# ---------------------------------------------------------------------------
# Helpers (exact ports from regressionMSR/estimators/regMSR.py)
# ---------------------------------------------------------------------------

def _msr_leverage_shap(X, y, prob_sampled, v0, v1, p):
    """
    Closed-form Leverage SHAP linear fit.
    Exact port of leverage_shap() from regressionMSR/estimators/regMSR.py.

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


# ---------------------------------------------------------------------------
# Shared base class
# ---------------------------------------------------------------------------

class _MSRBase(estimatorTemplate):
    """
    Shared base for RegressionMSR and TreeMSR.

    Implements the sampling distribution, k=10 fold cross-validation, and MSR
    correction exactly as in UniversalMSR.explain() from regMSR.py.
    Subclasses implement _setup_case() and _fit_surrogate().
    """

    def __init__(
        self,
        *,
        sampling_with_replacement=None,
        paired_sampling=None,
        use_special_surrogates=True,
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
        symmetric_weights = bool(np.allclose(p, p[::-1]))
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

        # Match UniversalMSR.explain() budget rounding to k=10 CV.
        self._kfold = 10
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
        self._current_estimate = np.zeros(n, dtype=np.float64)
        self._rng = np.random.Generator(np.random.PCG64(self.estimator_seed))
        self._sample_matrix = None
        self._sample_prob = None

        self.buffer  = np.empty((self.buffer_size, n + 2), dtype=np.float64)
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

    # ── easeshap interface ─────────────────────────────────────────────────────

    def _init_indiv(self):
        # In original regMSR.py, Leverage SHAP uses model predictions on baseline
        # (all-0) and explicand (all-1), not sampled boundary rows.
        if self._is_leverage_shap and (self._v0 is None or self._v1 is None):
            game = self.game_func(**self.game_args)
            n = self.num_player
            self._v0 = float(game.evaluate(np.zeros(n, dtype=bool)))
            self._v1 = float(game.evaluate(np.ones(n, dtype=bool)))

    def _add_sample(self, X, prob_arr, row_idx, indices, prob):
        if not self._pair_sampling:
            X[row_idx, indices] = True
            prob_arr[row_idx] = prob
        else:
            X[2 * row_idx, indices] = True
            prob_arr[2 * row_idx] = prob
            comp = np.setdiff1d(np.arange(self.num_player), np.asarray(indices, dtype=int))
            X[2 * row_idx + 1, comp] = True
            prob_arr[2 * row_idx + 1] = prob

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

    def _sample_with_replacement(self, target_samples):
        n = self.num_player
        # Paired mode writes two rows per draw. If target is odd, allocate one
        # extra row and truncate at the end to avoid leaving an unfilled zero-
        # probability row.
        work_target = target_samples
        if self._pair_sampling and (work_target % 2 == 1):
            work_target += 1

        X = np.zeros((work_target, n), dtype=bool)
        probs = np.zeros(work_target, dtype=np.float64)

        size_binom = np.array([float(special.comb(n, s, exact=False)) for s in self._valid_sizes])
        sample_dist_size = self._sample_dist * size_binom
        sample_dist_size = sample_dist_size / sample_dist_size.sum()
        num_draws = work_target // 2 if self._pair_sampling else work_target
        sampled_sizes = self._rng.choice(self._valid_sizes, size=num_draws, p=sample_dist_size)

        for idx, s in enumerate(sampled_sizes):
            s = int(s)
            indices = self._rng.choice(n, size=s, replace=False)
            self._add_sample(X, probs, idx, indices, self._sample_dist[s - self._adjust])
        return X[:target_samples], probs[:target_samples]

    def _sample_without_replacement(self, target_samples):
        n = self.num_player
        C = self._find_bernoulli_constant(target_samples)
        sample_dist_C = np.minimum(self._sample_dist * C, np.ones_like(self._sample_dist))

        size_binom = np.array([float(special.comb(n, s, exact=False)) for s in self._valid_sizes])
        m_total = int(np.sum([round(p * b) for p, b in zip(sample_dist_C, size_binom)]))
        if self._pair_sampling:
            m_total = (m_total // 2) * 2

        X = np.zeros((m_total, n), dtype=bool)
        probs = np.zeros(m_total, dtype=np.float64)
        idx = 0

        for s, prob in zip(self._valid_sizes, sample_dist_C):
            s = int(s)
            m_s = int(round(prob * float(special.comb(n, s, exact=False))))
            if m_s <= 0:
                continue

            if self._pair_sampling and s == n // 2 and n % 2 == 0:
                # Avoid duplicate complements when n even and s=n/2.
                gen = _msr_combination_generator(self._rng, n - 1, s - 1, m_s // 2)
                for combo in gen:
                    self._add_sample(X, probs, idx, list(combo) + [n - 1], prob)
                    idx += 1
                break

            gen = _msr_combination_generator(self._rng, n, s, m_s)
            for combo in gen:
                self._add_sample(X, probs, idx, combo, prob)
                idx += 1

            if self._pair_sampling and s == n // 2:
                break

        if idx < m_total:
            # In paired mode idx counts pair-draws, i.e., 2 rows each.
            used_rows = 2 * idx if self._pair_sampling else idx
            X = X[:used_rows]
            probs = probs[:used_rows]

        # Keep easeshap trajectory shape stable: force exact target length.
        cur = len(X)
        if cur > target_samples:
            X = X[:target_samples]
            probs = probs[:target_samples]
        elif cur < target_samples:
            add_X, add_p = self._sample_with_replacement(target_samples - cur)
            X = np.vstack([X, add_X]) if cur else add_X
            probs = np.concatenate([probs, add_p]) if cur else add_p
        return X, probs

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
        if use_with_replacement:
            self._sample_matrix, self._sample_prob = self._sample_with_replacement(self.num_sample)
        else:
            self._sample_matrix, self._sample_prob = self._sample_without_replacement(self.num_sample)

        N = len(self._sample_matrix)
        for start in range(0, N, self.batch_size):
            end = min(start + self.batch_size, N)
            batch = np.empty((end - start, self.num_player + 1), dtype=np.float64)
            batch[:, :self.num_player] = self._sample_matrix[start:end].astype(np.float64)
            batch[:, self.num_player] = self._sample_prob[start:end]
            yield batch

    def _generator(self):
        raise NotImplementedError("_generator is unused in _MSRBase; use sampling().")

    def run(self, samples):
        """Evaluate game for each coalition in the batch."""
        game    = self.game_func(**self.game_args)
        n       = self.num_player
        X = samples[:, :n].astype(bool)
        prob = samples[:, n]
        results = np.empty((len(samples), n + 2), dtype=np.float64)
        results[:, :n] = X.astype(np.float64)
        for i in range(len(samples)):
            results[i, n] = game.evaluate(X[i])
            results[i, n + 1] = prob[i]
        return results

    def _process(self, inputs):
        """Accumulate sampled (X, y, prob) chunks."""
        n     = self.num_player
        X_raw = inputs[:, :n].astype(np.float64)
        y_raw = inputs[:, n].copy()
        p_raw = inputs[:, n + 1].copy()
        self._X_chunks.append(X_raw)
        self._y_chunks.append(y_raw)
        self._prob_chunks.append(p_raw)

    def _estimate(self):
        """Run k=10 fold regression and return semivalue estimates."""
        return self._run_kfold(k=10)

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

            # Match original regressionMSR behavior: do not silently skip fold
            # failures. If surrogate fitting fails, propagate the exception.
            reg_phi, reg_pred = self._fit_surrogate(
                X_train, y_train, X_test, pr_train
            )

            residuals = y_test - reg_pred      # (|test|,)

            # ── MSR correction (per player, exact match of regMSR.py) ─────
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
    LinearMSR from Kolpaczki et al. (2025), arxiv:2506.11849.

    Fits a linear surrogate f(S) to sampled (coalition, utility) pairs and
    applies the MSR residual correction with k=10 fold cross-validation.
    Three special cases match the original authors' code exactly:

      • Shapley / Beta(1,1) → Leverage SHAP  (closed-form projected WLS)
      • Banzhaf (p=0.5)     → Kernel Banzhaf (centred OLS, no intercept)
      • All other semivalues → sklearn LinearRegression (with intercept)

    Hyperparameters exactly match the original authors' implementation.
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
    PolySHAP from Fumagalli, Witter, and Musco (2026), adapted to the
    easeshap estimator interface.

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
        for _ in range(remaining):
            if not self.switch:
                self.samples[count] = self._generator()
                self.switch = True
            else:
                self.samples[count] = 1 - self.samples[count - 1]   # complement
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
    TreeMSR from Kolpaczki et al. (2025), arxiv:2506.11849.

    Replaces the linear surrogate with XGBoost (default parameters, matching
    the original authors' code exactly: xgboost.XGBRegressor() with no args).
    Exact surrogate semivalues are extracted via tree traversal:
      • Shapley      → shap.TreeExplainer (matches tree_shap in treeshap.py)
      • Other svs    → tree_prob from regressionMSR.exact.treeprob (DP over
                       tree paths with general semivalue weights p[k])

    MSR correction and k=10 fold CV are inherited from _MSRBase.

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

        # Exactly matching original: XGBRegressor() with NO custom parameters
        model = xgb.XGBRegressor()
        model.fit(X_train, y_train)
        reg_pred = model.predict(X_test)

        baseline  = np.zeros((1, n), dtype=np.float64)
        explicand = np.ones( (1, n), dtype=np.float64)

        if self.semivalue == 'shapley':
            # TreeSHAP via shap library (matches tree_shap() in treeshap.py)
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


class EaseSHAP(estimatorTemplate):
    """
    Draft 19 Appendix B implementation with pooled-refit cross-fitting.

    Supported working-surrogate bases:

      - `surrogate_basis = d` for an integer d >= 0:
            intercept
          + all interaction monomials 1{T subseteq S} for 1 <= |T| <= d
          + optional nonlinear size terms.
        In particular, d=1 is the previous default singleton-indicator class,
        and d=0 is the intercept-only class.

      - `surrogate_basis = "size_player"`:
            intercept
          + size-by-player indicators h_{i,s}(S) = 1{|S| = s, i in S}
          + optional nonlinear size terms.

    The implementation uses the pooled shared-surrogate criterion from
    Eqs. (73)-(77), but replaces the paper's external auxiliary-training sample
    with K-fold cross-fitting on the collected evaluation coalitions.

    Optional exact boundary handling evaluates all coalitions with
    |S| in {0, 1, n-1, n} once up front, charges those utility evaluations
    against the total budget, restricts random sampling to interior sizes
    2, ..., n-2, and adds the exact boundary contribution back to every
    reported estimate.

    Stage 1 uses the initialization law q_init(S) ∝ sqrt(A_|S|). Stage 2 uses:
      - the single-coalition pilot residual MSE rule for generic semivalues;
      - the complement-pair pilot residual-difference rule from Eqs. (82)-(84)
        for symmetric semivalues when `use_complement_sampling=True`.

    For symmetric semivalues, the sampler emits complement pairs (S, S^c) and
    stores the coalition-level law q with q(S)=q(S^c)=pi({S,S^c})/2. The final
    estimator is recomputed at each tracked stage using the current cross-fitted
    surrogate(s), so previously collected observations are re-scored under the
    current refit rather than frozen at the time they arrived.
    """

    def __init__(
        self,
        *,
        pilot_nue=None,
        pilot_fraction=0.2,
        use_complement_sampling=True,
        surrogate_ridge_lambda=1.0,
        surrogate_ridge_schedule="fixed",
        stage2_size_floor=1e-8,
        stage2_min_size_count=2,
        num_folds=10,
        surrogate_basis=1,
        include_nonlinear_size_terms=True,
        exact_boundary_handling=True,
        **kwargs,
    ):
        if pilot_nue is not None:
            if not isinstance(pilot_nue, (int, np.integer)):
                raise ValueError(f"`pilot_nue` must be an integer or None, got {pilot_nue!r}.")
            if int(pilot_nue) < 0:
                raise ValueError(f"`pilot_nue` must be >= 0, got {pilot_nue!r}.")
        pilot_fraction = float(pilot_fraction)
        if not np.isfinite(pilot_fraction) or not (0.0 <= pilot_fraction <= 1.0):
            raise ValueError(
                "`pilot_fraction` must be finite and lie in [0, 1], "
                f"got {pilot_fraction!r}."
            )

        self.pilot_nue = None if pilot_nue is None else int(pilot_nue)
        self.pilot_fraction = pilot_fraction
        self.use_complement_sampling = bool(use_complement_sampling)

        self.surrogate_ridge_lambda = float(surrogate_ridge_lambda)
        if not np.isfinite(self.surrogate_ridge_lambda) or self.surrogate_ridge_lambda < 0.0:
            raise ValueError(
                "`surrogate_ridge_lambda` must be finite and >= 0, "
                f"got {surrogate_ridge_lambda!r}."
            )
        if not isinstance(surrogate_ridge_schedule, str):
            raise ValueError(
                "`surrogate_ridge_schedule` must be a string in "
                '{"fixed", "times_m"}, '
                f"got {surrogate_ridge_schedule!r}."
            )
        self.surrogate_ridge_schedule = surrogate_ridge_schedule.strip().lower()
        if self.surrogate_ridge_schedule not in {"fixed", "times_m"}:
            raise ValueError(
                "Unknown `surrogate_ridge_schedule`. Use "
                '"fixed" or "times_m". '
                f"Got {surrogate_ridge_schedule!r}."
            )

        self.stage2_size_floor = float(stage2_size_floor)
        if not np.isfinite(self.stage2_size_floor) or not (0.0 <= self.stage2_size_floor < 1.0):
            raise ValueError(
                "`stage2_size_floor` must be finite and lie in [0, 1), "
                f"got {stage2_size_floor!r}."
            )

        if not isinstance(stage2_min_size_count, (int, np.integer)):
            raise ValueError(
                "`stage2_min_size_count` must be an integer >= 1, "
                f"got {stage2_min_size_count!r}."
            )
        self.stage2_min_size_count = int(stage2_min_size_count)
        if self.stage2_min_size_count < 1:
            raise ValueError(
                f"`stage2_min_size_count` must be >= 1, got {stage2_min_size_count!r}."
            )

        if not isinstance(num_folds, (int, np.integer)):
            raise ValueError(f"`num_folds` must be an integer >= 2, got {num_folds!r}.")
        self.num_folds = int(num_folds)
        if self.num_folds < 2:
            raise ValueError(f"`num_folds` must be >= 2, got {num_folds!r}.")

        self.surrogate_basis = surrogate_basis
        self.include_nonlinear_size_terms = bool(include_nonlinear_size_terms)
        self.exact_boundary_handling = bool(exact_boundary_handling)
        self._surrogate_basis_kind = None
        self._interaction_degree = None
        self._parse_surrogate_basis()

        self._requires_serial_feedback = True

        super(EaseSHAP, self).__init__(**kwargs)

        n = self.num_player

        dist_card = self.distribution_cardinality()
        p = np.zeros(n, dtype=np.float64)
        for k in range(n):
            denom = float(special.comb(n - 1, k, exact=False))
            if denom > 0.0:
                p[k] = dist_card[k] / denom
        self._p = p

        self._symmetric_semivalue = bool(np.allclose(self._p, self._p[::-1]))
        self._pair_sampling = bool(self.use_complement_sampling and self._symmetric_semivalue)

        self._sampling_size_mask = np.ones(n + 1, dtype=bool)
        self._boundary_X = np.empty((0, n), dtype=bool)
        if self.exact_boundary_handling:
            self._sampling_size_mask[:] = False
            if n >= 4:
                self._sampling_size_mask[2:n - 1] = True
            self._boundary_X = self._build_boundary_subset_matrix()
        self._boundary_eval_count = int(len(self._boundary_X))

        total_budget = self.nue_avg * n
        if total_budget < self._boundary_eval_count:
            raise ValueError(
                "Strict boundary handling requires at least "
                f"{self._boundary_eval_count} utility evaluations, but "
                f"`nue_avg * num_player` is only {total_budget}."
            )

        remaining_budget = total_budget - self._boundary_eval_count
        if not np.any(self._sampling_size_mask):
            remaining_budget = 0

        self.num_sample = remaining_budget
        self.interval_track = self.nue_track_avg * n
        self.batch_size = self.nue_per_proc
        if self._pair_sampling:
            self.batch_size = max(2, (self.batch_size // 2) * 2)
            self.num_sample = (self.num_sample // 2) * 2
        self.nue_per_proc_run = self.batch_size

        self._rng = np.random.Generator(np.random.PCG64(self.estimator_seed))
        self._fold_rng = np.random.Generator(np.random.PCG64(self.estimator_seed + 9173))
        self._binom_by_size = np.array(
            [float(special.comb(n, s, exact=False)) for s in range(n + 1)],
            dtype=np.float64,
        )
        self._A_size = self._compute_A_size()
        self._build_feature_map_and_readout(dist_card)
        self._boundary_sizes = self._boundary_X.sum(axis=1).astype(np.int64)
        self._boundary_Z = (
            self._build_feature_block(self._boundary_X)[0]
            if len(self._boundary_X)
            else np.empty((0, self._feature_dim), dtype=np.float64)
        )
        self._boundary_raw_gamma = self._raw_gamma_matrix(self._boundary_X, self._boundary_sizes)
        self._boundary_values = self._evaluate_boundary_values()
        self._boundary_exact = self._boundary_raw_gamma.T @ self._boundary_values
        self._q_init_size = self._normalize_size_law(np.sqrt(self._A_size), enforce_symmetric=self._pair_sampling)
        self._q_stage2_size = self._q_init_size.copy()

        if self.pilot_nue is None:
            pilot_samples = int(round(self.pilot_fraction * self.num_sample))
        else:
            pilot_samples = self.pilot_nue * n
        pilot_samples = max(0, min(pilot_samples, self.num_sample))
        if self._pair_sampling:
            pilot_samples = max(0, (pilot_samples // 2) * 2)
        self._pilot_num_sample = pilot_samples
        self._num_batches_hint = self._count_phase_batches(self._pilot_num_sample) + self._count_phase_batches(
            self.num_sample - self._pilot_num_sample
        )

        self._current_estimate = self._boundary_exact.copy()

        dense_state_bytes = self._estimate_dense_state_bytes()
        if dense_state_bytes > 2_000_000_000:
            raise ValueError(
                "Selected surrogate basis is too large for the pooled-refit dense state "
                f"(estimated bytes={dense_state_bytes}). Reduce `surrogate_basis`, "
                "disable nonlinear size terms, or use a smaller budget."
            )

        # Stored evaluation sample up to self._num_obs.
        self._num_obs = 0
        self._X_obs = np.empty((self.num_sample, n), dtype=bool)
        self._y_obs = np.empty(self.num_sample, dtype=np.float64)
        self._q_obs = np.empty(self.num_sample, dtype=np.float64)
        self._size_obs = np.empty(self.num_sample, dtype=np.int64)
        self._fold_obs = np.empty(self.num_sample, dtype=np.int64)

        # Running pooled sufficient statistics over all observations, plus
        # fold-specific slices for leave-fold-out refits.
        d = self._feature_dim
        K = self.num_folds
        self._total_count = 0
        self._A_total = np.zeros((d, d), dtype=np.float64)
        self._c_total = np.zeros(d, dtype=np.float64)
        self._B_total = np.zeros((n, d), dtype=np.float64)
        self._b_total = np.zeros(n, dtype=np.float64)

        self._fold_count = np.zeros(K, dtype=np.int64)
        self._A_fold = np.zeros((K, d, d), dtype=np.float64)
        self._c_fold = np.zeros((K, d), dtype=np.float64)
        self._B_fold = np.zeros((K, n, d), dtype=np.float64)
        self._b_fold = np.zeros((K, n), dtype=np.float64)

        self._pilot_finalized = (self._pilot_num_sample == 0)
        self._record_traj_up_to_current_budget()

    def _parse_surrogate_basis(self):
        basis = self.surrogate_basis
        if isinstance(basis, (int, np.integer)):
            degree = int(basis)
            if degree < 0:
                raise ValueError(
                    f"`surrogate_basis` as an integer must be >= 0, got {basis!r}."
                )
            self._surrogate_basis_kind = "interactions"
            self._interaction_degree = degree
            return

        if not isinstance(basis, str):
            raise ValueError(
                "`surrogate_basis` must be either an integer interaction degree "
                'or one of {"none", "size_player"}, '
                f"got {basis!r}."
            )

        key = basis.strip().lower().replace("-", "_")
        if key in {"none", "constant", "intercept"}:
            self._surrogate_basis_kind = "interactions"
            self._interaction_degree = 0
            return
        if key in {"size_player", "size_by_player", "player_size", "is", "i_s"}:
            self._surrogate_basis_kind = "size_player"
            self._interaction_degree = None
            return

        raise ValueError(
            "Unknown `surrogate_basis`. Use an integer degree, "
            '"none", or "size_player". '
            f"Got {basis!r}."
        )

    def _build_feature_map_and_readout(self, dist_card):
        n = self.num_player
        self._interaction_blocks = []
        self._size_player_start = None
        self._log_col = None
        self._quad_col = None

        next_col = 1  # intercept is always column 0
        if self.include_nonlinear_size_terms:
            self._log_col = next_col
            next_col += 1
            self._quad_col = next_col
            next_col += 1

        if self._surrogate_basis_kind == "interactions":
            degree = self._interaction_degree
            if degree > n:
                raise ValueError(
                    f"`surrogate_basis={degree}` exceeds num_player={n}."
                )
            num_interactions = 0
            for r in range(1, degree + 1):
                num_interactions += math.comb(n, r)
            self._feature_dim = next_col + num_interactions
        elif self._surrogate_basis_kind == "size_player":
            self._size_player_start = next_col
            self._feature_dim = next_col + n * n
        else:
            raise RuntimeError(f"Unexpected surrogate basis kind {self._surrogate_basis_kind!r}.")

        zeta = np.zeros((n, self._feature_dim), dtype=np.float64)
        if self._log_col is not None:
            s_grid = np.arange(n + 1, dtype=np.float64)
            phi_log_const = float(dist_card @ (np.log1p(s_grid[1:]) - np.log1p(s_grid[:-1])))
            quad_grid = (s_grid / float(n)) ** 2
            phi_quad_const = float(dist_card @ (quad_grid[1:] - quad_grid[:-1]))
            zeta[:, self._log_col] = phi_log_const
            zeta[:, self._quad_col] = phi_quad_const

        if self._surrogate_basis_kind == "interactions":
            start_col = next_col
            alpha = self._p
            omega_by_degree = {}
            for r in range(1, self._interaction_degree + 1):
                s_idx = np.arange(r - 1, n, dtype=int)
                comb_terms = special.comb(n - r, s_idx - r + 1, exact=False)
                omega_by_degree[r] = float(np.dot(comb_terms, alpha[s_idx]))

            for r in range(1, self._interaction_degree + 1):
                m = math.comb(n, r)
                if m == 0:
                    continue
                flat = np.fromiter(
                    itertools.chain.from_iterable(itertools.combinations(range(n), r)),
                    dtype=np.int64,
                    count=m * r,
                )
                combos = flat.reshape(m, r)
                self._interaction_blocks.append((r, combos, start_col))
                cols = np.arange(start_col, start_col + m, dtype=np.int64)
                omega = omega_by_degree[r]
                for pos in range(r):
                    zeta[combos[:, pos], cols] = omega
                start_col += m
        else:
            for s in range(1, n + 1):
                diag = float(special.comb(n - 1, s - 1, exact=False)) * self._p[s - 1]
                alpha_cur = self._p[s] if s < n else 0.0
                off = (
                    float(special.comb(n - 2, s - 2, exact=False)) * self._p[s - 1]
                    - float(special.comb(n - 2, s - 1, exact=False)) * alpha_cur
                )
                block_start = self._size_player_start + (s - 1) * n
                block_end = block_start + n
                zeta[:, block_start:block_end] = off
                diag_idx = block_start + np.arange(n, dtype=np.int64)
                zeta[np.arange(n, dtype=np.int64), diag_idx] = diag

        self._zeta = zeta

    def _build_boundary_subset_matrix(self):
        n = self.num_player
        subsets = []
        seen = set()

        def add(mask):
            key = np.packbits(mask.astype(np.uint8), bitorder="little").tobytes()
            if key not in seen:
                seen.add(key)
                subsets.append(mask.copy())

        mask = np.zeros(n, dtype=bool)
        add(mask)
        for i in range(n):
            mask_i = np.zeros(n, dtype=bool)
            mask_i[i] = True
            add(mask_i)
        full = np.ones(n, dtype=bool)
        for i in range(n):
            mask_rm = full.copy()
            mask_rm[i] = False
            add(mask_rm)
        add(full)

        if not subsets:
            return np.empty((0, n), dtype=bool)
        return np.vstack(subsets)

    def _evaluate_boundary_values(self):
        if len(self._boundary_X) == 0:
            return np.empty(0, dtype=np.float64)

        game = self.game_func(**self.game_args)
        values = np.empty(len(self._boundary_X), dtype=np.float64)
        for idx, subset in enumerate(self._boundary_X):
            values[idx] = game.evaluate(subset)
        return values

    def _record_traj_up_to_current_budget(self):
        total_eval = self._boundary_eval_count + self._num_obs
        while (
            self.pos_traj < len(self.values_traj)
            and total_eval >= (self.pos_traj + 1) * self.interval_track
        ):
            self.values_traj[self.pos_traj] = self._current_estimate
            self.pos_traj += 1

    def _estimate_dense_state_bytes(self):
        d = self._feature_dim
        n = self.num_player
        K = self.num_folds
        obs_bytes = (
            self.num_sample * n +          # X_obs bool
            self.num_sample * 8 * 4        # y_obs, q_obs, size_obs, fold_obs
        )
        stats_bytes = (
            8 * d * d * (K + 1) +          # A_total / A_fold
            8 * d * (K + 1) +              # c_total / c_fold
            8 * n * d * (K + 1) +          # B_total / B_fold
            8 * n * (K + 1)                # b_total / b_fold
        )
        return int(obs_bytes + stats_bytes)

    def _compute_A_size(self):
        n = self.num_player
        out = np.zeros(n + 1, dtype=np.float64)
        for s in range(n + 1):
            val = 0.0
            if s > 0:
                val += float(s) * (self._p[s - 1] ** 2)
            if s < n:
                val += float(n - s) * (self._p[s] ** 2)
            out[s] = val
        return out

    def _normalize_size_law(self, size_factor, *, enforce_symmetric=False):
        size_factor = np.asarray(size_factor, dtype=np.float64)
        if size_factor.shape != (self.num_player + 1,):
            raise ValueError(
                f"Expected size_factor shape {(self.num_player + 1,)}, got {size_factor.shape}."
            )
        if not np.all(np.isfinite(size_factor)):
            raise ValueError("Encountered non-finite entries in size_factor.")

        size_factor = np.maximum(size_factor, 0.0)
        if enforce_symmetric:
            size_factor = 0.5 * (size_factor + size_factor[::-1])
        size_factor = np.where(self._sampling_size_mask, size_factor, 0.0)

        total = float(np.dot(self._binom_by_size, size_factor))
        if total <= 0.0:
            fallback = np.zeros_like(size_factor)
            total_mass = float(self._binom_by_size[self._sampling_size_mask].sum())
            if total_mass > 0.0:
                fallback[self._sampling_size_mask] = 1.0 / total_mass
            return fallback
        return size_factor / total

    def _mix_with_init_size_mass(self, size_factor, mix_weight, *, enforce_symmetric=False):
        size_factor = np.asarray(size_factor, dtype=np.float64)
        if size_factor.shape != (self.num_player + 1,):
            raise ValueError(
                f"Expected size_factor shape {(self.num_player + 1,)}, got {size_factor.shape}."
            )
        if not np.all(np.isfinite(size_factor)):
            raise ValueError("Encountered non-finite entries in size_factor.")

        size_factor = np.maximum(size_factor, 0.0)
        if enforce_symmetric:
            size_factor = 0.5 * (size_factor + size_factor[::-1])
        size_factor = np.where(self._sampling_size_mask, size_factor, 0.0)

        target_mass = self._binom_by_size * size_factor
        target_total = float(target_mass.sum())
        if target_total <= 0.0:
            return self._q_init_size.copy()

        target_mass = target_mass / target_total
        init_mass = self._binom_by_size * self._q_init_size
        init_total = float(init_mass.sum())
        if init_total > 0.0:
            init_mass = init_mass / init_total
            target_mass = (1.0 - mix_weight) * target_mass + mix_weight * init_mass

        q_size = np.zeros_like(size_factor)
        valid = (self._binom_by_size > 0.0) & self._sampling_size_mask
        q_size[valid] = target_mass[valid] / self._binom_by_size[valid]
        return q_size

    def _size_mass(self, q_size):
        mass = self._binom_by_size * q_size
        total = float(mass.sum())
        if total <= 0.0:
            raise ValueError("Encountered non-positive size mass.")
        return mass / total

    def _batch_rows(self, remaining):
        if not self._pair_sampling:
            return min(self.batch_size, remaining)
        cur = min(self.batch_size, remaining)
        if cur % 2 == 1:
            cur -= 1
        if cur <= 0:
            cur = min(remaining, 2)
        return cur

    def _count_phase_batches(self, num_rows):
        if num_rows <= 0:
            return 0
        return -(-num_rows // self.batch_size)

    def _sample_batch(self, num_rows, q_size):
        n = self.num_player
        mass = self._size_mass(q_size)

        out = np.empty((num_rows, n + 1), dtype=np.float64)
        if not self._pair_sampling:
            sampled_sizes = self._rng.choice(np.arange(n + 1), size=num_rows, p=mass)
            for row, s in enumerate(sampled_sizes):
                s = int(s)
                subset = np.zeros(n, dtype=bool)
                if s > 0:
                    subset[self._rng.choice(n, size=s, replace=False)] = True
                out[row, :n] = subset.astype(np.float64)
                out[row, n] = q_size[s]
            return out

        num_pairs = num_rows // 2
        sampled_sizes = self._rng.choice(np.arange(n + 1), size=num_pairs, p=mass)
        for pair_idx, s in enumerate(sampled_sizes):
            s = int(s)
            subset = np.zeros(n, dtype=bool)
            if s > 0:
                subset[self._rng.choice(n, size=s, replace=False)] = True
            comp = ~subset

            row = 2 * pair_idx
            out[row, :n] = subset.astype(np.float64)
            out[row, n] = q_size[s]
            out[row + 1, :n] = comp.astype(np.float64)
            out[row + 1, n] = q_size[int(comp.sum())]
        return out

    def sampling(self):
        if self.num_sample <= 0:
            return

        pilot_remaining = self._pilot_num_sample
        while pilot_remaining > 0:
            cur = self._batch_rows(pilot_remaining)
            yield self._sample_batch(cur, self._q_init_size)
            pilot_remaining -= cur

        main_remaining = self.num_sample - self._pilot_num_sample
        while main_remaining > 0:
            if not self._pilot_finalized:
                self._finalize_pilot_design()
            cur = self._batch_rows(main_remaining)
            yield self._sample_batch(cur, self._q_stage2_size)
            main_remaining -= cur

    def run(self, samples):
        game = self.game_func(**self.game_args)
        n = self.num_player

        results = np.empty((len(samples), n + 2), dtype=np.float64)
        results[:, :n] = samples[:, :n]
        results[:, n + 1] = samples[:, n]
        for idx in range(len(samples)):
            results[idx, n] = game.evaluate(samples[idx, :n].astype(bool))
        return results

    def _phi_from_beta(self, beta):
        phi = self._zeta @ beta
        if len(self._boundary_X):
            phi -= self._boundary_raw_gamma.T @ (self._boundary_Z @ beta)
        return phi

    def _build_feature_block(self, X):
        X = np.asarray(X, dtype=bool)
        sizes = X.sum(axis=1).astype(np.int64)

        Z = np.zeros((len(X), self._feature_dim), dtype=np.float64)
        Z[:, 0] = 1.0
        if self._log_col is not None:
            Z[:, self._log_col] = np.log1p(sizes.astype(np.float64))
            Z[:, self._quad_col] = (sizes.astype(np.float64) / float(self.num_player)) ** 2

        if self._surrogate_basis_kind == "interactions":
            for r, combos, start in self._interaction_blocks:
                m = combos.shape[0]
                if r == 1:
                    Z[:, start:start + m] = X[:, combos[:, 0]].astype(np.float64)
                else:
                    Z[:, start:start + m] = X[:, combos].all(axis=2).astype(np.float64)
        else:
            X_float = X.astype(np.float64)
            n = self.num_player
            for s in range(1, n + 1):
                mask = sizes == s
                if np.any(mask):
                    start = self._size_player_start + (s - 1) * n
                    Z[mask, start:start + n] = X_float[mask]
        return Z, sizes

    def _gamma_matrix(self, X, sizes, q_vals):
        return self._raw_gamma_matrix(X, sizes) / q_vals[:, None]

    def _raw_gamma_matrix(self, X, sizes):
        n = self.num_player
        X_float = X.astype(np.float64, copy=False)

        out_value = np.zeros(len(sizes), dtype=np.float64)
        mask_out = sizes < n
        if np.any(mask_out):
            out_value[mask_out] = -self._p[sizes[mask_out]]

        in_value = np.zeros(len(sizes), dtype=np.float64)
        mask_in = sizes > 0
        if np.any(mask_in):
            in_value[mask_in] = self._p[sizes[mask_in] - 1]

        gamma = out_value[:, None] + X_float * (in_value - out_value)[:, None]
        return gamma

    def _effective_surrogate_ridge_lambda(self, count):
        if self.surrogate_ridge_schedule == "fixed":
            return self.surrogate_ridge_lambda
        return float(count) * self.surrogate_ridge_lambda

    def _fit_from_stats(self, A_stat, c_stat, B_stat, b_stat, count):
        if count <= 0:
            beta = np.zeros(self._feature_dim, dtype=np.float64)
            return beta, self._phi_from_beta(beta)

        gram = A_stat - (B_stat.T @ B_stat) / float(count)
        rhs = c_stat - (B_stat.T @ b_stat) / float(count)
        gram = gram.copy()
        gram[np.diag_indices_from(gram)] += self._effective_surrogate_ridge_lambda(count)

        try:
            beta = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        return beta, self._phi_from_beta(beta)

    def _assign_folds(self, num_rows):
        if not self._pair_sampling:
            return self._fold_rng.integers(self.num_folds, size=num_rows)

        if num_rows % 2 != 0:
            raise ValueError("Complement-paired blocks must contain an even number of rows.")
        pair_folds = self._fold_rng.integers(self.num_folds, size=num_rows // 2)
        return np.repeat(pair_folds, 2)

    def _append_block(self, results_collect):
        n = self.num_player
        X = results_collect[:, :n].astype(bool)
        y = results_collect[:, n].astype(np.float64)
        q = results_collect[:, n + 1].astype(np.float64)
        if np.any(q <= 0.0):
            raise ValueError("Encountered non-positive q(S) in EaseSHAP.")

        Z, sizes = self._build_feature_block(X)
        gamma = self._gamma_matrix(X, sizes, q)
        weights = self._A_size[sizes] / (q ** 2)
        folds = self._assign_folds(len(y))

        start = self._num_obs
        end = start + len(y)
        self._X_obs[start:end] = X
        self._y_obs[start:end] = y
        self._q_obs[start:end] = q
        self._size_obs[start:end] = sizes
        self._fold_obs[start:end] = folds
        self._num_obs = end

        wz = weights[:, None] * Z
        self._A_total += Z.T @ wz
        self._c_total += Z.T @ (weights * y)
        self._B_total += gamma.T @ Z
        self._b_total += gamma.T @ y
        self._total_count += len(y)

        for k in np.unique(folds):
            mask = folds == k
            if not np.any(mask):
                continue
            Zk = Z[mask]
            yk = y[mask]
            wk = weights[mask]
            gk = gamma[mask]
            self._A_fold[k] += Zk.T @ (wk[:, None] * Zk)
            self._c_fold[k] += Zk.T @ (wk * yk)
            self._B_fold[k] += gk.T @ Zk
            self._b_fold[k] += gk.T @ yk
            self._fold_count[k] += int(mask.sum())

    def _finalize_pilot_design(self):
        if self._pilot_finalized:
            return

        if self._pilot_num_sample <= 0:
            self._q_stage2_size = self._q_init_size.copy()
            self._pilot_finalized = True
            return

        pilot_count = min(self._pilot_num_sample, self._num_obs)
        beta_pilot, _ = self._fit_from_stats(
            self._A_total,
            self._c_total,
            self._B_total,
            self._b_total,
            self._total_count,
        )
        Z_pilot, _ = self._build_feature_block(self._X_obs[:pilot_count])
        resid = self._y_obs[:pilot_count] - Z_pilot @ beta_pilot
        pilot_size = self._size_obs[:pilot_count]

        second_moment = np.zeros(self.num_player + 1, dtype=np.float64)
        counts = np.zeros(self.num_player + 1, dtype=np.int64)

        if not self._pair_sampling:
            rss = np.zeros(self.num_player + 1, dtype=np.float64)
            np.add.at(rss, pilot_size, resid * resid)
            np.add.at(counts, pilot_size, 1)
            total_count = int(counts.sum())
            if total_count <= 0:
                second_moment.fill(1.0)
            else:
                global_mse = max(float(rss.sum()) / float(total_count), 1e-12)
                second_moment.fill(global_mse)
                strong = counts >= self.stage2_min_size_count
                if np.any(strong):
                    second_moment[strong] = rss[strong] / counts[strong]
                second_moment = np.maximum(second_moment, 1e-12)
        else:
            if pilot_count % 2 != 0:
                raise ValueError("Paired pilot sample must contain an even number of rows.")
            rss = np.zeros(self.num_player + 1, dtype=np.float64)
            pair_diff = resid[0:pilot_count:2] - resid[1:pilot_count:2]
            pair_sq = pair_diff * pair_diff
            size_left = pilot_size[0:pilot_count:2]
            size_right = pilot_size[1:pilot_count:2]
            np.add.at(rss, size_left, pair_sq)
            np.add.at(counts, size_left, 1)
            np.add.at(rss, size_right, pair_sq)
            np.add.at(counts, size_right, 1)
            total_count = int(counts.sum())
            if total_count <= 0:
                second_moment.fill(1.0)
            else:
                global_mse = max(float(rss.sum()) / float(total_count), 1e-12)
                second_moment.fill(global_mse)
                strong = counts >= self.stage2_min_size_count
                if np.any(strong):
                    second_moment[strong] = rss[strong] / counts[strong]
                second_moment = np.maximum(second_moment, 1e-12)
                second_moment = 0.5 * (second_moment + second_moment[::-1])

        size_factor = np.sqrt(self._A_size * second_moment)
        if self.stage2_size_floor > 0.0:
            self._q_stage2_size = self._mix_with_init_size_mass(
                size_factor,
                self.stage2_size_floor,
                enforce_symmetric=self._pair_sampling,
            )
        else:
            self._q_stage2_size = self._normalize_size_law(
                size_factor,
                enforce_symmetric=self._pair_sampling,
            )

        self._pilot_finalized = True

    def _crossfit_estimate(self):
        n = self.num_player
        m = self._num_obs
        if m <= 0:
            return self._boundary_exact.copy()

        est_sum = np.zeros(n, dtype=np.float64)
        num_scored = 0
        fold_ids = np.unique(self._fold_obs[:m])
        for k in fold_ids:
            count_holdout = int(self._fold_count[k])
            if count_holdout <= 0:
                continue

            train_count = m - count_holdout
            if train_count <= 0:
                continue

            beta_k, phi_k = self._fit_from_stats(
                self._A_total - self._A_fold[k],
                self._c_total - self._c_fold[k],
                self._B_total - self._B_fold[k],
                self._b_total - self._b_fold[k],
                train_count,
            )

            mask = self._fold_obs[:m] == k
            X_hold = self._X_obs[:m][mask]
            q_hold = self._q_obs[:m][mask]
            size_hold = self._size_obs[:m][mask]
            Z_hold, _ = self._build_feature_block(X_hold)
            y_hold = self._y_obs[:m][mask]
            gamma_hold = self._gamma_matrix(X_hold, size_hold, q_hold)
            resid_hold = y_hold - Z_hold @ beta_k
            est_sum += float(count_holdout) * phi_k + gamma_hold.T @ resid_hold
            num_scored += count_holdout

        if num_scored <= 0:
            beta_all, phi_all = self._fit_from_stats(
                self._A_total,
                self._c_total,
                self._B_total,
                self._b_total,
                self._total_count,
            )
            X_all = self._X_obs[:m]
            q_all = self._q_obs[:m]
            size_all = self._size_obs[:m]
            Z_all, _ = self._build_feature_block(X_all)
            gamma_all = self._gamma_matrix(X_all, size_all, q_all)
            resid_all = self._y_obs[:m] - Z_all @ beta_all
            return self._boundary_exact + phi_all + (gamma_all.T @ resid_all) / float(max(m, 1))

        return self._boundary_exact + est_sum / float(num_scored)

    def aggregate(self, results_collect):
        pos = 0
        while pos < len(results_collect):
            remaining = len(results_collect) - pos
            take = remaining
            if self.pos_traj < len(self.values_traj):
                next_track = (self.pos_traj + 1) * self.interval_track
                to_track = max(next_track - (self._boundary_eval_count + self._num_obs), 0)
                if to_track > 0:
                    take = min(take, to_track)

            if self._pair_sampling and (take % 2 == 1):
                if take > 1:
                    take -= 1
                else:
                    take = min(remaining, 2)
            if take <= 0:
                break

            self._append_block(results_collect[pos:pos + take])
            pos += take

            if not self._pilot_finalized and self._num_obs >= self._pilot_num_sample:
                self._finalize_pilot_design()

            while (
                self.pos_traj < len(self.values_traj)
                and (self._boundary_eval_count + self._num_obs) >= (self.pos_traj + 1) * self.interval_track
            ):
                self._current_estimate = self._crossfit_estimate()
                self.values_traj[self.pos_traj] = self._current_estimate
                self.pos_traj += 1

    def finalize(self):
        if not self._pilot_finalized and self._num_obs >= self._pilot_num_sample:
            self._finalize_pilot_design()

        if self._num_obs > 0:
            self._current_estimate = self._crossfit_estimate()
        values_final = self._current_estimate.copy()
        if self.values_traj.size:
            if self.pos_traj < len(self.values_traj):
                self.values_traj[self.pos_traj:] = values_final
            else:
                self.values_traj[-1] = values_final
        return values_final, self.values_traj

    def _process(self, inputs):
        raise NotImplementedError("EaseSHAP processes batches in aggregate().")

    def _estimate(self):
        return self._current_estimate.copy()


# ---------------------------------------------------------------------------
# Unbiased RegressionMSR variants (paper Algorithm 1 correction)
# ---------------------------------------------------------------------------

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

    def _run_kfold(self, k=10):
        n = self.num_player
        if not self._X_chunks:
            return np.zeros(n)

        X = np.vstack(self._X_chunks)
        y = np.concatenate(self._y_chunks)
        prob_sampled = np.concatenate(self._prob_chunks)
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

            reg_phi, reg_pred = self._fit_surrogate(X_train, y_train, X_test, pr_train)
            residuals = y_test - reg_pred
            d_test = self._subset_density_by_size(sz_test)
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
