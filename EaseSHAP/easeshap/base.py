"""Base estimator templates shared by EaseSHAP and baseline estimators."""

import numpy as np
from scipy import special


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
        if self.nue_track_avg <= 0:
            raise ValueError("`nue_track_avg` must be positive")
        
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
        if self.interval_track is None or self.batch_size is None:
            raise RuntimeError(
                "`buffer_size` cannot be computed before `interval_track` "
                "and `batch_size` are initialized"
            )
        return self.interval_track + self.batch_size - 1

    def run(self):
        pass

    def _init_indiv(self):
        pass

    def sampling(self):
        self._init_indiv()
        np.random.seed(self.estimator_seed)

        count = 0
        pending_antithetic = None
        for _ in range(self.num_sample):
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

