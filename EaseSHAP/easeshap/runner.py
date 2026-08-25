"""Estimator runner orchestration."""

import multiprocessing as mp

from tqdm import tqdm

from .utils import vd_tqdm


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
        self.n_process = max(1, int(n_process))
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
        from .registry import get_estimator_class

        estimator_cls = get_estimator_class(self.estimator)
        estimator = estimator_cls(**estimator_args, **self.kwargs_estimator)
        print(f"The number of utility evalutions each process runs in one batch is {estimator.nue_per_proc_run}")
        requires_serial_feedback = bool(getattr(estimator, "_requires_serial_feedback", False))
        if hasattr(estimator, "_num_batches_hint"):
            num_batches_hint = int(estimator._num_batches_hint)
        elif estimator.batch_size:
            num_batches_hint = int(-(-estimator.num_sample // estimator.batch_size))
        else:
            num_batches_hint = 0
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
