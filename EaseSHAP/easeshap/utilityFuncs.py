import os
import math

import numpy as np
try:
    import torch
except ModuleNotFoundError:
    torch = None
try:
    from .datasets import *
except ModuleNotFoundError:
    pass
try:
    from . import models
except ModuleNotFoundError:
    models = None
from scipy import special
from scipy.special import ndtri


def _make_zobrist_keys(n, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, np.iinfo(np.uint64).max, size=n, dtype=np.uint64)


def _hash_to_uniform01_uint64(h):
    h = np.asarray(h, dtype=np.uint64)
    return (h.astype(np.float64) + 0.5) / (2.0 ** 64)


class AdditiveUtility:
    """
    U(S) = (1-lam) * transform(sum_{j in S} w_j) + lam * sigma * Z(S),
    where Z(S) is a deterministic pseudo-random N(0,1)-like value produced by
    Zobrist hashing of the subset indicator and mapping through ndtri.
    """
    def __init__(
        self,
        *,
        num_player,
        weights,
        transform=None,
        lam=0.0,
        sigma=1.0,
        zobrist_seed=0,
    ):
        if not isinstance(num_player, int) or num_player <= 0:
            raise ValueError(f"`num_player` must be a positive int, got {num_player!r}.")
        self.num_player = num_player

        w = np.asarray(weights, dtype=float)
        if w.shape != (num_player,):
            raise ValueError(f"`weights` must have shape ({num_player},), got {w.shape}.")
        self.weights = w

        self.transform = (lambda x: x) if transform is None else transform

        lam = float(lam)
        if not (0.0 <= lam <= 1.0):
            raise ValueError(f"`lam` must be in [0,1], got {lam!r}.")
        self.lam = lam

        sigma = float(sigma)
        if not np.isfinite(sigma):
            raise ValueError(f"`sigma` must be finite, got {sigma!r}.")
        self.sigma = sigma

        if not isinstance(zobrist_seed, int):
            raise ValueError(f"`zobrist_seed` must be int, got {zobrist_seed!r}.")
        self.zobrist_seed = zobrist_seed
        self._zobrist_keys = _make_zobrist_keys(num_player, zobrist_seed)

    def _zobrist_gaussian_from_mask(self, mask_bool):
        mask_bool = np.asarray(mask_bool, dtype=bool)
        if mask_bool.any():
            h = np.bitwise_xor.reduce(self._zobrist_keys[mask_bool], dtype=np.uint64)
        else:
            h = np.uint64(0)
        u = _hash_to_uniform01_uint64(h)
        return float(ndtri(u))

    def evaluate(self, subset):
        assert isinstance(subset[0], np.bool_)
        subset = subset[:self.num_player]
        score = float(subset @ self.weights)
        if self.lam == 0.0:
            return float(self.transform(score))
        z = self._zobrist_gaussian_from_mask(subset)
        return self.transform( (1.0 - self.lam) * score + (self.lam * self.sigma) * z )

class RidgeRegressionUtility:
    """
    Cooperative game where each player is a data point (X_i, Y_i).

    U(S) = || beta_hat(S) - beta ||_2^2

    where beta_hat(S) = (X_S^T X_S + penalty * I)^{-1} X_S^T Y_S is the ridge
    estimator trained on the subset S, and beta is the true parameter vector.

    For the empty set, X_S has no rows, so X_S^T X_S = 0 and
    beta_hat(empty) = 0, giving U(empty) = || beta ||^2.
    """

    def __init__(self, *, num_player, X, Y, beta, penalty):
        if not isinstance(num_player, int) or num_player <= 0:
            raise ValueError(f"`num_player` must be a positive int, got {num_player!r}.")
        self.num_player = num_player

        self.X = np.asarray(X, dtype=float)        # (n, d)
        self.Y = np.asarray(Y, dtype=float)        # (n,)
        self.beta = np.asarray(beta, dtype=float)  # (d,)
        self.d = self.X.shape[1]

        penalty = float(penalty)
        if penalty <= 0:
            raise ValueError(f"`penalty` must be positive, got {penalty!r}.")
        self.penalty = penalty

    def evaluate(self, subset):
        assert isinstance(subset[0], np.bool_)
        subset = subset[:self.num_player]
        X_S = self.X[subset]   # (s, d)
        Y_S = self.Y[subset]   # (s,)
        if len(X_S) == 0:
            beta_hat = np.zeros(self.d)
        else:
            A = X_S.T @ X_S + self.penalty * np.eye(self.d)
            beta_hat = np.linalg.solve(A, X_S.T @ Y_S)
        err = beta_hat - self.beta
        return float(err @ err)


class RidgeRegressionUtilityAvgLoss(RidgeRegressionUtility):
    """
    Same cooperative game as RidgeRegressionUtility, but beta_hat(S) is fitted
    with the averaged ridge objective

        (1 / |S|) ||Y_S - X_S b||_2^2 + penalty * ||b||_2^2.

    For nonempty S, this yields

        beta_hat(S) = (X_S^T X_S + |S| * penalty * I)^{-1} X_S^T Y_S.

    For the empty set we keep the same convention beta_hat(empty) = 0, so
    U(empty) = ||beta||_2^2.
    """

    def evaluate(self, subset):
        assert isinstance(subset[0], np.bool_)
        subset = subset[:self.num_player]
        X_S = self.X[subset]   # (s, d)
        Y_S = self.Y[subset]   # (s,)
        if len(X_S) == 0:
            beta_hat = np.zeros(self.d)
        else:
            A = X_S.T @ X_S + (len(X_S) * self.penalty) * np.eye(self.d)
            beta_hat = np.linalg.solve(A, X_S.T @ Y_S)
        err = beta_hat - self.beta
        return float(err @ err)


class gameTraining:
    def __init__(self, *, X_valued, y_valued, X_perf, y_perf, num_class, metric, arch, lr, game_seed=2026):
        self.X_train, self.y_train = X_valued, y_valued
        self.X_perf, self.y_perf = X_perf, y_perf
        self.arch = arch
        self.metric = metric
        self.game_seed = game_seed
        self.num_class = num_class
        self.num_player = len(y_valued)
        self.half_num_class = num_class // 2

        # load model and optimizer
        if arch == "logistic":
            self.model = models.LogisticRegression(self.X_perf.shape[1], num_class)
        elif arch == "LeNet":
            self.model = models.LeNet()
        else:
            raise NotImplementedError(f"Check {arch}")
        self.model.double() # float64 is used for more consistent reproducibility across platforms

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    def evaluate(self, subset):
        assert isinstance(subset[0], np.bool_)
        subset = subset[:self.num_player] # there may be a null player at the end.
        self.train_model(self.X_train[subset], self.y_train[subset])
        return self.output_score()

    def dist_evaluate(self, subset, z_extra):
        assert isinstance(subset[0], np.bool_)
        X_extra, y_extra = z_extra
        X = torch.cat((self.X_train[subset], X_extra))
        y = torch.cat((self.y_train[subset], y_extra))
        self.train_model(X, y)
        return self.output_score()

    def train_model(self, X, y):
        # to avoid that for some fixed cardinality the data point (self.X_train[-1], self.y_train[-1]) is always the
        # last one fed into the model in one-epoch-one-mini-batch learning.
        with set_numpy_seed((X > 0).sum().item() + (y < self.half_num_class).sum().item() + self.game_seed):
            pi = np.random.permutation(len(y))
        X, y = X[pi], y[pi]

        with set_torch_seed(self.game_seed):
            for layer in self.model.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        if len(y):
            for datum, label in zip(X, y):
                logit = self.model(datum)
                loss = self.criterion(logit, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def output_score(self):
        self.model.eval()
        with torch.no_grad():
            logit = self.model(self.X_perf)
        assert ~torch.isnan(logit.sum())
        self.model.train()
        if self.metric == "accuracy":
            predict = np.argmax(logit.numpy(), 1)
            label = self.y_perf.numpy()
            score = np.sum(predict == label) / len(label)
        elif self.metric == "cross_entropy":
            score = -self.criterion(logit, self.y_perf).numpy()
        else:
            raise NotImplementedError(f"Check {self.metric}")
        return score


class gameKNN:
    def __init__(self, *, X_valued, y_valued, X_perf, y_perf, K=20):
        self.X_valued, self.y_valued = X_valued, y_valued
        self.num_player = len(self.y_valued)
        self.X_perf, self.y_perf = X_perf, y_perf
        self.num_perf = len(self.y_perf)
        self.K = K
        self._alpha = None

    @property
    def alpha(self):
        if self._alpha is None:
            # dist = np.linalg.norm(self.X_perf[:, None, :] - self.X_valued[None, :, :], axis=2)
            # self._alpha = dist.argsort(axis=1).argsort(axis=1)
            self._alpha = np.empty((self.num_perf, self.num_player), dtype=np.int64)
            for i, xp in enumerate(self.X_perf):
                dist = np.linalg.norm(self.X_valued - xp[None, :], axis=1)
                self._alpha[i] = dist.argsort().argsort()
        return self._alpha

    def evaluate(self, subset):
        assert isinstance(subset[0], np.bool_)
        if subset.sum():
            y_sub = self.y_valued[subset]
            alpha = self.alpha[:, subset]
            acc = 0.
            for i in range(self.num_perf):
                alpha_sub = alpha[i].argsort()
                yt = self.y_perf[i]
                acc += (y_sub[alpha_sub][:self.K] == yt).sum() / self.K
            return acc / self.num_perf
        else:
            return 0

    def get_Shapley(self):
        tmp = np.arange(1, self.num_player + 1)
        coeff = np.divide(np.minimum(tmp, self.K), tmp)
        value_exact = np.zeros(self.num_player, dtype=np.float64)
        value_cur = np.zeros(self.num_player, dtype=np.float64)
        for i in range(self.num_perf):
            yt = self.y_perf[i]
            alpha = self.alpha[i]
            y_cur = self.y_valued[alpha.argsort()]
            value_cur[-1] = (y_cur[-1] == yt) / self.K * coeff[-1]
            for j in range(self.num_player - 2, -1, -1):
                value_cur[j] = value_cur[j + 1] + (int(y_cur[j] == yt) - int(y_cur[j + 1] == yt)) / self.K * coeff[j]
            value_exact += value_cur[alpha]
        return value_exact / self.num_perf


class gameSOU:
    """
    Legacy random-weight sum-of-unanimity game.

    This matches the older OFA codebase utility, not the structured SOU game
    used in the Faithful GSV paper benchmark. Use `gameSOUPaper` below for the
    paper-compatible benchmark utility.
    """
    def __init__(self, *, num_player, num_unanimity, path="exp/SOUGame", seed=2026):
        os.makedirs(path, exist_ok=True)
        data_saved = os.path.join(path, f"n_player={num_player};n_unanimity={num_unanimity};seed={seed}.npz")
        self.num_player = num_player
        if os.path.exists(data_saved):
            data = np.load(data_saved)
            self.coalitions = data["coalitions"]
            self.weights = data["weights"]
        else:
            rng = np.random.default_rng(seed)
            s_range = np.arange(1, num_player)
            pos_range = np.arange(num_player)
            self.coalitions = np.zeros((num_unanimity, num_player), dtype=bool)
            for i in range(num_unanimity):
                s = int(rng.choice(s_range))
                pos = rng.choice(pos_range, size=s, replace=False)
                self.coalitions[i, pos] = True
            self.weights = rng.uniform(low=-num_player, high=num_player, size=num_unanimity)
            np.savez_compressed(data_saved, coalitions=self.coalitions, weights=self.weights)
        self.sizes = self.coalitions.sum(axis=1)

    def evaluate(self, subset):
        assert isinstance(subset[0], np.bool_)
        subset = subset[:self.num_player]
        tmp = np.logical_and(self.coalitions, subset[None, :]).sum(axis=1) == self.sizes
        return (tmp * self.weights).sum()
        # index = subset.sum() >= self.sizes
        # tmp = np.logical_and(self.coalitions[index], subset[None, :]).sum(axis=1) == self.sizes[index]
        # return (tmp * self.weights[index]).sum()

    def get_semivalue(self, *, semivalue, semivalue_param=None):
        if semivalue == "shapley":
            tmp = 1 / self.sizes
        elif semivalue == "weighted_banzhaf":
            tmp = np.power(semivalue_param, self.sizes - 1, dtype=np.float128)
        elif semivalue == "beta_shapley":
            tmp = special.beta(semivalue_param[1] + self.sizes - 1, semivalue_param[0]) \
                  / special.beta(semivalue_param[1], semivalue_param[0])
        return np.dot(self.weights * tmp, self.coalitions)


class gameSOUPaper:
    """
    Structured SOU benchmark game from Faithful GSV Section 4.1.

    The game is a sum of unanimity terms

        U(S) = (1 / d) * sum_j alpha_j 1[T_j subseteq S],

    where the unanimity term weights are determined by fixed player-group labels
    `i mod 4 in {0, 1, 2, 3}` exactly as in
    `Faithful_GSV-main/exp1_benchmark/utils.py`.
    """

    def __init__(self, *, num_player, d=0, path="exp/SOUGamePaper", seed=42):
        os.makedirs(path, exist_ok=True)
        if d == 0:
            d = int(num_player) ** 2
        data_saved = os.path.join(path, f"n_player={num_player};d={d};seed={seed}.npz")
        self.num_player = int(num_player)
        self.d = int(d)

        if os.path.exists(data_saved):
            data = np.load(data_saved)
            self.coalitions = data["coalitions"]
            self.alpha = data["alpha"]
        else:
            rng = np.random.RandomState(seed)
            group_weight = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
            self.coalitions = np.zeros((self.d, self.num_player), dtype=bool)
            self.alpha = np.zeros(self.d, dtype=np.float64)
            pos_range = np.arange(self.num_player)
            for i in range(self.d):
                size = int(rng.randint(1, self.num_player))
                pos = rng.choice(pos_range, size=size, replace=False)
                self.coalitions[i, pos] = True
                self.alpha[i] = float(group_weight[pos % 4].mean())
            np.savez_compressed(data_saved, coalitions=self.coalitions, alpha=self.alpha)

        self.sizes = self.coalitions.sum(axis=1)

    def evaluate(self, subset):
        assert isinstance(subset[0], np.bool_)
        subset = subset[:self.num_player]
        active = np.logical_and(self.coalitions, subset[None, :]).sum(axis=1) == self.sizes
        return float((active * self.alpha).sum() / self.d)

    def get_semivalue(self, *, semivalue, semivalue_param=None):
        if semivalue == "shapley":
            tmp = 1.0 / self.sizes
        elif semivalue == "weighted_banzhaf":
            tmp = np.power(semivalue_param, self.sizes - 1, dtype=np.float128)
        elif semivalue == "beta_shapley":
            tmp = special.beta(semivalue_param[1] + self.sizes - 1, semivalue_param[0]) \
                  / special.beta(semivalue_param[1], semivalue_param[0])
        else:
            raise NotImplementedError(f"Unknown semivalue {semivalue!r}.")
        return np.dot((self.alpha / self.d) * tmp, self.coalitions)


class gameSOUStructuredGaussian:
    """
    Structured/unstructured Gaussian sum-of-unanimity game.

    The low-order component contains all singleton and pairwise unanimity
    terms. The high-order component follows the OFA-style random SOU
    construction, except each random coalition has cardinality at least 3.

        U(S) = sum_{T in T_low union T_high} w_T 1[T subseteq S].

    By default, sigma^2 is the total number of active terms. Low-order weights
    receive expected squared coefficient mass alpha^2 sigma^2, while high-order
    weights receive expected squared coefficient mass (1 - alpha^2) sigma^2.
    """

    def __init__(
        self,
        *,
        num_player,
        alpha,
        num_high_order=None,
        sigma2=None,
        path="exp/SOUGameStructuredGaussian",
        seed=2026,
        allow_duplicate_high_order=True,
    ):
        if not isinstance(num_player, int) or num_player < 4:
            raise ValueError(f"`num_player` must be an int at least 4, got {num_player!r}.")
        self.num_player = num_player

        alpha = float(alpha)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"`alpha` must be in [0, 1], got {alpha!r}.")
        self.alpha = alpha

        if num_high_order is None:
            num_high_order = num_player ** 2
        if not isinstance(num_high_order, int) or num_high_order <= 0:
            raise ValueError(f"`num_high_order` must be a positive int, got {num_high_order!r}.")
        self.num_high_order = num_high_order

        self.allow_duplicate_high_order = bool(allow_duplicate_high_order)
        if not self.allow_duplicate_high_order:
            num_possible_high_order = sum(math.comb(num_player, k) for k in range(3, num_player))
            if num_high_order > num_possible_high_order:
                raise ValueError(
                    "`num_high_order` exceeds the number of unique coalitions "
                    f"with cardinality in [3, {num_player - 1}]."
                )

        num_low_order = num_player + num_player * (num_player - 1) // 2
        if sigma2 is None:
            sigma2 = float(num_low_order + num_high_order)
        sigma2 = float(sigma2)
        if not np.isfinite(sigma2) or sigma2 < 0.0:
            raise ValueError(f"`sigma2` must be finite and nonnegative, got {sigma2!r}.")
        self.sigma2 = sigma2

        os.makedirs(path, exist_ok=True)
        data_saved = os.path.join(
            path,
            "n_player={};alpha={:.12g};n_high={};sigma2={:.12g};seed={};duplicate={}.npz".format(
                num_player,
                alpha,
                num_high_order,
                sigma2,
                seed,
                int(self.allow_duplicate_high_order),
            ),
        )

        if os.path.exists(data_saved):
            data = np.load(data_saved)
            self.coalitions = data["coalitions"]
            self.weights = data["weights"]
            self.component = data["component"]
        else:
            rng = np.random.default_rng(seed)
            low_coalitions = self._make_low_order_coalitions(num_player)
            high_coalitions = self._sample_high_order_coalitions(
                rng,
                num_player,
                num_high_order,
                self.allow_duplicate_high_order,
            )

            low_sd = np.sqrt((alpha ** 2) * sigma2 / len(low_coalitions))
            high_sd = np.sqrt((1.0 - alpha ** 2) * sigma2 / len(high_coalitions))
            low_weights = rng.normal(loc=0.0, scale=low_sd, size=len(low_coalitions))
            high_weights = rng.normal(loc=0.0, scale=high_sd, size=len(high_coalitions))

            self.coalitions = np.vstack((low_coalitions, high_coalitions))
            self.weights = np.concatenate((low_weights, high_weights))
            self.component = np.concatenate((
                np.zeros(len(low_coalitions), dtype=np.int8),
                np.ones(len(high_coalitions), dtype=np.int8),
            ))
            np.savez_compressed(
                data_saved,
                coalitions=self.coalitions,
                weights=self.weights,
                component=self.component,
            )

        self.sizes = self.coalitions.sum(axis=1)

    @staticmethod
    def _make_low_order_coalitions(num_player):
        num_low_order = num_player + num_player * (num_player - 1) // 2
        coalitions = np.zeros((num_low_order, num_player), dtype=bool)
        row = 0
        for i in range(num_player):
            coalitions[row, i] = True
            row += 1
        for i in range(num_player):
            for j in range(i + 1, num_player):
                coalitions[row, [i, j]] = True
                row += 1
        return coalitions

    @staticmethod
    def _sample_high_order_coalitions(rng, num_player, num_high_order, allow_duplicates):
        s_range = np.arange(3, num_player)
        pos_range = np.arange(num_player)
        coalitions = np.zeros((num_high_order, num_player), dtype=bool)

        if allow_duplicates:
            for row in range(num_high_order):
                size = int(rng.choice(s_range))
                pos = rng.choice(pos_range, size=size, replace=False)
                coalitions[row, pos] = True
            return coalitions

        seen = set()
        row = 0
        while row < num_high_order:
            size = int(rng.choice(s_range))
            pos = tuple(sorted(rng.choice(pos_range, size=size, replace=False)))
            if pos in seen:
                continue
            seen.add(pos)
            coalitions[row, list(pos)] = True
            row += 1
        return coalitions

    def evaluate(self, subset):
        assert isinstance(subset[0], np.bool_)
        subset = subset[:self.num_player]
        active = np.logical_and(self.coalitions, subset[None, :]).sum(axis=1) == self.sizes
        return float((active * self.weights).sum())

    def get_semivalue(self, *, semivalue, semivalue_param=None):
        if semivalue == "shapley":
            tmp = 1.0 / self.sizes
        elif semivalue == "weighted_banzhaf":
            tmp = np.power(semivalue_param, self.sizes - 1, dtype=np.float128)
        elif semivalue == "beta_shapley":
            tmp = special.beta(semivalue_param[1] + self.sizes - 1, semivalue_param[0]) \
                  / special.beta(semivalue_param[1], semivalue_param[0])
        else:
            raise NotImplementedError(f"Unknown semivalue {semivalue!r}.")
        return np.dot(self.weights * tmp, self.coalitions)







