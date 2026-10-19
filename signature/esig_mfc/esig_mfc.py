"""Expected-signature matching as a mean-field control problem, solved through its dual.

The primal problem (``doc/expected_signature_matching.md``, weighted by ``lambda_v > 0``) is

    V = inf_alpha E[1/2 int_0^T alpha_t^2 dt] + kappa/2 sum_v lambda_v (E[Sig(X)_T^v] - E^v)^2,
    dX = alpha dt + dW,  X_0 = 0,

with the time-augmented signature ``Sig(t, X)`` (letter ``1`` = time, letter ``2`` = state). Fenchel
linearisation of the penalty and the Boue--Dupuis formula reduce it to the finite-dimensional concave
dual

    J(p) = -kappa/2 p^T Lambda^{-1} p - kappa <p, E> - log E[exp(-kappa <p, Sig(W)_T>)],

maximised over the closed convex cone ``K = B cap T^N`` on which the exponential moment is finite
(spec ``spec/esig_mfc_spec.md``). The expectation is replaced by a sample average over a fixed
sample of Brownian signatures, so ``J`` is a deterministic smooth concave function
(:class:`DualProblem`), maximised by projected Newton or projected gradient
(:func:`optimize_dual`). The optimal control at ``p*`` is the signature feedback of
``sig_control`` with terminal tensor ``q = -kappa p*`` (:func:`optimal_policy`, :func:`rollout`);
the optimal path law is also reachable by resampling the Brownian paths (:func:`resample`).

.. note::
    The module expects ``jax_enable_x64=True`` (float64 everywhere): level-7 coordinates are
    ``O(1e-4)`` and the dual Hessian mixes scales over many orders of magnitude. It does not toggle
    the global JAX config itself; set it before importing (the notebook and tests do).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from ..tensor_sequence import TensorSequence
from ..words import number_of_words_up_to_trunc, index_to_word_vect
from .. import sig_control as sc


# ---------------------------------------------------------------------------------------------
# Word bookkeeping
# ---------------------------------------------------------------------------------------------

def _words(trunc: int, dim: int) -> np.ndarray:
    """Code words (decimal-digit encoding, ``0`` = empty word) of all indices up to ``trunc``."""
    n_sig = int(number_of_words_up_to_trunc(trunc, dim))
    return np.asarray(index_to_word_vect(jnp.arange(n_sig), dim)).astype(np.int64)


def _word_str(word: int) -> str:
    return "" if word == 0 else str(int(word))


def _letter_count(words: np.ndarray, letter: int) -> np.ndarray:
    """Number of occurrences of ``letter`` in each word."""
    return np.array([_word_str(w).count(str(letter)) for w in words], dtype=int)


def _word_lengths(words: np.ndarray) -> np.ndarray:
    return np.array([len(_word_str(w)) for w in words], dtype=int)


# ---------------------------------------------------------------------------------------------
# 3.1 Sampling and signatures
# ---------------------------------------------------------------------------------------------

def path_signatures(paths: np.ndarray, t_grid: np.ndarray, trunc: int,
                    batch_size: int = 10_000) -> np.ndarray:
    """Truncated Stratonovich signatures of the time-augmented piecewise-linear paths ``(t, X_t)``.

    The signature is translation invariant, so ``X_0 = paths[:, 0]`` need not be subtracted. The
    computation is done by ``iisignature`` in batches of ``batch_size`` paths to bound memory.

    :param paths: state values, shape ``(M, n_t)``.
    :param t_grid: time grid, shape ``(n_t,)``.
    :param trunc: truncation order ``N``.
    :param batch_size: number of paths per ``iisignature`` call.
    :return: signatures with the empty-word row prepended, shape ``(n_sig, M)``, in package word order.
    """
    import iisignature
    paths = np.asarray(paths, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)
    M, n_t = paths.shape
    if t_grid.shape != (n_t,):
        raise ValueError(f"t_grid has shape {t_grid.shape}, expected ({n_t},).")
    n_sig = int(number_of_words_up_to_trunc(trunc, 2))
    out = np.empty((n_sig, M))
    out[0] = 1.0
    for start in range(0, M, batch_size):
        stop = min(start + batch_size, M)
        aug = np.stack([np.broadcast_to(t_grid, (stop - start, n_t)), paths[start:stop]], axis=2)
        out[1:, start:stop] = iisignature.sig(aug, trunc).T
    return out


def brownian_paths(T: float, n_steps: int, M: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Standard Brownian paths on a uniform grid.

    :param T: horizon.
    :param n_steps: number of time steps.
    :param M: number of paths.
    :param seed: RNG seed.
    :return: ``(t_grid, W)`` with shapes ``(n_steps + 1,)`` and ``(M, n_steps + 1)``, ``W_0 = 0``.
    """
    t_grid = np.linspace(0.0, T, n_steps + 1)
    rng = np.random.default_rng(seed)
    W = np.zeros((M, n_steps + 1))
    W[:, 1:] = np.cumsum(rng.normal(size=(M, n_steps)) * np.sqrt(T / n_steps), axis=1)
    return t_grid, W


def target_paths(kind: str, T: float, n_steps: int, M: int, seed: int,
                 **params) -> Tuple[np.ndarray, np.ndarray]:
    """Sample paths of a target law on a uniform grid.

    * ``kind="ou"``: ``dX = theta (mu - X) dt + sigma dW``, ``X_0 = x0``, simulated with the exact
      Gaussian transition. Parameters ``theta=1.0, mu=0.5, sigma=1.0, x0=0.0``. With ``sigma=1``
      the OU drift is an admissible control, so the law is absolutely continuous w.r.t. Wiener
      measure and the matching problem is feasible.
    * ``kind="double_well"``: Langevin dynamics ``dX = -V'(X) dt + sigma dW`` in the double-well
      potential ``V(x) = h (x^2 - 1)^2``, simulated by Euler--Maruyama with ``substeps`` sub-steps per
      grid step. Parameters ``h=1.0, sigma=1.0, x0=0.0, substeps=10``. With ``sigma=1`` the law is
      absolutely continuous w.r.t. Wiener measure, and by Ito its density
      ``exp(-V(X_T) + V(X_0) + int (V''/2 - V'^2/2) dt)`` is a signature functional of state degree 6
      whose leading part is ``-int X^6 dt`` (the integrated leading word of the cone for ``N >= 7``).
    * ``kind="cir"``: Cox--Ingersoll--Ross ``dX = a (b - X) dt + sigma sqrt(X) dW``, simulated with
      the exact (scaled non-central chi-square) transition. Parameters ``a=2.0, b=1.0, sigma=1.0,
      x0=1.0``. The diffusion coefficient is state dependent, so the law is singular w.r.t. Wiener
      measure (stress test: only the truncated moments can be matched).
    * ``kind="w2"``: ``X_t = c (x0 + W_t)^2`` (stress test: not a Brownian-with-drift law).
      Parameters ``c=1.0, x0=0.0``. For ``x0 != 0`` the increments ``c (2 x0 W + W^2)`` are locally
      Brownian-like with volatility ``2 c x0``. The top signature moments scale like ``c^6``; for
      ``x0 = 0`` they are lighter than Brownian ones (target inside the cone) only for
      ``c < (15/10395)^{1/6} ~ 0.34``.

    :param kind: ``"ou"``, ``"double_well"``, ``"cir"`` or ``"w2"``.
    :param T: horizon.
    :param n_steps: number of time steps.
    :param M: number of paths.
    :param seed: RNG seed.
    :return: ``(t_grid, X)`` with shapes ``(n_steps + 1,)`` and ``(M, n_steps + 1)``.
    """
    t_grid = np.linspace(0.0, T, n_steps + 1)
    dt = T / n_steps
    rng = np.random.default_rng(seed)
    if kind == "ou":
        theta = float(params.pop("theta", 1.0))
        mu = float(params.pop("mu", 0.5))
        sigma = float(params.pop("sigma", 1.0))
        x0 = float(params.pop("x0", 0.0))
        if params:
            raise TypeError(f"unexpected OU parameters: {sorted(params)}")
        decay = np.exp(-theta * dt)
        std = sigma * np.sqrt((1.0 - decay ** 2) / (2.0 * theta))
        Z = rng.normal(size=(M, n_steps))
        X = np.empty((M, n_steps + 1))
        X[:, 0] = x0
        for k in range(n_steps):
            X[:, k + 1] = mu + (X[:, k] - mu) * decay + std * Z[:, k]
        return t_grid, X
    if kind == "double_well":
        h = float(params.pop("h", 1.0))
        sigma = float(params.pop("sigma", 1.0))
        x0 = float(params.pop("x0", 0.0))
        substeps = int(params.pop("substeps", 10))
        if params:
            raise TypeError(f"unexpected double_well parameters: {sorted(params)}")
        ds = dt / substeps
        X = np.empty((M, n_steps + 1))
        X[:, 0] = x0
        x = X[:, 0].copy()
        for k in range(n_steps):
            for _ in range(substeps):
                x = x - 4.0 * h * x * (x ** 2 - 1.0) * ds + sigma * np.sqrt(ds) * rng.normal(size=M)
            X[:, k + 1] = x
        return t_grid, X
    if kind == "cir":
        a = float(params.pop("a", 2.0))
        b = float(params.pop("b", 1.0))
        sigma = float(params.pop("sigma", 1.0))
        x0 = float(params.pop("x0", 1.0))
        if params:
            raise TypeError(f"unexpected cir parameters: {sorted(params)}")
        if x0 < 0 or a <= 0 or b <= 0 or sigma <= 0:
            raise ValueError("cir needs x0 >= 0 and a, b, sigma > 0.")
        decay = np.exp(-a * dt)
        c = sigma ** 2 * (1.0 - decay) / (4.0 * a)         # X_{t+dt} = c * chi'^2(df, nc)
        df = 4.0 * a * b / sigma ** 2
        X = np.empty((M, n_steps + 1))
        X[:, 0] = x0
        for k in range(n_steps):
            X[:, k + 1] = c * rng.noncentral_chisquare(df, X[:, k] * decay / c)
        return t_grid, X
    if kind == "w2":
        c = float(params.pop("c", 1.0))
        x0 = float(params.pop("x0", 0.0))
        if params:
            raise TypeError(f"unexpected w2 parameters: {sorted(params)}")
        _, W = brownian_paths(T, n_steps, M, seed)
        return t_grid, c * (x0 + W) ** 2
    raise ValueError(f"unknown target kind {kind!r}; expected 'ou', 'double_well', 'cir' or 'w2'.")


def expected_signature(S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sample mean and standard error of a signature sample.

    :param S: signatures, shape ``(n_sig, M)``.
    :return: ``(mean, se)``, each of shape ``(n_sig,)``.
    """
    S = np.asarray(S)
    M = S.shape[1]
    return S.mean(axis=1), S.std(axis=1, ddof=1) / np.sqrt(M)


def word_weights(kind: str, S_target: Optional[np.ndarray], cone: "ConeSpec",
                 floor: float = 1e-12) -> np.ndarray:
    """Positive word weights ``lambda_v`` of the penalty ``kappa/2 sum_v lambda_v (m^v - E^v)^2``.

    * ``"uniform"``: ``lambda_v = 1`` (the note).
    * ``"inverse_variance"``: ``lambda_v = 1 / max(Var_target(Sig^v_T), floor)``, a chi^2-type
      penalty that keeps the ``O(1e-4)`` top-level coordinates from being effectively unpenalised.
      Pure-time words ``1...1`` are deterministic (``Sig^v_T = T^k/k!`` for every path), so their
      variance is pure rounding noise; they get ``lambda_v = 1`` instead of ``1/floor``, which
      would only amplify that noise. Their residual vanishes identically, so this does not change
      the primal problem.

    The empty-word weight is irrelevant and set to ``1``.

    :param kind: ``"uniform"`` or ``"inverse_variance"``.
    :param S_target: target signature sample ``(n_sig, M_target)`` (needed for ``"inverse_variance"``).
    :param cone: the cone (gives the truncation and the word list).
    :param floor: lower bound on the variance.
    :return: weights, shape ``(n_sig,)``.
    """
    n_sig = cone.words.shape[0]
    if kind == "uniform":
        return np.ones(n_sig)
    if kind == "inverse_variance":
        if S_target is None:
            raise ValueError("'inverse_variance' weights need the target sample S_target.")
        S_target = np.asarray(S_target)
        if S_target.shape[0] != n_sig:
            raise ValueError(f"S_target has {S_target.shape[0]} rows, expected {n_sig}.")
        lam = 1.0 / np.maximum(S_target.var(axis=1, ddof=1), floor)
        lam[cone.pure_time_mask] = 1.0
        lam[0] = 1.0
        return lam
    raise ValueError(f"unknown weight kind {kind!r}; expected 'uniform' or 'inverse_variance'.")


# ---------------------------------------------------------------------------------------------
# 3.2 Cone K = B cap T^N
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class ConeSpec:
    """The closed convex set ``K_eps = B cap T^N`` of admissible dual tensors (spec §2.3).

    Every non-empty word is in exactly one of three classes: *free* (unconstrained coefficient),
    *lead* (coefficient ``>= eps``) or *forced* (coefficient ``0``). The empty word is in none of
    them and its coefficient is always ``0``.
    """
    trunc: int
    dim: int
    n: Optional[int]
    m: Optional[int]
    eps: float
    lead_idx: Tuple[int, ...]
    free_mask: np.ndarray        # bool (n_sig,)
    forced_mask: np.ndarray      # bool (n_sig,)
    lower: np.ndarray            # float (n_sig,): eps on leads, -inf elsewhere
    words: np.ndarray            # int (n_sig,): code words in package order
    time_letter: int = 1
    state_letter: int = 2

    @property
    def n_sig(self) -> int:
        return int(self.words.shape[0])

    @property
    def lead_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_sig, dtype=bool)
        mask[list(self.lead_idx)] = True
        return mask

    @property
    def active_mask(self) -> np.ndarray:
        """Free or lead words: the coordinates the optimiser works on."""
        return self.free_mask | self.lead_mask

    @property
    def active_idx(self) -> np.ndarray:
        return np.flatnonzero(self.active_mask)

    @property
    def pure_time_mask(self) -> np.ndarray:
        """Non-empty words made only of the time letter (deterministic coordinates)."""
        lengths = _word_lengths(self.words)
        return (lengths > 0) & (_letter_count(self.words, self.time_letter) == lengths)

    @property
    def state_degree(self) -> np.ndarray:
        """``|v|_2``: number of state letters of each word."""
        return _letter_count(self.words, self.state_letter)

    def word_class(self) -> np.ndarray:
        """Class label of each word: ``"empty"``, ``"free"``, ``"lead"`` or ``"forced"``."""
        labels = np.full(self.n_sig, "free", dtype=object)
        labels[self.forced_mask] = "forced"
        labels[list(self.lead_idx)] = "lead"
        labels[0] = "empty"
        return labels

    def describe(self) -> str:
        """Human-readable summary of the three word classes."""
        lengths = _word_lengths(self.words)
        lines = [f"Cone B ∩ T^{self.trunc} (dim={self.dim}, n={self.n}, m={self.m}, eps={self.eps:g})"]
        lead_words = ", ".join(_word_str(self.words[i]) for i in self.lead_idx) or "none"
        lines.append(f"  lead words   (p >= eps): {lead_words}")
        forced = [_word_str(w) for w in self.words[self.forced_mask]]
        lines.append(f"  forced words (p == 0)  : {len(forced)}"
                     + (f"  [{', '.join(forced)}]" if forced else ""))
        per_level = [int(np.sum(self.free_mask & (lengths == k))) for k in range(1, self.trunc + 1)]
        lines.append(f"  free words   (p in R)  : {int(self.free_mask.sum())}  "
                     f"(per level 1..{self.trunc}: {per_level})")
        lines.append(f"  active coordinates     : {int(self.active_mask.sum())} of {self.n_sig - 1} non-empty words")
        return "\n".join(lines)


def make_b_cone(trunc: int, n: Optional[int] = "auto", m: Optional[int] = "auto", eps: float = 1e-6,
                dim: int = 2, time_letter: int = 1, state_letter: int = 2) -> ConeSpec:
    """Build the cone ``K_eps = B cap T^N`` of the Laplace paper (Def. 4.2), spec §2.3.

    The leading words are ``state^{2n}`` and ``state^{2m} time``. Defaults: the largest that fit in
    ``T^N``, ``n = floor(N/2)`` (``None`` if ``N < 2``) and ``m = floor((N-1)/2)`` (``None`` if
    ``N < 3``). Free words: all words with ``|v|_2 <= 1`` (Gaussian directions); if ``n``, the pure
    state words of length ``< 2n``; if ``m``, the words ending with the time letter with fewer than
    ``2m`` state letters; if both, every word with ``|v|_2 < 2 min(m, n)``. All other non-empty
    words are forced to zero.

    :param trunc: truncation order ``N``.
    :param n: half the state degree of the pure-state leading word, ``None`` (absent) or ``"auto"``.
    :param m: half the state degree of the integrated leading word, ``None`` (absent) or ``"auto"``.
    :param eps: lower bound of the leading coefficients (``eps > 0`` keeps the mixed part legitimate).
    :param dim: path dimension (must be 2).
    :param time_letter: code letter of time.
    :param state_letter: code letter of the state.
    :return: a :class:`ConeSpec`.
    """
    if dim != 2:
        raise ValueError("only the time-augmented 1-D state (dim=2) is supported.")
    if n == "auto":
        n = trunc // 2 if trunc >= 2 else None
    if m == "auto":
        m = (trunc - 1) // 2 if trunc >= 3 else None
    if n is not None and not (1 <= n and 2 * n <= trunc):
        raise ValueError(f"n={n}: need n >= 1 and leading word of length 2n <= trunc={trunc}.")
    if m is not None and not (1 <= m and 2 * m + 1 <= trunc):
        raise ValueError(f"m={m}: need m >= 1 and leading word of length 2m+1 <= trunc={trunc}.")

    words = _words(trunc, dim)
    lengths = _word_lengths(words)
    k_state = _letter_count(words, state_letter)
    last = np.array([int(_word_str(w)[-1]) if w else 0 for w in words])
    nonempty = lengths > 0

    free = nonempty & (k_state <= 1)
    lead_words = []
    if n is not None:
        free |= nonempty & (k_state == lengths) & (lengths < 2 * n)
        lead_words.append(int(str(state_letter) * (2 * n)))
    if m is not None:
        free |= nonempty & (last == time_letter) & (k_state < 2 * m)
        lead_words.append(int(str(state_letter) * (2 * m) + str(time_letter)))
    if n is not None and m is not None:
        free |= nonempty & (k_state < 2 * min(m, n))

    word_pos = {int(w): i for i, w in enumerate(words)}
    lead_idx = tuple(word_pos[w] for w in lead_words)
    lead = np.zeros_like(free)
    lead[list(lead_idx)] = True
    free &= ~lead
    forced = nonempty & ~free & ~lead

    lower = np.full(words.shape[0], -np.inf)
    lower[list(lead_idx)] = eps
    return ConeSpec(trunc=trunc, dim=dim, n=n, m=m, eps=float(eps), lead_idx=lead_idx,
                    free_mask=free, forced_mask=forced, lower=lower, words=words,
                    time_letter=time_letter, state_letter=state_letter)


def project(p: jax.Array, cone: ConeSpec) -> jax.Array:
    """Euclidean projection onto ``K_eps``: zero the forced words and the empty word, clip the
    leading words at ``eps``.

    :param p: coefficients, shape ``(n_sig,)``.
    :param cone: the cone.
    :return: the projected coefficients, shape ``(n_sig,)``.
    """
    p = jnp.asarray(p) * jnp.asarray(cone.active_mask, dtype=float)
    return jnp.maximum(p, jnp.asarray(cone.lower))


def in_cone(p: jax.Array, cone: ConeSpec, tol: float = 0.0) -> bool:
    """Whether ``p`` belongs to ``K_eps`` up to ``tol``.

    :param p: coefficients, shape ``(n_sig,)``.
    :param cone: the cone.
    :param tol: tolerance on the zero and the lower-bound constraints.
    :return: ``True`` if every forced and empty entry is within ``tol`` of 0 and every lead is
        ``>= eps - tol``.
    """
    p = np.asarray(p)
    inactive = ~cone.active_mask
    lead = list(cone.lead_idx)
    return bool(np.all(np.abs(p[inactive]) <= tol) and np.all(p[lead] >= cone.eps - tol))


# ---------------------------------------------------------------------------------------------
# 3.3 Dual problem (sample-average approximation)
# ---------------------------------------------------------------------------------------------

class Tilt(NamedTuple):
    """Exponential tilt ``dQ_p/dW ∝ exp(-kappa <p, Sig(W)_T>)`` of the empirical Brownian measure."""
    logZ: jax.Array    # log mean exp(ell),  ell_i = -kappa <p, S_i>
    w: jax.Array       # (M,) self-normalised weights softmax(ell)
    m: jax.Array       # (n_sig,) tilted mean sum_i w_i S_i
    ess: jax.Array     # effective sample size 1 / sum w_i^2


@jax.jit
def _tilt_kernel(S: jax.Array, p: jax.Array, kappa: jax.Array) -> Tilt:
    ell = -kappa * (p @ S)
    lse = logsumexp(ell)
    w = jnp.exp(ell - lse)
    return Tilt(logZ=lse - jnp.log(S.shape[1]), w=w, m=S @ w, ess=1.0 / jnp.sum(w ** 2))


@jax.jit
def _value_kernel(S: jax.Array, p: jax.Array, E: jax.Array, inv_lam: jax.Array,
                  kappa: jax.Array) -> jax.Array:
    ell = -kappa * (p @ S)
    logZ = logsumexp(ell) - jnp.log(S.shape[1])
    return -0.5 * kappa * jnp.sum(inv_lam * p ** 2) - kappa * (p @ E) - logZ


@jax.jit
def _cov_kernel(S: jax.Array, w: jax.Array, m: jax.Array) -> jax.Array:
    Sc = S - m[:, None]
    return (Sc * w) @ Sc.T


class DualProblem:
    """Monte-Carlo (sample-average) dual of the weighted expected-signature matching problem.

    With ``ell_i = -kappa <p, S_i>``, ``w = softmax(ell)`` and ``Lambda = diag(lambda)``:

        log Z(p) = logsumexp(ell) - log M,
        J(p)     = -kappa/2 p^T Lambda^{-1} p - kappa <p, E> - log Z(p),
        grad J   = kappa (m(p) - E - Lambda^{-1} p),
        hess J   = -kappa Lambda^{-1} - kappa^2 C(p),

    with ``m``, ``C`` the tilted mean and covariance. These are the exact derivatives of the
    sample-average ``J``. All linear algebra is restricted to the *active* coordinates
    ``cone.active_idx`` (free and lead words).

    :param S: Brownian signatures at ``T``, shape ``(n_sig, M)`` (empty-word row included). Kept on
        device as float64; memory is about ``8 n_sig M`` bytes.
    :param E: target expected signature, shape ``(n_sig,)``. ``E[0]`` is ignored.
    :param kappa: penalty weight.
    :param cone: the feasible cone.
    :param weights: word weights ``lambda`` of shape ``(n_sig,)`` (default: ones).
    """

    def __init__(self, S: jax.Array, E: jax.Array, kappa: float, cone: ConeSpec,
                 weights: Optional[jax.Array] = None):
        self.S = jnp.asarray(S, dtype=jnp.float64)
        n_sig, self.M = self.S.shape
        if n_sig != cone.n_sig:
            raise ValueError(f"S has {n_sig} rows but the cone has {cone.n_sig} words.")
        self.E = jnp.asarray(E, dtype=jnp.float64).at[0].set(0.0)
        self.kappa = float(kappa)
        self.cone = cone
        lam = np.ones(n_sig) if weights is None else np.asarray(weights, dtype=float).copy()
        if lam.shape != (n_sig,) or np.any(lam[1:] <= 0):
            raise ValueError("weights must be positive with shape (n_sig,).")
        lam[0] = 1.0
        self.weights = jnp.asarray(lam)
        self.inv_lam = 1.0 / self.weights
        self.idx = cone.active_idx
        self._active = jnp.asarray(cone.active_mask, dtype=float)

    # -- core quantities ----------------------------------------------------------------------

    def tilt(self, p: jax.Array) -> Tilt:
        """Tilt weights, ``log Z``, tilted mean and effective sample size at ``p``."""
        return _tilt_kernel(self.S, jnp.asarray(p), self.kappa)

    def value(self, p: jax.Array) -> float:
        """The sample-average dual function ``J(p)`` (differentiable in ``p``)."""
        return _value_kernel(self.S, jnp.asarray(p), self.E, self.inv_lam, self.kappa)

    def value_from_tilt(self, p: jax.Array, tilt: Tilt) -> float:
        """``J(p)`` from an already computed tilt (avoids a second pass over ``S``)."""
        p = jnp.asarray(p)
        return float(-0.5 * self.kappa * jnp.sum(self.inv_lam * p ** 2)
                     - self.kappa * (p @ self.E) - tilt.logZ)

    def grad(self, p: jax.Array, tilt: Optional[Tilt] = None) -> jax.Array:
        """Closed-form gradient ``kappa (m - E - Lambda^{-1} p)``, zero on forced words and on the
        empty word. Shape ``(n_sig,)``."""
        p = jnp.asarray(p)
        tilt = self.tilt(p) if tilt is None else tilt
        return self.kappa * (tilt.m - self.E - self.inv_lam * p) * self._active

    def hessian(self, p: jax.Array, tilt: Optional[Tilt] = None) -> jax.Array:
        """Closed-form Hessian ``-kappa Lambda^{-1} - kappa^2 C`` on the active coordinates.
        Shape ``(n_active, n_active)``, rows/columns ordered as ``self.idx``."""
        tilt = self.tilt(p) if tilt is None else tilt
        C = _cov_kernel(self.S, tilt.w, tilt.m)[np.ix_(self.idx, self.idx)]
        return -self.kappa * jnp.diag(self.inv_lam[self.idx]) - self.kappa ** 2 * C

    # -- primal side and duality gap ----------------------------------------------------------

    def primal_cost(self, p: jax.Array, tilt: Optional[Tilt] = None) -> dict:
        """Cost of the tilted law ``Q_p`` in the **modified primal** of spec §2.4.

        ``entropy`` is ``H(Q_p | W_M) = sum_i w_i log(M w_i)`` on the empirical reference measure.
        The modified penalty is ``kappa sum_v pen_v(z_v)``, ``z = m(p) - E``, with
        ``pen_v(z) = lambda_v z^2 / 2`` on free words, ``0`` on forced words, and on leading words
        the conjugate of the one-sided constraint ``p_v >= eps``:
        ``lambda_v z^2 / 2`` if ``lambda_v z >= eps``, else ``eps z - eps^2 / (2 lambda_v)``.
        ``penalty_unmodified`` is the plain ``kappa/2 sum_v lambda_v z_v^2`` over all non-empty words.

        :return: dict with keys ``entropy``, ``penalty``, ``total``, ``penalty_unmodified``.
        """
        tilt = self.tilt(p) if tilt is None else tilt
        w = np.asarray(tilt.w)
        pos = w > 0
        entropy = float(np.sum(w[pos] * np.log(self.M * w[pos])))
        z = np.asarray(tilt.m - self.E)
        lam = np.asarray(self.weights)
        eps = self.cone.eps
        quad = 0.5 * lam * z ** 2
        pen = np.where(self.cone.free_mask, quad, 0.0)
        lead = list(self.cone.lead_idx)
        pen[lead] = np.where(lam[lead] * z[lead] >= eps, quad[lead],
                             eps * z[lead] - eps ** 2 / (2.0 * lam[lead]))
        penalty = self.kappa * float(np.sum(pen))
        return {"entropy": entropy, "penalty": penalty, "total": entropy + penalty,
                "penalty_unmodified": self.kappa * float(np.sum(quad[1:]))}

    def duality_gap(self, p: jax.Array, tilt: Optional[Tilt] = None) -> float:
        """``P(p) - J(p) >= 0`` with the modified primal; it vanishes at the constrained optimum."""
        tilt = self.tilt(p) if tilt is None else tilt
        return self.primal_cost(p, tilt)["total"] - self.value_from_tilt(p, tilt)


# ---------------------------------------------------------------------------------------------
# 3.4 Optimiser
# ---------------------------------------------------------------------------------------------

@dataclass
class DualResult:
    """Output of :func:`optimize_dual`."""
    p: jax.Array
    J: float
    gap: float
    converged: bool
    n_iter: int
    active_leads: Tuple[int, ...]          # indices of the leading words sitting at eps
    history: List[dict] = field(default_factory=list)   # per iter: J, grad_norm, ess_frac, step, gap
    kappa: Optional[float] = None


def _projected_grad_norm(problem: DualProblem, p: jax.Array, g: jax.Array) -> float:
    """``||p - Pi_K(p + Lambda grad / kappa)||_inf``."""
    step = problem.weights * g / problem.kappa
    return float(jnp.max(jnp.abs(p - project(p + step, problem.cone))))


def _active_leads(problem: DualProblem, p: jax.Array, atol: float) -> Tuple[int, ...]:
    p = np.asarray(p)
    eps = problem.cone.eps
    return tuple(int(i) for i in problem.cone.lead_idx if p[i] <= eps + atol)


def _newton_direction(problem: DualProblem, p: jax.Array, g: jax.Array, tilt: Tilt,
                      delta: float) -> np.ndarray:
    """Projected-Newton direction (Bertsekas): Newton step on the free set ``F``, zero on the
    binding leads ``B = {p_v <= eps + delta, g_v < 0}``."""
    idx = problem.idx
    p_np, g_np = np.asarray(p), np.asarray(g)
    binding = {i for i in problem.cone.lead_idx if p_np[i] <= problem.cone.eps + delta and g_np[i] < 0}
    keep = np.array([i not in binding for i in idx])
    negH = -np.asarray(problem.hessian(p, tilt))[np.ix_(keep, keep)]   # >= kappa Lambda^{-1} > 0
    gF = g_np[idx[keep]]
    # Jacobi scaling: the diagonal spans many orders of magnitude (kappa/lambda vs kappa^2 Var).
    s = 1.0 / np.sqrt(np.diag(negH))
    A = negH * s[:, None] * s[None, :]
    try:
        cf = scipy.linalg.cho_factor(A, lower=True, check_finite=False)
        dF = s * scipy.linalg.cho_solve(cf, s * gF, check_finite=False)
    except np.linalg.LinAlgError:
        dF = s * np.linalg.lstsq(A + 1e-12 * np.eye(A.shape[0]), s * gF, rcond=None)[0]
    d = np.zeros_like(p_np)
    d[idx[keep]] = dF
    return d


def optimize_dual(problem: DualProblem, p0: Optional[jax.Array] = None, method: str = "newton",
                  max_iter: int = 200, tol: float = 1e-8, step0: float = 1.0,
                  armijo: float = 1e-4, backtrack: float = 0.5, min_ess_frac: float = 1e-3,
                  verbose: bool = False, max_backtracks: int = 60) -> DualResult:
    """Maximise the sample-average dual ``J`` over ``K_eps``.

    * ``method="newton"`` (default): projected Newton with a box on the leading coordinates. The
      binding set is ``B = {lead at eps with dJ/dp_v < 0}``, the Newton direction
      ``d_F = (-H_FF)^{-1} g_F`` is taken on the rest (``d_B = 0``) and ``Pi_K(p + t d)`` is
      line-searched.
    * ``method="gradient"``: projected scaled ascent ``p <- Pi_K(p + t Lambda grad J / kappa)``. With
      ``t = eta`` this is the damped fixed point ``p <- (1-eta) p + eta Lambda (m - E)`` of the note §7.

    The step ``t`` starts at ``step0`` and is halved (``backtrack``) until the Armijo condition
    ``J(p_t) >= J(p) + armijo * <grad, p_t - p>`` holds and the ESS fraction of the tilt at ``p_t`` is ``>= min_ess_frac`` (or, from a point
    already below that floor, does not fall below the current fraction). Close to the optimum,
    where ``J(p_t) - J(p)`` is at rounding level (a few ulps of ``J``), a step is accepted instead
    if it decreases the projected-gradient norm. Iterations stop
    when ``||p - Pi_K(p + Lambda grad / kappa)||_inf <= tol`` (``converged=True``) or when no
    acceptable step exists (``converged=False``).

    :param problem: the dual problem.
    :param p0: starting point (projected onto ``K_eps``); ``None`` means ``project(0)``.
    :param method: ``"newton"`` or ``"gradient"``.
    :param max_iter: maximum number of iterations.
    :param tol: tolerance on the projected-gradient norm.
    :param step0: initial step of every line search.
    :param armijo: sufficient-increase constant.
    :param backtrack: step reduction factor.
    :param min_ess_frac: minimum accepted ESS / M.
    :param verbose: print one line per iteration.
    :param max_backtracks: maximum number of step reductions per line search.
    :return: a :class:`DualResult`.
    """
    if method not in ("newton", "gradient"):
        raise ValueError(f"unknown method {method!r}; expected 'newton' or 'gradient'.")
    cone = problem.cone
    p = project(jnp.zeros(cone.n_sig) if p0 is None else jnp.asarray(p0, dtype=jnp.float64), cone)
    tilt = problem.tilt(p)
    J = problem.value_from_tilt(p, tilt)
    history: List[dict] = []
    converged = False
    step = 0.0
    n_iter = 0

    for n_iter in range(max_iter + 1):
        g = problem.grad(p, tilt)
        pg = _projected_grad_norm(problem, p, g)
        gap = problem.duality_gap(p, tilt)
        ess_frac = float(tilt.ess) / problem.M
        history.append({"J": J, "grad_norm": pg, "ess_frac": ess_frac, "step": step, "gap": gap})
        if verbose:
            print(f"[{method}] it={n_iter:3d}  J={J:.12g}  |pg|={pg:.3e}  ess={ess_frac:.3f}  "
                  f"step={step:.3g}  gap={gap:.3e}")
        if pg <= tol:
            converged = True
            break
        if n_iter == max_iter:
            break

        if method == "newton":
            d = jnp.asarray(_newton_direction(problem, p, g, tilt, delta=min(1e-6, pg)))
        else:
            d = problem.weights * g / problem.kappa * problem._active

        slack = 64 * np.finfo(float).eps * (1.0 + abs(J))
        ess_floor = min(min_ess_frac, ess_frac)   # a start already below the floor may still improve
        t = step0
        accepted = False
        for _ in range(max_backtracks):
            p_new = project(p + t * d, cone)
            tilt_new = problem.tilt(p_new)
            if float(tilt_new.ess) / problem.M >= ess_floor:
                J_new = problem.value_from_tilt(p_new, tilt_new)
                if np.isfinite(J_new):
                    if J_new >= J + armijo * float(g @ (p_new - p)):
                        accepted = True
                    elif J_new >= J - slack:
                        # J differences are at rounding level: decide on the (exact, closed-form)
                        # projected gradient instead, which keeps decreasing near the optimum.
                        g_new = problem.grad(p_new, tilt_new)
                        accepted = _projected_grad_norm(problem, p_new, g_new) < pg
                    if accepted:
                        break
            t *= backtrack
        if not accepted:
            if verbose:
                print(f"[{method}] line search failed at it={n_iter}; stopping.")
            break
        p, tilt, J, step = p_new, tilt_new, J_new, t

    return DualResult(p=p, J=J, gap=problem.duality_gap(p, tilt), converged=converged,
                      n_iter=n_iter, active_leads=_active_leads(problem, p, atol=max(tol, 1e-12)),
                      history=history, kappa=problem.kappa)


def _warm_start(problem: DualProblem, candidates: Sequence[jax.Array], min_ess_frac: float) -> jax.Array:
    """The candidate starting point with the largest ``J`` among those with an acceptable ESS
    (falls back to the largest ESS if none is acceptable)."""
    best, best_key = None, None
    for c in candidates:
        c = project(c, problem.cone)
        tilt = problem.tilt(c)
        ok = float(tilt.ess) / problem.M >= min_ess_frac
        J = problem.value_from_tilt(c, tilt)
        key = (ok, J if (ok and np.isfinite(J)) else float(tilt.ess))
        if best_key is None or key > best_key:
            best, best_key = c, key
    return best


def continuation(S: jax.Array, E: jax.Array, cone: ConeSpec, kappas: Sequence[float],
                 weights: Optional[jax.Array] = None, warm_start: str = "best",
                 **opt_kwargs) -> List[DualResult]:
    """Warm-started sequence of dual solves for increasing ``kappa``.

    The starting point for ``kappa_{k+1}`` is built from ``p*_k``:

    * ``warm_start="residual"``: ``p*_k`` itself (``p*`` is a weighted residual, so it is the right
      scale while the residual does not shrink, i.e. for small ``kappa``);
    * ``warm_start="tilt"``: ``p*_k kappa_k / kappa_{k+1}``, which keeps the tilt ``q = -kappa p``
      fixed (the right scale for large ``kappa``, when ``kappa p*`` converges to the natural
      parameter of the I-projection and the residual decays like ``1/kappa``);
    * ``warm_start="best"`` (default): whichever of the two, or ``project(0)``, has the largest
      ``J`` among the points whose ESS fraction is at least ``min_ess_frac``. With large ``kappa``
      steps the residual warm start multiplies the tilt by ``kappa_{k+1}/kappa_k`` and can collapse
      the ESS, so this is the robust choice.

    :param S: Brownian signatures ``(n_sig, M)``.
    :param E: target expected signature ``(n_sig,)``.
    :param cone: the feasible cone.
    :param kappas: increasing penalty weights.
    :param weights: word weights (default: ones).
    :param warm_start: ``"best"``, ``"residual"`` or ``"tilt"``.
    :param opt_kwargs: forwarded to :func:`optimize_dual` (``p0`` is used for the first ``kappa`` only).
    :return: one :class:`DualResult` per ``kappa``.
    """
    if warm_start not in ("best", "residual", "tilt"):
        raise ValueError(f"unknown warm_start {warm_start!r}; expected 'best', 'residual' or 'tilt'.")
    S = jnp.asarray(S, dtype=jnp.float64)
    results: List[DualResult] = []
    p0 = opt_kwargs.pop("p0", None)
    min_ess_frac = opt_kwargs.get("min_ess_frac", 1e-3)
    for kappa in kappas:
        problem = DualProblem(S, E, kappa, cone, weights)
        if results:
            prev = results[-1]
            residual, tilt = prev.p, prev.p * (prev.kappa / kappa)
            p0 = {"residual": residual, "tilt": tilt}.get(warm_start)
            if p0 is None:
                p0 = _warm_start(problem, [residual, tilt, jnp.zeros(cone.n_sig)], min_ess_frac)
        res = optimize_dual(problem, p0=p0, **opt_kwargs)
        results.append(res)
    return results


# ---------------------------------------------------------------------------------------------
# 3.5 Optimal trajectories
# ---------------------------------------------------------------------------------------------

def control_tensor(p: jax.Array, kappa: float, trunc: int, dim: int = 2) -> TensorSequence:
    """The terminal tensor ``q = -kappa p`` of the ``sig_control`` (maximisation) convention,
    with empty-word coefficient ``0`` (spec §1)."""
    q = -kappa * jnp.asarray(p, dtype=jnp.float64)
    n_sig = int(number_of_words_up_to_trunc(trunc, dim))
    if q.shape != (n_sig,):
        raise ValueError(f"p has shape {q.shape}, expected ({n_sig},).")
    return TensorSequence(array=q.at[0].set(0.0), trunc=trunc, dim=dim)


def optimal_policy(p: jax.Array, kappa: float, sig_brownian_1: TensorSequence, T: float,
                   trunc: int) -> sc.Policy:
    """Monte-Carlo signature feedback of the auxiliary problem at ``p`` (``q = -kappa p``).

    ``sig_brownian_1`` must be an **independent** Brownian signature sample on ``[0, 1]``,
    truncated at ``>= deg(q)`` (empty-word row included); it is sliced to the words up to
    ``deg(q)``, as ``sc.monte_carlo_policy`` requires.

    :param p: dual tensor, shape ``(n_sig,)``.
    :param kappa: penalty weight.
    :param sig_brownian_1: sample of ``Sig(W)_{0,1}``, shape ``(n_sig_mc, N_MC)``.
    :param T: horizon.
    :param trunc: truncation order of ``p``.
    :return: a policy ``(i, t, X_sig) -> (control, value, extra)``.
    """
    q = control_tensor(p, kappa, trunc)
    nz = np.abs(np.asarray(q.array)) > 1e-12
    n_mc = sig_brownian_1.array.shape[0]
    if not nz.any():
        # q = 0: zero control and zero value (Z = 1); monte_carlo_policy cannot take a zero tensor.
        def zero_policy(i: int, t: float, X_sig: TensorSequence):
            size = X_sig.array.shape[1]
            return np.zeros(size), np.zeros(size), {}
        return zero_policy
    q_deg = int(np.asarray(q.get_lengths_array())[nz].max())
    n_need = int(number_of_words_up_to_trunc(q_deg, 2))
    if n_mc < n_need:
        raise ValueError(f"sig_brownian_1 has {n_mc} rows; deg(q)={q_deg} needs {n_need}.")
    sigW = TensorSequence(array=jnp.asarray(sig_brownian_1.array[:n_need]), trunc=q_deg, dim=2)
    return sc.monte_carlo_policy(q, sigW, T, trunc, dim=2)


def rollout(p: jax.Array, kappa: float, T: float, trunc: int, n_steps: int, size: int,
            sig_brownian_1: TensorSequence, seed: int, chunk: int = 500,
            x0: float = 0.0) -> sc.ControlResult:
    """Closed-loop rollout ``dX = alpha*(t, Sig(X)_t) dt + dW`` of the optimal feedback at ``p``.

    The cost per step is about ``2 size N_MC n_sig`` flops. Paths are simulated in chunks of
    ``chunk`` to bound the ``(chunk, N_MC)`` work arrays of the Monte-Carlo policy. The signature
    (hence the feedback) is translation invariant, so ``x0`` only shifts the state paths; set it to
    the target's initial point to compare paths on the same scale.

    :param p: dual tensor, shape ``(n_sig,)``.
    :param kappa: penalty weight.
    :param T: horizon.
    :param trunc: signature truncation order.
    :param n_steps: number of (uniform) time steps.
    :param size: number of controlled paths.
    :param sig_brownian_1: independent sample of ``Sig(W)_{0,1}`` (see :func:`optimal_policy`).
    :param seed: RNG seed of the Brownian increments.
    :param chunk: number of paths simulated at once.
    :param x0: initial state (does not affect the signature or the control).
    :return: a ``sc.ControlResult`` (state, signature trajectory, control, value).
    """
    t_grid = np.linspace(0.0, T, n_steps + 1)
    dW = np.random.default_rng(seed).normal(size=(size, n_steps)) * np.sqrt(T / n_steps)
    policy = optimal_policy(p, kappa, sig_brownian_1, T, trunc)
    parts = [sc.simulate(dW[s:s + chunk], t_grid, policy, trunc, dim=2, sigma=1.0, x0=x0)
             for s in range(0, size, chunk)]
    if len(parts) == 1:
        return parts[0]
    return sc.ControlResult(X=np.concatenate([r.X for r in parts], axis=0),
                            sig=np.concatenate([r.sig for r in parts], axis=1),
                            control=np.concatenate([r.control for r in parts], axis=0),
                            value=np.concatenate([r.value for r in parts], axis=0),
                            extra={k: np.concatenate([r.extra[k] for r in parts], axis=0)
                                   for k in parts[0].extra})


def resample(problem: DualProblem, p: jax.Array, paths: np.ndarray, n: int, seed: int,
             x0: Optional[float] = None) -> np.ndarray:
    """Multinomial resampling of the Brownian paths behind ``problem.S`` with the tilt weights.

    The tilted law ``Q_p`` is a reweighting of Wiener measure on path space, so the resampled
    paths are exact samples of the optimal path law up to Monte-Carlo error (note §8, item 2).
    ``paths[i]`` must be the path whose signature is ``problem.S[:, i]``.

    :param problem: the dual problem.
    :param p: dual tensor, shape ``(n_sig,)``.
    :param paths: Brownian paths, shape ``(M, n_t)``.
    :param n: number of resampled paths.
    :param seed: RNG seed.
    :param x0: if given, every resampled path is translated to start at ``x0`` (the signature is
        translation invariant, so the law of the increments is unchanged).
    :return: resampled paths, shape ``(n, n_t)``.
    """
    paths = np.asarray(paths)
    if paths.shape[0] != problem.M:
        raise ValueError(f"paths has {paths.shape[0]} rows but problem.S has {problem.M} samples.")
    w = np.asarray(problem.tilt(p).w, dtype=float)
    w = w / w.sum()
    idx = np.random.default_rng(seed).choice(problem.M, size=n, replace=True, p=w)
    out = paths[idx]
    if x0 is not None:
        out = out - out[:, :1] + x0
    return out
