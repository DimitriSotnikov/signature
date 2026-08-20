"""Cubature--log-sum-exp backward scheme for the signature log-Laplace/characteristic tensor.

This implements ``doc/cubature_lse_scheme_spec.md`` in its simplest framework: **two-point
cubature** + **shuffle log-sum-exp**. It is an alternative to the ODE integrators of
:mod:`signature.ode_integration` for the *same* object -- the tensor sequence ``psi_t`` with

    E[ exp<p, X (x) W_{t,T}> ] = exp<psi_t, X>,      psi_T = p,

which the ODE route obtains by time-stepping the shuffle Riccati equation
``R psi = psi|_time + 1/2 psi|_{ss} + 1/2 (psi|_s)^{sh 2}``.

**What it is instead.** The scheme is not a discretization of that ODE. It is the *exact*
backward dynamic-programming recursion for the log-transform of the **cubature random walk**
that replaces the Brownian increment on each step by a finite quantizer. In log coordinates
the DP step is a shuffle log-sum-exp,

    psi_{t-h} = log^sh ( sum_j w_j exp^sh( psi_t |_{g_j} ) ),                            (*)

with ``(w_j, g_j)`` the cubature weights and (group-like) increment signatures.

**Shift direction.** Draft v1 of the spec wrote the step with the *left* shift
``{}_{g_j}|psi_t``; this was wrong and is corrected in spec v2 (section 1). With the repo's Chen
convention (past on the left, ``Sig_{t-h,T} = g (x) Sig_{t,T}``), conditioning on ``xi`` gives

    Phi_{t-h}(g) = E_xi[ Phi_t(g (x) xi) ] = E_xi[ exp<psi_t, g (x) xi> ]
                 = E_xi[ exp<psi_t|_xi, g> ],

so (*) needs the **right** shift ``psi_t|_{g_j}`` (the ``proj``-flavoured one). The spec's own
composition identity in section 9.1, ``p <| (g1 (x) g2) = (p <| g2) <| g1``, holds only for the
right shift, so section 9.1 agrees with the derivation above against section 1. Note that at
``dim == 1`` concatenation is commutative and the two shifts coincide -- only a ``dim >= 2`` run
can tell them apart, which is why :func:`shift_matrix` also exposes ``side="left"``.

**Consistency.** With the symmetric two-point nodes ``g_pm = exp^(x)(h e_time +- sqrt(h) e_s)``
and weights ``1/2``, one step of (*) expands as ``psi + h R psi + O(h^2)``: the ``log 2`` coming
out of ``log(2 cosh)`` cancels the ``log(1/2)`` weight shifts, and ``1/2 (c + delta_bar)^{sh 2}``
reassembles ``h/2 (psi|_s)^{sh 2}``. The odd powers of ``sqrt(h)`` cancel by the node symmetry, so
the local error is ``O(h^2)`` and the scheme has **global weak order 1**. In particular it
converges as ``h -> 0`` to the *same truncated* Riccati solution as Euler/PECE/RK4 -- it is not
immune to the truncation-induced blowup of ``doc/Riccati_scheme_ideas.md``, it merely does not add
a step-size instability on top of it.

**Error budget** (spec section 8.4), separable at ``dim == 1`` thanks to :func:`walk_reference`:

* closure error in ``N``   -- ``solve(...) - walk_reference(...)``: the adjoint pulls levels above
  the truncation down into the retained ones, and those levels are missing;
* cubature error in ``h``  -- ``walk_reference(...) - truth``: weak error of the quantized walk;
* floating point.

The production path never materializes a shuffle exponential: :func:`shuffle_log` and
:func:`lse2_naive` exist only to validate :func:`lse2`.

Everything is written dtype-agnostically and runs on complex input, but the branch-continuity
tracking of spec section 4.5 is **not** implemented -- complex ``p`` with a large imaginary part
is out of scope for this version.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp

from .tensor_sequence import TensorSequence
from .factory import unit_like
from .shuffle_product import shuffle_prod
from .words import number_of_words_up_to_trunc, index_to_word_len, index_to_lam_prod_vect

# ``signature.shuffle_table.get_shuffle_table`` returns a NumPy array; JAX consumes it happily.
ShuffleTable = Union[np.ndarray, jax.Array]


# --------------------------------------------------------------------------------------------
# 1. Shift (adjoint) operators by a fixed group element
# --------------------------------------------------------------------------------------------

def _concat_index_table(trunc: int, dim: int) -> Tuple[int, np.ndarray, np.ndarray]:
    """Concatenation index table: ``U[i, k] = index(w_i . v_k)`` plus the validity mask.

    Uses the same arithmetic as :func:`signature.tensor_product.tensor_prod_word`,
    ``index(w v) = n(|w| + |v| - 1) + dim^{|v|} b_w + b_v`` with ``b_x`` the position of ``x``
    inside its length block, but in plain NumPy so that it is exact and word-integer free (no
    19-digit limit, no floating-point tolerance anywhere).
    """
    n_sig = int(number_of_words_up_to_trunc(trunc, dim))
    idx = jnp.arange(n_sig)
    lengths = np.asarray(index_to_word_len(idx, dim))
    base = np.asarray(idx - number_of_words_up_to_trunc(lengths - 1, dim))

    total_len = lengths[:, None] + lengths[None, :]                     # |w_i| + |v_k|
    offset = np.asarray(number_of_words_up_to_trunc(jnp.asarray(total_len) - 1, dim))
    U = offset + (dim ** lengths[None, :]) * base[:, None] + base[None, :]
    return n_sig, U, total_len <= trunc


def shift_matrix(g: TensorSequence, side: str = "right") -> jax.Array:
    """Dense matrix of the shift of a tensor sequence by the fixed element ``g``.

    ``side="right"`` (the one the scheme needs) returns ``A`` with

        (A psi)^w = sum_v psi^{wv} g^v,          i.e.  <A psi, x> = <psi, x (x) g>,

    the truncated **right shift** ``psi|_g`` -- the linear extension of
    :meth:`TensorSequence.proj` from words to the whole of ``g``, and the adjoint of
    ``x -> x (x) g``.

    ``side="left"`` returns ``B`` with ``(B psi)^v = sum_w psi^{wv} g^w`` (``{}_g|psi``, the
    recentering shift of :func:`signature.projection.left_proj_on_seq`). It is provided **only** to
    check numerically that the spec's section-1 notation is the wrong one.

    Because the two shifts coincide when the alphabet has a single letter, a ``dim == 1``
    experiment cannot distinguish them.

    .. note::
        Built from :func:`_concat_index_table` rather than from
        :func:`signature.tensor_product.tensor_prod`: the latter skips any coefficient of ``g``
        that ``jnp.allclose`` calls zero (``atol = 1e-8``), which silently truncates the node
        signature once ``sqrt(h)^k / k!`` drops below that -- an *absolute* threshold on a quantity
        whose contribution is only small *relative* to the large low-level coefficients it is
        multiplied against.

    :param g: the group-like element to shift by (a plain tensor sequence works too).
    :param side: ``"right"`` (default, used by the scheme) or ``"left"`` (diagnostic).
    :return: an ``(n_sig, n_sig)`` matrix acting on ``psi.array``.
    """
    if side not in ("right", "left"):
        raise ValueError(f"side must be 'right' or 'left'; got {side!r}.")
    trunc, dim = int(g.trunc), int(g.dim)
    n_sig, U, valid = _concat_index_table(trunc, dim)
    gv = np.asarray(g.array)

    rows_w, rows_v = np.meshgrid(np.arange(n_sig), np.arange(n_sig), indexing="ij")
    m = valid.ravel()
    i_w, i_v, u = rows_w.ravel()[m], rows_v.ravel()[m], U.ravel()[m]

    out = np.zeros((n_sig, n_sig), dtype=gv.dtype)
    if side == "right":
        np.add.at(out, (i_w, u), gv[i_v])       # (A psi)^{w} = sum_v g^v psi^{wv}
    else:
        np.add.at(out, (i_v, u), gv[i_w])       # (B psi)^{v} = sum_w g^w psi^{wv}
    return jnp.asarray(out)


# --------------------------------------------------------------------------------------------
# 2. Cubature measures (spec section 5.1)
# --------------------------------------------------------------------------------------------

def segment_signature(increment: np.ndarray, trunc: int, dim: int) -> TensorSequence:
    """Signature of the straight segment with the given ``increment`` vector, in closed form.

    ``Sig = exp^(x)(sum_m increment_m e_m)``, i.e. ``Sig^{i_1...i_k} = x_{i_1} ... x_{i_k} / k!``.
    Written out directly rather than through :func:`signature.tensor_product.tensor_exp` for the
    same reason as in :func:`shift_matrix`: ``tensor_exp`` goes through ``tensor_prod``, which drops
    coefficients below an absolute ``1e-8``.
    """
    increment = np.asarray(increment, dtype=float)
    n_sig = int(number_of_words_up_to_trunc(trunc, dim))
    idx = jnp.arange(n_sig)
    lengths = np.asarray(index_to_word_len(idx, dim))
    prods = np.asarray(index_to_lam_prod_vect(idx, dim, jnp.asarray(increment)))
    from scipy.special import factorial
    return TensorSequence(array=jnp.asarray(prods / factorial(lengths)), trunc=trunc, dim=dim)


def cubature_from_increments(increments: np.ndarray, weights: np.ndarray, trunc: int, dim: int,
                             side: str = "right") -> Tuple[jax.Array, jax.Array]:
    """Build the ``(weights, shift matrices)`` pair of a cubature from its node **increments**.

    Node ``j`` is the straight segment with increment vector ``increments[j]`` (one entry per
    letter), whose signature is ``g_j = exp^(x)(sum_m increments[j,m] e_m)``. The scheme only ever
    needs the shift matrices, so they are precomputed once here.

    :param increments: ``(m, dim)`` array of per-node increment vectors.
    :param weights: ``(m,)`` array of cubature weights (must sum to 1).
    :param trunc: truncation level.
    :param dim: alphabet size.
    :param side: shift direction, see :func:`shift_matrix`.
    :return: ``(weights, shifts)`` with ``shifts`` of shape ``(m, n_sig, n_sig)``.
    """
    increments = np.atleast_2d(np.asarray(increments, dtype=float))
    weights = np.asarray(weights, dtype=float)
    if increments.shape[1] != dim:
        raise ValueError(f"increments must have {dim} columns (one per letter); "
                         f"got shape {increments.shape}.")
    if increments.shape[0] != weights.shape[0]:
        raise ValueError("increments and weights must have the same number of nodes.")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"cubature weights must sum to 1; got {weights.sum()}.")

    shifts = [shift_matrix(segment_signature(row, trunc, dim), side=side) for row in increments]
    return jnp.asarray(weights), jnp.stack(shifts)


def two_point_cubature(h: float, trunc: int, dim: int, time_letter: Optional[int] = None,
                       side: str = "right") -> Tuple[jax.Array, jax.Array]:
    """Symmetric two-point (per Brownian direction) cubature over a step of length ``h``.

    Degree-3 cubature on Wiener space (spec section 5.1): with ``d`` Brownian letters, the ``2d``
    nodes are ``+- sqrt(d h) e_m`` with weights ``1/(2d)``. Moments ``1, 2, 3`` of the Brownian
    increment are matched; the first mismatch is the 4th (``h^2`` against ``3 h^2``), giving local
    weak error ``O(h^2)`` and global ``O(h)``.

    A ``time_letter`` (1-based), if given, is *deterministic*: it receives the increment ``h`` in
    every node rather than being quantized. This is the time-augmented case ``(t, W_t)`` used by the
    control problems -- ``dim=2, time_letter=1`` puts time on letter ``1`` and the state on ``2``,
    matching :mod:`signature.sig_control`.

    :param h: step length.
    :param trunc: truncation level.
    :param dim: alphabet size.
    :param time_letter: 1-based index of the deterministic time letter, or ``None`` if there is
        none (the ``dim=1`` state-only case of the Laplace problem).
    :param side: shift direction, see :func:`shift_matrix`.
    :return: ``(weights, shifts)`` as in :func:`cubature_from_increments`.
    """
    bm_letters = [m for m in range(1, dim + 1) if m != time_letter]
    d = len(bm_letters)
    if d == 0:
        raise ValueError("at least one non-time letter is required.")
    increments, weights = [], []
    for m in bm_letters:
        for sign in (+1.0, -1.0):
            row = np.zeros(dim)
            if time_letter is not None:
                row[time_letter - 1] = h
            row[m - 1] = sign * np.sqrt(d * h)
            increments.append(row)
            weights.append(1.0 / (2 * d))
    return cubature_from_increments(np.array(increments), np.array(weights), trunc, dim, side=side)


# --------------------------------------------------------------------------------------------
# 3. The log-cosh Taylor engine and the shuffle log-sum-exp (spec sections 4.1--4.2)
# --------------------------------------------------------------------------------------------

def logcosh_series(c, order: int) -> jax.Array:
    """Taylor coefficients ``(a_1, ..., a_order)`` of ``x -> logcosh(c + x) - logcosh(c)``.

    Generated from ``t(x) = tanh(c + x) = sum_k t_k x^k`` through ``t' = 1 - t^2``:

        t_1 = 1 - t_0^2,      (k+1) t_{k+1} = - sum_{j=0}^{k} t_j t_{k-j}   (k >= 1),

    then ``a_k = t_{k-1} / k``. At ``c = 0`` this reproduces the even series
    ``x^2/2 - x^4/12 + x^6/45 - 17 x^8/2520 + ...``.

    The radius of convergence is ``dist(c, i pi (Z + 1/2))`` (the zeros of ``cosh``); see
    :func:`logcosh_radius`. ``order`` is a static Python int -- the loop is unrolled.

    :param c: the (scalar) expansion centre; real or complex.
    :param order: number of coefficients to return.
    :return: array of shape ``(order,)``.
    """
    if order < 1:
        raise ValueError("order must be >= 1.")
    t = [jnp.tanh(c)]
    if order >= 2:
        t.append(1 - t[0] ** 2)
    for k in range(1, order - 1):
        # (k+1) t_{k+1} = - sum_{j=0}^{k} t_j t_{k-j}
        t.append(-sum(t[j] * t[k - j] for j in range(k + 1)) / (k + 1))
    return jnp.stack([t[k - 1] / k for k in range(1, order + 1)])


def logcosh_radius(c) -> jax.Array:
    """Conditioning radius ``rho(c) = dist(c, i pi (Z + 1/2))`` of :func:`logcosh_series`.

    The coefficients grow like ``rho^{-k}``, so a small ``rho`` means the ``lse2`` step is close to
    a zero of the averaged exponential -- i.e. to a genuine singularity of ``psi`` (spec section 7).
    For **real** ``c`` this is bounded below by ``pi/2``, so the series is always well conditioned;
    the diagnostic only bites for complex frequencies.
    """
    c = jnp.asarray(c)
    x, y = jnp.real(c), jnp.imag(c)
    k = jnp.round(y / jnp.pi - 0.5)
    return jnp.sqrt(x ** 2 + (y - jnp.pi * (k + 0.5)) ** 2)


def compose_series(coeffs: jax.Array, x_bar: TensorSequence, shuffle_table: ShuffleTable) -> TensorSequence:
    """Evaluate ``sum_{k=1}^{M} coeffs[k-1] * x_bar^{sh k}`` by Horner in the shuffle algebra.

    ``x_bar`` must lie in the augmentation ideal (zero empty-word coefficient), where it is
    nilpotent, so a finite number of terms is exact. Costs ``M`` shuffle products.
    """
    unit = unit_like(x_bar)
    order = coeffs.shape[0]
    acc = unit * coeffs[order - 1]
    for k in range(order - 2, -1, -1):
        acc = unit * coeffs[k] + shuffle_prod(x_bar, acc, shuffle_table)
    return shuffle_prod(x_bar, acc, shuffle_table)


def _log_two_cosh(c):
    """``log(2 cosh c)``, computed stably as ``|c| + log1p(exp(-2|c|))`` for real ``c``."""
    if jnp.issubdtype(jnp.asarray(c).dtype, jnp.complexfloating):
        return jnp.log(2 * jnp.cosh(c))              # principal branch, no continuity tracking
    a = jnp.abs(c)
    return a + jnp.log1p(jnp.exp(-2 * a))


def lse2(a: TensorSequence, b: TensorSequence, shuffle_table: ShuffleTable,
         order: Optional[int] = None) -> Tuple[TensorSequence, jax.Array]:
    """Two-term shuffle log-sum-exp: ``log^sh(exp^sh a + exp^sh b)`` (spec section 4.5).

    Exact in the (commutative) shuffle algebra, from
    ``exp^sh a + exp^sh b = exp^sh((a+b)/2) sh (2 cosh^sh delta)`` with ``delta = (a-b)/2``:

        log^sh(exp^sh a + exp^sh b) = (a+b)/2 + log(2 cosh c) + LC_c(delta_bar),

    where ``delta = c + delta_bar`` splits off the empty-word constant and ``LC_c`` is the series
    of :func:`logcosh_series` composed by :func:`compose_series`.

    This is *algebraically* identical to ``shuffle_log(shuffle_exp(a) + shuffle_exp(b))`` -- all the
    operations are graded, so truncation commutes with both routes -- but numerically far better:
    the exponential route builds large near-cancelling intermediates and then feeds a ``log(1+x)``
    composition with a non-small ``x``, contaminating the low levels that everything else is built
    on. :func:`lse2_naive` implements the bad route for the equality test only.

    :param a: first exponent.
    :param b: second exponent.
    :param shuffle_table: shuffle table of the ambient algebra.
    :param order: number of series terms; defaults to ``trunc`` (always safe, since ``delta_bar``
        is nilpotent of order ``trunc + 1``). Must be supplied explicitly when ``a.trunc`` is a
        JAX tracer, i.e. from inside a traced loop.
    :return: ``(result, c)`` -- the log-sum-exp and the merge constant ``c`` used by the
        diagnostics of spec section 7.
    """
    order = int(a.trunc) if order is None else order
    half_sum = (a + b) * 0.5
    delta = (a - b) * 0.5
    c = delta.array[0]
    delta_bar = TensorSequence(array=delta.array.at[0].set(0), trunc=a.trunc, dim=a.dim)

    out = TensorSequence(array=half_sum.array, trunc=a.trunc, dim=a.dim)
    out = out + unit_like(out) * _log_two_cosh(c)
    if order >= 1:
        out = out + compose_series(logcosh_series(c, order), delta_bar, shuffle_table)
    return out, c


def lse(weights: jax.Array, terms, shuffle_table: ShuffleTable,
        order: Optional[int] = None) -> Tuple[TensorSequence, jax.Array]:
    """Weighted ``m``-term shuffle log-sum-exp ``log^sh(sum_j w_j exp^sh(terms[j]))``.

    Weights are absorbed as constant shifts, ``w exp^sh(a) = exp^sh(a + log w)``, then merged
    pairwise with :func:`lse2`. For the symmetric two-point cubature with ``w = 1/2`` the ``log 2``
    produced by ``log(2 cosh)`` cancels the two ``log(1/2)`` shifts exactly; this is left to happen
    numerically so that the same code path serves every cubature.

    :return: ``(result, cs)`` with ``cs`` the ``(m-1,)`` vector of merge constants.
    """
    log_w = jnp.log(jnp.asarray(weights))
    acc = terms[0] + unit_like(terms[0]) * log_w[0]
    cs = []
    for j in range(1, len(terms)):
        acc, c = lse2(acc, terms[j] + unit_like(terms[j]) * log_w[j], shuffle_table, order)
        cs.append(c)
    return acc, jnp.stack(cs) if cs else jnp.zeros(0)


# --------------------------------------------------------------------------------------------
# 4. Reference routines -- validation only, never on the production path
# --------------------------------------------------------------------------------------------

def shuffle_log(u: TensorSequence, shuffle_table: ShuffleTable) -> TensorSequence:
    """Shuffle logarithm, the inverse of :func:`signature.shuffle_product.shuffle_exp`.

    ``log^sh(u) = log(u^empty) + sum_{k=1}^{trunc} (-1)^{k+1}/k * x^{sh k}`` with
    ``x = u / u^empty - empty``. Reference implementation, used to validate :func:`lse2`.
    """
    trunc = int(u.trunc)
    unit = unit_like(u)
    c = u.array[0]
    x = u * (1.0 / c) - unit
    coeffs = jnp.array([(-1.0) ** (k + 1) / k for k in range(1, trunc + 1)])
    return unit * jnp.log(c) + compose_series(coeffs, x, shuffle_table)


def lse2_naive(a: TensorSequence, b: TensorSequence, shuffle_table: ShuffleTable) -> TensorSequence:
    """``log^sh(exp^sh a + exp^sh b)`` via explicit shuffle exp/add/log -- the route spec section
    4.5 forbids in production. Used only to check :func:`lse2` against it (spec section 8.1)."""
    from .shuffle_product import shuffle_exp
    return shuffle_log(shuffle_exp(a, shuffle_table) + shuffle_exp(b, shuffle_table), shuffle_table)


def walk_reference(p_coef: float, T: float, n_steps: int) -> float:
    """Exact **untruncated** value of the two-point cubature walk for ``<p, Sig> = p_coef * W_T^4``,
    at ``dim == 1``.

    A one-dimensional path has a signature determined by its endpoint alone
    (``Sig^{1^k} = S^k / k!``), so every one of the ``2^K`` walk paths with the same endpoint
    contributes identically and the walk collapses to ``S_K = sqrt(h) (2B - K)``, ``B ~ Bin(K, 1/2)``:

        psi_0^empty = logsumexp_j [ log C(K,j) - K log 2 + p_coef * h^2 (2j - K)^4 ].

    This is the exact answer the scheme would give at infinite truncation, so
    ``solve(...) - walk_reference(...)`` isolates the **closure error in N** and
    ``walk_reference(...) - truth`` isolates the **cubature error in h** (spec section 8.4), with no
    tree enumeration and no Monte-Carlo noise. Note the resulting equality test of spec section 9.2
    is a *convergence* test in ``N``, not an exact identity: backward in time ``psi_t`` acquires
    coefficients of unboundedly high degree even though ``psi_T = p`` is finite.

    :param p_coef: coefficient of ``W_T^4`` in the exponent (``-1`` for ``E[exp(-W_T^4)]``).
    :param T: horizon.
    :param n_steps: number of walk steps ``K``.
    :return: ``psi_0^empty = log E[exp(p_coef S_K^4)]``.
    """
    from scipy.special import gammaln, logsumexp
    K = int(n_steps)
    h = T / K
    j = np.arange(K + 1)
    log_binom = gammaln(K + 1) - gammaln(j + 1) - gammaln(K - j + 1)
    return float(logsumexp(log_binom - K * np.log(2.0) + p_coef * h ** 2 * (2.0 * j - K) ** 4))


# --------------------------------------------------------------------------------------------
# 5. The backward loop (spec section 6)
# --------------------------------------------------------------------------------------------

def cubature_lse_step(psi: TensorSequence, weights: jax.Array, shifts: jax.Array,
                      shuffle_table: ShuffleTable, order: Optional[int] = None
                      ) -> Tuple[TensorSequence, jax.Array]:
    """One backward step ``psi_{t-h} = log^sh(sum_j w_j exp^sh(psi_t|_{g_j}))``.

    :return: ``(psi_next, cs)`` with ``cs`` the merge constants of the ``m-1`` :func:`lse2` calls.
    """
    terms = [TensorSequence(array=shifts[j] @ psi.array, trunc=psi.trunc, dim=psi.dim)
             for j in range(shifts.shape[0])]
    return lse(weights, terms, shuffle_table, order)


def cubature_lse_solve(p: TensorSequence, T: float, n_steps: int, shuffle_table: ShuffleTable,
                       time_letter: Optional[int] = None, side: str = "right",
                       order: Optional[int] = None,
                       cubature: Callable[..., Tuple[jax.Array, jax.Array]] = two_point_cubature
                       ) -> Tuple[TensorSequence, dict]:
    """Run the backward scheme on a **uniform** grid of ``n_steps`` steps of length ``h = T/n_steps``.

    The cubature nodes depend on ``h``, so the grid must be uniform -- this is why the scheme is a
    dedicated :func:`jax.lax.scan` driver rather than a ``step_fun`` for
    :func:`signature.ode_integration.ode_solver_traj` (a ``step_fun`` that ignores its ``dt`` would
    be silently wrong on a non-uniform grid).

    Output layout matches ``ode_solver_traj``: the returned trajectory has shape
    ``(n_sig, n_steps + 1)``, **column 0 is the terminal condition** ``psi_T = p`` and column ``-1``
    is ``psi_0`` -- i.e. the same reversed-time indexing (``tau = T - t``) as the ODE route, so a
    trajectory from either can be consumed identically.

    :param p: terminal condition ``psi_T = p``.
    :param T: horizon.
    :param n_steps: number of backward steps ``K``.
    :param shuffle_table: shuffle table of the ambient algebra.
    :param time_letter: 1-based deterministic time letter, or ``None``; see
        :func:`two_point_cubature`.
    :param side: shift direction; ``"right"`` is the correct one (see the module docstring),
        ``"left"`` is there to demonstrate that the spec's section-1 notation is wrong.
    :param order: number of ``logcosh`` series terms per merge; defaults to ``trunc``.
    :param cubature: node factory, called as ``cubature(h, trunc, dim, time_letter, side)``.
    :return: ``(psi_traj, diagnostics)``; ``diagnostics`` holds ``"c"`` (merge constants, shape
        ``(n_steps, m-1)``), ``"rho"`` (series conditioning radius, same shape) and ``"psi0"``
        (the empty-word coefficient along the trajectory, shape ``(n_steps + 1,)``).
    """
    trunc, dim = int(p.trunc), int(p.dim)
    order = trunc if order is None else order
    h = T / n_steps
    weights, shifts = cubature(h, trunc, dim, time_letter=time_letter, side=side)

    # The scan carries the raw coefficient array, not the TensorSequence: ``trunc``/``dim`` are
    # pytree *leaves* of TensorSequence, so carrying one would trace them and break the static
    # Python ints the series orders need.
    def scan_fun(arr, _):
        psi = TensorSequence(array=arr, trunc=trunc, dim=dim)
        psi_next, cs = cubature_lse_step(psi, weights, shifts, shuffle_table, order)
        return psi_next.array, (psi_next.array, cs)

    _, (arr, cs) = jax.lax.scan(scan_fun, init=p.array, xs=None, length=n_steps)
    arr = jnp.concatenate([p.array[None], arr], axis=0)          # (n_steps + 1, n_sig)
    traj = TensorSequence(array=jnp.moveaxis(arr, 0, -1), trunc=trunc, dim=dim)
    diagnostics = {"c": cs, "rho": logcosh_radius(cs), "psi0": arr[:, 0]}
    return traj, diagnostics
