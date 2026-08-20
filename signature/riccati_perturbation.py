"""Triangular perturbation solver for the signature Riccati equation (``doc/riccati_perturbation.md``).

The **diffusive dilation** ``delta_eps`` acts on letters by ``time -> eps * time`` and
``state -> sqrt(eps) * state``, extended as an algebra homomorphism; on a word it is therefore
diagonal, ``delta_eps v = eps^{||v||/2} v`` with the **diffusive grade**

    ||v|| = 2 |v|_time + sum_s |v|_s .

Because ``delta_eps W_hat_T =d W_hat_{eps T}``, expanding in ``eps^{1/2}`` is simultaneously a
*small terminal data* and a *small time* expansion. Substituting the ansatz
``psi^{delta_eps p} = sum_k psi_k eps^{k/2}`` into the Riccati equation and matching powers gives

    d/dt psi_k + L psi_k + 1/2 sum_{l=0}^{k} (psi_l|_s) sh (psi_{k-l}|_s) = 0,
    psi_k(T) = pi_k(p) := sum_{||v|| = k} p^v v,

with ``L ell = ell|_time + 1/2 sum_s ell|_ss``.

**Why this is triangular, and exactly solvable.** Three facts, each asserted at run time:

1. ``psi_k`` lives in ``F_k = span{v : ||v|| <= k}`` -- a *filtration*, not a grading: ``L`` lowers
   the grade by 2 and ``(psi_l|_s) sh (psi_m|_s)`` has grade ``<= l + m - 2``, so ``F_k`` is
   invariant and the terminal data ``pi_k(p)`` is homogeneous of grade ``k``.

   In fact more is true, and it is worth stating because it changes how the scheme is truncated
   (verified numerically, see :func:`order_of`): once the ``tau``-degree is tracked as well, the
   ``tau^j`` coefficient of ``psi_k`` is supported on words of grade **exactly** ``k - 2j``. The
   filtration ``F_k`` is the union of these over ``j``. Two consequences:

   * each pair (word grade ``gamma``, ``tau``-degree ``j``) is carried by **exactly one** order,
     ``k = gamma + 2 j``. The hierarchy is therefore not merely triangular but *diagonal*: no
     summation over orders is needed to reconstruct a given Taylor coefficient, and the degree bound
     ``j <= floor(k/2)`` of item 3 is just ``gamma = k - 2j >= 0``;
   * truncating at order ``K`` keeps the **triangle** ``{gamma + 2 j <= K}`` -- a *joint* truncation
     in word grade and time degree. This is a genuinely different (and, at small ``T``, better)
     truncation than "every word of length ``<= N``, every Taylor degree ``<= J``", which is what an
     ODE integrator on the truncated algebra commits to.
2. ``F_0 = span{empty}``, so ``psi_0`` is a constant multiple of the empty word and
   ``psi_0|_s = 0``. That kills **both** the ``l = 0`` and the ``l = k`` term of the quadratic sum,
   so the equation for ``psi_k`` is **linear in psi_k**, with a source built only from
   ``psi_1, ..., psi_{k-1}``. Solve ``k = 0, 1, 2, ...`` in order; every stage is linear.
3. ``L`` is nilpotent on ``F_k`` and the source is polynomial in ``tau = T - t``, so each ``psi_k``
   is a **polynomial in tau of degree <= floor(k/2)**. There is no ODE to integrate and no time
   grid: :func:`perturbation_hierarchy` returns exact polynomial coefficients.

**Relation to the plain Taylor method (important).** Summing the system over ``k`` at ``eps = 1``
and writing ``a_j = sum_k g_{k,j}`` for the ``tau^j`` coefficient of the total gives

    (j+1) a_{j+1} = L a_j + 1/2 sum_s sum_i (a_i|_s) sh (a_{j-i}|_s),

which is just the recursive ``tau``-Taylor coefficient relation for ``g' = R g``. So the graded
hierarchy and :func:`taylor_coefficients` produce **identical values wherever both truncations are
complete** -- namely on the triangle ``gamma + 2 j <= K`` of item 1, and in particular on the
empty-word row (``gamma = 0``) for every ``j <= K/2``, which is the quantity of interest. Outside
the triangle they legitimately differ, because the graded route has not yet computed the single
order ``k = gamma + 2j`` that carries those coefficients. :func:`assert_routes_agree` checks this.

The grading therefore buys structure (the linearity of item 2, the exact solvability of item 3, the
diagonality of item 1) and a better truncation -- not a different number.
:func:`taylor_coefficients` is the fast path (``O(J)`` batched shuffle calls against the hierarchy's
``O(K J)``); the graded route is what the specification describes and what exposes the per-order
structure.

**Convergence is problem-dependent, and the series can be purely asymptotic.** For
``<p, W_hat_T> = -W_T^4`` the resulting ``T``-series ``-3T^2 + 48T^4 - 1584T^6 + 78336T^8 - ...``
has *zero* radius of convergence (Laplace Example 5.1): it is an asymptotic series with an optimal
truncation order that depends on ``T``. For quadratic functionals such as
``-lam^2/2 int_0^T W_s^2 ds`` (log-transform ``-1/2 log cosh(lam T)``) it converges, with radius
``pi / (2 lam)``. Use :func:`optimal_order` to locate the truncation, and
:func:`perturbation_solve` (restart every ``h``) when the single-shot range is too short --
noting that the composed version is then a high-order Taylor integrator for the *truncated* system
and inherits its blow-up exactly as Euler/PECE do.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp

from .tensor_sequence import TensorSequence
from .factory import from_array
from .shuffle_product import shuffle_prod
from .words import number_of_words_up_to_trunc, index_to_lam_sum_vect

ShuffleTable = Union[np.ndarray, jax.Array]


# --------------------------------------------------------------------------------------------
# 1. The diffusive grading
# --------------------------------------------------------------------------------------------

def state_letters(dim: int, time_letter: Optional[int]) -> List[int]:
    """The (1-based) letters carrying Brownian coordinates, i.e. all of them bar ``time_letter``."""
    return [m for m in range(1, dim + 1) if m != time_letter]


def word_grades(trunc: int, dim: int, time_letter: Optional[int] = None) -> np.ndarray:
    """Diffusive grade ``||v|| = 2 |v|_time + sum_s |v|_s`` for every word index.

    With ``time_letter=None`` (the state-only ``dim=1`` case of the Laplace problem) the grade is
    just the word length, so the filtration ``F_k`` coincides with the ordinary truncation and the
    dimension reduction of :func:`grade_truncation_mask` is invisible. It is at ``dim >= 2``, where
    the time letter costs 2 and the state letters 1, that grade and length genuinely differ.
    """
    lam = jnp.array([2.0 if m == time_letter else 1.0 for m in range(1, dim + 1)])
    n_sig = int(number_of_words_up_to_trunc(trunc, dim))
    return np.asarray(index_to_lam_sum_vect(jnp.arange(n_sig), dim, lam)).round().astype(int)


def graded_part(p: TensorSequence, k: int, grades: np.ndarray) -> TensorSequence:
    """``pi_k(p) = sum_{||v|| = k} p^v v`` -- the grade-``k`` homogeneous part of ``p``."""
    mask = jnp.asarray((grades == k).astype(p.array.dtype))
    return TensorSequence(array=p.array * mask.reshape((-1,) + (1,) * (p.array.ndim - 1)),
                          trunc=p.trunc, dim=p.dim)


def grade_truncation_mask(trunc: int, dim: int, order: int,
                          time_letter: Optional[int] = None) -> Tuple[jax.Array, int]:
    """Projector onto ``F_order = span{v : ||v|| <= order}``, and its dimension.

    This is the truncation the diffusive scaling actually calls for: a word of grade ``||v||``
    enters the expansion at order ``eps^{||v||/2}``, so keeping ``||v|| <= K`` retains *exactly* the
    words that matter to order ``K`` and discards the rest. Compare the two truncations at
    ``dim = 2``: ``dim F_K = Fib(K+3) - 1`` against ``2^{K+1} - 1`` words of length ``<= K``
    (232 vs 2047 at ``K = 10``; 10945 vs 524287 at ``K = 18``), because the words that grade-2
    time letters push out of ``F_K`` are the ones contributing only at higher order.
    """
    grades = word_grades(trunc, dim, time_letter)
    mask = (grades <= order)
    return jnp.asarray(mask.astype(float)), int(mask.sum())


# --------------------------------------------------------------------------------------------
# 2. The Riccati operator, split into its linear and quadratic parts
# --------------------------------------------------------------------------------------------

def generator(ell: TensorSequence, time_letter: Optional[int], states: Sequence[int]) -> TensorSequence:
    """``L ell = ell|_time + 1/2 sum_s ell|_ss`` -- the diffusion generator of the signature.

    Lowers the diffusive grade by exactly 2 (stripping one time letter, or two state letters), which
    is why it is nilpotent on every ``F_k`` and why each ``psi_k`` is polynomial in ``tau``.
    """
    out = (TensorSequence(jnp.zeros_like(ell.array), ell.trunc, ell.dim) if time_letter is None
           else ell.proj(time_letter))
    for s in states:
        out = out + ell.proj(s * 10 + s) * 0.5          # word "ss"
    return out


def _polarized_quadratic(a: TensorSequence, b: TensorSequence, states: Sequence[int],
                         shuffle_table: ShuffleTable) -> TensorSequence:
    """``1/2 sum_s (a|_s) sh (b|_s)`` -- the Riccati nonlinearity, polarized in its two arguments.

    Batched: ``a`` and ``b`` may carry a trailing axis (used to evaluate a whole convolution of
    Taylor coefficients in one shuffle call).
    """
    out = None
    for s in states:
        term = shuffle_prod(a.proj(s), b.proj(s), shuffle_table) * 0.5
        out = term if out is None else out + term
    return out


# --------------------------------------------------------------------------------------------
# 3. The fast path: recursive tau-Taylor coefficients
# --------------------------------------------------------------------------------------------

def taylor_coefficients(init: TensorSequence, n_terms: int, shuffle_table: ShuffleTable,
                        time_letter: Optional[int] = None,
                        mask: Optional[jax.Array] = None) -> TensorSequence:
    """``tau``-Taylor coefficients ``a_0 .. a_{n_terms}`` of the Riccati solution ``g(tau)``.

    Solves ``g' = L g + 1/2 sum_s (g|_s)^{sh 2}``, ``g(0) = init``, in reversed time
    ``tau = T - t`` (the repo's convention: ``init`` is the *terminal* condition), by

        (j+1) a_{j+1} = L a_j + 1/2 sum_s sum_{i=0}^{j} (a_i|_s) sh (a_{j-i}|_s).

    This is the graded hierarchy of :func:`perturbation_hierarchy` summed at ``eps = 1``; the two
    agree to machine precision. The convolution over ``i`` is done in a single batched shuffle
    product per ``(j, s)``, so the cost is ``O(n_terms)`` shuffle calls rather than ``O(n_terms^2)``.

    :param init: terminal condition ``p`` (shape ``(n_sig,)``).
    :param n_terms: highest ``tau`` power to return; the result has ``n_terms + 1`` columns.
    :param shuffle_table: shuffle table of the ambient algebra.
    :param time_letter: 1-based time letter, or ``None`` for the state-only case.
    :param mask: optional projector (e.g. from :func:`grade_truncation_mask`) applied to every
        coefficient, to truncate by diffusive grade instead of by word length.
    :return: a :class:`TensorSequence` of shape ``(n_sig, n_terms + 1)``; column ``j`` is ``a_j``.
    """
    trunc, dim = int(init.trunc), int(init.dim)
    states = state_letters(dim, time_letter)
    apply_mask = (lambda ts: ts) if mask is None else \
        (lambda ts: TensorSequence(ts.array * mask.reshape((-1,) + (1,) * (ts.array.ndim - 1)),
                                   trunc, dim))

    a = [apply_mask(TensorSequence(jnp.asarray(init.array), trunc, dim))]
    for j in range(n_terms):
        # stack the convolution pairs (a_i, a_{j-i}) along a trailing axis and shuffle them at once
        lhs = TensorSequence(jnp.stack([a[i].array for i in range(j + 1)], axis=-1), trunc, dim)
        rhs = TensorSequence(jnp.stack([a[j - i].array for i in range(j + 1)], axis=-1), trunc, dim)
        quad = _polarized_quadratic(lhs, rhs, states, shuffle_table)
        source = TensorSequence(quad.array.sum(axis=-1), trunc, dim)
        a.append(apply_mask((generator(a[j], time_letter, states) + source) * (1.0 / (j + 1))))
    return TensorSequence(array=jnp.stack([x.array for x in a], axis=-1), trunc=trunc, dim=dim)


# --------------------------------------------------------------------------------------------
# 4. The graded hierarchy (the object the specification describes)
# --------------------------------------------------------------------------------------------

def perturbation_hierarchy(p: TensorSequence, order: int, shuffle_table: ShuffleTable,
                           time_letter: Optional[int] = None, check: bool = True
                           ) -> Tuple[List[TensorSequence], dict]:
    """Solve the triangular system order by order, exactly, as polynomials in ``tau = T - t``.

    Returns ``psi_0 .. psi_order``, each a :class:`TensorSequence` of shape
    ``(n_sig, floor(order/2) + 1)`` whose column ``j`` is the ``tau^j`` coefficient. Every stage is
    a *linear* ODE in ``psi_k`` (see the module docstring), integrated in closed form via

        (j+1) g_{k,j+1} = L g_{k,j} + S_{k,j},
        S_{k,j} = 1/2 sum_s sum_{l=1}^{k-1} sum_{i=0}^{j} (g_{l,i}|_s) sh (g_{k-l,j-i}|_s),

    starting from ``g_{k,0} = pi_k(p)``. The ``l = 0`` and ``l = k`` terms of the quadratic sum are
    omitted because ``psi_0|_s = 0``; with ``check=True`` that identity, the filtration
    ``psi_k in F_k`` and the degree bound ``deg psi_k <= floor(k/2)`` are all verified rather than
    assumed.

    :param p: terminal condition.
    :param order: highest order ``K`` in ``eps^{1/2}`` to compute.
    :param shuffle_table: shuffle table of the ambient algebra.
    :param time_letter: 1-based time letter, or ``None`` for the state-only case.
    :param check: verify the three structural claims (cheap; keep it on).
    :return: ``(psi_list, info)``; ``info`` carries ``"grades"``, ``"dim_F"`` (the dimensions of the
        filtration) and ``"max_tau_degree"`` per order.
    """
    trunc, dim = int(p.trunc), int(p.dim)
    states = state_letters(dim, time_letter)
    grades = word_grades(trunc, dim, time_letter)
    n_sig = grades.shape[0]
    J = order // 2                                  # tau-degree bound, see the module docstring

    psi: List[TensorSequence] = []
    shifted: List[List[TensorSequence]] = []        # shifted[l][s_idx] = psi_l|_s, batched over tau
    info = {"grades": grades, "dim_F": [int((grades <= k).sum()) for k in range(order + 1)],
            "max_tau_degree": []}

    for k in range(order + 1):
        cols = [graded_part(p, k, grades).array] + [jnp.zeros(n_sig) for _ in range(J + 1)]
        for j in range(J + 1):
            source = jnp.zeros(n_sig)
            if k >= 2 and j >= 0:
                # batch the (l, i) pairs of S_{k,j} into a single shuffle product per state letter
                pairs = [(l, i) for l in range(1, k) for i in range(j + 1)]
                if pairs:
                    lhs, rhs = [], []
                    for l, i in pairs:
                        lhs.append(shifted[l])                       # psi_l|_s  (all s, all tau)
                        rhs.append(shifted[k - l])
                    acc = None
                    for s_idx, s in enumerate(states):
                        A = jnp.stack([lhs[q][s_idx].array[:, i] for q, (l, i) in enumerate(pairs)], axis=-1)
                        B = jnp.stack([rhs[q][s_idx].array[:, j - i] for q, (l, i) in enumerate(pairs)], axis=-1)
                        term = shuffle_prod(TensorSequence(A, trunc, dim),
                                            TensorSequence(B, trunc, dim), shuffle_table).array
                        acc = term if acc is None else acc + term
                    source = 0.5 * acc.sum(axis=-1)
            lin = generator(TensorSequence(cols[j], trunc, dim), time_letter, states).array
            cols[j + 1] = (lin + source) / (j + 1)

        g_k = TensorSequence(jnp.stack(cols[:J + 2], axis=-1), trunc, dim)
        if check:
            _check_structure(g_k, k, J, grades, states, shuffle_table, trunc, dim)
        g_k = TensorSequence(g_k.array[:, :J + 1], trunc, dim)      # drop the (vanishing) tail
        psi.append(g_k)
        shifted.append([g_k.proj(s) for s in states])
        deg = np.flatnonzero(np.abs(np.asarray(g_k.array)).max(axis=0) > 1e-12)
        info["max_tau_degree"].append(int(deg.max()) if deg.size else -1)

    return psi, info


def _check_structure(g_k: TensorSequence, k: int, J: int, grades: np.ndarray,
                     states: Sequence[int], shuffle_table: ShuffleTable, trunc: int, dim: int):
    """Verify the three claims the triangularity rests on (see :func:`perturbation_hierarchy`)."""
    arr = np.asarray(g_k.array)
    tol = 1e-9 * max(1.0, np.abs(arr).max())
    # (1) psi_k lives in F_k = span{||v|| <= k}
    leak = np.abs(arr[grades > k]).max() if (grades > k).any() else 0.0
    if leak > tol:
        raise AssertionError(f"psi_{k} leaked outside F_{k} (max |coef| = {leak:.3e}); the "
                            f"filtration claim of doc/riccati_perturbation.md fails.")
    # (2) psi_0|_s = 0 -- what makes every stage linear
    if k == 0:
        for s in states:
            if np.abs(np.asarray(g_k.proj(s).array)).max() > tol:
                raise AssertionError("psi_0|_s != 0, so the l=0 / l=k terms of the quadratic sum do "
                                     "not vanish and the system is NOT triangular.")
    # (3) deg_tau psi_k <= floor(k/2): the allocated tail coefficient must vanish
    if np.abs(arr[:, min(k // 2 + 1, J + 1):]).max() > tol:
        raise AssertionError(f"psi_{k} has tau-degree above floor({k}/2) = {k // 2}; the nilpotency "
                             f"argument fails.")


# --------------------------------------------------------------------------------------------
# 5. Evaluation, single-shot and composed
# --------------------------------------------------------------------------------------------

def evaluate_poly(coeffs: TensorSequence, tau: Union[float, np.ndarray]) -> jax.Array:
    """Evaluate a ``tau``-polynomial ``(n_sig, J+1)`` at ``tau`` by Horner. Scalar or vector ``tau``."""
    arr = coeffs.array
    tau = jnp.atleast_1d(jnp.asarray(tau, dtype=float))
    out = jnp.zeros((arr.shape[0], tau.shape[0]))
    for j in range(arr.shape[-1] - 1, -1, -1):
        out = out * tau[None, :] + arr[:, j][:, None]
    return out


def hierarchy_value(psi: Sequence[TensorSequence], T: Union[float, np.ndarray],
                    up_to: Optional[int] = None) -> jax.Array:
    """``sum_{k <= up_to} psi_k(T)`` at ``eps = 1`` -- the perturbative Riccati solution at ``t = 0``.

    Setting ``eps = 1`` is not an approximation of the ansatz: by the scaling relation
    ``psi^{delta_eps p}(t) = delta_eps psi^p(eps t)``, the ``eps`` variable only reorganizes the
    series, and the value at ``eps = 1`` is the solution for the un-dilated data ``p`` at horizon
    ``T``. Truncating at ``up_to = K`` is what introduces the error.

    :return: array ``(n_sig, len(T))`` (or ``(n_sig, 1)`` for scalar ``T``).
    """
    up_to = len(psi) - 1 if up_to is None else up_to
    out = None
    for k in range(min(up_to, len(psi) - 1) + 1):
        v = evaluate_poly(psi[k], T)
        out = v if out is None else out + v
    return out


def partial_sums(psi: Sequence[TensorSequence], T: float) -> np.ndarray:
    """Empty-word coefficient of every partial sum ``sum_{k <= K} psi_k(T)``, ``K = 0 ..``.

    The diagnostic for an asymptotic series: it stops improving and then diverges.
    """
    per_order = np.array([float(evaluate_poly(psi[k], T)[0, 0]) for k in range(len(psi))])
    return np.cumsum(per_order)


def optimal_order(psi: Sequence[TensorSequence], T: float, reference: float) -> Tuple[int, float]:
    """``(K*, |error|)`` at the best truncation order for horizon ``T``, given a reference value.

    For a convergent series this is just the highest available order; for an asymptotic one it is an
    interior optimum, and reporting it is the honest way to quote the method's accuracy.
    """
    errs = np.abs(partial_sums(psi, T) - reference)
    k = int(np.argmin(errs))
    return k, float(errs[k])


def perturbation_solve(p: TensorSequence, T: float, n_steps: int, n_terms: int,
                       shuffle_table: ShuffleTable, time_letter: Optional[int] = None,
                       mask: Optional[jax.Array] = None, return_traj: bool = False):
    """**Composed** variant: restart the expansion every ``h = T / n_steps``.

    Each step propagates ``psi`` over ``h`` by its own ``tau``-Taylor polynomial, re-expanded around
    the new point -- legitimate because the hierarchy solves the Riccati equation for *any* terminal
    data. With ``n_steps = 1`` this is the single-shot expansion of :func:`hierarchy_value`; with
    ``n_steps > 1`` it is a Taylor-series integrator of order ``n_terms`` in ``h``, which is how to
    reach horizons beyond the (possibly zero) radius of convergence of the single-shot series.

    .. warning::
        Being an integrator for the *truncated* Riccati system, this converges as ``h -> 0`` to the
        same truncated solution as Euler/PECE/RK4 and inherits its truncation-induced blow-up at the
        same ``(trunc, T)``. It removes the series-divergence obstruction, not the closure one.

    :return: ``psi_0`` as a :class:`TensorSequence`, or ``(psi_0, trajectory)`` with the trajectory
        of shape ``(n_sig, n_steps + 1)`` in reversed-time order (column 0 = terminal ``p``) when
        ``return_traj`` is set.
    """
    trunc, dim = int(p.trunc), int(p.dim)
    h = T / n_steps
    psi = TensorSequence(jnp.asarray(p.array), trunc, dim)
    traj = [np.asarray(psi.array)]
    for _ in range(n_steps):
        coeffs = taylor_coefficients(psi, n_terms, shuffle_table, time_letter, mask)
        psi = TensorSequence(evaluate_poly(coeffs, h)[:, 0], trunc, dim)
        if return_traj:
            traj.append(np.asarray(psi.array))
    if return_traj:
        return psi, np.stack(traj, axis=-1)
    return psi


def order_of(grade: int, tau_degree: int) -> int:
    """The unique perturbation order carrying a word of grade ``gamma`` at ``tau``-degree ``j``:
    ``k = gamma + 2 j`` (see item 1 of the module docstring)."""
    return grade + 2 * tau_degree


def support_by_order(psi: Sequence[TensorSequence], grades: np.ndarray, tol: float = 1e-9) -> dict:
    """Which word grades actually carry each ``(order, tau-degree)`` cell -- the diagonality check.

    Returns ``{(k, j): sorted list of grades present}``. Every non-empty entry must be exactly
    ``[k - 2j]``; anything else falsifies item 1 of the module docstring.
    """
    out = {}
    for k, g_k in enumerate(psi):
        arr = np.asarray(g_k.array)
        scale = max(1.0, np.abs(arr).max())
        for j in range(arr.shape[-1]):
            present = np.unique(grades[np.abs(arr[:, j]) > tol * scale])
            if present.size:
                out[(k, j)] = [int(x) for x in present]
    return out


def assert_routes_agree(p: TensorSequence, order: int, shuffle_table: ShuffleTable,
                        time_letter: Optional[int] = None, tol: float = 1e-9) -> float:
    """Check the graded hierarchy against :func:`taylor_coefficients` on the triangle where both
    truncations are complete, i.e. ``grade + 2 * tau_degree <= order``.

    Outside that triangle the two *must* differ -- the graded route has not computed the order
    ``k = grade + 2j`` carrying those cells -- so comparing the raw arrays wholesale is meaningless.
    Returns the max absolute discrepancy inside the triangle (expected ~0).
    """
    psi, info = perturbation_hierarchy(p, order, shuffle_table, time_letter, check=True)
    J = order // 2
    a = np.asarray(taylor_coefficients(p, J, shuffle_table, time_letter).array)
    g = np.zeros_like(a)
    for g_k in psi:
        arr = np.asarray(g_k.array)
        g[:, :arr.shape[-1]] += arr
    inside = (info["grades"][:, None] + 2 * np.arange(J + 1)[None, :]) <= order
    diff = float(np.abs(a - g)[inside].max())
    if diff > tol * max(1.0, np.abs(a[inside]).max()):
        raise AssertionError(f"graded hierarchy and tau-Taylor recursion disagree by {diff:.3e} "
                             f"inside the triangle grade + 2j <= {order}.")
    return diff


def taylor_series_in_T(psi: Sequence[TensorSequence], word_index: int = 0) -> np.ndarray:
    """Coefficients of the exact ``T``-Taylor series of ``<psi_0, .>^{word}``, summed over orders.

    The headline analytic output: no ODE integrator produces this. For ``<p, W_hat_T> = -W_T^4`` and
    ``word_index = 0`` it returns ``[0, 0, -3, 0, 48, 0, -1584, 0, 78336, ...]``, i.e.
    ``log E[exp(-W_T^4)] = -3T^2 + 48T^4 - 1584T^6 + 78336T^8 - ...``.
    """
    J = max(x.array.shape[-1] for x in psi)
    out = np.zeros(J)
    for x in psi:
        arr = np.asarray(x.array)[word_index]
        out[:arr.shape[0]] += arr
    return out
