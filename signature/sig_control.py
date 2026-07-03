"""Closed-loop simulation of signature-feedback controls.

The controlled state ``X`` follows ``dX_t = alpha_t dt + sigma dW_t`` and is augmented to the
**time-augmented signature** ``Sig(t, X)_t`` (letter ``1`` = time, letter ``2`` = state). The
optimal control of the associated signature problems is a *Markovian feedback in the signature*,
``alpha_t = pi(t, Sig(X)_t)``; the different solution methods (local Riccati expansion,
Monte-Carlo) only change how ``pi`` is evaluated. So the rollout loop -- compute the control,
step the path, update its signature by Chen's identity -- is written **once** in :func:`simulate`,
and the method is supplied as a *policy*.

A policy is any callable

    policy(i, t, X_sig) -> (control, value, extra)

where ``i`` is the (1-based) step index, ``t`` the current calendar time, ``X_sig`` the current
``Sig(X)_t`` (a :class:`TensorSequence` batched over the ``size`` paths), and the outputs are the
control ``alpha_t`` and value ``psi_t`` (each of shape ``(size,)``) together with a dict of custom
per-step diagnostics (e.g. the Riccati norm tracker). Two factories are provided:
:func:`riccati_policy` and :func:`monte_carlo_policy`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from .tensor_sequence import TensorSequence
from .factory import from_array
from .tensor_product import tensor_prod
from .shuffle_product import shuffle_exp
from .operators import dilation
from .words import number_of_words_up_to_trunc, index_to_word, word_len
from .path_signature import __compute_inc_sig_constant_lam as _increment_signature
from .ode_integration import ode_solver_traj
from .projection import get_projection_matrix

# (control, value, extra-diagnostics), each batched over the ``size`` paths.
Policy = Callable[[int, float, TensorSequence], Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]]


@dataclass
class ControlResult:
    """Output of :func:`simulate`. All arrays are indexed ``[path, time]`` (state/control/value)
    or ``[word, path, time]`` (signature trajectory)."""
    X: np.ndarray                  # (size, n_dw)         controlled state path
    sig: np.ndarray                # (n_sig, size, n_dw)  signature trajectory Sig(X)_t
    control: np.ndarray            # (size, n_dw)         applied control alpha_t
    value: np.ndarray              # (size, n_dw)         value process psi_t
    extra: Dict[str, np.ndarray]   # custom per-policy diagnostics, each (size, n_dw)


def simulate(dW: np.ndarray, t_grid: np.ndarray, policy: Policy, trunc: int, dim: int = 2,
             sigma: float = 1.0, x0: float = 0.0) -> ControlResult:
    """Roll the signature-feedback closed loop ``dX = policy*dt + sigma dW`` forward in time.

    The loop is policy-agnostic: at every step it asks ``policy`` for the control/value at the
    current signature, advances the state, and updates the signature by Chen's identity. This is
    the single shared routine behind the Riccati and Monte-Carlo experiments.

    :param dW: Brownian increments of the state coordinate, shape ``(size, n_dw - 1)``.
    :param t_grid: time grid, shape ``(n_dw,)``.
    :param policy: feedback rule, called as ``policy(i, t_grid[i-1], Sig(X)_{t_{i-1}})``.
    :param trunc: signature truncation order.
    :param dim: path dimension (``2``: time letter ``1`` + state letter ``2``).
    :param sigma: diffusion coefficient of the state.
    :param x0: initial state.
    :return: a :class:`ControlResult` with the state, signature trajectory, control, value and extras.
    """
    size = dW.shape[0]
    n_dw = t_grid.shape[0]
    dt = np.diff(t_grid)
    n_sig = int(number_of_words_up_to_trunc(trunc, dim))

    X = np.zeros((size, n_dw)); X[:, 0] = x0
    control = np.zeros((size, n_dw))
    value = np.zeros((size, n_dw))
    sig = np.zeros((n_sig, size, n_dw))
    extra: Dict[str, np.ndarray] = {}

    X_sig = from_array(jnp.zeros((n_sig, size)).at[0].set(1.0), trunc=trunc, dim=dim)
    sig[:, :, 0] = np.asarray(X_sig.array)

    for i in range(1, n_dw):
        alpha, psi_t, diag = policy(i, float(t_grid[i - 1]), X_sig)
        control[:, i - 1] = alpha
        value[:, i - 1] = psi_t
        for key, val in diag.items():
            extra.setdefault(key, np.zeros((size, n_dw)))[:, i - 1] = val

        dX_state = alpha * dt[i - 1] + sigma * dW[:, i - 1]
        X[:, i] = X[:, i - 1] + dX_state

        dt_i = dt[i - 1] * jnp.ones(size)
        dX_path = jnp.stack([dt_i, jnp.asarray(dX_state)], axis=1)             # (size, dim): [d t, d X]
        X_sig = tensor_prod(X_sig, _increment_signature(dX=dX_path, dt=dt_i, lam=jnp.zeros(dim),
                                                        dim=dim, trunc=trunc))  # Chen's identity
        sig[:, :, i] = np.asarray(X_sig.array)

    # No step is taken out of t_{n-1}; carry the last control/value/diagnostics for plotting.
    control[:, -1] = control[:, -2]
    value[:, -1] = value[:, -2]
    for key in extra:
        extra[key][:, -1] = extra[key][:, -2]
    return ControlResult(X=X, sig=sig, control=control, value=value, extra=extra)


def simulate_with_recentering(dW: np.ndarray, t_grid: np.ndarray, riccati_data: dict, trunc: int,
                              dim: int = 2, sigma: float = 1.0, x0: float = 0.0,
                              recenter_threshold: float = 1.0) -> ControlResult:
    """Roll the signature-feedback closed loop ``dX = alpha*dt + sigma dW`` with **per-path
    recentering** of the local Riccati expansion (Control paper Thm 4.3; doc ``riccati_signature.md``
    §2.5/5).

    The local expansion ``<psi_t, Sig(X)_t>`` of the value/control has a finite radius of
    convergence and is valid only up to ``tau = inf{t : ||bar psi_t||_{Sig(X)_t} >= log 2}`` (the
    norm tracker reaching ``1``). At ``tau`` we *recenter*: by Chen's identity the problem data
    shifted by the full signature ``Sig(X)_{0,tau}`` defines a new Riccati problem whose solution is
    paired against the *reset* increment signature ``Sig(X)_{tau,t}`` for ``t >= tau``. Each path
    crosses ``tau`` at its own time, so every path carries its own data and Riccati trajectory; the
    whole batch is run at once, re-solving only the columns that recenter.

    **Running cost (source).** ``riccati_data`` may carry a *time-independent* running cost
    ``source = q`` (a tensor sequence of shape ``(n_sig,)``) alongside the terminal condition ``r``.
    The Riccati is then ``psi' = R(psi) + q`` (so ``ode_fun`` must be the **bare** operator ``R``),
    and recentering shifts **both** ``r`` and ``q`` by ``X_full`` -- Chen's identity applies to the
    running-cost integrand exactly as to the terminal. The reported value is the value process
    ``<psi_t, Sig(X)_{tau,t}> + int_0^t <q, Sig(X)_{0,s}> ds``: the in-ODE source already carries the
    future ``int_t^T`` part, and the incurred ``int_0^t`` is added against the *original* ``q`` and
    the full signature. With ``source=None`` the value is the value-to-go ``<psi_t, Sig(X)_{tau,t}>``.

    **Markovian 1-D mode (``dim=1``).** When the cost is a function of the current state ``X_t`` plus
    running integrals of such functions, the problem is Markovian and the time letter is unnecessary:
    the value ``v(t,x) = <psi_t, (x^k/k!)_k>`` lives in the **1-D** tensor algebra (single letter
    ``1`` = state, so ``state_letter=1``), with ``R psi = 1/2 psi|_11 + 1/2 (psi|_1)^{sh 2}`` and the
    running cost as the source ``q``. This has only ``trunc + 1`` coefficients instead of
    ``2^{trunc+1} - 1``, so far higher truncations are affordable. ``dim=2`` keeps the time-augmented
    signature (letters ``1`` = time, ``2`` = state); use it for genuinely path-dependent costs (Lévy
    area, signature matching, a time-dependent ``q``) that the 1-D Markovian reduction does not cover.

    Two group-like signatures are tracked per path, both starting at the unit ``Ø``:

    * ``X_full = Sig(X)_{0,t}`` -- only used to build the recentered data ``{}_{X_full}|r = P_r @
      X_full`` (and ``{}_{X_full}|q = P_q @ X_full``); ``P_r, P_q`` are the fixed projection matrices
      of the deterministic ``r, q``, so the whole batch is shifted by a single matrix product.
    * ``X_loc = Sig(X)_{tau(b),t}`` -- the signature since path ``b``'s last recentering; the state
      against which control ``alpha_t = <psi_t|_state, X_loc>``, value and the norm tracker are
      evaluated. Reset to ``Ø`` for the crossing paths at each recentering.

    :param dW: Brownian increments of the state coordinate, shape ``(size, n_dw - 1)``.
    :param t_grid: simulation grid, shape ``(n_dw,)``; must match ``riccati_data["t_grid"]`` in
        length so the reversed-time column ``-i`` of ``psi`` lines up with calendar time ``t_{i-1}``.
    :param riccati_data: dict with keys ``ode_fun`` (the **bare** operator ``R``), ``step_fun``,
        ``t_grid``, ``r`` (the **bare** terminal condition -- do *not* also fold ``q`` into it),
        ``shuffle_table``, ``state_letter`` and ``source`` (time-independent ``q`` of shape
        ``(n_sig,)``, or ``None``).
    :param trunc: signature truncation order.
    :param dim: ``1`` (state-only, Markovian) or ``2`` (time letter ``1`` + state letter ``2``).
    :param sigma: diffusion coefficient of the state.
    :param x0: initial state. Keep ``0``: both signatures encode increments, so cost functionals
        such as ``(e_state - mu)`` mean ``(X_t - mu)`` only when ``X_0 = 0``.
    :param recenter_threshold: recenter a path once its norm tracker reaches this value. ``1.0`` (the
        ``log 2`` radius bound of Remark 5.1) is the latest safe value; recentering *earlier* is
        always exact (Thm 3.3 holds for any ``X``), trading extra re-solves for a wider margin.
    :return: a :class:`ControlResult`; ``extra["norm_tracker"]`` holds the per-path radius tracker.
    """
    if dim not in (1, 2):
        raise ValueError(f"dim must be 1 (state-only, Markovian) or 2 (time-augmented); got {dim}.")
    ode_fun, step_fun = riccati_data["ode_fun"], riccati_data["step_fun"]
    t_grid_ric = riccati_data["t_grid"]
    r = riccati_data["r"]                          # bare terminal condition, array (n_sig,)
    q = riccati_data.get("source")                 # time-independent running cost, or None
    state_letter = riccati_data["state_letter"]
    shuffle_table = riccati_data["shuffle_table"]
    if q is not None and q.array.ndim not in (1, 2):
        raise ValueError("source (running cost q) must be a tensor sequence of shape (n_sig,) "
                         "[time-independent] or (n_sig, n_dw) [time-dependent, reversed-time order].")
    q_is_traj = q is not None and q.array.ndim == 2   # time-dependent running cost?

    size = dW.shape[0]
    n_dw = t_grid.shape[0]
    if t_grid_ric.shape[0] != n_dw:
        raise ValueError(f"riccati_data['t_grid'] has {t_grid_ric.shape[0]} points but the "
                         f"simulation grid has {n_dw}; they must match for the '-i' time alignment.")
    dt = np.diff(t_grid)
    n_sig = int(number_of_words_up_to_trunc(trunc, dim))

    if q_is_traj and q.array.shape[1] != n_dw:
        raise ValueError(f"time-dependent source has {q.array.shape[1]} columns, expected n_dw={n_dw}.")

    # The recentered terminal is {}_X|r = P_r @ X. The recentered source is {}_X|q: for a constant q
    # it is P_q @ X; for a time-dependent q (kept at its own -- typically low -- truncation ``n_src``)
    # we use {}_X|q = sum_w X^w q.left_proj(w), vectorised over q's time axis by precomputing the
    # per-prefix shifted trajectories LP[w] = q.left_proj(w) once. Words and indices are
    # truncation-independent, so X is simply sliced to its first n_src coefficients.
    P_r = get_projection_matrix(r)                 # (n_sig, n_sig)
    n_src = q.array.shape[0] if q is not None else 0
    P_q = get_projection_matrix(q) if (q is not None and not q_is_traj) else None
    LP = (jnp.stack([q.left_proj(int(index_to_word(w, dim))).array for w in range(n_src)])
          if q_is_traj else None)                  # (n_src, n_src, n_dw)

    def shifted_source(Xb: jax.Array) -> Optional[jax.Array]:
        """Recentered source {}_X|q for the full-signature columns ``Xb`` (n_sig, m): ``None`` if no
        source, ``(n_sig, m)`` for a constant q, ``(n_src, m, n_dw)`` for a time-dependent q."""
        if q is None:        return None
        if not q_is_traj:    return P_q @ Xb
        return jnp.einsum("wvs,wb->vbs", LP, Xb[:n_src])

    # Riccati RHS: bare operator R plus the (per-path) source, added to its first coefficients. The
    # source ``src`` is None, a constant (n_*, m), or a trajectory (n_*, m, n_dw) sliced by step i.
    def rhs(psi: TensorSequence, args: dict) -> TensorSequence:
        out = ode_fun(psi, args); src = args["src"]
        if src is None: return out
        add = src if src.ndim == 2 else src[..., args["i"]]    # (n_*, m)
        return TensorSequence(out.array.at[:add.shape[0]].add(add), out.trunc, out.dim)

    def solve_batch(r_arr: jax.Array, src: Optional[jax.Array]) -> jax.Array:
        """Solve ``psi' = R(psi) + q`` for a batch of terminal columns ``r_arr`` (n_sig, m) with the
        (recentered) source ``src``; returns ``(n_sig, m, n_dw)``. Called on the crossing subset only,
        so ``m`` is usually a handful (JAX retraces once per distinct ``m``)."""
        return ode_solver_traj(fun=rhs, step_fun=step_fun, t_grid=t_grid_ric,
                               init=TensorSequence(array=r_arr, trunc=trunc, dim=dim),
                               args={"shuffle_table": shuffle_table, "src": src}).array

    X = np.zeros((size, n_dw)); X[:, 0] = x0
    control = np.zeros((size, n_dw))
    value = np.zeros((size, n_dw))
    norm_track = np.zeros((size, n_dw))
    sig = np.zeros((n_sig, size, n_dw))

    unit_arr = jnp.zeros((n_sig, size)).at[0].set(1.0)
    X_full = from_array(unit_arr, trunc=trunc, dim=dim)   # Sig(X)_{0,t}
    X_loc = from_array(unit_arr, trunc=trunc, dim=dim)    # Sig(X)_{tau(b),t}
    sig[:, :, 0] = np.asarray(X_full.array)

    # Step 0: every path is centred at Ø with the same data (r, q), so one solve serves all --
    # solve once and broadcast. Thereafter each path carries its own Riccati trajectory in ``psi``
    # (shape (n_sig, size, n_dw)), recomputed only for the columns that recenter.
    src0 = None if q is None else (q.array[:, None, :] if q_is_traj else q.array[:, None])
    psi = jnp.array(jnp.broadcast_to(solve_batch(r.array[:, None], src0), (n_sig, size, n_dw)))
    psi_ctrl = TensorSequence(psi, trunc, dim).proj(state_letter).array     # psi|_state

    incurred = np.zeros(size)               # int_0^t <q, Sig(X)_{0,s}> ds (trapezoidal, original q)
    prev_rate = prev_t = None

    log2 = np.log(2.0)
    for i in range(1, n_dw):
        t = float(t_grid[i - 1])
        col = psi[:, :, -i]                          # psi_t,        (n_sig, size)
        col_ctrl = psi_ctrl[:, :, -i]                # psi_t|_state, (n_sig, size)

        # Radius-of-convergence tracker ||bar psi_t||_{X_loc} / log 2 (the empty-word term, which
        # equals |psi_t^Ø|, is removed); the expansion is valid while this stays < 1.
        mask = jnp.ones_like(col).at[0].set(0.0)
        psi_bar = TensorSequence(col * mask, trunc, dim)
        # exact radius condition (Remark 5.1): ||exp^sh(bar psi_t)||_{X_loc} >= 2, tracked as its
        # half so the threshold is 1. np.array (not np.asarray) => a writable copy, so the
        # norm[crossed] = 0 update below does not hit JAX's read-only buffer.
        norm = np.array(shuffle_exp(psi_bar, shuffle_table).seminorm(X_loc) / 2)

        crossed = (norm >= recenter_threshold)
        if crossed.any():
            b = np.flatnonzero(crossed)
            # Recenter only the crossing paths: shift r (and q) by their full signature
            # ({}_X|r = P_r @ X), re-solve just those columns (the others keep their trajectory),
            # and reset their local signature to Ø.
            psi = psi.at[:, b, :].set(solve_batch(P_r @ X_full.array[:, b], shifted_source(X_full.array[:, b])))
            psi_ctrl = TensorSequence(psi, trunc, dim).proj(state_letter).array
            X_loc = TensorSequence(X_loc.array.at[:, b].set(0.0).at[0, b].set(1.0), trunc, dim)
            col, col_ctrl = psi[:, :, -i], psi_ctrl[:, :, -i]   # use the fresh expansion this step
            norm[crossed] = 0.0                                 # bar psi vanishes against Ø

        control[:, i - 1] = np.asarray(jnp.einsum("vs,vs->s", col_ctrl, X_loc.array))
        value_t = np.asarray(jnp.einsum("vs,vs->s", col, X_loc.array))         # value-to-go
        if q is not None:                                                     # + int_0^t <q_s, Sig_{0,s}> ds
            q_slice = q.array[:, -i] if q_is_traj else q.array                 # original (unshifted) q_t
            rate = np.asarray(jnp.einsum("v,vs->s", q_slice, X_full.array[:n_src]))   # <q_t, Sig(X)_{0,t}>
            if prev_rate is not None:
                incurred = incurred + 0.5 * (prev_rate + rate) * (t - prev_t)
            prev_rate, prev_t = rate, t
            value_t = value_t + incurred
        value[:, i - 1] = value_t
        norm_track[:, i - 1] = norm

        dX_state = control[:, i - 1] * dt[i - 1] + sigma * dW[:, i - 1]
        X[:, i] = X[:, i - 1] + dX_state

        dt_i = dt[i - 1] * jnp.ones(size)
        if dim == 1:
            dX_path = jnp.asarray(dX_state)[:, None]                           # (size, 1): state only
        else:
            dX_path = jnp.stack([dt_i, jnp.asarray(dX_state)], axis=1)         # (size, 2): [d t, d X]
        dX_sig = _increment_signature(dX=dX_path, dt=dt_i, lam=jnp.zeros(dim), dim=dim, trunc=trunc)
        X_full = tensor_prod(X_full, dX_sig)   # Chen's identity, full signature
        X_loc = tensor_prod(X_loc, dX_sig)     # Chen's identity, recentered signature
        sig[:, :, i] = np.asarray(X_full.array)

    # No step is taken out of t_{n-1}; carry the last control/value/diagnostics for plotting.
    control[:, -1] = control[:, -2]
    value[:, -1] = value[:, -2]
    norm_track[:, -1] = norm_track[:, -2]
    return ControlResult(X=X, sig=sig, control=control, value=value,
                         extra={"norm_tracker": norm_track})


def _running_cost_integral(source: TensorSequence):
    """Stateful trapezoidal accumulator of the running-cost integral
    ``int_0^t <q_s, Sig(X)_s> ds`` along a forward-in-time rollout.

    ``source.array`` has shape ``(n_sig, n_dw)`` in the solver's reversed-time order, so at step
    ``i`` (calendar time ``t``) the active source slice is ``source.array[:, -i]`` -- the same
    ``-i`` indexing as the Riccati trajectory. Returns ``step(i, t, X_sig)`` that updates and
    returns the integral; call it once per step, in increasing time. ``source`` may be kept at a
    lower truncation than ``X_sig`` (a degree-``d`` running cost only needs the first words): the
    pairing then uses ``X_sig``'s first ``len(source)`` coefficients.
    """
    n_src = source.array.shape[0]
    state = {"integral": 0.0, "rate": None, "t": None}

    def step(i: int, t: float, X_sig: TensorSequence):
        rate = np.asarray(source.array[:, -i] @ X_sig.array[:n_src])  # <q_t, Sig(X)_t>, batched over paths
        if state["rate"] is not None:
            state["integral"] = state["integral"] + 0.5 * (state["rate"] + rate) * (t - state["t"])
        state["rate"], state["t"] = rate, t
        return state["integral"]

    return step


def riccati_policy(psi: TensorSequence, source: Optional[TensorSequence] = None, state_letter: int = 2) -> Policy:
    """Feedback from a precomputed Riccati trajectory ``psi`` (solved forward in reversed time
    ``tau = T - t``, so column ``-i`` is calendar time ``t_{i-1}``).

    Control ``alpha_t = <psi_t|_state, Sig(X)_t>`` and a norm tracker
    ``(||psi_t||_{Sig(X)_t} - |psi_t^empty|) / log 2`` monitoring the radius of convergence of the
    local expansion (it must stay ``< 1``).

    The reported value is the **value process**. ``<psi_t, Sig(X)_t>`` alone is the value-to-go;
    with a running-cost ``source`` ``q`` (the same ``(n_sig, n_dw)`` trajectory fed to the Riccati
    ODE) the already-incurred cost is added,
    ``value = <psi_t, Sig(X)_t> + int_0^t <q_s, Sig(X)_s> ds``. Without a source (``q = 0``) it
    reduces to the value-to-go. The accumulator is stateful, so use one policy per rollout.
    """
    psi_ctrl = psi.proj(state_letter)
    running = _running_cost_integral(source) if source is not None else None

    def policy(i: int, t: float, X_sig: TensorSequence):
        col = psi.array[:, -i]
        alpha = np.asarray(psi_ctrl.array[:, -i] @ X_sig.array)
        value = np.asarray(col @ X_sig.array)                       # value-to-go <psi_t, Sig(X)_t>
        if running is not None:
            value = value + running(i, t, X_sig)                    # + int_0^t <q_s, Sig(X)_s> ds
        norm_tracker = (np.asarray(psi.subsequence((-i,)).seminorm(X_sig))
                        - np.abs(np.asarray(col[0]))) / np.log(2)
        return alpha, value, {"norm_tracker": norm_tracker}

    return policy


def _prepend_letter(letter: int, word: int) -> int:
    """Concatenate a single ``letter`` in front of ``word`` (decimal-digit word encoding)."""
    return letter * 10 ** int(word_len(word)) + word


def monte_carlo_policy(p: TensorSequence, sig_brownian: TensorSequence, T: float,
                       trunc: int, dim: int = 2, state_letter: int = 2,
                       scaling: Callable[[float], jax.Array] = lambda h: jnp.array([h, jnp.sqrt(h)])
                       ) -> Policy:
    """Monte-Carlo feedback from a precomputed standardized Brownian-signature sample.

    Uses the recentering identities (see ``doc/riccati_signature.md``)

        value(t) = log E[ exp<p, Sig(X)_t (x) sigW_{t,T}> ],
        alpha_t  = E[ exp<p, Sig(X)_t (x) sigW_{t,T}> <p, Sig(X)_t (x) state (x) sigW_{t,T}> ] / exp(value),

    where ``sigW_{t,T} =d dilation(sigW_1, scaling(T - t))`` by Brownian scaling. ``p`` is shifted
    by the current signature (``q = {}_{Sig(X)_t}|p``, then by ``state``); since ``p`` has finite
    degree, only the first ``n_sig_mc`` rows of these shift maps are needed and they are built once
    by direct projections -- no dense projection matrix.

    The reported value is the **value process** directly, with no running-cost term added: because
    ``p = tensor_prod(q, e_time) + r`` encodes the running cost as a terminal functional through the
    time letter, ``<p, Sig(X)_T> = <r, Sig(X)_T> + int_0^T <q, Sig(X)_s> ds`` already carries the
    full ``int_0^T`` (so this equals ``<psi_t, Sig(X)_t> + int_0^t <q, Sig(X)_s> ds``).

    .. warning::
        The time-letter identity ``<tensor_prod(q, e_time), Sig_T> = int_0^T <q, Sig_s> ds`` holds
        only for a **time-independent** ``q``. For a time-dependent running cost ``q_s`` the correct
        exponent is ``<r, Sig(X)_T> + int_t^T <q_s, Sig(X)_s> ds``, which a single ``p`` cannot
        represent; that case needs the intermediate future signatures ``sigW_{t,s}`` (see the module
        notes / use :func:`riccati_policy`, which handles a time-dependent source exactly).

    :param p: terminal-functional coefficient sequence (``p = tensor_prod(q, e_time) + r``,
        valid for time-independent ``q``).
    :param sig_brownian: a sample of ``sigW_1`` on ``[0, 1]``, shape ``(n_sig_mc, N_MC)``,
        truncated at ``deg(p)`` (enough to reproduce every term of the expectation exactly).
    :param T: horizon (the remaining horizon at time ``t`` is ``T - t``).
    :param scaling: ``h -> (c_time, c_state)`` dilation factors; defaults to Brownian scaling
        ``(h, sqrt(h))`` for the time-augmented motion.
    """
    p_deg = int(p.get_lengths_array()[np.abs(np.asarray(p.array)) > 1e-12].max())
    n_sig_mc = int(number_of_words_up_to_trunc(dim=dim, trunc=p_deg))

    # Shift maps as (n_sig_mc, n_sig) matrices acting on the full signature Sig(X)_t:
    #   q = {}_X | p          ->  q^v   = (p|_v)         . X    (right shift of p by v)
    #   r = q |^state         ->  r^v   = (p|_{state v}) . X    (right shift of p by state.v)
    words = [int(index_to_word(i, dim)) for i in range(n_sig_mc)]
    P_q = jnp.stack([p.proj(v).array for v in words])
    P_r = jnp.stack([p.proj(_prepend_letter(state_letter, v)).array for v in words])
    sigW1 = sig_brownian.array
    p_deg_trunc = sig_brownian.trunc

    @jax.jit
    def _eval(X_arr: jax.Array, sigW: jax.Array):
        pair_q = jnp.einsum("vs,vm->sm", P_q @ X_arr, sigW)        # <q, sigW>   (size, N_MC)
        pair_r = jnp.einsum("vs,vm->sm", P_r @ X_arr, sigW)        # <r, sigW>
        m = jnp.max(pair_q, axis=1, keepdims=True)                # log-sum-exp stabilisation
        w = jnp.exp(pair_q - m)
        psi = m[:, 0] + jnp.log(jnp.mean(w, axis=1))              # psi_t = log E[exp<p,.>]
        alpha = jnp.sum(w * pair_r, axis=1) / jnp.sum(w, axis=1)  # E[exp<p,.> <p,.state.>] / exp(psi_t)
        return psi, alpha

    def policy(i: int, t: float, X_sig: TensorSequence):
        sigW = dilation(TensorSequence(array=sigW1, trunc=p_deg_trunc, dim=dim), scaling(T - t)).array
        value, alpha = _eval(X_sig.array, sigW)                     # value process directly (constant q baked in p)
        return np.asarray(alpha), np.asarray(value), {}

    return policy


def _max_degree(ts: TensorSequence, tol: float = 1e-12) -> int:
    """Highest word length carried by a (possibly time-batched) tensor sequence."""
    nz = np.abs(np.asarray(ts.array))
    while nz.ndim > 1:
        nz = nz.max(axis=-1)
    lengths = np.asarray(ts.get_lengths_array())
    mask = nz > tol
    return int(lengths[mask].max()) if mask.any() else 0


def brownian_signature_trajectory(t_grid: np.ndarray, N_MC: int, trunc_mc: int, dim: int = 2,
                                  sigma: float = 1.0, seed: int = 0) -> jax.Array:
    """Precompute the time-augmented Brownian signature **trajectory** on ``t_grid``.

    Returns an array of shape ``(n_sig_mc, n_dw, N_MC)`` whose ``[:, k, :]`` slice is the signature
    ``Sig(W)_{0, t_grid[k]}`` of the time-augmented Brownian motion ``(s, sigma W_s)`` (truncated at
    ``trunc_mc``), with ``Sig(W)_{0,0} = unit``. Because Brownian increments are stationary,
    ``Sig(W)_{t,s} =d Sig(W)_{0, s-t}`` -- i.e. the future-signature trajectory from any ``t`` is a
    *slice* of this one array -- which is what :func:`monte_carlo_source_policy` consumes (no
    dilation needed). The build uses a single batched ``iisignature`` stream call.
    """
    import iisignature
    t_grid = np.asarray(t_grid)
    n_dw = t_grid.shape[0]
    dt = np.diff(t_grid)
    n_sig_mc = int(number_of_words_up_to_trunc(trunc_mc, dim))
    rng = np.random.default_rng(seed)
    W = np.zeros((N_MC, n_dw))
    W[:, 1:] = np.cumsum(rng.normal(size=(N_MC, n_dw - 1)) * (sigma * np.sqrt(dt)), axis=1)
    paths = np.stack([np.broadcast_to(t_grid, (N_MC, n_dw)), W], axis=2)        # (N_MC, n_dw, 2)
    stream = iisignature.sig(paths, trunc_mc, 2)                                # (N_MC, n_dw-1, n_sig_mc-1)
    traj = np.zeros((n_sig_mc, n_dw, N_MC))
    traj[0] = 1.0                                                              # empty word
    traj[1:, 1:, :] = np.transpose(stream, (2, 1, 0))                          # [word, time, sample]
    return jnp.asarray(traj)


def monte_carlo_source_policy(r: TensorSequence, source: TensorSequence,
                              sig_brownian_traj: jax.Array, t_grid: np.ndarray,
                              dim: int = 2, state_letter: int = 2) -> Policy:
    """Monte-Carlo feedback for a **time-dependent** running cost ``q`` (a path-dependent source).

    Unlike :func:`monte_carlo_policy` -- which bakes a *constant* ``q`` into ``p = tensor_prod(q,
    e_time) + r`` -- this evaluates the value-to-go with the genuine time-dependent running cost,

        value-to-go(t) = log E[ exp( <r, Sig(X)_t (x) sigW_{t,T}>
                                     + int_t^T <q_s, Sig(X)_t (x) sigW_{t,s}> ds ) ],

    and the control numerator inserts ``state`` after ``Sig(X)_t`` in each pairing. By stationary
    increments ``sigW_{t,s} =d Sig(W)_{0, s-t}``, so the whole future-signature trajectory is a slice
    of ``sig_brownian_traj`` (from :func:`brownian_signature_trajectory`) -- no dilation. The shift
    trick is applied per integration time ``s`` (``q_s`` is shifted by ``Sig(X)_t``). The integral is
    a trapezoid on ``t_grid``; ``t_grid`` must be **uniform** so that ``s - t`` lands on the grid.

    The reported value is the value process: ``int_0^t <q_s, Sig(X)_s> ds`` is added to the
    value-to-go (matching :func:`riccati_policy`).

    :param r: terminal coefficient sequence.
    :param source: running-cost trajectory ``q`` of shape ``(n_sig, n_dw)`` (reversed-time order,
        the same object fed to the Riccati ODE: column ``c`` is ``q`` at calendar time ``T - t_grid[c]``).
    :param sig_brownian_traj: precomputed ``(n_sig_mc, n_dw, N_MC)`` Brownian signature trajectory,
        truncated at ``>= max(deg r, deg q)``.
    :param t_grid: the (uniform) time grid of shape ``(n_dw,)``.
    """
    V = int(number_of_words_up_to_trunc(max(_max_degree(r), _max_degree(source)), dim))
    if sig_brownian_traj.shape[0] < V:
        raise ValueError(f"sig_brownian_traj truncated too low: need >= {V} words, "
                         f"got {sig_brownian_traj.shape[0]} (raise trunc_mc to max(deg r, deg q)).")
    n_dw = sig_brownian_traj.shape[1]
    dt = float(t_grid[1] - t_grid[0])
    Fut = jnp.asarray(sig_brownian_traj[:V])                                   # (V, n_dw, N_MC)

    # Shift maps restricted to the first V words: row v acts on Sig(X)_t to give
    #   ({}_X|c)^v = (c|_v) . X    (terminal)   and    (c|_{state v}) . X   (control numerator).
    words = [int(index_to_word(v, dim)) for v in range(V)]
    P_r_term = jnp.stack([r.proj(v).array[:V] for v in words])                 # (V, V)
    P_r_ctrl = jnp.stack([r.proj(_prepend_letter(state_letter, v)).array[:V] for v in words])
    P_q_term = jnp.stack([source.proj(v).array[:V] for v in words])           # (V, V, n_dw)
    P_q_ctrl = jnp.stack([source.proj(_prepend_letter(state_letter, v)).array[:V] for v in words])
    running = _running_cost_integral(source)

    @jax.jit
    def _eval(Xv, weights, cols, term_idx):
        # shift each q-column by X, then realign column -> relative-time index k and weight it
        QXt = jnp.take(jnp.einsum("vwc,ws->vsc", P_q_term, Xv), cols, axis=2) * weights[None, None, :]
        QXc = jnp.take(jnp.einsum("vwc,ws->vsc", P_q_ctrl, Xv), cols, axis=2) * weights[None, None, :]
        fut_T = Fut[:, term_idx, :]                                            # sigW_{t,T} =d Sig(W)_{0,T-t}
        E = jnp.einsum("vs,vm->sm", P_r_term @ Xv, fut_T) + jnp.einsum("vsk,vkm->sm", QXt, Fut)
        D = jnp.einsum("vs,vm->sm", P_r_ctrl @ Xv, fut_T) + jnp.einsum("vsk,vkm->sm", QXc, Fut)
        m = jnp.max(E, axis=1, keepdims=True)
        w = jnp.exp(E - m)
        value_to_go = m[:, 0] + jnp.log(jnp.mean(w, axis=1))
        alpha = jnp.sum(w * D, axis=1) / jnp.sum(w, axis=1)
        return value_to_go, alpha

    def policy(i: int, t: float, X_sig: TensorSequence):
        K = n_dw - i + 1                                                       # relative-time points in [t, T]
        # relative time k <-> source column (n_dw-i-k); trapezoid weights over [0, T-t], 0 beyond
        cols = jnp.asarray(np.clip(n_dw - i - np.arange(n_dw), 0, n_dw - 1))
        w = np.zeros(n_dw)
        if K >= 2:
            w[:K] = dt; w[0] = dt / 2; w[K - 1] = dt / 2
        value_to_go, alpha = _eval(jnp.asarray(X_sig.array[:V]), jnp.asarray(w), cols, n_dw - i)
        value = np.asarray(value_to_go) + running(i, t, X_sig)                 # value process
        return np.asarray(alpha), value, {}

    return policy
