# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Expected-signature matching as a mean-field control problem
#
# We steer $dX_t=\alpha_t\,dt+dW_t$, $X_0=0$, so that the expected signature of $\widehat X_t=(t,X_t)$
# matches a target $\mathcal E$:
# $$
# V=\inf_\alpha\ \mathbb E\Big[\tfrac12\int_0^T\alpha_t^2dt\Big]+\frac\kappa2\sum_{v\in\mathcal W_N}\lambda_v\big(\mathbb E[\widehat{\mathbb X}^v_T]-\mathcal E^v\big)^2 .
# $$
# Fenchel duality and Boué–Dupuis turn it into the finite-dimensional concave problem
# $$
# \sup_{\boldsymbol p\in\mathcal B\cap T^N} J(\boldsymbol p),\qquad
# J(\boldsymbol p)=-\tfrac\kappa2\,\boldsymbol p^\top\Lambda^{-1}\boldsymbol p-\kappa\langle\boldsymbol p,\mathcal E\rangle-\log\mathbb E\big[e^{-\kappa\langle\boldsymbol p,\widehat{\mathbb W}_T\rangle}\big],
# $$
# whose maximiser $\boldsymbol p^*$ is the weighted residual $\Lambda(m^*-\mathcal E)$. The optimal control is
# the signature feedback of the classical problem with terminal tensor $-\kappa\boldsymbol p^*$, and the
# optimal path law is the exponential tilt $dQ^*/d\mathbb W\propto e^{-\kappa\langle\boldsymbol p^*,\widehat{\mathbb W}_T\rangle}$.
#
# Theory: [doc/expected_signature_matching.md](../doc/expected_signature_matching.md);
# spec: [spec/esig_mfc_spec.md](../spec/esig_mfc_spec.md); code: `signature/esig_mfc`.
#
# Letters: code letter `1` = time, `2` = state (paper letters $0$, $1$).

# %%
import os
import sys
import time

for _cand in ("..", "."):
    if os.path.isdir(os.path.join(_cand, "signature")) and os.path.abspath(_cand) not in sys.path:
        sys.path.insert(0, os.path.abspath(_cand))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import signature.tensor_algebra as ta
import signature.esig_mfc as em
from signature.expected_signature import expected_bm_signature

PALETTE = ["#5C7B55", "#C0714A", "#C9A84C", "#A8C5A0", "#8B5A6B", "#4A5568"]
plt.rcParams["figure.figsize"] = ((1 + np.sqrt(5)) / 2 * 5, 5)
plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=PALETTE)
LAW_COLORS = {"target": PALETTE[0], "rollout": PALETTE[1], "resampled": PALETTE[4], "BM": PALETTE[2]}
CLASS_COLORS = {"free": PALETTE[0], "lead": PALETTE[1], "forced": PALETTE[5]}

# ---- parameters (spec §4) ----
N = 7                    # truncation: n_sig = 255, 247 active coordinates
T = 1.0
N_STEPS = 200            # grid of the target, the dual sample and the resampled paths
TARGET = "ou"            # "ou" | "double_well" | "cir" | "w2"  (see Part 1; case-insensitive)
TARGET = TARGET.lower()
TARGET_PARAMS = {
    # theta = 5: Var X_T ~ 1/(2 theta) = 0.1 vs Var W_T = 1, clearly not the uncontrolled BM
    "ou": dict(theta=5.0, mu=0.5, sigma=1.0, x0=0.0),
    # V(x) = h (x^2 - 1)^2, started at the barrier top: bimodal around +-1
    "double_well": dict(h=1.0, sigma=1.0, x0=0.0),
    # dX = a (b - X) dt + sigma sqrt(X) dW. Feller condition violated (2ab/sigma^2 = 0.6 < 1): 0 is
    # attainable (reflecting); ~1/3 of the paths come within 0.01 of 0 and ~20% end below 0.05
    "cir": dict(a=1.0, b=0.3, sigma=1.0, x0=0.5),
    # X_t = c (x0 + W_t)^2; c < 0.34 puts the target inside the cone for x0 = 0 (see Part 1)
    "w2": dict(c=0.2, x0=0.0),
}[TARGET]
M_TARGET = 20_000
M_DUAL = 100_000         # Brownian paths of the dual (SAA) sample, kept for resampling
M_OOS = 50_000           # independent Brownian sample for the out-of-sample check
WEIGHTS = "uniform"      # "uniform" (the note) or "inverse_variance" (chi^2; see the caveat in Part 3)
KAPPAS = [10.0, 100.0, 1000.0, 10_000.0]
CONE_N, CONE_M, EPS = 3, 3, 1e-6
ROLL_SIZE, N_MC, ROLL_STEPS = 2000, 20_000, 100

SEED_TARGET, SEED_DUAL, SEED_OOS, SEED_MC, SEED_ROLL, SEED_RESAMPLE = 0, 1, 2, 3, 4, 5

# %% [markdown]
# ## Part 1 — Target expected signature
#
# The target $\mathcal E=\mathbb E^\mu[\widehat{\mathbb X}_T]$ is estimated from $M_{\rm target}$ paths of a
# reference law $\mu$, chosen with `TARGET`:
#
# | `TARGET` | dynamics | relation to Wiener measure |
# |---|---|---|
# | `"ou"` | $dX=\theta(\mu-X)dt+dW$ | $\mu\ll\mathbb W$ (the drift is an admissible control): feasible |
# | `"double_well"` | $dX=-V'(X)dt+dW$, $V(x)=h(x^2-1)^2$ | $\mu\ll\mathbb W$; by Itô the density $e^{-V(X_T)+V(X_0)+\int(V''-V'^2)/2\,dt}$ is a signature functional whose leading term $-8h^2\int X^6dt$ is the integrated leading word `2222221` |
# | `"cir"` | $dX=a(b-X)dt+\sigma\sqrt X\,dW$ | singular (state-dependent diffusion): stress test. With $2ab<\sigma^2$ (default) the boundary $0$ is reached and reflected, which a Brownian tilt (no hard wall) can only mimic |
# | `"w2"` | $X_t=c\,(x_0+W_t)^2$ | singular. The top signature moments scale like $c^6$. With $c=1$ they are far heavier than Brownian ones ($\mathbb E[W^{12}]=10395$ against $\mathbb E[W^6]=15$), so the target is **outside the cone**: the leading residuals at $\boldsymbol p=0$ are negative, and matching would need a tilt concentrated on paths with $|W_1|\gtrsim4.7$, which the sample barely contains. The ESS collapses and the dual stalls (note §6.5). For $x_0=0$ the target is inside the cone when $c<(15/10395)^{1/6}\approx0.34$; the default $c=0.2$ converges. It stays expensive, because $cW^2$ has almost no volatility near $t=0$, which the control must cancel: low ESS (~0.5%) and a sizeable penalty at the optimum |
#
# The signature only sees increments. The generated paths (rollout, resampled) and the uncontrolled
# BM are therefore translated to start at the target's initial point $X_0$; this changes neither the
# signature nor the control.
#
# For the feasible laws the optimal tilt can reproduce the target law itself (spec §3.1). For the
# singular ones there is no Gibbs representation of $\mu$, and only the truncated moments can be
# matched. The dual problem lives on the cone $K=\mathcal B\cap T^N$ of the Laplace paper
# (note §6.2–6.3). The leading words $\mathbf 1^{\otimes 2n}$ and $\mathbf 1^{\otimes 2m}\mathbf 0$ (code
# `222222`, `2222221`) need coefficients $\ge\varepsilon$. The words with too many state letters are
# forced to $0$, and everything else is free. Since $\boldsymbol p^*$ is a residual, a positive leading
# residual at $\boldsymbol p=0$ (Brownian moments heavier than the target's) suggests that the target
# lies inside the cone.

# %%
t0 = time.time()
t_grid, X_target = em.target_paths(TARGET, T, N_STEPS, M_TARGET, SEED_TARGET, **TARGET_PARAMS)

fig, ax = plt.subplots()
ax.plot(t_grid, X_target[:50].T, color=LAW_COLORS["target"], lw=0.6, alpha=0.5)
mean_t, std_t = X_target.mean(axis=0), X_target.std(axis=0)
ax.plot(t_grid, mean_t, color="k", lw=2, label="mean")
ax.fill_between(t_grid, mean_t - std_t, mean_t + std_t, color=LAW_COLORS["target"], alpha=0.2, label="± std")
X0 = float(X_target[0, 0])               # deterministic initial point, reused by all generated laws
assert np.allclose(X_target[:, 0], X0), "the target must start from a deterministic point"
ax.plot(t_grid, X0 + np.sqrt(t_grid), ls="--", color=LAW_COLORS["BM"], lw=1.5,
        label=r"uncontrolled BM: $X_0\pm\sqrt{t}$")
ax.plot(t_grid, X0 - np.sqrt(t_grid), ls="--", color=LAW_COLORS["BM"], lw=1.5)
ax.set_title(f"Target law ({TARGET}): 50 sample paths")
ax.set_xlabel("t"); ax.legend()
plt.show()

# %%
S_target = em.path_signatures(X_target, t_grid, N)
E, E_se = em.expected_signature(S_target)
cone = em.make_b_cone(N, n=CONE_N, m=CONE_M, eps=EPS)
words = cone.words
levels = np.array([len(str(w)) if w else 0 for w in words])
word_labels = np.array([str(w) if w else "Ø" for w in words])
print(f"target signatures: {S_target.shape}, {time.time() - t0:.1f}s")

# %%
def iw(word):
    """Index of a code word."""
    return int(np.flatnonzero(words == word)[0])

# Closed forms of the first words (increments X_T - X_0: word 2 = E[X_T - X_0],
# word 22 = E[(X_T - X_0)^2] / 2), and the target drift used as a reference for the optimal control
# in Part 3 (None when the law is not a unit-diffusion drift law).
target_drift = None
if TARGET == "ou":
    th, mu_, sg, x0 = (TARGET_PARAMS[k] for k in ("theta", "mu", "sigma", "x0"))
    mean_T = mu_ + (x0 - mu_) * np.exp(-th * T)
    var_T = sg ** 2 * (1 - np.exp(-2 * th * T)) / (2 * th)
    closed = {1: T, 2: mean_T - x0, 22: ((mean_T - x0) ** 2 + var_T) / 2}
    target_drift = (r"OU drift $\theta(\mu-X_t)$", lambda x: th * (mu_ - x))
elif TARGET == "cir":
    a_, b_, sg, x0 = (TARGET_PARAMS[k] for k in ("a", "b", "sigma", "x0"))
    e_ = np.exp(-a_ * T)
    mean_T = b_ + (x0 - b_) * e_
    var_T = x0 * sg ** 2 / a_ * (e_ - e_ ** 2) + b_ * sg ** 2 / (2 * a_) * (1 - e_) ** 2
    closed = {1: T, 2: mean_T - x0, 22: ((mean_T - x0) ** 2 + var_T) / 2}
elif TARGET == "double_well":
    h_, x0 = TARGET_PARAMS["h"], TARGET_PARAMS["x0"]
    closed = {1: T, **({2: 0.0} if x0 == 0 else {})}      # symmetric potential: E[X_T] = 0 from x0 = 0
    target_drift = (r"double-well drift $-V'(X_t)$", lambda x: -4 * h_ * x * (x ** 2 - 1))
else:  # X = c (x0 + W)^2: X_T - X_0 = c (2 x0 W_T + W_T^2)
    cw, x0w = TARGET_PARAMS.get("c", 1.0), TARGET_PARAMS.get("x0", 0.0)
    closed = {1: T, 2: cw * T, 22: cw ** 2 * (4 * x0w ** 2 * T + 3 * T ** 2) / 2}
print("Sanity check against closed forms")
print(f"{'word':>6} {'E (MC)':>12} {'s.e.':>10} {'closed form':>12} {'z':>7}")
for w, val in closed.items():
    i = iw(w)
    z = (E[i] - val) / E_se[i] if E_se[i] > 0 else 0.0
    print(f"{w:>6} {E[i]:12.6f} {E_se[i]:10.2e} {val:12.6f} {z:7.2f}")

print("\nE per level")
print(f"{'level':>5} {'#words':>7} {'||E||_2':>10} {'max|E|':>10} {'median s.e.':>12}")
for k in range(1, N + 1):
    sel = levels == k
    print(f"{k:>5} {sel.sum():>7} {np.linalg.norm(E[sel]):10.3e} {np.abs(E[sel]).max():10.3e} {np.median(E_se[sel]):12.2e}")

# %%
weights = em.word_weights(WEIGHTS, S_target, cone)
print(cone.describe())
print(f"\nweights '{WEIGHTS}': min {weights[1:].min():.3g}, max {weights[1:].max():.3g}")

E_bm = np.asarray(expected_bm_signature(T, N).array).ravel()
print("\nLeading residuals at p = 0 (Brownian minus target); positive = inside the cone:")
for i in cone.lead_idx:
    r0 = E_bm[i] - E[i]
    print(f"  {word_labels[i]:>8}: E_BM = {E_bm[i]:.4e}, E = {E[i]:.4e} ± {E_se[i]:.1e}, "
          f"residual = {r0:+.4e}  ({'inside' if r0 > 0 else 'OUTSIDE'})")

# %% [markdown]
# ## Part 2 — Dual optimisation for $\boldsymbol p^*$
#
# On a fixed sample of $M$ Brownian signatures, $\widehat J$ is a deterministic, smooth, strongly concave
# function (sample-average approximation, note §7):
# $\nabla\widehat J=\kappa(\widehat m-\mathcal E-\Lambda^{-1}\boldsymbol p)$ and
# $\nabla^2\widehat J=-\kappa\Lambda^{-1}-\kappa^2\widehat C$. We maximise it over $K_\varepsilon$ by projected
# Newton (spec §3.4), with a continuation in $\kappa$ warm-started from the previous optimum. On the
# empirical measure the Gibbs principle is exact, so the gap $\widehat P-\widehat J\ge0$ with the modified
# primal of spec §2.4 certifies optimality. An active leading constraint means that the target asks
# for heavier top moments than the cone allows (note §6.5).

# %%
t0 = time.time()
_, W_dual = em.brownian_paths(T, N_STEPS, M_DUAL, SEED_DUAL)
S_dual = jnp.asarray(em.path_signatures(W_dual, t_grid, N))
print(f"dual sample: {S_dual.shape}, {time.time() - t0:.1f}s")

t0 = time.time()
results = em.continuation(S_dual, E, cone, KAPPAS, weights, method="newton", tol=1e-8, verbose=False)
print(f"continuation: {time.time() - t0:.1f}s")
for r in results:
    print(f"kappa={r.kappa:>7g}: converged={r.converged}, iters={r.n_iter:>3}, J={r.J:.6f}, "
          f"gap={r.gap:.2e}, ESS/M={r.history[-1]['ess_frac']:.3f}, "
          f"active leads={[word_labels[i] for i in r.active_leads]}")

# %%
fig, axes = plt.subplots(1, 4, figsize=(22, 4.5))
for r, c in zip(results, PALETTE):
    h = r.history
    it = np.arange(len(h))
    lab = f"κ={r.kappa:g}"
    axes[0].plot(it, [x["J"] for x in h], "o-", color=c, ms=3, label=lab)
    axes[1].semilogy(it, [max(x["grad_norm"], 1e-16) for x in h], "o-", color=c, ms=3, label=lab)
    axes[2].plot(it, [x["ess_frac"] for x in h], "o-", color=c, ms=3, label=lab)
    axes[3].semilogy(it, [max(abs(x["gap"]), 1e-17) for x in h], "o-", color=c, ms=3, label=lab)
for ax, title in zip(axes, [r"$\widehat J$", r"projected gradient $\|\cdot\|_\infty$", "ESS / M",
                            r"gap $|\widehat P-\widehat J|$"]):
    ax.set_title(title); ax.set_xlabel("iteration"); ax.legend()
axes[0].set_ylim(min(r.history[1]["J"] if len(r.history) > 1 else r.J for r in results) - 0.05,
                 max(r.J for r in results) + 0.02)
plt.show()

# %%
final = results[-1]
KAPPA = final.kappa
p_star = np.asarray(final.p)
problem = em.DualProblem(S_dual, E, KAPPA, cone, weights)
tilt_star = problem.tilt(final.p)
cls = cone.word_class()

fig, axes = plt.subplots(2, 1, figsize=(22, 8), sharex=True)
x = np.arange(cone.n_sig)
for name, col in CLASS_COLORS.items():
    sel = cls == name
    axes[0].bar(x[sel], p_star[sel], color=col, width=0.9, label=name)
    axes[1].bar(x[sel], (p_star / weights)[sel], color=col, width=0.9, label=name)
axes[0].set_yscale("symlog", linthresh=1e-3)
axes[0].set_title(rf"$\mathbf{{p}}^*$ at κ={KAPPA:g} (symlog)")
axes[1].set_yscale("symlog", linthresh=1e-6)
axes[1].set_title(r"predicted residual $\Lambda^{-1}\mathbf{p}^* = m^*-\mathcal{E}$ (symlog)")
for k in range(1, N + 1):
    for ax in axes:
        ax.axvline(np.flatnonzero(levels == k)[0] - 0.5, color="grey", lw=0.5)
axes[1].set_xticks([int(np.flatnonzero(levels == k).mean()) for k in range(1, N + 1)],
                   [f"L{k}" for k in range(1, N + 1)])
axes[1].set_xlabel("words, grouped by level")
axes[0].legend()
plt.show()

print("Leading words at the optimum:")
g_star = np.asarray(problem.grad(final.p))
for i in cone.lead_idx:
    state = "ACTIVE (at eps)" if i in final.active_leads else "inactive"
    print(f"  {word_labels[i]:>8}: p* = {p_star[i]:.3e}, dJ/dp = {g_star[i]:+.3e}  -> {state}")

# %%
# Out-of-sample check: the tilt weights depend on the sample only through <p*, S_i>, so m-hat(p*)
# can be evaluated on an independent Brownian sample.
_, W_oos = em.brownian_paths(T, N_STEPS, M_OOS, SEED_OOS)
problem_oos = em.DualProblem(em.path_signatures(W_oos, t_grid, N), E, KAPPA, cone, weights)
tilt_oos = problem_oos.tilt(final.p)
del W_oos


def tilted_se(prob, tilt):
    """Self-normalised importance-sampling standard error of the tilted mean, per word."""
    w = tilt.w
    return np.sqrt(np.asarray(jnp.sum(w ** 2 * (prob.S - tilt.m[:, None]) ** 2, axis=1)))


m_in, m_oos = np.asarray(tilt_star.m), np.asarray(tilt_oos.m)
se_in, se_oos = tilted_se(problem, tilt_star), tilted_se(problem_oos, tilt_oos)
print(f"ESS/M: in-sample {float(tilt_star.ess) / problem.M:.3f}, "
      f"out-of-sample {float(tilt_oos.ess) / problem_oos.M:.3f}")
print(f"log Z: in-sample {float(tilt_star.logZ):.5f}, out-of-sample {float(tilt_oos.logZ):.5f}")
print(f"{'level':>5} {'rel diff ||m_oos-m_in||/||m_in||':>34} {'median |z|':>11} {'max |z|':>8}")
for k in range(1, N + 1):
    sel = (levels == k) & ~cone.pure_time_mask       # pure-time words are deterministic
    z = np.abs(m_oos - m_in)[sel] / np.sqrt(se_in[sel] ** 2 + se_oos[sel] ** 2)
    rel = np.linalg.norm((m_oos - m_in)[levels == k]) / np.linalg.norm(m_in[levels == k])
    print(f"{k:>5} {rel:34.2e} {np.median(z) if z.size else 0:11.2f} {z.max() if z.size else 0:8.2f}")

# %% [markdown]
# ## Part 3 — Optimal trajectories at $\boldsymbol p^*$
#
# Two generators of the optimal law $Q^*$ (note §8):
# 1. **Closed loop.** The feedback $\alpha^*_t=\pi(t,\widehat{\mathbb X}_t)$ of the auxiliary problem with
#    terminal tensor $\boldsymbol q=-\kappa\boldsymbol p^*$ is evaluated by `sig_control.monte_carlo_policy`.
#    It uses an *independent* sample of $\widehat{\mathbb W}_{0,1}$, Brownian scaling, and the recentering
#    identities, and is rolled out with fresh noise.
# 2. **Resampling.** $Q^*$ is a tilt of Wiener measure on path space, so multinomial resampling of the
#    dual Brownian paths with weights $w_i(\boldsymbol p^*)$ gives exact samples of $Q^*$, up to MC error.
#
# By strong duality (note §5.3), the energy $\tfrac12\mathbb E\int(\alpha^*)^2$ equals the entropy
# $H(Q^*|\mathbb W)$, and energy plus penalty equals $J(\boldsymbol p^*)$.
#
# **Diverging rollout paths.** The cone guarantees that $\langle-\kappa\boldsymbol p,\widehat{\mathbb X}_T\rangle$ is
# bounded above pathwise, but the bound $C_T$ grows like $|\boldsymbol s|\sim$ (mixed coefficients)$/(\alpha\beta)$.
# When a leading constraint is active ($\boldsymbol p^{\mathbf 1^{\otimes 2n}}=\varepsilon$, so $\alpha=\kappa\varepsilon$)
# and the top-level coefficients are large, the bound is only formal. This happens with
# `"inverse_variance"` weights, where $\lambda_v$ reaches $\sim10^7$ at level 7. From some atypical states,
# the shifted tilt $\langle\boldsymbol q,\widehat{\mathbb X}_t\otimes\widehat{\mathbb W}_{t,T}\rangle$ then rewards far-tail
# futures. The conditional ESS of the feedback collapses to a single sample, the drift turns repulsive,
# and the path runs away. The dual sample never visits that region, so neither the ESS nor the gap
# of Part 2 can detect it: **the closed-loop rollout is the detector**. Diverged paths (non-finite or
# $|X|>3\max|X^{\rm target}|$) are counted and reported, and the rollout statistics below are computed on
# the retained paths, i.e. *conditional on non-divergence* (biased by the excluded fraction). In our
# runs with the OU target, `"inverse_variance"` weights let 0.5% (θ=1) to about 7% (θ=5) of the
# paths diverge, while `WEIGHTS = "uniform"` gave none.

# %%
t0 = time.time()
tg_mc, W_mc = em.brownian_paths(1.0, N_STEPS, N_MC, SEED_MC)
sig_brownian_1 = ta.TensorSequence(array=jnp.asarray(em.path_signatures(W_mc, tg_mc, N)), trunc=N, dim=2)
del W_mc

roll = em.rollout(final.p, KAPPA, T, N, ROLL_STEPS, ROLL_SIZE, sig_brownian_1, SEED_ROLL, x0=X0)
t_roll = np.linspace(0.0, T, ROLL_STEPS + 1)
print(f"rollout ({ROLL_SIZE} paths x {ROLL_STEPS} steps, N_MC={N_MC}): {time.time() - t0:.1f}s")

X_BOUND = 3 * np.abs(X_target - X0).max()           # bound on the excursion |X_t - X_0|
with np.errstate(invalid="ignore"):
    diverged = ~np.all(np.isfinite(roll.X) & (np.abs(roll.X - X0) <= X_BOUND), axis=1)
keep = ~diverged
n_div = int(diverged.sum())
print(f"diverged paths (non-finite or |X - X0| > {X_BOUND:.2f}): {n_div} / {ROLL_SIZE} = {n_div / ROLL_SIZE:.2%}"
      + ("   -> rollout statistics below are conditional on non-divergence" if n_div else ""))
roll_X, roll_control = roll.X[keep], roll.control[keep]
S_roll = roll.sig[:, keep, -1]
n_roll = int(keep.sum())
HAS_ROLL = n_roll >= 10          # otherwise the rollout is unusable and is left out of Parts 3-4
if not HAS_ROLL:
    print("WARNING: (almost) every rollout path diverged: the feedback at p* is not usable. This is "
          "expected when the target lies outside the cone (negative leading residuals in Part 1, note §6.5). "
          "The rollout is left out of the checks and of the validation below.")

if HAS_ROLL:
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    axes[0].plot(t_roll, roll_X[:50].T, color=LAW_COLORS["rollout"], lw=0.6, alpha=0.5)
    mr, sr = roll_X.mean(axis=0), roll_X.std(axis=0)
    axes[0].plot(t_roll, mr, "k", lw=2, label="mean")
    axes[0].fill_between(t_roll, mr - sr, mr + sr, color=LAW_COLORS["rollout"], alpha=0.2, label="± std")
    axes[0].set_title(f"Closed-loop rollout: 50 of {n_roll} retained paths"); axes[0].legend()
    ma, sa = roll_control[:, :-1].mean(axis=0), roll_control[:, :-1].std(axis=0)
    axes[1].plot(t_roll[:-1], ma, color=LAW_COLORS["rollout"], lw=2, label=r"mean $\alpha^*_t$")
    axes[1].fill_between(t_roll[:-1], ma - sa, ma + sa, color=LAW_COLORS["rollout"], alpha=0.2, label="± std")
    if target_drift is not None:
        # reference: the target's own drift evaluated along the rollout paths
        drift_label, drift_fn = target_drift
        axes[1].plot(t_roll[:-1], drift_fn(roll_X[:, :-1]).mean(axis=0), "--", color=LAW_COLORS["target"],
                     label=f"mean {drift_label}")
    axes[1].set_title("Optimal control"); axes[1].legend()
    for ax in axes:
        ax.set_xlabel("t")
    plt.show()

# %%
X_resampled = em.resample(problem, final.p, W_dual, n=M_TARGET, seed=SEED_RESAMPLE, x0=X0)
S_res = em.path_signatures(X_resampled, t_grid, N)
m_res, se_res = em.expected_signature(S_res)
print(f"resampled {X_resampled.shape[0]} paths from the dual sample "
      f"(ESS = {float(tilt_star.ess):.0f} of {problem.M})")

# %%
pc = problem.primal_cost(final.p)


def empirical_penalty(m, se_m):
    """kappa/2 sum_v lambda_v (m^v - E^v)^2, bias-corrected for the sampling noise of m."""
    return 0.5 * KAPPA * np.sum(weights[1:] * ((m - E)[1:] ** 2 - se_m[1:] ** 2))


pen_res = empirical_penalty(m_res, se_res)
print(f"  policy value at t=0 (independent MC sample) : {roll.value[0, 0]: .5f}")
print(f"  log Z(p*) on the dual sample  (= -V_p*)    : {float(tilt_star.logZ): .5f}")
if HAS_ROLL:
    dt_roll = np.diff(t_roll)
    energy_path = 0.5 * np.sum(roll_control[:, :-1] ** 2 * dt_roll, axis=1)
    energy, energy_se = energy_path.mean(), energy_path.std() / np.sqrt(n_roll)
    m_roll, se_roll = em.expected_signature(S_roll)
    pen_roll = empirical_penalty(m_roll, se_roll)
    aux_path = energy_path + KAPPA * (p_star @ S_roll)
    print(f"Value checks (rollout quantities on the {n_roll} retained paths)")
    print(f"  auxiliary cost E[1/2∫α² + κ<p*,X_T>]       : {aux_path.mean(): .5f} ± {aux_path.std() / np.sqrt(n_roll):.5f}"
          f"   vs  V_p* = {-float(tilt_star.logZ):.5f}")
    print(f"  energy 1/2 E∫α² (rollout)                  : {energy: .5f} ± {energy_se:.5f}"
          f"   vs  entropy H(Q*|W) = {pc['entropy']:.5f}")
    print(f"  empirical primal cost (energy + penalty)    : {energy + pen_roll: .5f}"
          f"   vs  J(p*) = {final.J:.5f}")
else:
    energy = energy_se = pen_roll = np.nan
    print("  (no usable rollout: energy and auxiliary-cost checks skipped)")
print(f"  penalty: at Q* (dual sample) {pc['penalty']:.5f} | rollout estimate {pen_roll:.5f}"
      f" | resampled estimate {pen_res:.5f}")
print(f"  (the empirical penalty multiplies any bias of m by kappa*lambda, up to {KAPPA * weights[1:].max():.1e} here,\n"
      f"   so at large kappa the energy-vs-entropy and auxiliary-cost lines are the informative checks)")

# %% [markdown]
# ## Part 4 — Validation
#
# The generated laws should reproduce the target paths in distribution, and their expected signatures
# should match $\mathcal E+\Lambda^{-1}\boldsymbol p^*$, the theoretical residual at finite $\kappa$
# (note §8), up to Monte-Carlo error. Forced-zero words are not matched by design: they carry no dual
# coefficient, so their moments are whatever the tilt produces.
#
# The **uncontrolled Brownian motion** ($\alpha\equiv0$, i.e. $\boldsymbol p=0$, the reference measure of the
# tilt) is shown alongside as a baseline: it shows how far the control has to move the law. Its paths
# are the first dual-sample paths, and its expected signature is the dual-sample mean (same grid).

# %%
X_bm = X0 + W_dual[:M_TARGET]                      # uncontrolled BM started at the target's X_0
m_bm, se_bm = em.expected_signature(np.asarray(S_dual))
laws = {"target": (t_grid, X_target)}
if HAS_ROLL:
    laws["rollout"] = (t_roll, roll_X)
laws.update({"resampled": (t_grid, X_resampled), "BM": (t_grid, X_bm)})
fig, axes = plt.subplots(2, 4, figsize=(26, 10))
for ax in axes[0, len(laws):]:
    ax.axis("off")
for j, (name, (tg, Xp)) in enumerate(laws.items()):
    ax = axes[0, j]
    ax.plot(tg, Xp[:50].T, color=LAW_COLORS[name], lw=0.6, alpha=0.4)
    for other, (tg_o, X_o) in laws.items():
        mo, so = X_o.mean(axis=0), X_o.std(axis=0)
        ax.plot(tg_o, mo, color=LAW_COLORS[other], lw=2 if other == name else 1,
                ls="-" if other == name else "--", label=f"{other} mean ± std")
        ax.fill_between(tg_o, mo - so, mo + so, color=LAW_COLORS[other], alpha=0.12 if other == name else 0.0)
        ax.plot(tg_o, mo + so, color=LAW_COLORS[other], lw=0.8, ls=":")
        ax.plot(tg_o, mo - so, color=LAW_COLORS[other], lw=0.8, ls=":")
    ax.set_title(f"{name if name != 'BM' else 'uncontrolled BM'}: 50 paths"); ax.set_xlabel("t")
    if j == 0:
        ax.legend(fontsize=8)
y_lo = min(np.percentile(Xp, 0.5) for _, Xp in laws.values())
y_hi = max(np.percentile(Xp, 99.5) for _, Xp in laws.values())
for ax in axes[0]:
    ax.set_ylim(y_lo, y_hi)
grid_y = np.linspace(y_lo, y_hi, 300)
for j, t_eval in enumerate([0.25, 0.5, 1.0]):
    ax = axes[1, j]
    for name, (tg, Xp) in laws.items():
        k = int(np.argmin(np.abs(tg - t_eval)))
        vals = Xp[:, k]
        ax.hist(vals, bins=60, range=(y_lo, y_hi), density=True, alpha=0.25, color=LAW_COLORS[name])
        ax.plot(grid_y, gaussian_kde(vals)(grid_y), color=LAW_COLORS[name], lw=2, label=name)
    ax.set_title(f"marginal of $X_t$ at t = {t_eval}"); ax.legend()
ax = axes[1, 3]
for name, (tg, Xp) in laws.items():
    ax.plot(tg, Xp.std(axis=0), color=LAW_COLORS[name], lw=2, label=name)
ax.set_title("std of $X_t$"); ax.set_xlabel("t"); ax.legend()
plt.show()

# %%
predicted = E + p_star / weights
predicted[0] = 1.0
generated = {"rollout": (m_roll, se_roll)} if HAS_ROLL else {}
generated["resampled"] = (m_res, se_res)
reference = {"BM": (m_bm, se_bm)}               # uncontrolled baseline, not expected to match
matched = cone.active_mask & ~cone.pure_time_mask   # free and lead words that carry randomness
forced = cone.forced_mask


def level_stats(m, se_m, ref, se_ref, sel):
    """rel L2 error, max abs error, median |z| and fraction of words with |z| < 2."""
    if not sel.any():
        return (np.nan,) * 4
    rel = np.linalg.norm((m - ref)[sel]) / np.linalg.norm(ref[sel])
    mx = np.abs(m - ref)[sel].max()
    z = np.abs(m - ref)[sel] / np.sqrt(se_m[sel] ** 2 + se_ref[sel] ** 2)
    return rel, mx, np.median(z), np.mean(z < 2)


# A level is "within MC error" if its median |z| is <= 1 (0.67 for exact N(0,1) z-scores).
within = {}
for name, (m, se_m) in {**generated, **reference}.items():
    print(f"\n=== {name}{' (uncontrolled baseline)' if name in reference else ''} ===")
    print(f"{'level':>6} | {'vs E: rel L2':>12} {'max|Δ|':>9} {'med z':>6} {'|z|<2':>6} | "
          f"{'vs E+Λ⁻¹p*: rel L2':>18} {'max|Δ|':>9} {'med z':>6} {'|z|<2':>6}")
    within[name] = []
    for k in range(1, N + 1):
        sel = matched & (levels == k)
        a = level_stats(m, se_m, E, E_se, sel)
        b = level_stats(m, se_m, predicted, E_se, sel)
        within[name].append(b[2] <= 1.0)
        print(f"{k:>6} | {a[0]:12.2e} {a[1]:9.2e} {a[2]:6.2f} {a[3]:6.0%} | "
              f"{b[0]:18.2e} {b[1]:9.2e} {b[2]:6.2f} {b[3]:6.0%}")
    a = level_stats(m, se_m, E, E_se, forced)
    print(f"{'forced':>6} | {a[0]:12.2e} {a[1]:9.2e} {a[2]:6.2f} {a[3]:6.0%} |   "
          f"(not matched by design: {forced.sum()} words)")

# %%
def grouped_bars(ax, sel_idx, level_idx, show_labels=True):
    """Grouped bars over the words ``sel_idx``. The y-range is set by target / rollout / resampled
    over the whole level ``level_idx`` (BM excluded); BM bars beyond it are clipped and marked."""
    width = 0.21
    x = np.arange(len(sel_idx))
    series = {"target": (E, E_se), **generated, "BM": (m_bm, se_bm)}
    shift = (len(series) - 1) / 2
    for j, (name, (m, se_m)) in enumerate(series.items()):
        ax.bar(x + (j - shift) * width, m[sel_idx], width=width, color=LAW_COLORS[name],
               yerr=2 * se_m[sel_idx], error_kw=dict(lw=0.6, capsize=0), label=name)
    vals = np.concatenate([E[level_idx], *[m[level_idx] for m, _ in generated.values()], [0.0]])
    lo, hi = vals.min(), vals.max()
    pad = 0.12 * (hi - lo) if hi > lo else 1.0
    lo, hi = lo - pad, hi + pad
    ax.set_ylim(lo, hi)
    x_bm = x + shift * width
    for side, edge, marker in ((m_bm[sel_idx] > hi, hi, "^"), (m_bm[sel_idx] < lo, lo, "v")):
        if side.any():
            ax.plot(x_bm[side], np.full(side.sum(), edge - 0.04 * (hi - lo) * (1 if marker == "^" else -1)),
                    marker, color=LAW_COLORS["BM"], mec="k", mew=0.5, ms=6, ls="none")
    for xi, i in zip(x, sel_idx):
        if cone.forced_mask[i]:
            ax.axvspan(xi - 0.45, xi + 0.45, color=CLASS_COLORS["forced"], alpha=0.12, lw=0)
        elif i in cone.lead_idx:
            ax.axvspan(xi - 0.45, xi + 0.45, color=CLASS_COLORS["lead"], alpha=0.15, lw=0)
    ax.set_xlim(-0.6, len(sel_idx) - 0.4)
    ax.set_xticks(x, word_labels[sel_idx] if show_labels else [""] * len(sel_idx),
                  rotation=90 if len(sel_idx) > 8 else 0, fontsize=7 if len(sel_idx) > 16 else 9)


fig = plt.figure(figsize=(26, 4.2 * 6))
fig.suptitle("Expected signature per level: error bars = 2 s.e.; shaded words: orange = lead, grey = forced; "
             "y-range set by target / rollout / resampled, ▲▼ = BM bar beyond the axis", fontsize=13)
gs = fig.add_gridspec(6, 4)
layout = {1: gs[0, 0], 2: gs[0, 1], 3: gs[0, 2:], 4: gs[1, :2], 5: gs[1, 2:], 6: gs[2, :]}
for k in range(1, 7):
    ax = fig.add_subplot(layout[k])
    lvl = np.flatnonzero(levels == k)
    grouped_bars(ax, lvl, lvl)
    ax.set_title(f"level {k}")
    if k == 1:
        ax.legend()
lvl7 = np.flatnonzero(levels == 7)
for r, half in enumerate(np.array_split(lvl7, 2)):
    ax = fig.add_subplot(gs[3 + r, :])
    grouped_bars(ax, half, lvl7)
    ax.set_title(f"level 7, part {r + 1}/2")
ax = fig.add_subplot(gs[5, :2])
cmap = plt.get_cmap("viridis", N)
markers = {"rollout": "o", "resampled": "^", "BM": "x"}
for name, (m, _) in {**generated, **reference}.items():
    sel = np.abs(E) > 0
    sc_ = ax.scatter(np.abs(E[sel]), np.abs(m[sel]), c=levels[sel], cmap=cmap, s=14 if name != "BM" else 22,
                     marker=markers[name], alpha=0.7, label=name, vmin=0.5, vmax=N + 0.5)
lim = [np.abs(E[np.abs(E) > 0]).min() / 2, max(np.abs(E).max(), np.abs(m_bm).max()) * 2]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel(r"$|\mathcal{E}^v|$ (target)"); ax.set_ylabel(r"$|m^v|$ (generated)")
ax.set_title("parity: o = rollout, ^ = resampled, x = uncontrolled BM; colour = level"); ax.legend()
fig.colorbar(sc_, ax=ax, ticks=range(1, N + 1), label="level")
ax = fig.add_subplot(gs[5, 2:])
for name, (m, se_m) in generated.items():
    sel = matched
    z = (m - predicted)[sel] / np.sqrt(se_m[sel] ** 2 + E_se[sel] ** 2)
    ax.hist(z, bins=40, range=(-6, 6), alpha=0.5, density=True, color=LAW_COLORS[name], label=name)
zz = np.linspace(-6, 6, 200)
ax.plot(zz, np.exp(-zz ** 2 / 2) / np.sqrt(2 * np.pi), "k--", label="N(0,1)")
ax.set_title(r"z-scores of $m^v-(\mathcal{E}+\Lambda^{-1}\mathbf{p}^*)^v$ over matched words"); ax.legend()
plt.show()

# %%
print("=" * 72)
print("Summary")
print("=" * 72)
print(f"target                 : {TARGET} {TARGET_PARAMS}, M_target = {M_TARGET}")
print(f"truncation / cone      : N = {N}, n = {cone.n}, m = {cone.m}, eps = {cone.eps:g}, weights = {WEIGHTS}")
print(f"final kappa            : {KAPPA:g}  (converged: {final.converged}, {final.n_iter} Newton iterations)")
print(f"J(p*)                  : {final.J:.6f}")
print(f"duality gap            : {final.gap:.2e}")
print(f"entropy H(Q*|W)        : {pc['entropy']:.6f}   (rollout energy {energy:.6f} ± {energy_se:.6f})")
print(f"penalty at Q*          : {pc['penalty']:.6f}")
print(f"active leads           : {[word_labels[i] for i in final.active_leads] or 'none'}")
print(f"ESS / M at p*          : {float(tilt_star.ess) / problem.M:.3f}")
print(f"diverged rollout paths : {n_div} / {ROLL_SIZE} ({n_div / ROLL_SIZE:.2%})"
      + ("  -- rollout statistics are conditional on non-divergence" if n_div else ""))
for name in generated:
    ok = [k for k, flag in zip(range(1, N + 1), within[name]) if flag]
    print(f"levels within MC error ({name:>9}, median |z| <= 1 vs E+Λ⁻¹p*): {ok}")
z_bm = np.abs(m_bm - E)[matched] / np.sqrt(se_bm[matched] ** 2 + E_se[matched] ** 2)
print(f"uncontrolled BM vs E   : median |z| = {np.median(z_bm):.1f} over matched words "
      f"(std X_T: BM {X_bm[:, -1].std():.3f}, target {X_target[:, -1].std():.3f}, "
      + (f"rollout {roll_X[:, -1].std():.3f}, " if HAS_ROLL else "")
      + f"resampled {X_resampled[:, -1].std():.3f})")

# %%
