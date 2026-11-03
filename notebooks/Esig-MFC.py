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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import gaussian_kde

import signature.tensor_algebra as ta
import signature.esig_mfc as em
from signature.expected_signature import expected_bm_signature

PALETTE = ["#5C7B55", "#C0714A", "#C9A84C", "#A8C5A0", "#8B5A6B", "#4A5568"]
plt.rcParams["figure.figsize"] = ((1 + np.sqrt(5)) / 2 * 5, 5)
plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=PALETTE)
LAW_COLORS = {"target": PALETTE[0], "generated": PALETTE[1], "BM": PALETTE[2]}
CLASS_COLORS = {"free": PALETTE[0], "lead": PALETTE[1], "forced": PALETTE[5]}
IMG = ("notebooks/" if os.path.isdir("notebooks") else "") + "images/esig-matching"
os.makedirs(IMG, exist_ok=True)

# ---- parameters (spec §4) ----
N = 5                    # truncation: n_sig = 255, 247 active coordinates
T = 1.0
N_STEPS = 500            # grid of the target, the dual sample and the resampled paths
TARGET = "rl_fbm_h01"    # "ou" | "double_well" | "cir" | "w2" | "rl_fbm_h01|h06|h07|h08"  (see Part 1)
TARGET = TARGET.lower()
TARGETS = {              # name -> (kind of em.target_paths, parameters)
    # theta = 5: Var X_T ~ 1/(2 theta) = 0.1 vs Var W_T = 1, clearly not the uncontrolled BM
    "ou": ("ou", dict(theta=5.0, mu=0.5, sigma=1.0, x0=0.0)),
    # V(x) = h (x^2 - 1)^2, started at the barrier top: bimodal around +-1
    "double_well": ("double_well", dict(h=1.0, sigma=1.0, x0=0.0)),
    # dX = a (b - X) dt + sigma sqrt(X) dW. Feller condition violated (2ab/sigma^2 = 0.6 < 1): 0 is
    # attainable (reflecting); ~1/3 of the paths come within 0.01 of 0 and ~20% end below 0.05
    "cir": ("cir", dict(a=1.0, b=0.3, sigma=1.0, x0=0.5)),
    # X_t = c (x0 + W_t)^2; c < 0.34 puts the target inside the cone for x0 = 0 (see Part 1)
    "w2": ("w2", dict(c=0.2, x0=0.0)),
    # X_t = int_0^t K(t-s) dW_s, K(u) = eta (u + delta)^(H - 1/2). delta > 0 makes X a semimartingale;
    # eta = delta^(1/2 - H) normalises K(0) = 1, i.e. unit instantaneous variance as dW (feasible law)
    "rl_fbm_h01": ("rl_fbm", dict(H=0.1, delta=0.1, eta=0.1 ** 0.4, x0=0.0)),
    "rl_fbm_h06": ("rl_fbm", dict(H=0.6, delta=0.1, eta=0.1 ** -0.1, x0=0.0)),
    "rl_fbm_h07": ("rl_fbm", dict(H=0.7, delta=0.01, eta=0.01 ** -0.2, x0=0.0)),
    "rl_fbm_h08": ("rl_fbm", dict(H=0.8, delta=0.1, eta=0.1 ** -0.3, x0=0.0)),
}
TARGET_KIND, TARGET_PARAMS = TARGETS[TARGET]
M_TARGET = 100_000
M_DUAL = 100_000         # Brownian paths of the dual (SAA) sample, kept for resampling
M_OOS = 100_000           # independent Brownian sample for the out-of-sample check
WEIGHTS = "uniform"      # "uniform" (the note) or "inverse_variance" (chi^2; see the caveat in Part 3)
KAPPAS = [10.0, 100.0, 1000.0, 10_000.0]
CONE_N, CONE_M, EPS = N // 2, N //2, 1e-6
GEN_SIZE, N_MC, GEN_STEPS = 2000, 20_000, 100

SEED_TARGET, SEED_DUAL, SEED_OOS, SEED_MC, SEED_GEN, SEED_RESAMPLE = 0, 1, 2, 3, 4, 5


def save_pdf(fig, name):
    """Save ``fig`` under ``IMG``; the file name carries TARGET and N so that runs do not overwrite each other."""
    fig.savefig(f"{IMG}/{TARGET}_N{N}_{name}.pdf", format="pdf", transparent=True)


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
# | `"rl_fbm_h01"`, `"rl_fbm_h06"`, `"rl_fbm_h07"`, `"rl_fbm_h08"` | Riemann–Liouville fBM with shifted kernel, $X_t=\int_0^tK(t-s)\,dW_s$, $K(u)=\eta\,(u+\delta)^{H-1/2}$, normalised by $\eta=\delta^{1/2-H}$ so that $K(u)=(1+u/\delta)^{H-1/2}$ and $K(0)=1$. $H\in\{0.1,0.6,0.7,0.8\}$, all with $\delta=0.1$. Simulated exactly on the grid (Gaussian vector); $\operatorname{Var}X_t=\eta^2\big((t+\delta)^{2H}-\delta^{2H}\big)/(2H)$ | for $\delta=0$, RL fBM is not a semimartingale ($H\ne\frac12$). The shift $\delta>0$ makes $K$ smooth on $[0,T]$, so $dX_t=K(0)\,dW_t+\big(\int_0^tK'(t-s)dW_s\big)dt$ is a semimartingale. With $K(0)=1$ its instantaneous variance is that of $dW$ and the drift is admissible: $\mu\ll\mathbb W$, feasible (with the default $\eta=\sqrt{2H}$ the law would be singular). For $H<\frac12$ the kernel decreases from $1$: lighter moments, inside the cone. For $H>\frac12$ it increases from $1$, so $\operatorname{Var}X_T>T$ for every $\delta$ and both leading words are **outside the cone** (they end up active at $\varepsilon$). The match then only holds below the top state degree. How far outside grows with $H$ (and as $\delta$ decreases): with $\delta=0.1$, $\operatorname{Var}X_T\approx1.4$, $2.0$ and $2.8$ for $H=0.6$, $0.7$, $0.8$. The resampled law reproduces the target in all cases. The closed-loop feedback works for $H=0.6$ (ESS ≈ 90%, no divergence, energy ≈ entropy). For $H=0.7$ it degrades: ESS ≈ 40%, about 4% of the paths diverge, and the energy overshoots the entropy. For $H=0.8$ it breaks down: ESS ≈ 5%, about two thirds of the paths diverge |
#
# The signature only sees increments. The generated paths (closed loop, resampled) and the uncontrolled
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
t_grid, X_target = em.target_paths(TARGET_KIND, T, N_STEPS, M_TARGET, SEED_TARGET, **TARGET_PARAMS)

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

# Target drift as a function of the state (Markov drift laws only), a reference for the feedback in Part 3.
target_drift = None
if TARGET_KIND == "ou":
    th, mu_ = TARGET_PARAMS["theta"], TARGET_PARAMS["mu"]
    target_drift = (r"OU drift $\theta(\mu-X_t)$", lambda x: th * (mu_ - x))
elif TARGET_KIND == "double_well":
    h_ = TARGET_PARAMS["h"]
    target_drift = (r"double-well drift $-V'(X_t)$", lambda x: -4 * h_ * x * (x ** 2 - 1))

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

fig, ax = plt.subplots(figsize=(22, 4.5))
x = np.arange(cone.n_sig)
for name, col in CLASS_COLORS.items():
    sel = cls == name
    ax.bar(x[sel], p_star[sel], color=col, width=0.9, label=name)
ax.set_yscale("symlog", linthresh=1e-3)
ax.set_title(rf"$\mathbf{{p}}^*$ at κ={KAPPA:g} (symlog)")
for k in range(1, N + 1):
    ax.axvline(np.flatnonzero(levels == k)[0] - 0.5, color="grey", lw=0.5)
ax.set_xticks([int(np.flatnonzero(levels == k).mean()) for k in range(1, N + 1)],
              [f"L{k}" for k in range(1, N + 1)])
ax.set_xlabel("words, grouped by level")
ax.legend()
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
print(f"ESS/M: in-sample {float(tilt_star.ess) / problem.M:.3f}, "
      f"out-of-sample {float(tilt_oos.ess) / problem_oos.M:.3f}")
print(f"log Z: in-sample {float(tilt_star.logZ):.5f}, out-of-sample {float(tilt_oos.logZ):.5f}")

# %% [markdown]
# ## Part 3 — Optimal trajectories at $\boldsymbol p^*$
#
# Two generators of the optimal law $Q^*$ (note §8):
# 1. **Closed loop.** The feedback $\alpha^*_t=\pi(t,\widehat{\mathbb X}_t)$ of the auxiliary problem with
#    terminal tensor $\boldsymbol q=-\kappa\boldsymbol p^*$ is evaluated by `sig_control.monte_carlo_policy`.
#    It uses an *independent* sample of $\widehat{\mathbb W}_{0,1}$, Brownian scaling, and the recentering
#    identities, and drives fresh Brownian noise: these are the **generated** paths.
# 2. **Resampling.** $Q^*$ is a tilt of Wiener measure on path space, so multinomial resampling of the
#    dual Brownian paths with weights $w_i(\boldsymbol p^*)$ gives exact samples of $Q^*$, up to MC error.
#
# By strong duality (note §5.3), the energy $\tfrac12\mathbb E\int(\alpha^*)^2$ equals the entropy
# $H(Q^*|\mathbb W)$, and energy plus penalty equals $J(\boldsymbol p^*)$.
#
# **The control.** The figure below shows:
# - **left:** a few generated paths, coloured by the terminal value $X_T$;
# - **right:** $\alpha^*_t$ against $X_t$ at a few times. For a Markov target this should follow the
#   target's drift function (dashed). For a path-dependent target such as the fBM, the vertical
#   spread at a given state is the dependence on the past.
#
# **Diverging generated paths.** The cone guarantees that $\langle-\kappa\boldsymbol p,\widehat{\mathbb X}_T\rangle$ is
# bounded above pathwise, but the bound $C_T$ grows like $|\boldsymbol s|\sim$ (mixed coefficients)$/(\alpha\beta)$.
# When a leading constraint is active ($\boldsymbol p^{\mathbf 1^{\otimes 2n}}=\varepsilon$, so $\alpha=\kappa\varepsilon$)
# and the top-level coefficients are large, the bound is only formal. This happens with
# `"inverse_variance"` weights, where $\lambda_v$ reaches $\sim10^7$ at level 7. From some atypical states,
# the shifted tilt $\langle\boldsymbol q,\widehat{\mathbb X}_t\otimes\widehat{\mathbb W}_{t,T}\rangle$ then rewards far-tail
# futures. The conditional ESS of the feedback collapses to a single sample, the drift turns repulsive,
# and the path runs away. The dual sample never visits that region, so neither the ESS nor the gap
# of Part 2 can detect it: **the closed-loop generation is the detector**. Diverged paths (non-finite or
# $|X|>3\max|X^{\rm target}|$) are counted and reported, and the generated statistics below are computed on
# the retained paths, i.e. *conditional on non-divergence* (biased by the excluded fraction). In our
# runs with the OU target, `"inverse_variance"` weights let 0.5% (θ=1) to about 7% (θ=5) of the
# paths diverge, while `WEIGHTS = "uniform"` gave none.

# %%
t0 = time.time()
tg_mc, W_mc = em.brownian_paths(1.0, N_STEPS, N_MC, SEED_MC)
sig_brownian_1 = ta.TensorSequence(array=jnp.asarray(em.path_signatures(W_mc, tg_mc, N)), trunc=N, dim=2)
del W_mc

gen = em.rollout(final.p, KAPPA, T, N, GEN_STEPS, GEN_SIZE, sig_brownian_1, SEED_GEN, x0=X0)
t_gen = np.linspace(0.0, T, GEN_STEPS + 1)
print(f"generated ({GEN_SIZE} paths x {GEN_STEPS} steps, N_MC={N_MC}): {time.time() - t0:.1f}s")

X_BOUND = 3 * np.abs(X_target - X0).max()           # bound on the excursion |X_t - X_0|
with np.errstate(invalid="ignore"):
    diverged = ~np.all(np.isfinite(gen.X) & (np.abs(gen.X - X0) <= X_BOUND), axis=1)
keep = ~diverged
n_div = int(diverged.sum())
print(f"diverged paths (non-finite or |X - X0| > {X_BOUND:.2f}): {n_div} / {GEN_SIZE} = {n_div / GEN_SIZE:.2%}"
      + ("   -> generated statistics below are conditional on non-divergence" if n_div else ""))
gen_X, gen_control = gen.X[keep], gen.control[keep]
S_gen = gen.sig[:, keep, -1]
n_gen = int(keep.sum())
HAS_GEN = n_gen >= 10            # otherwise the generated law is unusable and is left out of Parts 3-4
if not HAS_GEN:
    print("WARNING: (almost) every generated path diverged: the feedback at p* is not usable. This is "
          "expected when the target lies outside the cone (negative leading residuals in Part 1, note §6.5). "
          "The generated law is left out of the checks and of the validation below.")

if HAS_GEN:
    N_SHOW = 25
    alpha_t, t_a = gen_control[:, :-1], t_gen[:-1]           # the control acts on [t_k, t_{k+1})
    a_lo, a_hi = np.percentile(alpha_t, [0.5, 99.5])
    cmap_xt = LinearSegmentedColormap.from_list("xt", [PALETTE[5], PALETTE[0], PALETTE[2], PALETTE[1]])
    norm_xt = Normalize(*np.percentile(gen_X[:, -1], [5, 95]))
    colors_xt = cmap_xt(norm_xt(gen_X[:N_SHOW, -1]))
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    ax = axes[0]
    for x_, c_ in zip(gen_X[:N_SHOW], colors_xt):
        ax.plot(t_gen, x_, color=c_, lw=1)
    ax.set_title(f"generated paths $X_t$ ({N_SHOW} of {n_gen}), coloured by $X_T$"); ax.set_xlabel("t")
    fig.colorbar(ScalarMappable(norm_xt, cmap_xt), ax=ax, label="$X_T$")

    ax = axes[1]
    x_lo, x_hi = np.percentile(gen_X, [0.5, 99.5])
    for t_eval, c_ in zip((0.1, 0.5, 0.9), (PALETTE[5], PALETTE[0], PALETTE[1])):
        k = int(np.argmin(np.abs(t_a - t_eval)))
        ax.scatter(gen_X[:, k], alpha_t[:, k], s=5, alpha=0.3, color=c_, label=f"t = {t_eval}")
    if target_drift is not None:
        xs = np.linspace(x_lo, x_hi, 200)
        ax.plot(xs, target_drift[1](xs), "--", color="k", lw=1.5, label=target_drift[0])
    ax.axhline(0.0, color="grey", lw=0.6)
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(a_lo, a_hi); ax.legend(fontsize=8, markerscale=3)
    ax.set_title(r"feedback: $\alpha^*_t$ against the state $X_t$"); ax.set_xlabel("$X_t$"); ax.set_ylabel(r"$\alpha^*_t$")
    save_pdf(fig, "paths_feedback")
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
print(f"  policy value at t=0 (independent MC sample) : {gen.value[0, 0]: .5f}")
print(f"  log Z(p*) on the dual sample  (= -V_p*)    : {float(tilt_star.logZ): .5f}")
if HAS_GEN:
    dt_gen = np.diff(t_gen)
    energy_path = 0.5 * np.sum(gen_control[:, :-1] ** 2 * dt_gen, axis=1)
    energy, energy_se = energy_path.mean(), energy_path.std() / np.sqrt(n_gen)
    m_gen, se_gen = em.expected_signature(S_gen)
    pen_gen = empirical_penalty(m_gen, se_gen)
    aux_path = energy_path + KAPPA * (p_star @ S_gen)
    print(f"Value checks (generated quantities on the {n_gen} retained paths)")
    print(f"  auxiliary cost E[1/2∫α² + κ<p*,X_T>]       : {aux_path.mean(): .5f} ± {aux_path.std() / np.sqrt(n_gen):.5f}"
          f"   vs  V_p* = {-float(tilt_star.logZ):.5f}")
    print(f"  energy 1/2 E∫α² (generated)                : {energy: .5f} ± {energy_se:.5f}"
          f"   vs  entropy H(Q*|W) = {pc['entropy']:.5f}")
    print(f"  empirical primal cost (energy + penalty)    : {energy + pen_gen: .5f}"
          f"   vs  J(p*) = {final.J:.5f}")
else:
    energy = energy_se = pen_gen = np.nan
    print("  (no usable generated law: energy and auxiliary-cost checks skipped)")
print(f"  penalty: at Q* (dual sample) {pc['penalty']:.5f} | generated estimate {pen_gen:.5f}"
      f" | resampled estimate {pen_res:.5f}")
print(f"  (the empirical penalty multiplies any bias of m by kappa*lambda, up to {KAPPA * weights[1:].max():.1e} here,\n"
      f"   so at large kappa the energy-vs-entropy and auxiliary-cost lines are the informative checks)")

# %% [markdown]
# ## Part 4 — Validation
#
# The generated law should reproduce the target paths in distribution, and its expected signature
# should match $\mathcal E+\Lambda^{-1}\boldsymbol p^*$, the theoretical residual at finite $\kappa$
# (note §8), up to Monte-Carlo error. Forced-zero words are not matched by design: they carry no dual
# coefficient, so their moments are whatever the tilt produces.

# %%
X_bm = X0 + W_dual[:M_TARGET]                      # uncontrolled BM started at the target's X_0
laws = {"target": (t_grid, X_target)}
if HAS_GEN:
    laws["generated"] = (t_gen, gen_X)
y_lo = min(np.percentile(Xp, 0.5) for _, Xp in laws.values())
y_hi = max(np.percentile(Xp, 99.5) for _, Xp in laws.values())
fig = plt.figure(figsize=(26, 10))
gs = fig.add_gridspec(2, 4)
span = 4 // len(laws)
for j, (name, (tg, Xp)) in enumerate(laws.items()):
    ax = fig.add_subplot(gs[0, j * span:(j + 1) * span])
    ax.plot(tg, Xp[:50].T, color=LAW_COLORS[name], lw=0.6, alpha=0.4)
    for other, (tg_o, X_o) in laws.items():
        mo, so = X_o.mean(axis=0), X_o.std(axis=0)
        ax.plot(tg_o, mo, color=LAW_COLORS[other], lw=2 if other == name else 1,
                ls="-" if other == name else "--", label=f"{other} mean ± std")
        ax.fill_between(tg_o, mo - so, mo + so, color=LAW_COLORS[other], alpha=0.12 if other == name else 0.0)
        ax.plot(tg_o, mo + so, color=LAW_COLORS[other], lw=0.8, ls=":")
        ax.plot(tg_o, mo - so, color=LAW_COLORS[other], lw=0.8, ls=":")
    ax.set_title(f"{name}: 50 paths"); ax.set_xlabel("t"); ax.set_ylim(y_lo, y_hi)
    if j == 0:
        ax.legend(fontsize=8)
grid_y = np.linspace(y_lo, y_hi, 300)
for j, t_eval in enumerate([0.25, 0.5, 1.0]):
    ax = fig.add_subplot(gs[1, j])
    for name, (tg, Xp) in laws.items():
        k = int(np.argmin(np.abs(tg - t_eval)))
        vals = Xp[:, k]
        ax.hist(vals, bins=60, range=(y_lo, y_hi), density=True, alpha=0.25, color=LAW_COLORS[name])
        ax.plot(grid_y, gaussian_kde(vals)(grid_y), color=LAW_COLORS[name], lw=2, label=name)
    ax.set_title(f"marginal of $X_t$ at t = {t_eval}"); ax.legend()
ax = fig.add_subplot(gs[1, 3])
for name, (tg, Xp) in laws.items():
    ax.plot(tg, Xp.std(axis=0), color=LAW_COLORS[name], lw=2, label=name)
ax.set_title("std of $X_t$"); ax.set_xlabel("t"); ax.legend()
save_pdf(fig, "laws")
plt.show()

# %% [markdown]
# ### Autocorrelation
#
# Left: autocorrelation of the process, $\rho(s,t)=\operatorname{Corr}(X_s,X_t)$, as a function of $t$ for
# fixed $s$. Right: autocorrelation of the increments
# $\Delta X_t=X_{t+\Delta}-X_t$ at lag $k\Delta$, $\operatorname{Corr}(\Delta X_t,\Delta X_{t+k\Delta})$, pooled over $t$
# (each time slice is centred across paths, so a time-dependent mean does not count as correlation).
# Both laws are compared on the grid of the generated paths, $\Delta=T/$`GEN_STEPS`; the finer target is
# subsampled. Brownian increments are uncorrelated. For the RL fBM the increment autocorrelation is
# negative for $H<\tfrac12$ and positive for $H>\tfrac12$. The generated law is a Brownian motion with
# drift, with unit volatility. When the target's local volatility $K(0)$ differs from $1$ (singular
# target), the unit-volatility noise dominates its increments at the grid scale, whatever the matched
# signature moments are.

# %%
def on_grid(tg, Xp, t_common):
    """Values of the paths ``Xp`` (grid ``tg``) at the times ``t_common``, a sub-grid of ``tg``."""
    idx = np.clip(np.searchsorted(tg, t_common - 1e-12), 0, tg.size - 1)
    assert np.allclose(tg[idx], t_common), "t_common must be a sub-grid of every law's grid"
    return Xp[:, idx]


def process_autocorr(Xc, k_s):
    """Corr(X_s, X_t) for all t of the grid, s = grid point k_s (NaN where X_t is deterministic)."""
    Xc = Xc - Xc.mean(axis=0)
    sd = Xc.std(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (Xc[:, [k_s]] * Xc).mean(axis=0) / (sd[k_s] * sd)


def increment_acf(Xc, max_lag):
    """Pooled autocorrelation of the grid increments at lags 0..max_lag (slices centred over paths)."""
    d = np.diff(Xc, axis=1)
    d = d - d.mean(axis=0)
    n = d.shape[1]
    return np.array([np.sum(d[:, :n - k] * d[:, k:]) / np.sqrt(np.sum(d[:, :n - k] ** 2) * np.sum(d[:, k:] ** 2))
                     for k in range(max_lag + 1)])


t_common = np.linspace(0.0, T, GEN_STEPS + 1)
dt_common = T / GEN_STEPS
MAX_LAG = GEN_STEPS // 4
on_common = {name: on_grid(tg, Xp, t_common) for name, (tg, Xp) in laws.items()}

fig, axes = plt.subplots(1, 2, figsize=(18, 5))
ax = axes[0]
for s_eval, ls in ((0.25, "--"), (0.5, "-")):
    k_s = int(np.argmin(np.abs(t_common - s_eval)))
    for name, Xc in on_common.items():
        ax.plot(t_common, process_autocorr(Xc, k_s), color=LAW_COLORS[name], ls=ls, lw=2,
                label=f"{name}, s = {s_eval}")
    ax.axvline(t_common[k_s], color="grey", lw=0.6, ls=ls)
ax.set_title(r"process autocorrelation $\mathrm{Corr}(X_s, X_t)$"); ax.set_xlabel("t"); ax.legend(fontsize=8, ncol=2)
ax = axes[1]
lags = dt_common * np.arange(MAX_LAG + 1)
for name, Xc in on_common.items():
    ax.plot(lags[1:], increment_acf(Xc, MAX_LAG)[1:], "o-", ms=3, color=LAW_COLORS[name], lw=1.5, label=name)
ax.axhline(0.0, color="k", lw=0.8)
ax.set_title(rf"increment autocorrelation, $\Delta$ = {dt_common:g}"); ax.set_xlabel("lag"); ax.legend()
save_pdf(fig, "autocorrelation")
plt.show()

# %%
gen_moments = {"generated": (m_gen, se_gen)} if HAS_GEN else {}


def grouped_bars(ax, sel_idx, level_idx, show_labels=True):
    """Grouped bars over the words ``sel_idx``. The y-range is set by target / generated over the
    whole level ``level_idx``, so that the parts of a split level share it."""
    x = np.arange(len(sel_idx))
    series = {"target": (E, E_se), **gen_moments}
    width = 0.84 / len(series)
    shift = (len(series) - 1) / 2
    for j, (name, (m, se_m)) in enumerate(series.items()):
        ax.bar(x + (j - shift) * width, m[sel_idx], width=width, color=LAW_COLORS[name],
               yerr=2 * se_m[sel_idx], error_kw=dict(lw=0.6, capsize=0), label=name)
    vals = np.concatenate([E[level_idx], *[m[level_idx] for m, _ in gen_moments.values()], [0.0]])
    lo, hi = vals.min(), vals.max()
    pad = 0.12 * (hi - lo) if hi > lo else 1.0
    lo, hi = lo - pad, hi + pad
    ax.set_ylim(lo, hi)
    for xi, i in zip(x, sel_idx):
        if cone.forced_mask[i]:
            ax.axvspan(xi - 0.45, xi + 0.45, color=CLASS_COLORS["forced"], alpha=0.12, lw=0)
        elif i in cone.lead_idx:
            ax.axvspan(xi - 0.45, xi + 0.45, color=CLASS_COLORS["lead"], alpha=0.15, lw=0)
    ax.set_xlim(-0.6, len(sel_idx) - 0.4)
    ax.set_xticks(x, word_labels[sel_idx] if show_labels else [""] * len(sel_idx),
                  rotation=90 if len(sel_idx) > 8 else 0, fontsize=7 if len(sel_idx) > 16 else 9)


# Panels on a 4-column grid: levels 1-5 share rows, from level 6 on each panel is a full row of at most 64 words
panels, row, col = [], 0, 0
for k in range(1, N + 1):
    span = 1 if k <= 2 else 2 if k <= 5 else 4
    n_parts = max(2 ** (k - 6), 1)
    for part in range(n_parts):
        if col + span > 4:
            row, col = row + 1, 0
        panels.append((k, part, n_parts, row, col, span))
        col += span

fig = plt.figure(figsize=(26, 4.2 * (row + 1)), layout="constrained")
fig.suptitle("Expected signature per level: error bars = 2 s.e.; shaded words: orange = lead, grey = forced",
             fontsize=13)
gs = fig.add_gridspec(row + 1, 4)
for k, part, n_parts, r, c, span in panels:
    ax = fig.add_subplot(gs[r, c:c + span])
    lvl = np.flatnonzero(levels == k)
    grouped_bars(ax, np.array_split(lvl, n_parts)[part], lvl)
    ax.set_title(f"level {k}" if n_parts == 1 else f"level {k}, part {part + 1}/{n_parts}")
    if k == 1:
        ax.legend()
save_pdf(fig, "expected_signature")
plt.show()

# %%
print("=" * 72)
print("Summary")
print("=" * 72)
print(f"target                 : {TARGET} ({TARGET_KIND}, {TARGET_PARAMS}), M_target = {M_TARGET}")
print(f"truncation / cone      : N = {N}, n = {cone.n}, m = {cone.m}, eps = {cone.eps:g}, weights = {WEIGHTS}")
print(f"final kappa            : {KAPPA:g}  (converged: {final.converged}, {final.n_iter} Newton iterations)")
print(f"J(p*)                  : {final.J:.6f}")
print(f"duality gap            : {final.gap:.2e}")
print(f"entropy H(Q*|W)        : {pc['entropy']:.6f}   (generated energy {energy:.6f} ± {energy_se:.6f})")
print(f"penalty at Q*          : {pc['penalty']:.6f}")
print(f"active leads           : {[word_labels[i] for i in final.active_leads] or 'none'}")
print(f"ESS / M at p*          : {float(tilt_star.ess) / problem.M:.3f}")
print(f"diverged generated paths: {n_div} / {GEN_SIZE} ({n_div / GEN_SIZE:.2%})"
      + ("  -- generated statistics are conditional on non-divergence" if n_div else ""))
print(f"std X_T                : target {X_target[:, -1].std():.3f}, "
      + (f"generated {gen_X[:, -1].std():.3f}, " if HAS_GEN else "")
      + f"resampled {X_resampled[:, -1].std():.3f}, uncontrolled BM {X_bm[:, -1].std():.3f}")

# %%
