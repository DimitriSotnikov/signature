# Spec — Expected-signature matching MFC: dual optimisation over $\mathcal B\cap T^N$

**Theory:** [doc/expected_signature_matching.md](../doc/expected_signature_matching.md) (below: "the note").
**Deliverables:**

| file | content |
|------|---------|
| `signature/esig_mfc/__init__.py` | re-exports the public API of `esig_mfc.py` |
| `signature/esig_mfc/esig_mfc.py` | self-contained module: cone $\mathcal B\cap T^N$, Monte-Carlo dual problem, optimiser, trajectory generation |
| `tests/test_esig_mfc.py` | unit tests (§6) |
| `notebooks/Esig-MFC.py` | notebook in percent format (`# %%` cells, runnable in VS Code / Jupytext), four parts (§7) |

The module depends only on the existing package. The value function and optimal feedback reuse
[signature/sig_control.py](../signature/sig_control.py) (`monte_carlo_policy`, `simulate`); no Riccati
solver is required for v1.

---

## 1. Conventions

- **Letters.** As in the code base: `1` = time, `2` = state. Paper word $\mathbf 1^{\otimes k}$ (state) is
  code word `22…2`, and the integrated word $\mathbf 1^{\otimes 2m}\mathbf 0$ is code word `22…21`.
  $|v|_2$ := number of state letters of a code word.
- **Arrays.** A tensor in $T^N$ is stored as a full coefficient array of length
  `n_sig = number_of_words_up_to_trunc(N, 2)` $=2^{N+1}-1$ (255 for $N=7$), in the package word
  order (identical to `iisignature` order after prepending the empty word). The empty-word entry
  of `p` is always `0`; the empty-word entry of signatures is `1`.
- **Signatures.** Stratonovich signature of the piecewise-linear interpolation (`iisignature.sig`),
  consistent with the rollout of `sc.simulate` and with `monte_carlo_policy`.
- **Precision.** JAX with `jax_enable_x64=True` (set by the notebook and the tests; the module
  must not toggle global config itself, but must document the requirement).
- **Sign convention towards `sig_control`.** `sig_control` maximises:
  $\psi=\log\mathbb E e^{\langle\boldsymbol q,\cdot\rangle}$, $\alpha=+\partial\psi$. Our inner
  problem minimises, so the tensor passed to `sig_control` is always
  $$\boldsymbol q=-\kappa\,\boldsymbol p,\qquad V_{\boldsymbol p}=-\psi_0 .$$
  (Note §4.3.)

---

## 2. Mathematical problem implemented

### 2.1 Weighted primal (generalises the note, which is $\lambda\equiv1$)

$$
V=\inf_\alpha\ \mathbb E\Big[\tfrac12\int_0^T\alpha_t^2dt\Big]+\frac\kappa2\sum_{v\in\mathcal W_N}\lambda_v\big(\mathbb E[\widehat{\mathbb X}^v_T]-\mathcal E^v\big)^2,
\qquad dX=\alpha\,dt+dW,\ X_0=0 .
$$

Positive word weights $\lambda_v$ are needed in practice: level-7 coordinates are $O(10^{-4})$, so
with $\lambda\equiv1$ they are effectively unpenalised. Options: `"uniform"` ($\lambda_v=1$, the note) and
`"inverse_variance"` ($\lambda_v=1/\mathrm{Var}_{\text{target}}(\widehat{\mathbb X}^v_T)$, i.e. a
$\chi^2$-type penalty, estimated in Part 1). Weights do not affect the cone, since the sign of $\boldsymbol p$ is
the sign of the residual.

### 2.2 Dual (sample-average approximation)

Given a sample $S=(\widehat{\mathbb W}^{(i)}_T)_{i\le M}$ of Brownian signatures, write
$\ell_i=-\kappa\langle\boldsymbol p,\widehat{\mathbb W}^{(i)}_T\rangle$,
$w_i=\mathrm{softmax}(\ell)_i$, $\Lambda=\mathrm{diag}(\lambda)$:

$$
\begin{aligned}
\log\widehat Z(\boldsymbol p)&=\mathrm{logsumexp}(\ell)-\log M,\\
\widehat J(\boldsymbol p)&=-\tfrac\kappa2\,\boldsymbol p^\top\Lambda^{-1}\boldsymbol p-\kappa\langle\boldsymbol p,\mathcal E\rangle-\log\widehat Z(\boldsymbol p),\\
\widehat m(\boldsymbol p)&=\textstyle\sum_i w_i\widehat{\mathbb W}^{(i)}_T,\qquad
\widehat C(\boldsymbol p)=\sum_iw_i(\widehat{\mathbb W}^{(i)}_T-\widehat m)(\cdot)^\top,\\
\nabla\widehat J&=\kappa\big(\widehat m-\mathcal E-\Lambda^{-1}\boldsymbol p\big),\qquad
\nabla^2\widehat J=-\kappa\Lambda^{-1}-\kappa^2\widehat C .
\end{aligned}
$$

These formulas are *exact* derivatives of $\widehat J$ (testable against autodiff, §6).

**SAA primal and duality gap.** On the empirical reference measure (uniform on the $M$ samples)
the Gibbs variational principle holds exactly, so for the tilt weights $w(\boldsymbol p)$

$$
\widehat P(\boldsymbol p):=\underbrace{\textstyle\sum_iw_i\log(Mw_i)}_{\widehat H(Q_{\boldsymbol p}|\mathbb W)}
+\frac\kappa2\sum_v\lambda_v\big(\widehat m^v-\mathcal E^v\big)^2\ \ge\ \widehat J(\boldsymbol p'),\quad\forall\boldsymbol p'
$$

and $\mathrm{gap}(\boldsymbol p)=\widehat P(\boldsymbol p)-\widehat J(\boldsymbol p)\ge0$ vanishes at the
unconstrained optimum (note §5.3). For the cone-restricted problem the gap must be computed
with the **modified primal** of §2.4.

### 2.3 Feasible set $K=\mathcal B\cap T^N$ (note §6.2–6.3)

Built by `make_b_cone(N, n=None, m=None, eps=1e-6)`:

- Defaults: $n=\lfloor N/2\rfloor$, $m=\lfloor (N-1)/2\rfloor$ (largest leading words fitting in
  $\mathcal W_N$). For $N=7$: $n=m=3$, leading words `222222` and `2222221`. Either may be
  `None` (absent); `n=None` is forced if $N<2$, `m=None` if $N<3$.
- **Free words** (unconstrained coefficient):
  - Gaussian words: $|v|_2\le1$ (note §6.2 item 4, $D_{\mathcal B}+D\subset D$);
  - if $n$: pure state words `2…2` with $k<2n$ letters ($\boldsymbol q$);
  - if $m$: words `w1` ending in the time letter with $|w|_2<2m$ ($\boldsymbol r\mathbf 0$);
  - if $n$ and $m$: every word with $|v|_2<2\min(m,n)$ ($\boldsymbol s$).
- **Leading words:** coefficient $\ge\varepsilon$ (box lower bound). $\varepsilon>0$ keeps the
  mixed $\boldsymbol s$-part legitimate (it requires $\alpha\beta>0$).
- **Forced-zero words:** all others. For $N=7$, $n=m=3$ these are `2222222` and the six words with
  six `2`s and one `1` other than `2222221` (7 words in total). Words with $|v|_2=2$ are free here
  because $2\min(m,n)=6>2$.
- `project(p)`: zero the forced words and the empty word, clip leading words at $\varepsilon$. It is
  the Euclidean projection onto the closed convex set $K_\varepsilon$.

`ConeSpec` (frozen dataclass): `trunc, dim, n, m, eps, lead_idx (tuple[int]), free_mask (bool[n_sig]),
forced_mask (bool[n_sig]), lower (float[n_sig], eps on leads, -inf elsewhere)`, plus
`words (int[n_sig])` and `describe() -> str` (human-readable summary of the three word classes).

### 2.4 Exact meaning of the restricted dual

The restriction is not just a heuristic. With $\sup_{p\ge\varepsilon}(pz-\tfrac{p^2}{2\lambda})$ and
$p\equiv0$ on forced words, $\sup_{\boldsymbol p\in K_\varepsilon}\widehat J$ is the exact dual of the
**modified primal**:
- the penalty is dropped on forced-zero words;
- on leading words, the two-sided penalty $\tfrac\lambda2z^2$ is replaced by the one-sided
  $\varepsilon z-\tfrac{\varepsilon^2}{2\lambda}$ for $z\le\varepsilon/\lambda$, and by $\tfrac\lambda2z^2$
  above. So only overshoot $z=m^v-\mathcal E^v>0$ is penalised quadratically.

`DualProblem.primal_cost(p)` implements this modified penalty, so that the gap is $\ge0$ and
$\to0$ at the constrained optimum. The notebook reports which leading constraints are active:
an active constraint means the target asks for heavier top moments than the cone allows (note §6.5).

---

## 3. Module API — `signature/esig_mfc/esig_mfc.py`

All public functions have type hints and reST docstrings in the style of `sig_control.py`.

### 3.1 Sampling and signatures

```python
def path_signatures(paths: np.ndarray, t_grid: np.ndarray, trunc: int) -> np.ndarray
    # paths (M, n_t) state values, X_0 = paths[:,0]; time-augment with t_grid, iisignature.sig,
    # prepend empty-word row -> (n_sig, M). Translation invariant: no need to subtract X_0.

def brownian_paths(T: float, n_steps: int, M: int, seed: int) -> tuple[np.ndarray, np.ndarray]
    # (t_grid (n_steps+1,), W (M, n_steps+1)), W_0 = 0

def target_paths(kind: str, T: float, n_steps: int, M: int, seed: int, **params) -> tuple[t_grid, X]
    # kind="ou":  dX = theta (mu - X) dt + sigma dW, X_0 = x0 (exact OU transition, not Euler)
    #             defaults theta=1.0, mu=0.5, sigma=1.0, x0=0.0
    # kind="w2":  X_t = W_t**2 (stress test, see §7 Part 1)

def expected_signature(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]
    # (mean, standard error) over the sample axis, shapes (n_sig,), (n_sig,)

def word_weights(kind: str, S_target: np.ndarray | None, cone: ConeSpec, floor: float = 1e-12) -> np.ndarray
    # "uniform" -> ones; "inverse_variance" -> 1/max(var, floor); empty word weight irrelevant (set 1)
```

`target_paths("ou")` with `sigma=1` keeps the target law absolutely continuous w.r.t. Wiener
measure (the OU drift is an admissible control), so the problem is feasible. Its moments are
lighter than Brownian ones, so the leading residuals at $\boldsymbol p=0$ are positive: inside the cone.

### 3.2 Cone

```python
def make_b_cone(trunc: int, n: int | None = "auto", m: int | None = "auto", eps: float = 1e-6,
                dim: int = 2, time_letter: int = 1, state_letter: int = 2) -> ConeSpec
def project(p: jax.Array, cone: ConeSpec) -> jax.Array
def in_cone(p: jax.Array, cone: ConeSpec, tol: float = 0.0) -> bool
```

### 3.3 Dual problem

```python
class DualProblem:
    def __init__(self, S: jax.Array, E: jax.Array, kappa: float, cone: ConeSpec,
                 weights: jax.Array | None = None): ...
    # S: (n_sig, M) Brownian signatures at T; E: (n_sig,) target (E[0] ignored)
    def tilt(self, p) -> Tilt          # NamedTuple(logZ, w (M,), m (n_sig,), ess)  — jitted
    def value(self, p) -> float        # J-hat
    def grad(self, p) -> jax.Array     # closed form, zero on forced words and empty word
    def hessian(self, p) -> jax.Array  # (n_free, n_free) on cone.free|lead coordinates
    def primal_cost(self, p) -> dict   # {"entropy", "penalty", "total"} — modified primal §2.4
    def duality_gap(self, p) -> float
```

Implementation notes:
- Restrict all linear algebra to the *active* coordinates `idx = free ∪ lead` (dropping forced
  words and the empty word). For $N=7$ that is 247 coordinates.
- `ess = 1 / sum(w**2)` (report also `ess / M`).
- `S` is kept on device as `float64`; memory $\approx 8\cdot n_{sig}\cdot M$ bytes (200 MB for
  $M=10^5$, $N=7$).

### 3.4 Optimiser

```python
@dataclass
class DualResult:
    p: jax.Array; J: float; gap: float; converged: bool; n_iter: int
    active_leads: tuple[int, ...]          # leading words sitting at eps
    history: list[dict]                    # per iter: J, grad_norm (projected), ess_frac, step, gap

def optimize_dual(problem: DualProblem, p0: jax.Array | None = None, method: str = "newton",
                  max_iter: int = 200, tol: float = 1e-8, step0: float = 1.0,
                  armijo: float = 1e-4, backtrack: float = 0.5, min_ess_frac: float = 1e-3,
                  verbose: bool = False) -> DualResult
```

- `p0=None` means `project(0)`, i.e. leads at $\varepsilon$.
- **`method="gradient"`:** projected ascent $\boldsymbol p\leftarrow\Pi_K(\boldsymbol p+t\,\Lambda\nabla\widehat J/\kappa)$.
  With $t=\eta$ this is the damped fixed point of note §7, written with $\Lambda$. $t$ comes from
  Armijo backtracking on $\widehat J$.
- **`method="newton"`** (default): projected Newton with a box on the leading coordinates (Bertsekas).
  1. Find the binding set $B=\{$lead at $\varepsilon$ with $\partial_v\widehat J<0\}$.
  2. Take the Newton direction $d=-H_{FF}^{-1}g_F$ on the free set $F=\mathrm{idx}\setminus B$, with $d_B=0$.
  3. Line search on $\Pi_K(\boldsymbol p+td)$ with Armijo.

  $-H\succeq\kappa\Lambda^{-1}$, so use a Cholesky solve of $-H_{FF}$.
- **Stopping:** the projected-gradient norm $\|\boldsymbol p-\Pi_K(\boldsymbol p+\Lambda\nabla\widehat J/\kappa)\|_\infty\le$ `tol`.
- **Safeguard:** reject a step whose ESS fraction falls below `min_ess_frac`; halve the step instead.
  If no acceptable step is found, return with `converged=False`.

```python
def continuation(S, E, cone, kappas: Sequence[float], weights=None, **opt_kwargs) -> list[DualResult]
    # warm-started sequence kappa_1 < kappa_2 < ...; p* of the previous kappa is the next p0
    # (p* is a residual, so it needs no rescaling across kappa)
```

### 3.5 Optimal trajectories (Part 3)

```python
def control_tensor(p: jax.Array, kappa: float, trunc: int, dim: int = 2) -> TensorSequence
    # q = -kappa * p as a TensorSequence (empty word 0)

def optimal_policy(p, kappa, sig_brownian_1: TensorSequence, T: float, trunc: int) -> sc.Policy
    # sc.monte_carlo_policy(control_tensor(p, kappa, trunc), sig_brownian_1, T, trunc, dim=2)

def rollout(p, kappa, T, trunc, n_steps: int, size: int, sig_brownian_1, seed: int) -> sc.ControlResult
    # dW ~ N(0, dt), uniform grid, sc.simulate(dW, t_grid, policy, trunc, dim=2, sigma=1, x0=0)

def resample(problem: DualProblem, p, paths: np.ndarray, n: int, seed: int) -> np.ndarray
    # multinomial resampling of the Brownian paths behind problem.S with the tilt weights:
    # exact samples of Q_p up to MC error (note §8, item 2). Requires the same ordering as S.
```

- `sig_brownian_1` is an **independent** Brownian signature sample on $[0,1]$ at level $N$ (empty
  row prepended), as in `riccati_monte_carlo.md` §2.1. Using the optimisation sample would bias
  the validation.
- Cost of `rollout` per step is $\approx 2\cdot\text{size}\cdot N_{MC}\cdot n_{sig}$ flops.
  Defaults: `size=2000`, `N_MC=2·10^4`, `n_steps=100`, about $10^{12}$ flops in total (minutes on CPU).
  Expose all three sizes as parameters.

---

## 4. Numerical defaults (notebook)

| parameter | value | comment |
|-----------|-------|---------|
| $N$ | 7 | `n_sig = 255`, active coordinates 247 |
| $T$ | 1.0 | |
| grid | 200 steps | same grid for target, Brownian dual sample and resampled paths (same discretisation bias) |
| target | OU, $\theta=1,\mu=0.5,\sigma=1,x_0=0$ | `M_target = 2·10^4` (≥ 10^4 as required) |
| dual sample | `M = 10^5` Brownian paths | paths kept for resampling |
| weights | `"inverse_variance"` (switchable to `"uniform"`) | |
| $\kappa$ | continuation `[1, 10, 100, 1000]` | |
| cone | $n=m=3$, $\varepsilon=10^{-6}$ | |
| rollout | `size=2000`, `N_MC=2·10^4`, `n_steps=100` | |

---

## 5. Error budget and diagnostics to display

- Target estimation error: standard error `E_se` from Part 1, per word.
- Dual SAA error: compare $\widehat m(\boldsymbol p^*)$ on the optimisation sample with the same
  quantity on an independent Brownian sample (out-of-sample check).
- ESS fraction along the optimisation; gap $\widehat P-\widehat J$ at the optimum.
- Rollout error: Euler step plus MC error of the policy (`N_MC`). The rollout's expected signature
  should match $\mathcal E+\Lambda^{-1}\boldsymbol p^*$ (the theoretical residual), **not** $\mathcal E$.

---

## 6. Tests — `tests/test_esig_mfc.py`

`unittest`, `jax_enable_x64`, small sizes, fixed seeds. Tests tagged *(slow)* are skipped unless
`RUN_SLOW=1`.

| # | name | what it checks | tolerance |
|---|------|----------------|-----------|
| 1 | `test_cone_bookkeeping` | $N=7$: leads are `222222`, `2222221`; forced set is exactly the 7 words of §2.3; free count 245. $N=1$: no leads, everything free. $N=2$: lead `22`, `m=None`. | exact |
| 2 | `test_project` | `project` is idempotent; its output is `in_cone`; it is the identity on cone points; forced and empty entries are 0. | exact |
| 3 | `test_pathwise_upper_bound` | Prop. 4.3 / note §6.4. For random $\boldsymbol p\in K$ with leads $\ge0.1$, evaluate $\langle-\boldsymbol p,\widehat{\mathbb X}\rangle$ on linear paths $(t,ct)$ (exact signature $\exp^{\otimes}(T e_1+cTe_2)$) for $c\in[-50,50]$: it is bounded above and $\to-\infty$ as $\lvert c\rvert\to\infty$. Flipping a lead's sign makes it $\to+\infty$. | qualitative (monotone tail) |
| 4 | `test_zero_tilt_matches_bm_expectation` | At $\boldsymbol p=0$: $\log\widehat Z=0$, $w$ uniform, and $\widehat m$ ≈ `expected_bm_signature(T, N)`. | $\lvert z\text{-score}\rvert<5$ per word |
| 5 | `test_gaussian_closed_forms` | Tilt on word `2` only: $\log Z=\kappa^2p^2T/2$, $m^{2}=-\kappa pT$. Tilt on `22`: $\log Z=-\tfrac12\log(1+\kappa pT)$. | MC, 5 s.e. |
| 6 | `test_grad_hessian_autodiff` | `grad`, `hessian` vs `jax.grad`/`jax.hessian` of `value` on the same sample (restricted to active coords). | rtol 1e-8 |
| 7 | `test_concavity` | $t\mapsto\widehat J(\boldsymbol p+t\boldsymbol d)$ has non-positive second differences for random $\boldsymbol p,\boldsymbol d$. | 1e-10 |
| 8 | `test_weak_duality_saa` | $\widehat P(\boldsymbol p)\ge\widehat J(\boldsymbol p')$ for random cone points $\boldsymbol p,\boldsymbol p'$. | exact inequality |
| 9 | `test_planted_solution` | Pick $\boldsymbol p_{\rm true}\in K$ (leads $>\varepsilon$). Set $\mathcal E:=\widehat m(\boldsymbol p_{\rm true})-\Lambda^{-1}\boldsymbol p_{\rm true}$ on the sample. Then `optimize_dual` (both methods) from 0 recovers $\boldsymbol p_{\rm true}$, gap ≈ 0, `converged`. | rel 1e-6 (newton), 1e-4 (gradient) |
| 10 | `test_active_constraint` | Planted problem with a lead whose stationary value is $<\varepsilon$. The optimiser stops with that lead at $\varepsilon$, reports it in `active_leads`, its partial derivative is $\le0$, and the modified-primal gap is ≈ 0. | 1e-6 |
| 11 | `test_closed_form_mfc_N1` | $N=1$, $\mathcal E=(T, e)$, uniform weights. The note's problem is solvable by hand: $p^*_{2}=-e/(1+\kappa T)$, $V=\kappa e^2/(2(1+\kappa T))$, optimal control constant $a=\kappa e/(1+\kappa T)$. Check `optimize_dual` and `optimal_policy(...)(i, t, X_sig)` at a few states. | MC, 5 s.e. |
| 12 | `test_policy_value_consistency` | With `sig_brownian_1 = problem.S` ($T=1$), the policy value at $t=0$, $X=\varnothing$ equals $-\widehat V_{\boldsymbol p}=\log\widehat Z(\boldsymbol p)$. This pins the sign convention $\boldsymbol q=-\kappa\boldsymbol p$. | 1e-10 |
| 13 | `test_resample_moments` | The mean signature of resampled paths ≈ $\widehat m(\boldsymbol p)$. | 5 s.e. |
| 14 | *(slow)* `test_end_to_end_ou_small` | $N=3$, $\kappa=100$, OU target, small sizes. The rollout's mean signature ≈ $\mathcal E+\Lambda^{-1}\boldsymbol p^*$, and $\tfrac12\mathbb E\int\alpha^2$ + penalty ≈ $\widehat J(\boldsymbol p^*)$. | 5 s.e. / 5 % |

Tests 9–12 are the core correctness checks. 9–10 validate the optimiser exactly on the SAA
problem. 11 validates the theory, including the Boué–Dupuis log and the sign. 12 validates the
interface to `sig_control`.

---

## 7. Notebook — `notebooks/Esig-MFC.py`

Percent format: `# %% [markdown]` for text cells and `# %%` for code cells. Each part starts with a
markdown header and a short recap of the corresponding note section. Cell 0 contains the imports,
`jax.config.update("jax_enable_x64", True)`, the parameter block of §4, and the seeds. No cell
depends on state created by a later cell.

**Part 1 — Target expected signature**
1. Simulate `M_target = 2·10^4` target paths (`TARGET = "ou"`; `"w2"` is available as a stress
   test). Plot 50 sample paths and the mean ± std band.
2. `S_target = path_signatures(...)`, then `E, E_se = expected_signature(S_target)`.
3. Show a sanity table: level-1 words vs closed form ($\mathbb E X_T$ for OU), and E per level.
4. Compute the word weights and build the cone. Print `cone.describe()`.
5. Report the leading residual at $\boldsymbol p=0$: its sign predicts whether the target is inside the cone.

**Part 2 — Dual optimisation for $\boldsymbol p^*$**
1. Simulate `M = 10^5` Brownian paths on the same grid and compute `S`. Keep the paths.
2. Run `continuation` over $\kappa$ (Newton) and plot per $\kappa$: $\widehat J$ by iteration,
   projected-gradient norm (log scale), ESS fraction, and gap.
3. Show the final $\boldsymbol p^*$ as bars per word, coloured by class (free / lead / forced), and
   list the active leads.
4. Out-of-sample check: $\widehat m(\boldsymbol p^*)$ on an independent Brownian sample.

**Part 3 — Optimal trajectories at $\boldsymbol p^*$**
1. Build `sig_brownian_1` (independent, $N_{MC}$ samples, level $N$).
2. Closed loop: `res = rollout(...)`, then plot sample paths and the control $\alpha_t$ (mean ± std).
3. Resampling: draw `resample(problem, p*, W_paths, n=M_target)`.
4. Value check: the policy value at $t=0$ vs $-\widehat V_{\boldsymbol p^*}$; the empirical primal cost
   vs $\widehat J(\boldsymbol p^*)$.

**Part 4 — Validation**
1. **Paths.** Side by side (shared axes), for target, rollout and resampled: 50 paths each,
   mean ± std bands overlaid, and marginal histograms/KDEs at $t\in\{0.25,0.5,1\}$.
2. **Expected signature, quantitative.** A per-level table for each generated law:
   - $\|m-\mathcal E\|_2/\|\mathcal E\|_2$;
   - $\max_v\lvert m^v-\mathcal E^v\rvert$;
   - the median $z$-score $\lvert m^v-\mathcal E^v\rvert/\sqrt{se_m^2+se_{\mathcal E}^2}$;
   - the same against the predicted $\mathcal E+\Lambda^{-1}\boldsymbol p^*$.

   Mark forced-zero words separately: they are not matched by design.
3. **Expected signature, visual.** One subplot per level $1..N$: grouped bars (target, rollout,
   resampled) with error bars at 2 s.e. and x tick labels = words `v` (code letters; rotate for
   levels ≥ 5; for levels 6–7 with 64/128 words, use a narrow bar width or split into two rows).
   Add a parity scatter of $m^v$ vs $\mathcal E^v$ over all words, log–log on absolute values and
   coloured by level.
4. Closing summary cell: final $\kappa$, $\widehat J(\boldsymbol p^*)$, gap, active leads, entropy
   $\widehat H$, and the levels where the match is within MC error.

---

## 8. Out of scope for v1 / follow-ups

- An exact $Z$ and $\nabla Z$ via the signature Riccati equation (instead of SAA).
- `riccati_policy` / `simulate_with_recentering` as an alternative to the MC feedback.
- $\sigma\neq1$ in the controlled dynamics (requires `scaling` in `monte_carlo_policy` and a
  matching dual sample).
- The note currently treats $\lambda\equiv1$ and does not state §2.4. Update the note (§5–6) once
  the implementation confirms it.
