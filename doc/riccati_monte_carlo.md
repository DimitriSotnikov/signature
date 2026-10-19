# Monte-Carlo evaluation of signature-feedback controls

This note explains how to use the Monte-Carlo (MC) policies of
[signature/sig_control.py](../signature/sig_control.py) as a benchmark for the Riccati solver,
following the examples of [notebooks/SigControlRiccati.ipynb](../notebooks/SigControlRiccati.ipynb).
For the theory (recentering, shift maps, the control formula) see
[riccati_signature.md](riccati_signature.md).

**Main rule: precompute a batch of Brownian signatures once, before the rollout.** The sample is
reused at every time step and for every controlled path; it is never regenerated inside the loop.

---

## 1. Idea

The closed loop $dX_t = \alpha_t\,dt + \sigma\,dW_t$ is rolled out by `sc.simulate`, which only
asks a *policy* for $(\alpha_t, V_t)$ given the current signature $\widehat{\mathbb X}_t$. The MC
policies evaluate this feedback without solving the Riccati ODE, using Chen's identity
$\widehat{\mathbb X}_T = \widehat{\mathbb X}_t \otimes \widehat{\mathbb W}_{t,T}$:

$$
V_t = \log \mathbb E\big[\exp\langle \boldsymbol p, \widehat{\mathbb{X}}_t \otimes \widehat{\mathbb W}_{t,T}\rangle\big],
\qquad
\alpha_t = \frac{\mathbb E\big[e^{\langle \boldsymbol p, \widehat{\mathbb X}_t \otimes \widehat{\mathbb W}_{t,T}\rangle}
           \langle \boldsymbol p, \widehat{\mathbb X}_t \otimes \mathbf 2 \otimes \widehat{\mathbb W}_{t,T}\rangle\big]}{e^{V_t}} .
$$

The pairing $\langle \boldsymbol p, \widehat{\mathbb X}_t \otimes \cdot\,\rangle = \langle {}_{\widehat{\mathbb X}_t}|\boldsymbol p, \cdot\,\rangle$
is a shift of $\boldsymbol p$ by the current signature, so the only random input is the
**future Brownian signature** $\widehat{\mathbb W}_{t,T}$. Its law does not depend on the
controlled path, hence one fixed sample of it serves all paths and all times. Expectations are
sample means with a log-sum-exp stabilisation.

Since $\boldsymbol p$ has finite degree, the sample only needs to be truncated at
$\deg \boldsymbol p$ — this is exact, not an approximation (the shifted functional never has
higher degree). Higher truncation only costs memory.

## 2. Two policies, two kinds of precomputed sample

| policy | running cost $q$ | precomputed sample | shape |
|--------|------------------|--------------------|-------|
| `monte_carlo_policy` | time-independent, folded into $\boldsymbol p = q \otimes \mathbf 1 + r$ | terminal signature $\widehat{\mathbb W}_{0,1}$ on $[0,1]$ | `(n_sig_mc, N_MC)` |
| `monte_carlo_source_policy` | time-dependent $q_s$ | signature **trajectory** $\widehat{\mathbb W}_{0,t_k}$ on `t_grid` | `(n_sig_mc, n_dw, N_MC)` |

### 2.1 Constant running cost: one terminal sample + Brownian scaling

Only $\widehat{\mathbb W}_{t,T}$ is needed, and by Brownian scaling
$\widehat{\mathbb W}_{t,T} \overset{d}{=} \delta_{(h,\sqrt h)}\widehat{\mathbb W}_{0,1}$ with
$h = T - t$. So a single sample on $[0,1]$ is drawn once, and the policy rescales it at each step
with `dilation` (cheap: one multiplication per word).

There is no library helper for this sample; the notebook (Example 1) builds it with
`iisignature` on a fine grid:

```python
p_deg = int(p.get_lengths_array()[np.abs(np.array(p.array)) > 1e-12].max())
N_MC, n_fine = 500_000, 80
rng_mc = np.random.default_rng(1); u = np.linspace(0.0, 1.0, n_fine)
W1 = np.cumsum(rng_mc.normal(size=(N_MC, n_fine - 1)) * np.sqrt(1.0 / (n_fine - 1)), axis=1)
paths = np.zeros((N_MC, n_fine, dim)); paths[:, 1:, 0] = u[1:][None, :]; paths[:, 1:, 1] = W1
SigW1 = ta.from_array(jnp.asarray(np.vstack([np.ones((1, N_MC)), iisignature.sig(paths, p_deg).T])),
                      trunc=p_deg, dim=dim)

policy = sc.monte_carlo_policy(p, SigW1, T, trunc, dim)
res    = sc.simulate(dW, t_grid, policy, trunc, dim)
```

- `iisignature` returns the signature of the piecewise-linear interpolation, i.e. a
  Stratonovich signature, consistent with the generator
  $\mathscr R\psi = \psi|_1 + \tfrac12\psi|_{22} + \tfrac12(\psi|_2)^{ш2}$. `n_fine` controls the
  discretisation bias, `N_MC` the statistical error.
- The empty word (row 0, value 1) must be prepended by hand: `iisignature` omits it.
- The sample is for $\sigma = 1$. For another $\sigma$, pass
  `scaling=lambda h: jnp.array([h, sigma * jnp.sqrt(h)])`, and the same `sigma` to `simulate`.
- The value returned is the full value process: the running cost is already contained in
  $\boldsymbol p$ through the time letter. This identity only holds for a **constant** $q$.

### 2.2 Time-dependent running cost: a signature trajectory

With $q_s$ depending on time, the exponent is
$\langle r, \widehat{\mathbb X}_t\otimes\widehat{\mathbb W}_{t,T}\rangle + \int_t^T \langle q_s, \widehat{\mathbb X}_t\otimes\widehat{\mathbb W}_{t,s}\rangle\,ds$,
which needs the whole future trajectory $s \mapsto \widehat{\mathbb W}_{t,s}$. By stationary
increments $\widehat{\mathbb W}_{t,s} \overset{d}{=} \widehat{\mathbb W}_{0,s-t}$, so one trajectory
from $0$, computed on the simulation grid, gives every future window as a slice (no dilation).
It is built by `brownian_signature_trajectory` (Example 2):

```python
deg  = max(sc._max_degree(r), sc._max_degree(q_traj))
traj = sc.brownian_signature_trajectory(t_grid, N_MC=50_000, trunc_mc=deg, dim=dim, seed=7)

policy = sc.monte_carlo_source_policy(r, q_traj, traj, t_grid, dim=dim)
res    = sc.simulate(dW, t_grid, policy, trunc, dim)
```

- `t_grid` must be **uniform** and the same grid as the rollout, so that $s - t$ falls on grid
  points; the $ds$-integral is a trapezoid on that grid.
- `trunc_mc` must be at least $\max(\deg r, \deg q)$; the policy checks this and raises otherwise.
- `brownian_signature_trajectory` takes a `sigma` argument; use the same value as in `simulate`.
- The incurred cost $\int_0^t \langle q_s, \widehat{\mathbb X}_s\rangle ds$ is added inside the
  policy (stateful accumulator), so create **one policy per rollout**.

## 3. Practical points

- **Precompute first, then simulate.** Building the sample (thousands to hundreds of thousands of
  `iisignature` calls) is the expensive step; the policy itself is a jitted pair of matrix
  products per step. Keep the sample and reuse it across rollouts and parameter sweeps that
  share $\deg\boldsymbol p$ and $T$.
- **Use independent noise.** The MC sample represents the *future* noise and must be independent
  of the increments `dW` driving the closed loop — use a different seed. Reusing the same sample
  at every step (common random numbers) gives a control that is smooth in $t$.
- **Memory.** The terminal sample costs `n_sig_mc × N_MC` floats (e.g. $\deg\boldsymbol p = 5$,
  `dim = 2`: 63 words × 500 000 ≈ 250 MB). The trajectory adds a factor `n_dw`
  (31 words × 300 steps × 50 000 ≈ 3.7 GB), which is why Example 2 uses a smaller `N_MC`.
  Inside a step the policy also forms `(size, N_MC)` arrays, so keep `size` moderate.
- **Benchmark role.** MC has no radius-of-convergence constraint, so it serves as the reference
  in the notebook: `report(runs, show)` prints $\max|\alpha - \alpha_{\rm MC}|$ for the Riccati
  runs with and without recentering.
