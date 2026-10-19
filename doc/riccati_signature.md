# The signature Riccati equation: theory and implementation

This document explains how the code in [signature/ode_integration.py](../signature/ode_integration.py)
and [signature/projection.py](../signature/projection.py) solves the **infinite-dimensional
Riccati equation on the tensor algebra**, and how that connects to the two companion papers:

- **Laplace paper** — *Fourier–Laplace Transforms of the Brownian Signature via Riccati
  Equations on the Tensor Algebra*,
  sections 4–5.
- **Control paper** — *Stochastic Control with Signatures via Riccati Equations on the
  Tensor Algebra*, sections 4–5.

The reference notebook is [notebooks/SigRiccati.ipynb](../notebooks/SigRiccati.ipynb).

---

## 1. What problem is being solved

We want the conditional Fourier–Laplace transform of the time-augmented Brownian signature,
and especially its logarithm:

$$
\mathbb{E}\!\left[\exp\langle \boldsymbol p, \widehat{\mathbb W}_T\rangle \mid \mathcal F_t\right]
   = \langle \boldsymbol u_t, \widehat{\mathbb W}_t\rangle
   = \exp\big(\langle \boldsymbol\psi_t, \widehat{\mathbb W}_t\rangle\big),
$$

where $\widehat{\mathbb W}$ is the signature of the time-augmented Brownian motion
$\widehat W_t = (t, W_t)$, and $\boldsymbol p \in T((\mathbb C^2))$ is a tensor sequence with
finitely many non-zero coefficients (a "signature polynomial" in the path).

Two deterministic families of tensor-algebra coefficients appear:

| object | role | equation |
|--------|------|----------|
| $\boldsymbol u_t$ | coefficients of the transform itself | **linear** equation |
| $\boldsymbol\psi_t = \log^{\text{ш}}\boldsymbol u_t$ | coefficients of its logarithm | **Riccati** equation |

The code solves the **Riccati** equation for $\boldsymbol\psi_t$, because — as explained below —
the linear equation has a closed-form solution that **diverges** for the terminal conditions of
interest, while the Riccati formulation is numerically stable.

In the Control paper the same $\boldsymbol\psi$ is the **value process** of a path-dependent
stochastic control problem, and its left shift $\boldsymbol\psi|_1$ gives the **optimal feedback
control** in closed loop.

---

## 2. Mathematical background

### 2.1 Tensor algebra and linear functionals

Words index iterated integrals of the path coordinates. A **tensor sequence**
$\boldsymbol\ell$ is a family of coefficients $(\boldsymbol\ell^v)_v$ indexed by words $v$, paired
with a signature $\mathbb X$ through the linear functional
$\langle\boldsymbol\ell,\mathbb X\rangle = \sum_v \boldsymbol\ell^v \mathbb X^v$.

> **Letter convention — papers vs. code.** The papers number the letters of the time-augmented
> Brownian motion $\widehat W=(t,W_t)$ as $\{0,1\}$, with `0` the time coordinate and `1` the
> Brownian coordinate, so the natural space is $T((\mathbb C^2))$.
> **In the code, letters are integers numbered from `1`** — there is no letter `0`. A word is
> literally an integer whose digits are letters (e.g. `11`, `1212`), and letter `k` is the `k`-th
> coordinate of the path (the `k`-th column passed to the signature routine). So:
> - the **1-D power-series example** in the notebook uses `dim = 1`: a single coordinate (the
>   Brownian path $W$), carried by the single letter `1` — there is no time coordinate;
> - a genuinely **time-augmented, 2-D signature** uses `dim = 2` with letters `1` and `2`, one per
>   coordinate of $(t,W_t)$.
>
> Below, the mathematical background keeps the papers' $\{0,1\}$ notation; the code-facing sections
> translate to the integer letters actually used.

The product that mirrors pointwise multiplication of functionals is the **shuffle product**
$\text{ш}$:
$\langle\boldsymbol\ell,\mathbb X\rangle\langle\boldsymbol\ell',\mathbb X\rangle
= \langle\boldsymbol\ell\text{ш}\boldsymbol\ell',\mathbb X\rangle$.
The shuffle exponential / logarithm $\exp^{\text{ш}}$, $\log^{\text{ш}}$ are the tensor-algebra
analogues of $\exp/\log$, and they realise the **Cole–Hopf** change of variables
$\boldsymbol u = \exp^{\text{ш}}(\boldsymbol\psi)$.

### 2.2 Shifts as differentiation

For a letter $\mathsf{i}\in\{0,1\}$ the **right shift** $\boldsymbol\ell\mapsto\boldsymbol\ell|_{\mathsf i}$,
defined by $(\boldsymbol\ell|_v)^w = \boldsymbol\ell^{wv}$, plays the role of differentiating the
power-series coefficients (Laplace §3.3). Itô's formula on $\langle\boldsymbol\ell,\widehat{\mathbb W}\rangle$
produces the **diffusion generator** of the signature:

$$
\mathscr L\boldsymbol\ell \;=\; \boldsymbol\ell|_0 + \tfrac12\,\boldsymbol\ell|_{11}.
$$

### 2.3 The linear equation and why we avoid its closed form

The transform coefficients satisfy the **linear** evolution on the tensor algebra (Laplace
Theorem 5.2):

$$
\dot{\boldsymbol u}_t + \mathscr L\,\boldsymbol u_t = 0,\qquad
\boldsymbol u_T = \exp^{\text{ш}}(\boldsymbol p).
$$

When the terminal condition has finitely many terms there is even a closed form
$\boldsymbol u_t = \boldsymbol\sigma|_{\widehat{\mathcal E}_{T-t}}$ via the expected signature. **But** for the terminal conditions we care about — e.g. $\boldsymbol p = -24\cdot\mathbf{1111}$,
i.e. $\langle\boldsymbol p,\widehat{\mathbb W}_T\rangle = -W_T^4$ — this representation **diverges**:
the series $\sum_k \frac{(-1)^k T^{2k}(4k)!}{2^{2k}(2k)!\,k!}$ does not converge.
This is why one works with the logarithm instead.

### 2.4 The Riccati equation (the equation the code solves)

Passing to $\boldsymbol\psi = \log^{\text{ш}}\boldsymbol u$ (Cole–Hopf) gives the **signature
Riccati equation** (Laplace 5.14–5.15):

$$
\boxed{\;\dot{\boldsymbol\psi}_t + \mathscr R\,\boldsymbol\psi_t = 0,\qquad \boldsymbol\psi_T=\boldsymbol p\;}
\qquad
\mathscr R\boldsymbol\ell \;=\; \underbrace{\boldsymbol\ell|_0 + \tfrac12\boldsymbol\ell|_{11}}_{\text{linear part }\mathscr L}
   \;+\; \underbrace{\tfrac12\,(\boldsymbol\ell|_1)^{\text{ш} 2}}_{\text{nonlinearity}}.
$$

The quadratic term $(\boldsymbol\ell|_1)^{\text{ш} 2} = (\boldsymbol\ell|_1)\text{ш}(\boldsymbol\ell|_1)$
is the only nonlinearity; it comes from the shuffle structure. This is the tensor-algebra analogue
of the scalar Hamilton–Jacobi–Bellman / Cole–Hopf relation

$$
\partial_t\psi + \tfrac12\partial_{xx}\psi + \tfrac12(\partial_x\psi)^2 = 0.
$$

### 2.5 Locality and the recentering technique

A structural subtlety: even when $\boldsymbol p$ is finite, the series
$\langle\boldsymbol\psi_t,\widehat{\mathbb W}_t\rangle$ has only a **finite radius of convergence**.
The transform $u(t,\cdot)$ can have **zeros in the complex plane** (the explicit $-x^4$ example), and the logarithm cannot be entire past the first zero. So the local expansion
$\boldsymbol\psi_t$ is only valid up to a positive stopping time.

To recover a **global** representation one **recenters** the expansion at the running signature
$\widehat{\mathbb W}_t$ using Chen's identity. For any group-like
$\mathbb X$,

$$
\mathbb{E}\!\left[\exp\langle\boldsymbol p, \mathbb X\otimes\widehat{\mathbb W}_{t,T}\rangle\right]
   = \exp\big((\boldsymbol\psi^{\mathbb X}_t)^{\varnothing}\big),
$$

where $\boldsymbol\psi^{\mathbb X}$ solves the **same** Riccati equation but with a **shifted
terminal condition**:

$$
\dot{\boldsymbol\psi}^{\mathbb X}_t + \mathscr R\,\boldsymbol\psi^{\mathbb X}_t = 0,
\qquad \boldsymbol\psi^{\mathbb X}_T = {}_{\mathbb X}|\,\boldsymbol p,
$$

with $({}_{\mathbb X}|\boldsymbol p)^v = \sum_w \boldsymbol p^{wv}\,\mathbb X^w$ the **left
projection** of $\boldsymbol p$ onto $\mathbb X$. Taking $\mathbb X=\widehat{\mathbb W}_t$ yields the
global, path-dependent representation — at the cost of solving a **new Riccati equation for each
$t$** (or each value of the state). Only the empty-word component $(\boldsymbol\psi^{\mathbb X})^\varnothing$
is needed for the transform value.

> **Note — three different "projections" in the code, do not conflate them:**
> 1. `proj` / `TensorSequence.proj(v)` — the **right shift** $\boldsymbol\ell\mapsto\boldsymbol\ell|_v$,
>    used to *build the operator* $\mathscr R$.
> 2. `left_proj_on_seq(p, X)` — the **recentering** left projection ${}_{\mathbb X}|\boldsymbol p$,
>    used to *build the shifted terminal condition*.
> 3. `get_projection_matrix` — the shared square matrix $P$ that both of the above use internally.

### 2.6 The control problem

For the controlled Brownian motion $dX^\alpha_t = \alpha_t\,dt + dW_t$ and the cost
$\inf_\alpha \mathbb E[\int_0^T\alpha_s^2\,ds + F((X^\alpha_s)_{s\le T})]$, the Boué–Dupuis
representation turns the value into a Laplace transform of a signature functional. With
$F(X^\alpha)=\langle\boldsymbol p,\widehat{\mathbb X}^\alpha_T\rangle$, the **value process** and the
**optimal control** are:

$$
\psi^*_t = \langle\boldsymbol\psi^{\widehat{\mathbb X}^{\alpha^*}_t},\varnothing\rangle,\qquad
\alpha^*_t = \langle\boldsymbol\psi^{\widehat{\mathbb X}^{\alpha^*}_t}_t|_1,\varnothing\rangle,
$$

i.e. the optimal control is the **left shift $\boldsymbol\psi|_1$** of the recentered Riccati
solution evaluated at the current path — an explicit closed-loop, signature-Markovian feedback.

---

## 3. From equation to code

### 3.1 Data structure

A `TensorSequence` ([signature/tensor_sequence.py](../signature/tensor_sequence.py)) stores the
coefficients as a flat array indexed by words. A **word is an integer whose digits are letters**,
letters being numbered from `1` (one per path coordinate); there is no letter `0`. Word ↔ index
bookkeeping lives in [signature/words.py](../signature/words.py): `11` is literally the integer
`11`, `from_word(1111, ...)` builds $\mathbf{1111}$. Arithmetic operators (`+`, `*`, scalar `/`)
act coefficientwise; `@` is the inner product $\langle\cdot,\cdot\rangle$.

### 3.2 The operator $\mathscr R$ in code

The right shift $\boldsymbol\ell|_v$ is `ts.proj(v)`. Reading the index arithmetic in
[projection.py](../signature/projection.py): `proj(v)` keeps the coefficients of words **ending**
in `v` and maps $wv \mapsto w$, i.e. $(\boldsymbol\ell|_v)^w = \boldsymbol\ell^{wv}$ — exactly the
right shift. The shuffle square is `shuffle_pow(·, 2, shuffle_table)`.

So the Riccati right-hand side $\mathscr R\boldsymbol\psi$ is, in the notebook:

```python
@jax.jit
def ode_fun(psi, args):
    return psi.proj(11) * 0.5 + ta.shuffle_pow(psi.proj(1), 2, args["shuffle_table"]) * 0.5
```

This is $\tfrac12\boldsymbol\psi|_{11} + \tfrac12(\boldsymbol\psi|_1)^{\text{ш} 2}$.

> **Why is the time-shift term missing?** The papers' general operator carries an extra shift
> along the **time** letter, $\boldsymbol\ell|_0$ in their $\{0,1\}$ notation:
> $\mathscr R\boldsymbol\ell = \boldsymbol\ell|_0 + \tfrac12\boldsymbol\ell|_{11} + \tfrac12(\boldsymbol\ell|_1)^{\text{ш}2}$.
> The notebook runs with `dim = 1` — the 1-D power-series case: the path has a **single coordinate**
> (the Brownian motion $W$), carried by letter `1`, and **there is no time coordinate at all**, so
> the time-shift term is simply absent. The Brownian letter is `1` here, which is why the
> nonlinearity is `shuffle_pow(psi.proj(1), 2)` and the second-order term is `psi.proj(11)`.
>
> In a genuinely time-augmented run you build a 2-D path and add the shift along the letter carrying
> the time coordinate. With the path ordered as $(t, W_t)$ — so letters are `1` = time, `2` =
> Brownian — `ode_fun` becomes
> `psi.proj(1) + 0.5*psi.proj(22) + 0.5*shuffle_pow(psi.proj(2), 2)`.

### 3.3 The time-reversal convention — the main gotcha

The Riccati equation is **terminal-valued** ($\boldsymbol\psi_T=\boldsymbol p$, integrate backward).
The solvers integrate **forward** from `init = p`. The two are reconciled by the change of variable
$\tau = T - t$, $g(\tau) := \boldsymbol\psi_{T-\tau}$:

$$
\dot{\boldsymbol\psi}_t = -\mathscr R\boldsymbol\psi_t
\quad\Longleftrightarrow\quad
g'(\tau) = +\mathscr R\,g(\tau),\qquad g(0)=\boldsymbol p .
$$

That is why `ode_fun` returns $+\mathscr R\boldsymbol\psi$ and the Euler step is `psi + fun(psi)*dt`.
**Consequences for the reader/maintainer:**

- `psi_T = psi.subsequence([-1,])` is $g(T) = \boldsymbol\psi_{t=0}$ — the value at the *initial*
  calendar time, which is what the transform/value at $t=0$ needs.
- In the control loop, `psi.array[:, -i]` reverse-indexes the trajectory precisely to undo this
  reversal (the comment in the notebook, *"(-i) as we changed the time in Riccati"*).

### 3.4 Empirical check

In the notebook, the Riccati solve gives `exp(psi_T[0]) = 0.9928039072775828`, matching the
Gauss–Laguerre quadrature `0.9928039079789808` to ~7e-10 — independent confirmation that the sign
and reversal conventions are correct.

---

## 4. The numerical schemes

Two drivers in [ode_integration.py](../signature/ode_integration.py):

- `ode_solver` — `jax.lax.fori_loop`, returns only the final $\boldsymbol\psi$.
- `ode_solver_traj` — `jax.lax.scan`, returns the whole trajectory (shape `(n_coef, n_time)`),
  needed for the control feedback and for plotting convergence.

Both take a one-step map `step_fun(psi, dt, fun, args)`. Two families are provided.

### 4.1 Plain schemes (used in the notebook)

These integrate $g' = \mathscr R g$ directly. With $f := \mathscr R$:

| `step_fun` | update | order | notes |
|------------|--------|-------|-------|
| `step_fun_euler` | $\psi + f(\psi)\,dt$ | 1 | explicit Euler |
| `step_fun_pece` | predict $\tilde\psi=\psi+f(\psi)dt$; correct $\psi + \tfrac{dt}{2}\,(f(\psi)+f(\tilde\psi))$ | 2 | Heun / trapezoidal predictor–corrector (PECE) |
| `step_fun_rk4` | classical $\tfrac{dt}{6}(k_1+2k_2+2k_3+k_4)$ | 4 | Runge–Kutta 4 |

`step_fun_pece` is the default in the notebook — it is the sweet spot of accuracy vs. cost for the
quadratic nonlinearity. `step_fun_rk4` evaluates the (relatively expensive) shuffle nonlinearity
four times per step.

### 4.2 Semi-integrated (exponential-integrator) schemes

`step_fun_semi_int_euler / _pece / _rk4` target a slightly more general equation in which a
**diagonal linear part** $G$ is split off:

$$
\psi' = -G(\psi) + F .
$$

Here (see [operators.py](../signature/operators.py)) $G$ multiplies the coefficient of word $v$ by
$\lambda(v) = \sum_{i\in v}\lambda_i$ (a mean-reversion / discounting rate supplied as
`args["lam"]`), and the linear part is integrated **exactly** with an integrating factor:

- `D(ts, dt, lam)` $= e^{-\lambda(v)\,dt}\,\cdot$ — the discounting operator;
- `semi_integrated_scheme(ts, dt, lam)` $= G^{-1}(\mathrm{Id}-D) = \dfrac{1-e^{-\lambda(v)dt}}{\lambda(v)}\,\cdot$
  (and $\to dt$ as $\lambda(v)\to 0$) — the exact integral of the source over the step.

A semi-integrated Euler step is then
$\boldsymbol\psi_{\text{next}} = D(\psi) + \text{semi\_int}\big(f(\psi)\big)$, and analogously for
PECE/RK4 with the nonlinear stages fed through `semi_integrated_scheme`.

> **Scope for these papers.** The Riccati operator $\mathscr R$ in 
> contains **only shift terms** — there is no diagonal $G$. So one sets `lam = 0`, and then
> `D` = identity and `semi_integrated_scheme` = multiply-by-`dt`, which makes
> `step_fun_semi_int_*` **identical** to the corresponding plain `step_fun_*`. The semi-integrated
> family is general infrastructure (useful when a mean-reverting/stationary linear term *is*
> present, e.g. in other notebooks of this repo); for the Brownian signature Riccati it reduces to
> the plain schemes, which is why the notebook uses the plain ones.

### 4.3 Choosing a scheme

- **Default:** `step_fun_pece` with a few hundred to ~1000 steps (the notebook uses 1000 steps on
  $[0,T]$ with $T=0.05$).
- Use `rk4` if you need higher accuracy at fixed step count and can afford the extra shuffle
  evaluations.
- Use the `semi_int_*` variants **only** if your operator has a genuine diagonal $G$
  (`lam != 0`); otherwise they buy nothing over the plain ones here.

---

## 5. Recentering in code

Recentering replaces the terminal condition $\boldsymbol p$ by ${}_{\mathbb X}|\boldsymbol p$ and
re-solves. The left projection ${}_{\mathbb X}|\boldsymbol p$ is `left_proj_on_seq(p, X)` in
[projection.py](../signature/projection.py): it builds the projection matrix $P$ of `p` via
`get_projection_matrix` and applies it on the left, returning $({}_{\mathbb X}|\boldsymbol p)^v
= \sum_w \boldsymbol p^{wv}\,\mathbb X^w$.

In the notebook's recentered run, "evaluating at the state $x$" means projecting onto the
**1-dimensional signature** of $x$, i.e. the vector of powers $x^k/k!$:

```python
factorials = np.array([factorial(i) for i in range(N + 1)])
x_arr = (x_grid[None, :] ** np.arange(N + 1)[:, None]) / factorials[:, None]   # signature columns
p_left_proj_x_arr = ta.left_proj_on_seq(ts=p, proj_on=x_arr)                   # shifted terminal conditions
```

Then, **for each state $x$**, a separate Riccati equation is solved with that shifted terminal
condition, and the empty-word component of the solution is the (log-)transform at $x$:

```python
for i, p_x in enumerate(p_left_proj_x_arr.array.T):
    p_left_proj_x = ta.from_array(p_x, N, dim)
    psi_centered = ode_solver_traj(fun=ode_fun, step_fun=step_fun_pece,
                                   t_grid=t_grid, init=p_left_proj_x, args=args)
    psi_centered_arr[i] = psi_centered.subsequence([-1,])[0]   # empty-word component
```

The payoff (notebook figure `psi_series_recentered.pdf`): the **non-recentered** truncated series
$\langle\boldsymbol\psi_T, \text{sig}(x)\rangle$ blows up for $|x|$ beyond the first zero $r$ of the
transform — exactly the finite radius of convergence of §4.4 — whereas the **recentered** solution
tracks the Monte-Carlo benchmark globally across the whole $x$-grid.

---

## 6. Worked examples (notebook map)

1. **Scalar Laplace transform via Riccati.** $\boldsymbol p = -24\cdot\mathbf{1111}$ so
   $\langle\boldsymbol p,\widehat{\mathbb W}_T\rangle = -W_T^4$. Solve the Riccati with PECE; compare
   `exp(psi_T[0])` against Monte-Carlo and Gauss–Laguerre quadrature. Demonstrates §2.3 (closed-form
   linear solution would diverge) and the stability of the Riccati route.

2. **Transform over a grid of states $x$ — locality.** Plot truncated series for truncation orders
   $M=5..19$; they diverge past the first zero $r$.

3. **Recentered Riccati — global representation.** §5 above; recovers the transform for all $x$.

4. **Optimal control.** Terminal cost $\boldsymbol p$ combining $\mathbf{1111}$ and $\mathbf{11}$;
   solve the Riccati once, take `psi_proj_W = psi.proj(1)` (the left shift $\boldsymbol\psi|_1$), and
   roll the closed-loop control $\alpha_t = \langle\boldsymbol\psi_t|_1, \text{sig}(X^\alpha_t)\rangle$
   forward along simulated Brownian paths. The reverse indexing
   `psi.array[:, -i]` reflects the time-reversal of §3.3. A **norm tracker**
   $\|\overline{\boldsymbol\psi_t}\|_{\mathbb X^\alpha_t}/\log 2$ monitors the convergence-radius
   condition $\tau^\psi = \inf\{t:\ \|\overline{\boldsymbol\psi_t}\|_{\widehat{\mathbb W}_t}\ge\log2\}$
   from Control §5.2; crossing it signals that the local expansion is no longer valid and the
   strategy should be recentered.

---

## 7. Quick reference

| Concept (paper) | Symbol | Code |
|---|---|---|
| linear functional | $\langle\boldsymbol\ell,\mathbb X\rangle$ | `ts @ X` |
| right shift / derivation | $\boldsymbol\ell\vert_v$ | `ts.proj(v)` |
| shuffle product / power | $\text{ш}$, $\cdot^{\text{ш} n}$ | `shuffle_prod`, `shuffle_pow` |
| Riccati RHS (dim=1 example) | $\mathscr R\boldsymbol\ell$ | `ode_fun` = `proj(11)/2 + shuffle_pow(proj(1),2)/2` (add the time-letter shift in a time-augmented run) |
| terminal condition | $\boldsymbol\psi_T=\boldsymbol p$ | `init = p` (solved forward in $\tau=T-t$) |
| recentering shift | ${}_{\mathbb X}\vert\boldsymbol p$ | `left_proj_on_seq(p, X)` |
| transform value | $\exp((\boldsymbol\psi_t)^\varnothing)$ | `np.exp(psi_T[0])` |
| optimal control | $\boldsymbol\psi_t\vert_1$ | `psi.proj(1)` |
| discounting $e^{-\lambda(v)dt}$ | $D$ | `operators.D` |
| exact step integral | $G^{-1}(\mathrm{Id}-D)$ | `operators.semi_integrated_scheme` |

**Single biggest gotcha:** integration runs in reversed time $\tau = T-t$. `init` is the terminal
condition, the last trajectory column is the solution at calendar time $0$, and any indexing into a
saved trajectory must account for the reversal.
