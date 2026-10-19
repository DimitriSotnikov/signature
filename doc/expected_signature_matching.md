# Expected signature matching: a mean-field control problem and its dual

This note formalises a mean-field control (MFC) problem. The control steers a Brownian particle so
that the **expected signature** of its time-augmented path matches a prescribed target. A Fenchel
linearisation reduces the problem to a finite-dimensional concave maximisation over a tensor
$\boldsymbol p$. Each evaluation of that problem is a *classical* path-dependent control problem,
solved in closed form by the Boué–Dupuis formula. The same problem is solved numerically by the
signature-Riccati / Monte-Carlo machinery of the repo.

Companion notes:
- [riccati_signature.md](riccati_signature.md): theory of the signature Riccati equation and of
  the control problem $\sup_\alpha\mathbb E[\langle\boldsymbol q,\widehat{\mathbb X}_T\rangle-\tfrac12\int\alpha^2]$;
- [riccati_monte_carlo.md](riccati_monte_carlo.md): Monte-Carlo signature-feedback policies of
  [signature/sig_control.py](../signature/sig_control.py).

---

## 1. Setting and notation

**Dynamics.** On a filtered probability space carrying a 1-D Brownian motion $W$,

$$
dX_t = \alpha_t\,dt + dW_t,\qquad X_0 = 0,\qquad t\in[0,T],
$$

with $\alpha$ progressively measurable and $\mathbb E\int_0^T\alpha_t^2\,dt<\infty$.
The initial point does not matter: the signature is translation invariant.

**Signature.** $\widehat X_t=(t,X_t)$ is the time-augmented path and

$$
\widehat{\mathbb X}_t := \mathrm{Sig}_{0,t}\big(s\mapsto (s,X_s)\big)
$$

its (Stratonovich) signature. Its coordinates $\widehat{\mathbb X}^v_t$ are indexed by words $v$ in
the alphabet $\{0,1\}$: letter $0$ is time, letter $1$ the state. $\widehat{\mathbb W}$ denotes the
signature of $(t,W_t)$, i.e. the uncontrolled case $\alpha\equiv0$.

> **Letters in the code.** As in [riccati_signature.md](riccati_signature.md), the code numbers
> letters from `1`. Paper letter $0$ (time) is code letter `1`, and paper letter $1$ (state) is
> code letter `2` (`state_letter=2` in `sig_control`).

**Truncated tensors.** Let $\mathcal W_N=\{v:\ 1\le|v|\le N\}$ be the non-empty words of length
at most $N$. $T^N:=\mathbb R^{\mathcal W_N}$ carries the Euclidean pairing
$\langle\boldsymbol p,\boldsymbol x\rangle=\sum_{v\in\mathcal W_N}\boldsymbol p^v\boldsymbol x^v$ and the norm
$\|\boldsymbol p\|^2=\langle\boldsymbol p,\boldsymbol p\rangle$. The empty word is excluded: its
coordinate is identically $1$.

**Target.** $\mathcal E\in T^N$ is a target expected signature. Typically
$\mathcal E^v = \mathbb E^{\mu}[\mathbb S^v_T]$ for some reference path law $\mu$, or a calibrated
or empirical estimate of it.

---

## 2. The MFC problem

$$
\boxed{\;
V = \inf_{\alpha}\ \mathbb E\Big[\tfrac12\int_0^T\alpha_t^2\,dt\Big]
  + \frac\kappa2\sum_{v\in\mathcal W_N}\Big(\mathbb E\big[\widehat{\mathbb X}^v_T\big]-\mathcal E^v\Big)^2
\;}
$$

with penalty weight $\kappa>0$. Write $m(\alpha):=\mathbb E[\widehat{\mathbb X}_T]\in T^N$.

- The problem is **mean-field**: the terminal cost is a nonlinear (quadratic) function of
  the *law* of $X$, through the vector of moments $m(\alpha)$. It is not an expectation of a path
  functional, so dynamic programming on $X$ alone does not apply.
- It is a **soft** moment-matching problem. As $\kappa\to\infty$ it approaches the
  minimum-energy (equivalently minimum-entropy, see §4) control that reproduces $\mathcal E$ exactly.
- On pure-time words $v=0^k$, $\widehat{\mathbb X}^v_T=T^k/k!$ for every $\alpha$. Those terms add
  a constant, and they can be dropped from $\mathcal W_N$ (or $\mathcal E^{0^k}:=T^k/k!$ can be set).

---

## 3. Linearisation by Fenchel duality

For $z\in\mathbb R$, $\tfrac12 z^2=\sup_{p\in\mathbb R}\big(pz-\tfrac12p^2\big)$, attained at $p=z$.
Applying this componentwise gives

$$
\frac\kappa2\sum_{v\in\mathcal W_N}\big(m^v-\mathcal E^v\big)^2
 = \kappa\sup_{\boldsymbol p\in T^N}\Big(\langle\boldsymbol p,m-\mathcal E\rangle-\tfrac12\|\boldsymbol p\|^2\Big),
$$

with optimiser $\boldsymbol p=m-\mathcal E$, the **residual**. Since
$\langle\boldsymbol p,m(\alpha)\rangle=\mathbb E\langle\boldsymbol p,\widehat{\mathbb X}_T\rangle$,
the problem is written as an inf–sup of a Lagrangian that is *linear in the law of $X$*:

$$
V=\inf_\alpha\sup_{\boldsymbol p\in T^N} L(\alpha,\boldsymbol p),\qquad
L(\alpha,\boldsymbol p):=-\frac\kappa2\|\boldsymbol p\|^2-\kappa\langle\boldsymbol p,\mathcal E\rangle
+\mathbb E\Big[\tfrac12\int_0^T\alpha_t^2\,dt+\kappa\langle\boldsymbol p,\widehat{\mathbb X}_T\rangle\Big].
$$

### 3.1 Exchanging sup and inf: weak duality

$\inf\sup\ge\sup\inf$ always holds, so

$$
V\ \ge\ \sup_{\boldsymbol p\in T^N} J(\boldsymbol p),\qquad
J(\boldsymbol p):=-\frac\kappa2\|\boldsymbol p\|^2-\kappa\langle\boldsymbol p,\mathcal E\rangle+V_{\boldsymbol p},
$$

$$
V_{\boldsymbol p}:=\inf_\alpha\ \mathbb E\Big[\tfrac12\int_0^T\alpha_t^2\,dt+\kappa\langle\boldsymbol p,\widehat{\mathbb X}_T\rangle\Big].
$$

$V_{\boldsymbol p}$ is the value of an **auxiliary classical control problem**. It has no mean-field
term, and its terminal cost is a linear functional of the signature. Equality (strong duality) is
not automatic. §5.3 gives a verification argument showing that it holds whenever the dual maximum
is attained in the interior of the domain of $J$.

---

## 4. The auxiliary problem: Boué–Dupuis

### 4.1 Value

Let $F(\omega):=\kappa\langle\boldsymbol p,\mathrm{Sig}_{0,T}(t,\omega_t)\rangle$ be the terminal
cost as a functional of the path. $X=W+\int_0^\cdot\alpha$, so the cost is $F(W+\int\alpha)$. The
Boué–Dupuis formula,
$-\log\mathbb E[e^{-F(W)}]=\inf_\alpha\mathbb E\big[F(W+\int\alpha)+\tfrac12\int_0^T\alpha^2\big]$, gives

$$
\boxed{\;V_{\boldsymbol p}=-\log Z(\boldsymbol p),\qquad
Z(\boldsymbol p):=\mathbb E\Big[\exp\big(-\kappa\langle\boldsymbol p,\widehat{\mathbb W}_T\rangle\big)\Big].\;}
$$

The formula is classical for $F$ bounded measurable. It extends to $F$ bounded below, or with
$e^{-F}\in L^{1+\varepsilon}$, which in our setting is a condition on $\boldsymbol p$. For
$-\kappa\boldsymbol p$ in the class $\mathcal B$ of the Laplace paper, $F$ is bounded below
pathwise (§6.2).

### 4.2 Optimal law and optimal control

Equivalently, by the Gibbs variational principle,

$$
V_{\boldsymbol p}=\min_{Q\ll\mathbb W}\Big\{H(Q\,|\,\mathbb W)+\kappa\,\mathbb E_Q\langle\boldsymbol p,\widehat{\mathbb W}_T\rangle\Big\},
\qquad
\frac{dQ_{\boldsymbol p}}{d\mathbb W}=\frac{e^{-\kappa\langle\boldsymbol p,\widehat{\mathbb W}_T\rangle}}{Z(\boldsymbol p)},
$$

where $\mathbb W$ is Wiener measure. The relative entropy is the energy:
$H(\mathrm{Law}(X^\alpha)\,|\,\mathbb W)\le\tfrac12\mathbb E\int\alpha^2$, with equality for
the Föllmer drift. So **the optimally controlled path has law $Q_{\boldsymbol p}$**, the
exponential tilt of Wiener measure with the signature as sufficient statistic.

The optimal drift is $\alpha^*_t=\partial_x\log h_t$, with
$h_t=\mathbb E[e^{-\kappa\langle\boldsymbol p,\widehat{\mathbb W}_T\rangle}\mid\mathcal F_t]$. Chen's identity
$\widehat{\mathbb X}_T=\widehat{\mathbb X}_t\otimes\widehat{\mathbb X}_{t,T}$, together with the
independence of $\widehat{\mathbb W}_{t,T}$ from $\mathcal F_t$, makes it a **signature-Markovian
feedback**:

$$
\log h_t=\log\mathbb E\Big[\exp\big\langle -\kappa\boldsymbol p,\ \widehat{\mathbb X}_t\otimes\widehat{\mathbb W}_{t,T}\big\rangle\Big]\Big|_{\widehat{\mathbb X}_t},
\qquad
\alpha^*_t=\frac{\mathbb E\big[e^{\langle -\kappa\boldsymbol p,\widehat{\mathbb X}_t\otimes\widehat{\mathbb W}_{t,T}\rangle}
\langle -\kappa\boldsymbol p,\widehat{\mathbb X}_t\otimes\mathbf 1\otimes\widehat{\mathbb W}_{t,T}\rangle\big]}{h_t}.
$$

Equivalently, $\log h$ solves the signature Riccati equation with terminal condition
$-\kappa\boldsymbol p$, and $\alpha^*$ is its shift in the state letter
([riccati_signature.md](riccati_signature.md) §2.6).

### 4.3 Mapping to the code

[sig_control.py](../signature/sig_control.py) implements the **maximisation** convention
$\psi_t=\log\mathbb E[\exp\langle\boldsymbol q,\cdot\rangle]$, $\alpha_t=+\partial\psi_t$. Its
Riccati generator $\psi|_{0}+\tfrac12\psi|_{11}+\tfrac12(\psi|_1)^{ш2}$ (paper letters) is the HJB
of $\sup_\alpha\mathbb E[\langle\boldsymbol q,\widehat{\mathbb X}_T\rangle-\tfrac12\int\alpha^2]$.
That HJB uses the same $\tfrac12$ energy constant as here. (§2.6 of
[riccati_signature.md](riccati_signature.md) writes $\int\alpha^2$; the generator shows the
constant is $\tfrac12$.) Therefore:

| here | code |
|------|------|
| terminal tensor $\boldsymbol p$, weight $\kappa$ | `p` $=\boldsymbol q:=-\kappa\boldsymbol p$ |
| $V_{\boldsymbol p}=-\log Z(\boldsymbol p)$ | $-\psi_0$ (value returned at $t=0$) |
| optimal drift $\alpha^*_t$ | returned control $\alpha_t$ (unchanged sign) |

---

## 5. The dual problem

### 5.1 Dual function

$$
\boxed{\;J(\boldsymbol p)=-\frac\kappa2\|\boldsymbol p\|^2-\kappa\langle\boldsymbol p,\mathcal E\rangle-\log\mathbb E\Big[e^{-\kappa\langle\boldsymbol p,\widehat{\mathbb W}_T\rangle}\Big],
\qquad \boldsymbol p\in D:=\{Z<\infty\}.\;}
$$

This is a finite-dimensional problem: $\dim T^N=2^{N+1}-2$. Its only random input is the
**uncontrolled** Brownian signature $\widehat{\mathbb W}_T$, whose law is fixed.

### 5.2 Gradient, Hessian, optimality condition

Write $m(\boldsymbol p):=\mathbb E_{Q_{\boldsymbol p}}[\widehat{\mathbb W}_T]$ for the tilted mean
and $C(\boldsymbol p):=\mathrm{Cov}_{Q_{\boldsymbol p}}(\widehat{\mathbb W}_T)$. On the interior of
$D$ (standard exponential-family calculus):

$$
\nabla J(\boldsymbol p)=\kappa\big(m(\boldsymbol p)-\mathcal E-\boldsymbol p\big),\qquad
\nabla^2 J(\boldsymbol p)=-\kappa I-\kappa^2 C(\boldsymbol p)\ \preceq\ -\kappa I .
$$

- $J$ is **$\kappa$-strongly concave**, so the maximiser $\boldsymbol p^*$ is unique if it exists.
- **Optimality condition:**
  $$\boldsymbol p^*=m(\boldsymbol p^*)-\mathcal E,$$
  i.e. the dual variable is the moment residual under the optimally controlled law. This matches
  the Fenchel optimiser of §3 and the first-order condition of the MFC problem: its optimal
  control is the optimal control of the *classical* problem whose terminal cost is the
  linearisation $\kappa\langle m^*-\mathcal E,\widehat{\mathbb X}_T\rangle$ of the mean-field
  penalty.

### 5.3 Strong duality by verification

Suppose $\boldsymbol p^*\in\mathrm{int}\,D$ satisfies $\nabla J(\boldsymbol p^*)=0$. Let
$Q^*:=Q_{\boldsymbol p^*}$, $\alpha^*$ its drift, $m^*:=m(\boldsymbol p^*)=\mathcal E+\boldsymbol p^*$.
By §4.2, $V_{\boldsymbol p^*}=H(Q^*|\mathbb W)+\kappa\langle\boldsymbol p^*,m^*\rangle$, hence

$$
J(\boldsymbol p^*)=H(Q^*|\mathbb W)+\kappa\langle\boldsymbol p^*,m^*-\mathcal E\rangle-\frac\kappa2\|\boldsymbol p^*\|^2
=\underbrace{H(Q^*|\mathbb W)}_{=\frac12\mathbb E\int(\alpha^*)^2}+\frac\kappa2\|m^*-\mathcal E\|^2 .
$$

The right-hand side is the primal cost of $\alpha^*$, so $V\le J(\boldsymbol p^*)\le V$:
**strong duality holds, and $\alpha^*$ (the optimal control of the auxiliary problem at $\boldsymbol p^*$)
solves the MFC problem.** The optimal value also splits as

$$
V = H(Q^*\,|\,\mathbb W)+\frac\kappa2\|\boldsymbol p^*\|^2 .
$$

The primal is also equivalent to the convex problem
$\inf_Q\{H(Q|\mathbb W)+\frac\kappa2\|\mathbb E_Q[\widehat{\mathbb W}_T]-\mathcal E\|^2\}$, a
penalised **I-projection** (maximum-entropy) problem with signature moments. $J$ is its
Fenchel–Rockafellar dual, which explains the exponential-family structure.

---

## 6. Well-posedness: the domain of $Z$

The dual function is finite on $D=\{\boldsymbol p: Z(\boldsymbol p)<\infty\}$. $D$ is convex
(by Hölder) and contains $0$. This section gives explicit sufficient conditions for
$\boldsymbol p\in D$, and a matching necessary condition. Together they decide which targets the
method can handle.

**Notation** (as in the Laplace paper,
[references/Sig_Riccati-Laplace.pdf](../references/Sig_Riccati-Laplace.pdf)):
- $|v|_1$ is the number of state letters in a word $v$ and $|v|_0$ the number of time letters;
  $\deg_1,\deg_0$ are the corresponding maxima over the support of a tensor.
- $\mathbf 1^{\otimes k}$ is the word of $k$ state letters, so
  $\widehat{\mathbb X}^{\mathbf 1^{\otimes k}}_T=X_T^k/k!$.
- $\boldsymbol r\mathbf 0$ appends the time letter to every word of $\boldsymbol r$, so
  $\langle\boldsymbol r\mathbf 0,\widehat{\mathbb X}_T\rangle=\int_0^T\langle\boldsymbol r,\widehat{\mathbb X}_s\rangle\,ds$.

### 6.1 Low state degree: Gaussian computations

- **Words with at most one state letter.** $\langle\boldsymbol p,\widehat{\mathbb W}_T\rangle$ is
  Gaussian (first Wiener chaos plus a constant), so $Z<\infty$ for every coefficient.
- **Words with two state letters** (e.g. $\mathbf{11}$: $\widehat{\mathbb W}^{\mathbf{11}}_T=W_T^2/2$) lie in
  the second chaos. Then $Z<\infty$ iff a quadratic-form condition holds. For instance
  $\mathbb E[e^{-\kappa p W_T^2/2}]<\infty\iff 1+\kappa pT>0$.

### 6.2 Longer words: the class $\mathcal B$

For words with three or more state letters, Gaussian arguments are not available. Section 4.1 of
the Laplace paper introduces a class $\mathcal B$ of tensors whose signature functional is
**bounded from above pathwise**.

**Definition (Laplace paper, Def. 4.2, real coefficients).** $\boldsymbol\ell\in\mathcal B$ if there
exist $\alpha,\beta\ge0$, $\gamma\in\mathbb R$, $n,m\in\mathbb N^*$ and tensors
$\boldsymbol q,\boldsymbol r,\boldsymbol s$ of finite degree with

$$
\deg_1(\boldsymbol q)<2n,\quad \deg_0(\boldsymbol q)=0,\qquad
\deg_1(\boldsymbol r)<2m,\qquad
\deg_1(\boldsymbol s)<2m\wedge 2n,
$$

such that

$$
\boldsymbol\ell=-\alpha\big(\mathbf 1^{\otimes 2n}+\boldsymbol q\big)-\beta\big(\mathbf 1^{\otimes 2m}+\boldsymbol r\big)\mathbf 0+\alpha\beta\,\boldsymbol s+\gamma\,\varnothing .
$$

Reading the definition term by term:
- the first term is a polynomial in $X_T$ with negative even leading coefficient;
- the second is a time integral $\int_0^T(\cdot)\,ds$ of a functional dominated by $-X_s^{2m}$;
- the mixed term $\boldsymbol s$ (arbitrary time letters) is allowed only when *both* leading terms
  are present, and only below their state degree.

**Proposition (Laplace paper, Prop. 4.3).** If $\boldsymbol\ell\in\mathcal B$, there is $C_T<\infty$ such
that $\langle\boldsymbol\ell,\widehat{\mathbb X}_{s,t}\rangle\le C_T$ for **every** continuous
semimartingale $X$ and all $0\le s\le t\le T$. The bound comes from Lemma 4.1 of the paper:
$|\widehat{\mathbb X}^v_{s,t}|\lesssim 1+|X_{s,t}|^{|v|_1}+\int_s^t|X_{s,r}|^{|v|_1}dr$, and
$|\widehat{\mathbb X}^{v\mathbf 0}_{s,t}|\lesssim 1+\int_s^t|X_{s,r}|^{|v|_1}dr$.

**Application.** The exponent in $Z$ is $\langle-\kappa\boldsymbol p,\cdot\rangle$, so set

$$
D_{\mathcal B}:=\{\boldsymbol p\in T^N:\ -\kappa\boldsymbol p+c\,\varnothing\in\mathcal B\ \text{for some } c\in\mathbb R\}.
$$

$T^N$ has no empty-word coordinate; the constant $c$ only absorbs $\gamma$. For
$\boldsymbol p\in D_{\mathcal B}$:

1. **Finiteness.** $Z(\boldsymbol p)\le e^{C_T}<\infty$, so $D_{\mathcal B}\subset D$.
2. **Inner problem bounded below, pathwise.** The terminal cost satisfies
   $F=\kappa\langle\boldsymbol p,\widehat{\mathbb X}^\alpha_T\rangle\ge-C_T$ for *every* control
   $\alpha$. Hence $V_{\boldsymbol p}\ge-C_T$ directly, and the Boué–Dupuis formula of §4.1 applies
   in its bounded-below form. This is the integrability required there.
3. **Bounded importance weights.** $dQ_{\boldsymbol p}/d\mathbb W\le e^{C_T}/Z(\boldsymbol p)$.
   All signature moments under $Q_{\boldsymbol p}$ are therefore finite, so $m(\boldsymbol p)$ and
   $C(\boldsymbol p)$ of §5.2 are well defined. The self-normalised estimators of §7 have
   bounded weights.
4. **Stability under perturbation.** For any $\boldsymbol p'$,
   $Z(\boldsymbol p+\boldsymbol p')\le e^{C_T}Z(\boldsymbol p')$, so
   $$D_{\mathcal B}+D\subset D .$$
   The Gaussian directions of §6.1 can be added freely on top of a $\mathcal B$-part. This
   matters because $\mathcal B$ alone does not contain, for example, the word $\mathbf{01}$ without
   a leading term.
5. **Cone.** $D_{\mathcal B}$ is a cone: scale $\alpha,\beta$ by $t>0$ and $\boldsymbol s$ by $1/t$.
   For fixed $(n,m)$ and fixed support pattern it is moreover convex. The constraints are the sign
   conditions below plus a support condition.

### 6.3 What this means for the target words

Translate $-\kappa\boldsymbol p\in\mathcal B$ into conditions on the coefficients of $\boldsymbol p$
(up to Gaussian directions, by item 4). The **leading words** are
$\mathbf 1^{\otimes2n}$ (length $2n$) and $\mathbf 1^{\otimes2m}\mathbf 0$ (length $2m+1$). They must
belong to $\mathcal W_N$, and

$$
\boldsymbol p^{\mathbf 1^{\otimes 2n}}=\alpha/\kappa\ \ge0,\qquad
\boldsymbol p^{\mathbf 1^{\otimes 2m}\mathbf 0}=\beta/\kappa\ \ge0 .
$$

Every other word $v$ in the support of $\boldsymbol p$ must be dominated:

| leading terms present | allowed non-leading words |
|---|---|
| $\alpha>0,\ \beta=0$ | pure state words $\mathbf 1^{\otimes k}$, $k<2n$ |
| $\alpha=0,\ \beta>0$ | words $w\mathbf 0$ ending with the time letter, $\lvert w\rvert_1<2m$ |
| $\alpha>0,\ \beta>0$ | the two families above, plus **any** word with $\lvert v\rvert_1<2m\wedge2n$ |

In every case, words with $|v|_1\le1$ can be added (item 4).

**Consequences for the design of the target.**
- A general (mixed) word with $k\ge3$ state letters is covered only through $\boldsymbol s$. That
  needs both leading words with $2m\wedge2n>k$, hence a truncation $N\ge 2m+1\ge k+2$. In practice,
  the top levels of $\mathcal W_N$ are reserved for the confining words $\mathbf 1^{\otimes2n}$ and
  $\mathbf 1^{\otimes2m}\mathbf 0$, and the "informative" target words live at lower state degree.
- The sign conditions apply to the **dual** variable, which at the optimum is the residual
  $\boldsymbol p^*=m^*-\mathcal E$ (§5.2). So $\mathcal B$ requires the optimally controlled
  law to *overshoot* the target on the leading even moments, $\mathbb E[X_T^{2n}]$ and
  $\mathbb E\int_0^TX_s^{2m}ds$. The Gibbs tilt then acts as a confining potential
  $e^{-\alpha X_T^{2n}/(2n)!-\beta\int X_s^{2m}ds/(2m)!}$.

### 6.4 A matching necessary condition

The sign structure is not an artefact of the proof. Test the inner problem with a constant drift
$\alpha\equiv c$ and ignore the noise: the path $(t,ct)$ is linear, so
$\widehat{\mathbb X}^v_T=T^{|v|}c^{|v|_1}/|v|!$, and

$$
\kappa\langle\boldsymbol p,\widehat{\mathbb X}_T\rangle=\kappa\sum_{k}c^k\,a_k,\qquad
a_k:=\sum_{|v|_1=k}\boldsymbol p^v\,\frac{T^{|v|}}{|v|!} .
$$

The energy cost is only $\tfrac12c^2T$. Let $k^*:=\max\{k:a_k\ne0\}$. If $k^*\ge3$ and either
$k^*$ is odd or $a_{k^*}<0$, then $V_{\boldsymbol p}=-\infty$ by letting $c\to\pm\infty$, so
$\boldsymbol p\notin D$. (The noise only adds lower-order terms; this is a heuristic test, not a
proof.) For $\boldsymbol p\in D_{\mathcal B}$ the leading part has $k^*$ even with $a_{k^*}>0$,
consistent with this test. The gap between $D_{\mathcal B}$ and $D$ consists of functionals whose
top state-degree part is positive along linear paths but not dominated in the sense of Lemma 4.1.

### 6.5 Consequences for the dual problem

- **Restricting the dual.** Let $K\subset\overline{D_{\mathcal B}}$ be a closed convex cone (fixed
  $n,m$ and support, closed sign constraints), possibly enlarged by the Gaussian directions. Then
  $V\ge\sup_{\boldsymbol p\in K}J(\boldsymbol p)$ is still a valid lower bound, by weak duality. It is
  an equality (verification of §5.3) iff the constrained maximiser satisfies
  $\nabla J(\boldsymbol p^*)=0$, i.e. the residual $m^*-\mathcal E$ itself lies in the interior
  of $K$. Gradient ascent is then run as **projected** gradient ascent onto $K$.
- **Targets outside the cone.** If the target forces a negative residual on a leading word, i.e.
  the controlled law must have *heavier* top moments than $\mathcal E$, or charges an undominated
  word, the Gibbs form breaks down, and a duality gap is expected.

**Monte-Carlo warning.** A finite sample always gives a finite $\widehat Z$. The algorithm *runs*
even when $\boldsymbol p\notin D$, but the estimates are dominated by a few paths and are
meaningless. Inside $D_{\mathcal B}$ the weights are bounded (item 3). Outside it, monitor the
effective sample size (§7).

---

## 7. Algorithm: Monte-Carlo gradient ascent on $J$

**Precompute once** a sample $\{\widehat{\mathbb W}^{(i)}_T\}_{i=1}^M$ of truncated Brownian
signatures on $[0,T]$ at level $N$, as in [riccati_monte_carlo.md](riccati_monte_carlo.md) §2.1
(`iisignature` on a fine grid; prepend the empty word). The sample is reused at every iteration.

**Estimators** (log-sum-exp stabilised, self-normalised weights). Let
$\ell_i=-\kappa\langle\boldsymbol p,\widehat{\mathbb W}^{(i)}_T\rangle$ and
$w_i=e^{\ell_i-\max_j\ell_j}/\sum_k e^{\ell_k-\max_j\ell_j}$:

$$
\widehat V_{\boldsymbol p}=-\Big(\max_j\ell_j+\log\tfrac1M\textstyle\sum_i e^{\ell_i-\max_j\ell_j}\Big),\qquad
\widehat m(\boldsymbol p)=\textstyle\sum_i w_i\,\widehat{\mathbb W}^{(i)}_T,\qquad
\widehat C(\boldsymbol p)=\textstyle\sum_i w_i(\widehat{\mathbb W}^{(i)}_T-\widehat m)^{\otimes2}.
$$

**Iteration.** Gradient ascent with step $\eta/\kappa$ is exactly a **damped fixed point** on the
optimality condition:

$$
\boldsymbol p_{k+1}=\boldsymbol p_k+\frac{\eta}{\kappa}\nabla J(\boldsymbol p_k)
=(1-\eta)\,\boldsymbol p_k+\eta\,\big(\widehat m(\boldsymbol p_k)-\mathcal E\big).
$$

- **Step size.** $J$ is $\kappa$-strongly concave and $L$-smooth locally, with
  $L=\kappa(1+\kappa\lambda_{\max}(C))$. A safe choice is $\eta\le1/(1+\kappa\lambda_{\max}(\widehat C))$,
  with a backtracking line search on $\widehat J$.
- **Newton variant.** $\boldsymbol p_{k+1}=\boldsymbol p_k+(I+\kappa\widehat C)^{-1}(\widehat m-\mathcal E-\boldsymbol p_k)$.
  The matrix is small ($\dim T^N$) and well conditioned ($\succeq I$), so this converges in a few
  iterations near the optimum.
- **Initialisation.** $\boldsymbol p_0=0$, i.e. $Q_0=\mathbb W$ and $\widehat m=\mathbb E[\widehat{\mathbb W}_T]$.
  An optional continuation in $\kappa$ (small to large) keeps the tilt mild.
- **Diagnostics.** Effective sample size $\mathrm{ESS}=1/\sum_iw_i^2$. Stop or reduce the step when
  $\mathrm{ESS}/M$ becomes small, since that indicates leaving $D$ (§6) or an over-strong tilt.
  Also track $\|\nabla\widehat J\|$ and the duality check of §5.3.
- **Errors.** There are two: discretisation bias from the fine grid (`n_fine`) and statistical error
  ($M$). The same sample makes $\widehat J$ a deterministic smooth concave function (sample-average
  approximation), so standard deterministic optimisers apply.

---

## 8. Generating optimal trajectories at $\boldsymbol p^*$

1. **Closed-loop rollout.** Run `sc.simulate` with `sc.monte_carlo_policy(p=-κ p*, SigW1, T, trunc, dim)`,
   or with the Riccati policy solved for terminal condition $-\kappa\boldsymbol p^*$ (§4.3). This
   produces genuinely controlled paths $dX=\alpha^*_t\,dt+dW_t$ for *fresh* Brownian increments.
2. **Reweighting / resampling.** $Q^*$ is a tilt of Wiener measure on *path space*. Resampling
   the precomputed Brownian paths with weights $w_i(\boldsymbol p^*)$ therefore gives exact
   samples of the optimal path law, up to MC error. It is cheap but limited by the ESS.

**Validation.** On the generated paths, check that
$\frac1n\sum\widehat{\mathbb X}_T\approx\mathcal E+\boldsymbol p^*$ (the residual should equal
$\boldsymbol p^*$, not zero, for finite $\kappa$). Also check that the empirical primal cost
$\tfrac12\mathbb E\int\alpha^2+\tfrac\kappa2\|\cdot\|^2$ matches $J(\boldsymbol p^*)$ (§5.3).

---

## 9. Open points

- Strong duality and existence of $\boldsymbol p^*$ beyond the interior case of §5.3. For example:
  is $J$ steep at $\partial D$, so that the maximiser is automatically interior?
- Targets outside the $\mathcal B$-cone (§6.3–6.5): either close the gap between $D_{\mathcal B}$
  and $D$, or find a reformulation that keeps the closed-form inner problem. Candidates are
  bounded controls, normalised signatures, and adding confining leading words to the penalty.
- Exact $Z$, $\nabla Z$ via the signature Riccati equation instead of MC. That removes the sampling
  error in $J$, but Riccati blow-up for some $\boldsymbol p$ mirrors the boundary of $D$.
- Bias–variance trade-off between `n_fine`, $M$, and the truncation level $N$.
