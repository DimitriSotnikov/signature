"""Unit tests for signature.esig_mfc (expected-signature matching MFC, dual optimisation).

Tests 9--12 are the core correctness checks: 9--10 validate the optimiser exactly on the
sample-average problem, 11 validates the theory (Boue--Dupuis log and sign) on a problem solvable
by hand, and 12 pins the sign convention ``q = -kappa p`` of the interface to ``sig_control``.
Test 14 is slow and runs only with ``RUN_SLOW=1``.
"""
from __future__ import annotations

import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import signature.tensor_algebra as ta
import signature.esig_mfc as em
from signature.expected_signature import expected_bm_signature
from signature.words import word_to_index


RUN_SLOW = os.environ.get("RUN_SLOW", "0") == "1"
T = 1.0


def idx(word: int) -> int:
    return int(word_to_index(word, 2))


def brownian_sample(N: int, M: int, n_steps: int = 50, seed: int = 0, T: float = T):
    t_grid, W = em.brownian_paths(T, n_steps, M, seed)
    return em.path_signatures(W, t_grid, N), W


def discrete_bm_expected_signature(cone: em.ConeSpec, n_steps: int, T: float = T) -> np.ndarray:
    """Exact expected signature of the piecewise-linear interpolation of (t, W_t) on a uniform grid:
    the n-fold tensor power of E[exp(dt e1 + dW e2)], whose word-v coefficient is
    dt^{|v|_1} E[dW^{|v|_2}] / |v|! with Gaussian moments E[dW^{2j}] = (2j-1)!! dt^j."""
    from math import factorial
    dt = T / n_steps
    lengths = np.array([len(str(w)) if w else 0 for w in cone.words])
    k2 = cone.state_degree
    k1 = lengths - k2

    def gauss_moment(j):
        return 0.0 if j % 2 else float(np.prod(np.arange(j - 1, 0, -2))) * dt ** (j / 2)

    seg = np.array([dt ** k1[i] * gauss_moment(k2[i]) / factorial(lengths[i]) for i in range(cone.n_sig)])
    seg_ts = ta.TensorSequence(array=jnp.asarray(seg), trunc=cone.trunc, dim=2)
    acc = seg_ts
    for _ in range(n_steps - 1):
        acc = ta.tensor_prod(acc, seg_ts)
    return np.asarray(acc.array)


def random_cone_point(cone: em.ConeSpec, rng, scale: float, lead_low: float, lead_high: float,
                      level_decay: float = 1.0) -> jnp.ndarray:
    """A random point of K with leads in [lead_low, lead_high] and free coefficients of size
    ``scale * level_decay**(|v| - 1)``."""
    lengths = np.array([len(str(w)) if w else 0 for w in cone.words])
    p = rng.normal(size=cone.n_sig) * scale * level_decay ** np.maximum(lengths - 1, 0)
    p[list(cone.lead_idx)] = rng.uniform(lead_low, lead_high, size=len(cone.lead_idx))
    p = em.project(jnp.asarray(p), cone)
    assert em.in_cone(p, cone)
    return p


class TestTargetPaths(unittest.TestCase):
    def test_target_paths(self):
        M, n_steps = 20_000, 50
        # OU and CIR: exact transitions, terminal mean and variance in closed form
        th, mu, x0 = 3.0, 0.5, 0.2
        _, X = em.target_paths("ou", T, n_steps, M, seed=0, theta=th, mu=mu, sigma=1.0, x0=x0)
        self.assertEqual(X.shape, (M, n_steps + 1))
        mean, var = mu + (x0 - mu) * np.exp(-th * T), (1 - np.exp(-2 * th * T)) / (2 * th)
        self.assertLess(abs(X[:, -1].mean() - mean), 5 * np.sqrt(var / M))
        self.assertLess(abs(X[:, -1].var() / var - 1), 5 * np.sqrt(2 / M))

        a, b, sg, x0 = 2.0, 1.0, 1.0, 0.5
        _, X = em.target_paths("cir", T, n_steps, M, seed=1, a=a, b=b, sigma=sg, x0=x0)
        e = np.exp(-a * T)
        mean = b + (x0 - b) * e
        var = x0 * sg ** 2 / a * (e - e ** 2) + b * sg ** 2 / (2 * a) * (1 - e) ** 2
        self.assertTrue(np.all(X >= 0))
        self.assertLess(abs(X[:, -1].mean() - mean), 5 * np.sqrt(var / M))
        self.assertLess(abs(X[:, -1].var() / var - 1), 0.1)

        # double well started at the barrier top: symmetric and bimodal around +-1
        _, X = em.target_paths("double_well", T, n_steps, M, seed=2, h=1.0)
        XT = X[:, -1]
        self.assertLess(abs(XT.mean()), 5 * XT.std() / np.sqrt(M))
        hist, edges = np.histogram(XT, bins=40, range=(-2, 2), density=True)
        centers = 0.5 * (edges[1:] + edges[:-1])
        peak = hist[np.abs(np.abs(centers) - 1) < 0.25].mean()
        self.assertGreater(peak, 1.5 * hist[np.abs(centers) < 0.1].mean())

        # W^2 from x0: X = (x0 + W)^2, so E[X_T - X_0] = T
        _, X = em.target_paths("w2", T, n_steps, M, seed=3, x0=1.0)
        np.testing.assert_allclose(X[:, 0], 1.0)
        self.assertLess(abs((X[:, -1] - 1.0).mean() - T), 5 * (X[:, -1] - 1.0).std() / np.sqrt(M))

        with self.assertRaises(ValueError):
            em.target_paths("nope", T, n_steps, 10, seed=0)
        with self.assertRaises(TypeError):
            em.target_paths("ou", T, n_steps, 10, seed=0, bogus=1.0)


class TestCone(unittest.TestCase):
    def test_cone_bookkeeping(self):
        cone = em.make_b_cone(7)
        self.assertEqual((cone.n, cone.m), (3, 3))
        self.assertEqual([int(cone.words[i]) for i in cone.lead_idx], [222222, 2222221])
        forced = sorted(int(w) for w in cone.words[cone.forced_mask])
        expected = sorted([2222222] + [int("2" * k + "1" + "2" * (6 - k)) for k in range(6)])
        self.assertEqual(forced, expected)
        self.assertEqual(int(cone.free_mask.sum()), 245)
        self.assertEqual(int(cone.active_mask.sum()), 247)
        # the three classes partition the non-empty words
        classes = cone.free_mask.astype(int) + cone.forced_mask.astype(int) + cone.lead_mask.astype(int)
        np.testing.assert_array_equal(classes[1:], 1)
        self.assertEqual(classes[0], 0)
        np.testing.assert_array_equal(cone.lower[list(cone.lead_idx)], cone.eps)
        self.assertTrue(np.all(np.isneginf(np.delete(cone.lower, list(cone.lead_idx)))))
        self.assertIn("222222", cone.describe())

        cone1 = em.make_b_cone(1)
        self.assertEqual((cone1.n, cone1.m, cone1.lead_idx), (None, None, ()))
        self.assertTrue(cone1.free_mask[1:].all() and not cone1.forced_mask.any())

        cone2 = em.make_b_cone(2)
        self.assertEqual((cone2.n, cone2.m), (1, None))
        self.assertEqual([int(cone2.words[i]) for i in cone2.lead_idx], [22])
        self.assertFalse(cone2.forced_mask.any())

        with self.assertRaises(ValueError):
            em.make_b_cone(5, n=3)

    def test_project(self):
        rng = np.random.default_rng(0)
        cone = em.make_b_cone(7)
        for _ in range(5):
            p = jnp.asarray(rng.normal(size=cone.n_sig))
            pp = em.project(p, cone)
            np.testing.assert_array_equal(np.asarray(em.project(pp, cone)), np.asarray(pp))
            self.assertTrue(em.in_cone(pp, cone))
            self.assertEqual(float(pp[0]), 0.0)
            np.testing.assert_array_equal(np.asarray(pp)[cone.forced_mask], 0.0)
            self.assertTrue(np.all(np.asarray(pp)[list(cone.lead_idx)] >= cone.eps))
            q = random_cone_point(cone, rng, 1.0, 0.1, 1.0)
            np.testing.assert_array_equal(np.asarray(em.project(q, cone)), np.asarray(q))
        self.assertFalse(em.in_cone(jnp.zeros(cone.n_sig), cone))   # leads must be >= eps

    def test_pathwise_upper_bound(self):
        """Prop. 4.3 / note §6.4 along linear paths (t, ct): exp(T e1 + cT e2) has coordinates
        T^{|v|} c^{|v|_2} / |v|!, and <-p, .> must tend to -inf as |c| -> inf for p in K."""
        N = 7
        cone = em.make_b_cone(N)
        lengths = np.array([len(str(w)) if w else 0 for w in cone.words])
        k2 = cone.state_degree
        fact = np.array([float(np.prod(np.arange(1, k + 1))) for k in lengths])

        def sig_line(c):
            return T ** lengths * c ** k2 / fact

        # the closed form is the tensor exponential of the increment
        c0 = 1.7
        exact = ta.tensor_exp(ta.from_dict({1: T, 2: c0 * T}, trunc=N, dim=2))
        np.testing.assert_allclose(np.asarray(exact.array).ravel(), sig_line(c0), rtol=1e-12, atol=1e-14)

        cs = np.linspace(-50, 50, 401)
        tail = np.abs(cs) >= 25
        rng = np.random.default_rng(1)
        for _ in range(5):
            p = np.asarray(random_cone_point(cone, rng, 0.05, 0.1, 1.0))
            f = np.array([-p @ sig_line(c) for c in cs])
            self.assertTrue(np.all(np.isfinite(f)))
            right, left = f[cs >= 25], f[cs <= -25]
            self.assertTrue(np.all(np.diff(right) < 0), "not decreasing on the right tail")
            self.assertTrue(np.all(np.diff(left) > 0), "not increasing on the left tail")
            self.assertLess(max(f[0], f[-1]), f[~tail].max() - 1.0)

            # flipping the sign of the pure-state lead makes the functional unbounded above
            p_bad = p.copy()
            i6, i7 = cone.lead_idx
            p_bad[i6] = -rng.uniform(0.5, 1.0)
            p_bad[i7] = rng.uniform(0.1, 1.0)
            g = np.array([-p_bad @ sig_line(c) for c in cs])
            self.assertTrue(np.all(np.diff(g[cs >= 25]) > 0))
            self.assertTrue(np.all(np.diff(g[cs <= -25]) < 0))
            self.assertGreater(min(g[0], g[-1]), g[~tail].max() + 1.0)


class TestDualProblem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.N = 5
        cls.cone = em.make_b_cone(cls.N)
        cls.S, cls.W = brownian_sample(cls.N, 20_000, n_steps=50, seed=0)

    def _problem(self, kappa=1.0, weights=None, E=None, seed=0):
        if E is None:
            E = np.asarray(expected_bm_signature(T, self.N).array).ravel()
        return em.DualProblem(self.S, E, kappa, self.cone, weights)

    def _weights(self, seed=0):
        w = np.exp(np.random.default_rng(seed).uniform(-1, 1, size=self.cone.n_sig))
        return w

    def test_zero_tilt_matches_bm_expectation(self):
        problem = self._problem()
        tilt = problem.tilt(jnp.zeros(self.cone.n_sig))
        self.assertAlmostEqual(float(tilt.logZ), 0.0, places=12)
        np.testing.assert_allclose(np.asarray(tilt.w), 1.0 / problem.M, rtol=1e-10)
        self.assertAlmostEqual(float(tilt.ess), problem.M, delta=1e-6 * problem.M)
        ref = np.asarray(expected_bm_signature(T, self.N).array).ravel()
        # The sample is the signature of the piecewise-linear interpolation on 50 steps, whose
        # expectation differs from exp(T(1 + 22/2)) by O(dt) on mixed words (e.g. 11212), more
        # than 5 s.e. at M = 2e4. So the z-scores are taken against the exact discrete law, and
        # that law is checked to be within O(dt) of the continuous one.
        disc = discrete_bm_expected_signature(self.cone, n_steps=self.W.shape[1] - 1)
        self.assertLess(np.abs(disc - ref).max(), T * T / (self.W.shape[1] - 1))
        _, se = em.expected_signature(self.S)
        m = np.asarray(tilt.m)
        det = se < 1e-12                   # pure-time words (and the empty word) are deterministic
        np.testing.assert_allclose(m[det], ref[det], rtol=1e-10)
        z = np.abs(m[~det] - disc[~det]) / se[~det]
        self.assertLess(z.max(), 5.0, f"max z-score {z.max():.2f}")

    def test_gaussian_closed_forms(self):
        cone, n_sig, M = self.cone, self.cone.n_sig, self.S.shape[1]
        problem = self._problem(kappa=1.0)
        S2 = self.S[idx(2)]
        # word 2: <p, S> = p W_T, Gaussian: log Z = kappa^2 p^2 T / 2, m^2 = -kappa p T
        p_val = 0.3
        tilt = problem.tilt(jnp.zeros(n_sig).at[idx(2)].set(p_val))
        ell = -p_val * S2
        e = np.exp(ell - ell.max())
        se_logZ = e.std(ddof=1) / (e.mean() * np.sqrt(M))
        self.assertLess(abs(float(tilt.logZ) - p_val ** 2 * T / 2), 5 * se_logZ)
        w = np.asarray(tilt.w)
        se_m = np.sqrt(np.sum(w ** 2 * (S2 - float(tilt.m[idx(2)])) ** 2))
        self.assertLess(abs(float(tilt.m[idx(2)]) + p_val * T), 5 * se_m)

        # word 22: <p, S> = p W_T^2 / 2: log Z = -1/2 log(1 + kappa p T)
        p_val = 0.5
        tilt = problem.tilt(jnp.zeros(n_sig).at[idx(22)].set(p_val))
        ell = -p_val * self.S[idx(22)]
        e = np.exp(ell - ell.max())
        se_logZ = e.std(ddof=1) / (e.mean() * np.sqrt(M))
        self.assertLess(abs(float(tilt.logZ) + 0.5 * np.log(1 + p_val * T)), 5 * se_logZ)

    def test_grad_hessian_autodiff(self):
        rng = np.random.default_rng(3)
        problem = self._problem(kappa=2.0, weights=self._weights(1),
                                E=np.asarray(self.S[:, :5000].mean(axis=1)))
        ix = problem.idx
        for _ in range(2):
            p = random_cone_point(self.cone, rng, 0.05, 0.01, 0.1, level_decay=0.5)
            g_ad = np.asarray(jax.grad(problem.value)(p))
            g = np.asarray(problem.grad(p))
            np.testing.assert_allclose(g[ix], g_ad[ix], rtol=1e-8, atol=1e-10 * np.abs(g_ad).max())
            np.testing.assert_array_equal(g[~self.cone.active_mask], 0.0)
            H_ad = np.asarray(jax.hessian(problem.value)(p))[np.ix_(ix, ix)]
            H = np.asarray(problem.hessian(p))
            self.assertEqual(H.shape, (len(ix), len(ix)))
            np.testing.assert_allclose(H, H_ad, rtol=1e-8, atol=1e-8 * np.abs(H_ad).max())

    def test_concavity(self):
        rng = np.random.default_rng(4)
        problem = self._problem(kappa=3.0, weights=self._weights(2))
        h = 0.05
        for _ in range(5):
            p = random_cone_point(self.cone, rng, 0.05, 0.01, 0.1, level_decay=0.5)
            d = jnp.asarray(rng.normal(size=self.cone.n_sig) * 0.05) * jnp.asarray(self.cone.active_mask)
            vals = np.array([float(problem.value(p + t * d)) for t in np.arange(-5, 6) * h])
            self.assertTrue(np.all(np.diff(vals, 2) <= 1e-10), np.diff(vals, 2).max())

    def test_weak_duality_saa(self):
        rng = np.random.default_rng(5)
        E = np.asarray(self.S[:, 10_000:].mean(axis=1)) * 1.05
        problem = self._problem(kappa=5.0, weights=self._weights(3), E=E)
        for _ in range(10):
            p = random_cone_point(self.cone, rng, 0.05, 1e-6, 0.1, level_decay=0.5)
            p2 = random_cone_point(self.cone, rng, 0.05, 1e-6, 0.1, level_decay=0.5)
            cost = problem.primal_cost(p)
            self.assertGreaterEqual(cost["total"], float(problem.value(p2)))
            self.assertGreaterEqual(problem.duality_gap(p), -1e-12)
            # the modified penalty is a relaxation of the plain one
            self.assertLessEqual(cost["penalty"], cost["penalty_unmodified"] + 1e-12)

    def _planted(self, p_true, kappa, weights):
        problem = self._problem(kappa=kappa, weights=weights)
        m = np.asarray(problem.tilt(p_true).m)
        E = m - np.asarray(p_true) / weights
        return em.DualProblem(self.S, E, kappa, self.cone, weights)

    def test_planted_solution(self):
        rng = np.random.default_rng(6)
        weights = self._weights(4)
        p_true = random_cone_point(self.cone, rng, 0.2, 0.02, 0.1, level_decay=0.5)
        problem = self._planted(p_true, kappa=1.0, weights=weights)
        self.assertGreater(float(problem.tilt(p_true).ess) / problem.M, 0.3)
        for method, rel, tol in (("newton", 1e-6, 1e-10), ("gradient", 1e-4, 1e-7)):
            res = em.optimize_dual(problem, method=method, tol=tol, max_iter=5000)
            self.assertTrue(res.converged, f"{method} did not converge: {res.history[-1]}")
            err = float(jnp.linalg.norm(res.p - p_true) / jnp.linalg.norm(p_true))
            self.assertLess(err, rel, f"{method}: rel err {err:.2e}")
            self.assertLess(abs(res.gap), 1e-8)
            self.assertEqual(res.active_leads, ())
            self.assertTrue(em.in_cone(res.p, self.cone))

    def test_active_constraint(self):
        rng = np.random.default_rng(7)
        weights = self._weights(5)
        p_true = np.array(random_cone_point(self.cone, rng, 0.2, 0.02, 0.1, level_decay=0.5))
        i_lead = self.cone.lead_idx[0]
        p_true[i_lead] = -0.05                      # stationary value below eps
        problem = self._planted(jnp.asarray(p_true), kappa=1.0, weights=weights)
        res = em.optimize_dual(problem, tol=1e-10)
        self.assertTrue(res.converged, res.history[-1])
        self.assertAlmostEqual(float(res.p[i_lead]), self.cone.eps, delta=1e-6)
        self.assertIn(i_lead, res.active_leads)
        self.assertLessEqual(float(problem.grad(res.p)[i_lead]), 0.0)
        self.assertLess(abs(res.gap), 1e-6)
        self.assertGreater(float(jnp.linalg.norm(res.p - p_true)), 1e-3)   # not the planted point

    def test_policy_value_consistency(self):
        """At t=0, X=Ø and with sig_brownian_1 = problem.S (T=1), the policy value is
        log E[exp<-kappa p, S>] = log Z = -V_p: this pins q = -kappa p."""
        rng = np.random.default_rng(8)
        kappa = 2.0
        problem = self._problem(kappa=kappa)
        p = random_cone_point(self.cone, rng, 0.05, 0.01, 0.1, level_decay=0.5)
        sigW1 = ta.TensorSequence(array=jnp.asarray(self.S), trunc=self.N, dim=2)
        policy = em.optimal_policy(p, kappa, sigW1, T, self.N)
        X0 = ta.from_array(jnp.zeros((self.cone.n_sig, 3)).at[0].set(1.0), trunc=self.N, dim=2)
        _, value, _ = policy(1, 0.0, X0)
        np.testing.assert_allclose(value, float(problem.tilt(p).logZ), rtol=0, atol=1e-10)

    def test_resample_moments(self):
        rng = np.random.default_rng(9)
        problem = self._problem(kappa=1.0)
        p = random_cone_point(self.cone, rng, 0.1, 0.01, 0.1, level_decay=0.5)
        n = 20_000
        paths = em.resample(problem, p, self.W, n=n, seed=10)
        shifted = em.resample(problem, p, self.W, n=n, seed=10, x0=0.7)
        np.testing.assert_allclose(shifted, paths + 0.7)            # Brownian paths start at 0
        t_grid = np.linspace(0.0, T, self.W.shape[1])
        S_res = em.path_signatures(paths, t_grid, self.N)
        mean, se = em.expected_signature(S_res)
        m = np.asarray(problem.tilt(p).m)
        det = se < 1e-12
        np.testing.assert_allclose(mean[det], m[det], rtol=1e-10)
        z = np.abs(mean[~det] - m[~det]) / se[~det]
        self.assertLess(z.max(), 5.0, f"max z-score {z.max():.2f}")


class TestClosedFormN1(unittest.TestCase):
    def test_closed_form_mfc_N1(self):
        """N=1, E=(T, e), uniform weights: p*_2 = -e/(1+kappa T), V = kappa e^2 / (2(1+kappa T)),
        optimal control constant a = kappa e / (1+kappa T)."""
        kappa, e, M = 2.0, 0.5, 50_000
        cone = em.make_b_cone(1)
        S, _ = brownian_sample(1, M, n_steps=10, seed=11)
        E = np.array([1.0, T, e])
        problem = em.DualProblem(S, E, kappa, cone)
        res = em.optimize_dual(problem, tol=1e-12)
        self.assertTrue(res.converged)

        p_star = -e / (1 + kappa * T)
        V = kappa * e ** 2 / (2 * (1 + kappa * T))
        a = kappa * e / (1 + kappa * T)
        tilt = problem.tilt(res.p)
        se_p = np.sqrt(T / float(tilt.ess)) / (1 + kappa * T)   # se of m-hat^2, propagated
        self.assertAlmostEqual(float(res.p[idx(1)]), 0.0, places=12)
        self.assertLess(abs(float(res.p[idx(2)]) - p_star), 5 * se_p)
        ell = -kappa * np.asarray(res.p) @ S
        w = np.exp(ell - ell.max())
        se_V = w.std(ddof=1) / (w.mean() * np.sqrt(M))
        self.assertLess(abs(res.J - V), 5 * se_V + 1e-12)
        self.assertLess(abs(res.gap), 1e-10)

        # the feedback is the constant a; an independent sample at level 1 suffices
        S_mc, _ = brownian_sample(1, 5_000, n_steps=10, seed=12)
        sigW1 = ta.TensorSequence(array=jnp.asarray(S_mc), trunc=1, dim=2)
        policy = em.optimal_policy(res.p, kappa, sigW1, T, 1)
        X = ta.from_array(jnp.asarray(np.array([[1.0, 1.0, 1.0], [0.0, 0.3, 0.7], [0.0, -0.4, 1.2]])),
                          trunc=1, dim=2)
        for i, t in ((1, 0.0), (5, 0.3), (9, 0.7)):
            alpha, _, _ = policy(i, t, X)
            np.testing.assert_allclose(alpha, -kappa * float(res.p[idx(2)]), rtol=1e-12)
            self.assertTrue(np.all(np.abs(alpha - a) < 5 * kappa * se_p))


@unittest.skipUnless(RUN_SLOW, "slow; set RUN_SLOW=1")
class TestEndToEnd(unittest.TestCase):
    def test_end_to_end_ou_small(self):
        N, kappa, n_steps = 3, 100.0, 50
        cone = em.make_b_cone(N)
        t_grid, X = em.target_paths("ou", T, n_steps, 20_000, seed=0)
        E, E_se = em.expected_signature(em.path_signatures(X, t_grid, N))
        _, W = em.brownian_paths(T, n_steps, 50_000, seed=1)
        S = em.path_signatures(W, t_grid, N)
        weights = em.word_weights("uniform", None, cone)
        res = em.continuation(S, E, cone, [1.0, 10.0, kappa], weights)[-1]
        self.assertTrue(res.converged)
        problem = em.DualProblem(S, E, kappa, cone, weights)
        tilt = problem.tilt(res.p)
        predicted = E + np.asarray(res.p) / weights
        predicted[0] = 1.0

        S_mc, _ = brownian_sample(N, 20_000, n_steps=n_steps, seed=2)
        sigW1 = ta.TensorSequence(array=jnp.asarray(S_mc), trunc=N, dim=2)
        size = 4000
        roll = em.rollout(res.p, kappa, T, N, n_steps, size, sigW1, seed=3)
        S_roll = roll.sig[:, :, -1]
        m_roll, se_roll = em.expected_signature(S_roll)

        # rollout mean signature vs the theoretical residual E + Lambda^{-1} p*
        w = np.asarray(tilt.w)
        se_dual = np.sqrt(np.sum(w ** 2 * (S - np.asarray(tilt.m)[:, None]) ** 2, axis=1))
        se = np.sqrt(se_roll ** 2 + se_dual ** 2)
        det = se < 1e-12
        np.testing.assert_allclose(m_roll[det], predicted[det], rtol=1e-8)
        z = np.abs(m_roll[~det] - predicted[~det]) / se[~det]
        self.assertLess(z.max(), 5.0, f"max z {z.max():.2f}")

        # energy of the controlled paths vs the entropy of the optimal law (note §5.3)
        dt = np.diff(t_grid)
        energy_path = 0.5 * np.sum(roll.control[:, :-1] ** 2 * dt, axis=1)
        energy = energy_path.mean()
        entropy = problem.primal_cost(res.p)["entropy"]
        self.assertLess(abs(energy - entropy) / entropy, 0.05, f"energy {energy:.4f} vs H {entropy:.4f}")

        # auxiliary problem: E[1/2 int alpha^2 + kappa <p*, Sig_T>] = V_p* = -log Z(p*)
        aux_path = energy_path + kappa * (np.asarray(res.p) @ S_roll)
        self.assertLess(abs(aux_path.mean() + float(tilt.logZ)), 5 * aux_path.std() / np.sqrt(size))

        # empirical primal cost (energy + bias-corrected penalty) vs J(p*). The penalty estimate is
        # noisy (kappa times squared residuals of the rollout mean), so the tolerance is 5 % or 5
        # bootstrap standard errors, whichever is larger.
        def cost(rows):
            m, se_m = em.expected_signature(S_roll[:, rows])
            pen = 0.5 * kappa * np.sum(weights[1:] * ((m - E)[1:] ** 2 - se_m[1:] ** 2))
            return energy_path[rows].mean() + pen
        boot_rng = np.random.default_rng(4)
        boot = [cost(boot_rng.integers(0, size, size)) for _ in range(200)]
        total = cost(np.arange(size))
        self.assertLess(abs(total - res.J), max(0.05 * res.J, 5 * np.std(boot)),
                        f"primal cost {total:.4f} (boot se {np.std(boot):.4f}) vs J {res.J:.4f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
