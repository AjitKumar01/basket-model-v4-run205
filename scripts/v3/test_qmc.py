"""Regression tests for the full-catalogue normaliser path.

Run from the repository root with:
    python -m unittest scripts.v3.test_qmc
"""
import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ragged import (RaggedIndex, RaggedModel, esp_bucketed, log_f_ragged,
                    log_f_sparse, gh_grid, set_quad, sparse_prepare,
                    poly_tree, poly_tree_degree_aware, _poly_mul_trunc,
                    _poly_mul_trunc_eager, _esp_product_tree)
from fit import phi_control_adjustment
from qmc_multimode_probe import resolve_checkpoint_R


torch.set_default_dtype(torch.float64)


def one_row_model(J, Kz, nmax=None):
    nmax = J if nmax is None else nmax
    ix = RaggedIndex(torch.arange(J), torch.zeros(J, dtype=torch.long),
                     torch.tensor([0]), torch.tensor([0]), 1)
    m = RaggedModel(J, 1, 1, K=2, Kz=Kz, nmax=nmax, R=nmax,
                    S=1, Kp=2, Kt=2, Ks=2)
    m.house, m.ctx = torch.tensor([0]), None
    return m, ix


class FullCatalogueQmcTest(unittest.TestCase):
    def test_native_balanced_esp_matches_subtraction_free_reference_and_gradient(self):
        try:
            from poly_degree_native import esp_tree_native
        except ImportError as exc:
            self.skipTest(str(exc))
        generator = torch.Generator().manual_seed(194)
        weights = (0.2 * torch.rand(3, 5, 96, generator=generator)).requires_grad_()
        native_weights = weights.detach().clone().requires_grad_()
        expected = _esp_product_tree(weights, 96)
        actual = esp_tree_native(native_weights, 96)
        direction = torch.randn(expected.shape, generator=generator)
        expected_gradient = torch.autograd.grad((expected * direction).sum(), weights)[0]
        actual_gradient = torch.autograd.grad(
            (actual * direction).sum(), native_weights)[0]
        self.assertLess(float((actual - expected).norm() / expected.norm()), 1e-14)
        self.assertLess(float((actual_gradient - expected_gradient).norm()
                              / expected_gradient.norm()), 1e-14)

    def test_log_native_polynomials_match_reference_values_and_gradients(self):
        try:
            from poly_degree_native import (esp_tree_log_native,
                                             log_poly_tree_degree_native)
        except ImportError as exc:
            self.skipTest(str(exc))
        generator = torch.Generator().manual_seed(508)

        logw = (-7.0 * torch.rand(2, 4, 32, generator=generator)).requires_grad_()
        ref_logw = logw.detach().clone().requires_grad_()
        got_esp = esp_tree_log_native(logw, 24)
        ref_esp = torch.log(_esp_product_tree(torch.exp(ref_logw), 24))
        esp_direction = torch.randn(got_esp.shape, generator=generator)
        got_grad = torch.autograd.grad((got_esp * esp_direction).sum(), logw)[0]
        ref_grad = torch.autograd.grad((ref_esp * esp_direction).sum(), ref_logw)[0]
        self.assertLess(float((got_esp - ref_esp).abs().max()), 2e-14)
        self.assertLess(float((got_grad - ref_grad).abs().max()), 2e-12)

        nmax, categories = 16, 7
        degrees = torch.tensor([2, 5, 3, 8, 1, 6, 4]).expand(3, -1).contiguous()
        raw = -5.0 * torch.rand(2, 3, categories, nmax + 1, generator=generator)
        live = torch.arange(nmax + 1).view(1, 1, 1, -1) <= degrees.view(1, 3, -1, 1)
        raw = raw.masked_fill(~live, -float("inf"))
        logg = raw.clone().requires_grad_(True)
        ref_logg = raw.clone().requires_grad_(True)
        got = log_poly_tree_degree_native(logg, degrees, nmax)
        ref = torch.log(poly_tree_degree_aware(torch.exp(ref_logg), degrees, nmax))
        direction = torch.randn(got.shape, generator=generator)
        got_grad = torch.autograd.grad((got * direction).sum(), logg)[0]
        ref_grad = torch.autograd.grad((ref * direction).sum(), ref_logg)[0]
        self.assertLess(float((got - ref).abs().max()), 3e-13)
        self.assertLess(float((got_grad[live.expand_as(got_grad)]
                               - ref_grad[live.expand_as(ref_grad)]).abs().max()), 3e-11)

    def test_checkpoint_support_metadata_overrides_best_score_summary(self):
        """A *_best.json summary without R must not collapse complete support to 23."""
        self.assertEqual(resolve_checkpoint_R({"R": 120}, {"iter": 400, "R": 23}), 120)
        self.assertEqual(resolve_checkpoint_R({"R": 120}, {"iter": 400}), 120)
        self.assertEqual(resolve_checkpoint_R({}, {"R": 96}), 96)

    def test_augmented_gibbs_marginal_is_original_version4_energy(self):
        """Integrating z must recover the Gram interaction, including its Phi score."""
        torch.manual_seed(173)
        J, C, Kz = 7, 3, 4
        category = torch.tensor([0, 0, 1, 1, 1, 2, 2])
        m = RaggedModel(J, 1, C, K=2, Kz=Kz, nmax=J, R=J,
                        S=1, Kp=2, Kt=2, Ks=2, seed=47)
        with torch.no_grad():
            m.lam.copy_(torch.linspace(-1.8, -0.3, J))
            m.phi.normal_(0.0, 0.17)
            m.rho_c.copy_(torch.tensor([-0.12, 0.08, -0.04]))
            m.rho_0_free.copy_(0.03 * torch.arange(1, J + 1).square())

        masks = torch.arange(1, 1 << J)
        x = ((masks[:, None] >> torch.arange(J)) & 1).to(m.lam.dtype)
        n = x.sum(1).long()
        nc = torch.stack([x[:, category == c].sum(1) for c in range(C)], 1)
        mu = x @ m.phi

        original_pair = 0.5 * (mu.square().sum(1)
                               - x @ m.phi.square().sum(1))
        original = (x @ m.lam + original_pair
                    - (m.rho_c * nc * (nc - 1.0) / 2.0).sum(1)
                    - m.rho_0()[n])

        # In the augmented law, integrating N(z;0,I) exp(z'mu) contributes
        # exp(||mu||^2/2) exactly.  Everything else is the conditional S|z energy.
        integrated = (x @ m.lam - 0.5 * x @ m.phi.square().sum(1)
                      + 0.5 * mu.square().sum(1)
                      - (m.rho_c * nc * (nc - 1.0) / 2.0).sum(1)
                      - m.rho_0()[n])
        self.assertLess(float((integrated - original).detach().abs().max()), 2e-15)

        got = torch.logsumexp(integrated, 0)
        want = torch.logsumexp(original, 0)
        got_grad = torch.autograd.grad(got, m.phi, retain_graph=True)[0]
        want_grad = torch.autograd.grad(want, m.phi)[0]
        self.assertLess(abs(float((got - want).detach())), 2e-15)
        self.assertLess(float((got_grad - want_grad).abs().max()), 2e-14)

    def test_degree_aware_category_product_matches_dense_tree_and_gradient(self):
        """Removing proven zero coefficients must be an algebraic kernel change only."""
        torch.manual_seed(179)
        nmax = 10
        degrees = [1, 4, 2, 7, 3, 6]
        raw = torch.rand(2, 3, len(degrees), nmax + 1)
        for row, degree in enumerate(degrees):
            raw[:, :, row, degree + 1:] = 0.0
        dense_input = raw.clone().requires_grad_(True)
        degree_input = raw.clone().requires_grad_(True)

        dense = poly_tree(dense_input, nmax)
        degree_matrix = torch.tensor(degrees).expand(3, -1)
        running = poly_tree_degree_aware(degree_input, degree_matrix, nmax)

        self.assertLess(float((running - dense).abs().max()), 2e-13)
        probe = torch.randn_like(dense)
        dense_grad = torch.autograd.grad((dense * probe).sum(), dense_input)[0]
        degree_grad = torch.autograd.grad((running * probe).sum(), degree_input)[0]
        live = torch.stack([torch.arange(nmax + 1) <= degree for degree in degrees])
        live = live.view(1, 1, len(degrees), nmax + 1).expand_as(degree_grad)
        self.assertLess(float((degree_grad[live] - dense_grad[live]).abs().max()),
                        2e-12)

    def test_multilevel_conditional_control_cycle_targets_exact_joint_score(self):
        """Averaging over correction positions must recover the joint gradient exactly."""
        torch.manual_seed(167)
        J, C, Kz = 6, 2, 3
        category = torch.tensor([0, 0, 0, 1, 1, 1])
        ix = RaggedIndex(torch.arange(J), category,
                         torch.zeros(C, dtype=torch.long), torch.arange(C), 1)
        m = RaggedModel(J, 1, C, K=2, Kz=Kz, nmax=J, R=J,
                        S=1, Kp=2, Kt=2, Ks=2, seed=43)
        m.house, m.ctx = torch.tensor([0]), None
        with torch.no_grad():
            m.cat_of.copy_(category)
            m.lam.copy_(torch.tensor([-1.2, -0.3, -0.8, -1.0, -0.4, -1.5]))
            m.phi.copy_(torch.tensor([[0.22, -0.07, 0.04], [0.11, 0.18, -0.09],
                                      [-0.14, 0.06, 0.20], [0.08, 0.23, -0.10],
                                      [0.17, -0.03, 0.15], [-0.05, 0.12, 0.19]]))
            m.rho_c.copy_(torch.tensor([-0.13, 0.09]))
            m.rho_0_free.copy_(torch.tensor([0.0, 0.05, 0.16, 0.34, 0.61, 0.95]))

        masks = torch.arange(1, 1 << J)
        subsets = ((masks[:, None] >> torch.arange(J)) & 1).to(m.lam.dtype)
        observations = [torch.tensor(x, dtype=m.lam.dtype) for x in (
            [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1],
            [1, 1, 0, 0, 1, 0], [0, 0, 1, 1, 0, 1])]

        def direct_energy(x):
            summed = x @ m.phi
            pair = 0.5 * (summed.square().sum() - x @ m.phi.square().sum(1))
            counts = torch.stack([x[category == c].sum() for c in range(C)])
            return (x @ m.lam + pair
                    - (m.rho_c * counts * (counts - 1.0) / 2.0).sum()
                    - m.rho_0()[x.sum().long()])

        high, cheap = [], []
        for observed in observations:
            logz = torch.logsumexp(torch.stack([direct_energy(x) for x in subsets]), 0)
            high.append(torch.autograd.grad(direct_energy(observed) - logz, m.phi)[0])
            bought = torch.nonzero(observed, as_tuple=False).flatten().long()
            pseudo = m.pseudo_loglik(
                ix, bought, torch.zeros(len(bought), dtype=torch.long),
                category[bought], 1, line_ctx=None).sum()
            cheap.append(0.5 * torch.autograd.grad(pseudo, m.phi)[0])

        high = torch.stack(high)
        cheap = torch.stack(cheap)
        target = high.mean(0)
        # One correction position is selected uniformly per cycle.  The cheap mean uses
        # all k batches; high_i-cheap_i is evaluated only at the selected position.
        estimates = cheap.mean(0).unsqueeze(0) + high - cheap
        self.assertLess(float((estimates.mean(0) - target).abs().max()), 2e-16)
        for correction in range(len(observations)):
            ordered = [cheap[i] for i in range(len(observations)) if i != correction]
            ordered.append(cheap[correction])
            implemented = high[correction] + phi_control_adjustment(ordered)
            self.assertLess(float((implemented - estimates[correction]).abs().max()),
                            2e-16)

    def test_pseudolikelihood_is_exact_single_site_conditional_of_joint_energy(self):
        """The cheap conditional objective must not define a different basket law."""
        torch.manual_seed(163)
        J, C, Kz = 6, 2, 3
        category = torch.tensor([0, 0, 0, 1, 1, 1])
        ix = RaggedIndex(torch.arange(J), category,
                         torch.zeros(C, dtype=torch.long), torch.arange(C), 1)
        m = RaggedModel(J, 1, C, K=2, Kz=Kz, nmax=J, R=J,
                        S=1, Kp=2, Kt=2, Ks=2, seed=41)
        m.house, m.ctx = torch.tensor([0]), None
        with torch.no_grad():
            m.cat_of.copy_(category)
            m.lam.copy_(torch.tensor([-1.2, -0.3, -0.8, -1.0, -0.4, -1.5]))
            m.phi.copy_(torch.tensor([[0.22, -0.07, 0.04], [0.11, 0.18, -0.09],
                                      [-0.14, 0.06, 0.20], [0.08, 0.23, -0.10],
                                      [0.17, -0.03, 0.15], [-0.05, 0.12, 0.19]]))
            m.rho_c.copy_(torch.tensor([-0.13, 0.09]))
            m.rho_0_free.copy_(torch.tensor([0.0, 0.05, 0.16, 0.34, 0.61, 0.95]))

        observed = torch.tensor([1., 0., 1., 0., 1., 0.])
        bought = torch.nonzero(observed, as_tuple=False).flatten().long()
        line_trip = torch.zeros(len(bought), dtype=torch.long)
        got = m.pseudo_loglik(
            ix, bought, line_trip, category[bought], 1, line_ctx=None).sum()

        def direct_energy(x):
            summed = x @ m.phi
            pair = 0.5 * (summed.square().sum() - x @ m.phi.square().sum(1))
            counts = torch.stack([x[category == c].sum() for c in range(C)])
            return (x @ m.lam + pair
                    - (m.rho_c * counts * (counts - 1.0) / 2.0).sum()
                    - m.rho_0()[x.sum().long()])

        terms = []
        for j in range(J):
            absent, present = observed.clone(), observed.clone()
            absent[j], present[j] = 0.0, 1.0
            delta = direct_energy(present) - direct_energy(absent)
            terms.append(torch.nn.functional.logsigmoid(
                delta if bool(observed[j]) else -delta))
        want = torch.stack(terms).sum()
        got_grad = torch.autograd.grad(got, (m.lam, m.phi, m.rho_c, m.rho_0_free),
                                       retain_graph=True)
        want_grad = torch.autograd.grad(want, (m.lam, m.phi, m.rho_c, m.rho_0_free))
        self.assertLess(abs(float((got - want).detach())), 2e-14)
        for actual, expected in zip(got_grad, want_grad):
            self.assertLess(float((actual - expected).abs().max()), 2e-13)

    def test_latent_hessian_is_full_pair_covariance(self):
        """The Laplace geometry is Cov(sum phi_j x_j | z) - I, cross terms included."""
        torch.manual_seed(151)
        J, C, Kz = 6, 2, 3
        category = torch.tensor([0, 0, 0, 1, 1, 1])
        ix = RaggedIndex(torch.arange(J), category,
                         torch.zeros(C, dtype=torch.long), torch.arange(C), 1)
        m = RaggedModel(J, 1, C, K=2, Kz=Kz, nmax=J, R=J,
                        S=1, Kp=2, Kt=2, Ks=2, seed=29)
        m.house, m.ctx = torch.tensor([0]), None
        with torch.no_grad():
            m.cat_of.copy_(category)
            m.lam.copy_(torch.tensor([-1.1, -0.4, -1.3, -0.7, -0.9, -0.2]))
            m.phi.copy_(torch.tensor([[0.28, -0.11, 0.04], [0.21, 0.17, -0.08],
                                      [-0.16, 0.13, 0.19], [0.10, 0.25, -0.12],
                                      [0.18, -0.04, 0.23], [-0.09, 0.20, 0.15]]))
            m.rho_c.copy_(torch.tensor([-0.11, 0.06]))
            nval = torch.arange(1, J + 1, dtype=m.lam.dtype)
            m.rho_0_free.copy_(0.025 * nval.square())

        cache = sparse_prepare(m, ix)
        z0 = torch.tensor([0.17, -0.09, 0.13])

        def latent_gradient(zvec):
            # esp_bucketed's custom backward is first-order only.  Asking PyTorch for a
            # second autograd derivative differentiates the implementation of that custom
            # backward and does not produce the Hessian of log f.  The production audit
            # therefore measures the Hessian by central differences of the exact first
            # gradient, as qmc_multimode_probe.py does.
            zvec = zvec.detach().requires_grad_(True)
            z = zvec.view(1, 1, Kz)
            value = (log_f_sparse(m, z, ix, cache, True, detach_params=True).sum()
                     - 0.5 * zvec.square().sum())
            return torch.autograd.grad(value, zvec)[0]

        eps = 1e-4
        columns = []
        for k in range(Kz):
            direction = torch.zeros(Kz)
            direction[k] = eps
            columns.append((latent_gradient(z0 + direction)
                            - latent_gradient(z0 - direction)) / (2.0 * eps))
        got = torch.stack(columns, dim=1)

        mask = torch.arange(1, 1 << J)
        X = ((mask[:, None] >> torch.arange(J)) & 1).to(m.lam.dtype)
        n = X.sum(1).to(torch.long)
        mu = X @ m.phi.detach()
        nc = torch.stack([X[:, category == c].sum(1) for c in range(C)], 1)
        # These are the conditional S|z weights inside f(z).  The Gram pair term is
        # represented by z'mu and the Hubbard--Stratonovich diagonal correction.
        logits = (X @ m.b_flat(ix).detach()
                  - 0.5 * X @ m.phi.detach().square().sum(1)
                  + mu @ z0
                  - (m.rho_c.detach() * nc * (nc - 1.0) / 2.0).sum(1)
                  - m.rho_0().detach()[n])
        probability = torch.softmax(logits, 0)
        mean = (probability[:, None] * mu).sum(0)
        centred = mu - mean
        covariance = torch.einsum("s,si,sj->ij", probability, centred, centred)
        want = covariance - torch.eye(Kz)
        self.assertLess(float((got - want).abs().max()), 2e-10)

        # At full sketch rank, the production retry frame must reconstruct that same
        # covariance without differentiating the custom ESP backward a second time.
        m.quad_subspace_rank = Kz
        m.quad_subspace_iters = 0
        m.quad_subspace_eps = eps
        Q, sd = m._size_subspace_frame(ix, z0.view(1, 1, Kz), cache, True)
        implied_covariance = (Q[0, 0]
                              @ torch.diag(1.0 - sd[0, 0].square().reciprocal())
                              @ Q[0, 0].T)
        self.assertLess(float((implied_covariance - covariance).abs().max()), 2e-10)
        self.assertLess(float((Q[0, 0].T @ Q[0, 0] - torch.eye(Kz)).abs().max()),
                        2e-14)
        m.quad_subspace_rank = 0

        # The diagonal-Bernoulli shortcut omits nonzero cross-item covariances here.
        incidence = (probability[:, None] * X).sum(0)
        diagonal_proxy = ((m.phi.detach()
                           * (incidence * (1.0 - incidence)).sqrt()[:, None]).T
                          @ (m.phi.detach()
                             * (incidence * (1.0 - incidence)).sqrt()[:, None]))
        self.assertGreater(float((diagonal_proxy - covariance).abs().max()), 1e-3)

    def test_interaction_spectral_direction_uses_contextual_model_score(self):
        """At Phi=0, the O(epsilon^2) gain is the exact pair-score quadratic form."""
        torch.manual_seed(157)
        J = 7
        mask = torch.arange(1, 1 << J)
        X = ((mask[:, None] >> torch.arange(J)) & 1).to(torch.float64)
        n = X.sum(1).to(torch.long)
        b = torch.tensor([-1.5, -0.3, -1.1, -0.6, -1.8, -0.8, -0.4])
        rho0 = 0.035 * torch.arange(J + 1, dtype=torch.float64).square()
        base_energy = X @ b - rho0[n]
        p0 = torch.softmax(base_energy, 0)

        observed = torch.tensor([1., 1., 0., 0., 1., 0., 1.])
        observed_T = torch.outer(observed, observed) - torch.diag(observed)
        model_T = torch.einsum("s,si,sj->ij", p0, X, X) - torch.diag(
            (p0[:, None] * X).sum(0))
        residual = observed_T - model_T
        eigenvalue, eigenvector = torch.linalg.eigh(residual)
        v = eigenvector[:, -1]
        self.assertGreater(float(eigenvalue[-1]), 0.0)
        score_coefficient = 0.5 * torch.einsum("i,ij,j->", v, residual, v)

        def exact_loglik(epsilon):
            phi = epsilon * v
            summed = X @ phi
            pair = 0.5 * (summed.square() - X @ phi.square())
            energy = base_energy + pair
            observed_pair = 0.5 * ((observed @ phi).square()
                                   - observed @ phi.square())
            return observed @ b - rho0[int(observed.sum())] + observed_pair \
                - torch.logsumexp(energy, 0)

        eps = 1e-3
        finite_coefficient = (exact_loglik(eps) - exact_loglik(0.0)) / eps ** 2
        self.assertLess(abs(float(finite_coefficient - score_coefficient)), 2e-6)

    def test_context_gauge_projection_preserves_utility_and_sparse_gradients(self):
        """Gauge fixing must not turn sparse minibatches into dense Adam updates."""
        torch.manual_seed(37)
        m = RaggedModel(7, 5, 1, K=3, Kz=2, nmax=4, R=4,
                        S=4, Kp=2, Kt=2, Ks=2, n_week=6)
        item = torch.tensor([0, 2, 4])
        trip = torch.tensor([0, 0, 1])
        m.house = torch.tensor([1, 3])
        ctx = dict(dlp=torch.zeros(3), disp=torch.zeros(3), mail=torch.zeros(3),
                   week=torch.tensor([2, 2, 4]), store=torch.tensor([1, 1, 3]))
        before = m.b_at(item, trip, ctx).detach().clone()
        m.zero_grad(set_to_none=True)
        m.b_at(item, trip, ctx).sum().backward()

        active_house = torch.tensor([False, True, False, True, False])
        active_week = torch.tensor([False, False, True, False, True, False])
        active_store = torch.tensor([False, True, False, True])
        self.assertEqual(float(m.theta.grad[~active_house].abs().max()), 0.0)
        self.assertEqual(float(m.delta.grad[~active_week].abs().max()), 0.0)
        self.assertEqual(float(m.xi.grad[~active_store].abs().max()), 0.0)

        m.project_context_gauges()
        after = m.b_at(item, trip, ctx).detach()
        self.assertLess(float((after - before).abs().max()), 2e-14)
        self.assertLess(float(m.theta.mean(0).abs().max()), 2e-16)
        self.assertLess(float(m.delta.mean(0).abs().max()), 2e-16)
        self.assertLess(float(m.xi.mean(0).abs().max()), 2e-16)

    def test_original_joint_rollout_matches_enumerated_size_and_incidence(self):
        """Corollary 3 must sample the same joint law whose log Z is fitted."""
        torch.manual_seed(23)
        B, J, C, Kz = 2000, 5, 2, 2
        item = torch.arange(J).repeat(B)
        local_row = torch.tensor([0, 0, 1, 1, 1]).repeat(B)
        row_base = torch.arange(B).repeat_interleave(J) * C
        row_of = row_base + local_row
        row_trip = torch.arange(B).repeat_interleave(C)
        row_cat = torch.tensor([0, 1]).repeat(B)
        ix = RaggedIndex(item, row_of, row_trip, row_cat, B)
        m = RaggedModel(J, 1, C, K=2, Kz=Kz, nmax=J, R=J,
                        S=1, Kp=2, Kt=2, Ks=2, seed=5)
        m.house, m.ctx = torch.zeros(B, dtype=torch.long), None
        category = torch.tensor([0, 0, 1, 1, 1])
        with torch.no_grad():
            m.cat_of.copy_(category)
            m.lam.copy_(torch.tensor([-1.1, -0.8, -1.4, -0.5, -1.0]))
            m.phi.copy_(torch.tensor([[0.22, -0.08], [0.16, 0.12],
                                      [-0.11, 0.19], [0.07, 0.24], [0.18, 0.03]]))
            m.rho_c.copy_(torch.tensor([-0.12, 0.07]))
            m.rho_0_free.copy_(torch.tensor([0.0, 0.08, 0.22, 0.45, 0.80]))
        # Positive dense GH weights make stage 1 a directly sampled discrete posterior.
        # At this small interaction strength q=9 agrees with analytic subset weights far
        # below the Monte Carlo tolerance being tested.
        m.quad_z = gh_grid(Kz, 9)

        mask = torch.arange(1, 1 << J)
        X = ((mask[:, None] >> torch.arange(J)) & 1).to(m.lam.dtype)
        n = X.sum(1).to(torch.long)
        summed_phi = X @ m.phi
        pair = 0.5 * (summed_phi.square().sum(1)
                      - X @ m.phi.square().sum(1))
        nc = torch.stack([X[:, category == c].sum(1) for c in range(C)], dim=1)
        energy = (X @ m.lam + pair
                  - (m.rho_c[None, :] * nc * (nc - 1.0) / 2.0).sum(1)
                  - m.rho_0()[n])
        probability = torch.softmax(energy, 0)
        exact_mean = float((probability * n).sum())
        exact_incidence = (probability[:, None] * X).sum(0)

        draws = m.sample(ix, generator=torch.Generator().manual_seed(901))
        sampled_n = torch.tensor([len(s) for s in draws], dtype=m.lam.dtype)
        sampled_incidence = torch.tensor(
            [[float(j in basket) for j in range(J)] for basket in draws]).mean(0)
        self.assertLess(abs(float(sampled_n.mean()) - exact_mean), 0.07)
        self.assertLess(float((sampled_incidence - exact_incidence).abs().max()), 0.035)

    def test_sampler_retains_full_support_beyond_linear_rho_boundary(self):
        """Rollout and likelihood must share the R=120 log-domain category potential."""
        torch.manual_seed(613)
        m, ix = one_row_model(120, 2, nmax=120)
        m._esp_native = True
        m._poly_degree_native = True
        with torch.no_grad():
            m.lam.fill_(-2.0)
            m.phi.zero_()
            m.phi[:, 0] = 0.01
            m.rho_c.fill_(-0.20)  # exp[-rho*C(120,2)] is far beyond float64
            # Keep several sizes relevant instead of making n=120 deterministic.
            n = torch.arange(1, 121, dtype=m.lam.dtype)
            m.rho_0_free.copy_(0.20 * n * (n - 1.0) / 2.0 + 0.02 * n.square())
        set_quad(m, qmc_n=32, qmc_seed=17, qmc_reps=4, Kz=2,
                 probe=-1, steps=2, chunk=16, size_bands=1, size_steps=2,
                 mode_logtol=4.0, mode_sep=1.0, mix_n=64)
        basket = m.sample(ix, generator=torch.Generator().manual_seed(29))[0]
        self.assertGreaterEqual(len(basket), 1)
        self.assertLessEqual(len(basket), 120)
        self.assertEqual(len(basket), len(set(basket)))

    def test_original_joint_value_gradients_size_and_incidence_match_enumeration(self):
        """Audit the complete version-4 law, not only the Gaussian integral's value.

        A correct log Z must give the same size law and parameter derivatives as direct
        enumeration.  In particular d log Z / d b_j is marginal incidence and the sum of
        those incidences is E[n].  This catches an estimator that looks accurate in value
        while optimizing the wrong model.
        """
        torch.manual_seed(113)
        J, C, Kz = 9, 3, 6
        category = torch.arange(J) // 3
        ix = RaggedIndex(torch.arange(J), category,
                         torch.zeros(C, dtype=torch.long), torch.arange(C), 1)
        m = RaggedModel(J, 1, C, K=2, Kz=Kz, nmax=J, R=J,
                        S=1, Kp=2, Kt=2, Ks=2, seed=17)
        m.house, m.ctx = torch.tensor([0]), None
        with torch.no_grad():
            m.cat_of.copy_(category)
            m.lam.copy_(torch.linspace(-2.1, -0.5, J))
            m.phi.normal_()
            m.phi.mul_(0.42 / m.phi.norm(dim=1, keepdim=True))
            m.rho_c.copy_(torch.tensor([-0.10, 0.04, -0.06]))
            n = torch.arange(1, J + 1, dtype=m.lam.dtype)
            m.rho_0_free.copy_(0.035 * n.square())

        # Differentiable exact enumeration of every non-empty subset.
        mask = torch.arange(1, 1 << J)
        X = ((mask[:, None] >> torch.arange(J)) & 1).to(m.lam.dtype)
        n = X.sum(1).to(torch.long)
        b = m.b_flat(ix)
        summed_phi = X @ m.phi
        pair = 0.5 * (summed_phi.square().sum(1)
                      - X @ m.phi.square().sum(1))
        nc = torch.stack([X[:, category == c].sum(1) for c in range(C)], dim=1)
        cat_pen = (m.rho_c[None, :] * nc * (nc - 1.0) / 2.0).sum(1)
        exact_energy = X @ b + pair - cat_pen - m.rho_0()[n]
        exact_lz = torch.logsumexp(exact_energy, 0)
        exact_prob = torch.softmax(exact_energy, 0)
        exact_pn = torch.zeros(J, dtype=m.lam.dtype).index_add_(
            0, n - 1, exact_prob)
        exact_incidence = (exact_prob[:, None] * X).sum(0)
        exact_grad = torch.autograd.grad(
            exact_lz, (m.lam, m.phi, m.rho_c, m.rho_0_free), retain_graph=True)

        set_quad(m, qmc_n=4096, qmc_seed=31, qmc_reps=4, Kz=Kz,
                 probe=Kz, steps=6, chunk=128, size_bands=1, size_steps=3,
                 mode_logtol=8.0, mode_sep=1.0, mix_n=4096)
        q_lz, q_pn = m.log_Z(ix, drop_empty=True, return_size=True)
        q_grad = torch.autograd.grad(
            q_lz, (m.lam, m.phi, m.rho_c, m.rho_0_free))

        self.assertLess(abs(float(q_lz - exact_lz)), 8e-4)
        self.assertLess(float((q_pn[0] - exact_pn).abs().max()), 8e-4)
        self.assertLess(float((q_grad[0] - exact_incidence).abs().max()), 8e-4)
        self.assertLess(abs(float(q_grad[0].sum() - (q_pn[0] *
                            torch.arange(1, J + 1)).sum())), 2e-12)
        self.assertLess(abs(float(exact_incidence.sum() - (exact_pn *
                            torch.arange(1, J + 1)).sum())), 2e-12)
        for got, want in zip(q_grad, exact_grad):
            self.assertLess(float((got - want).abs().max()), 2e-3)

        # Proposition 1 must survive the estimator: a common utility shift delta adds
        # n*delta to every basket, hence d E[n]/d delta = Var(n).  This is the mechanism
        # through which the original joint model obtains aggregate price response.
        eps = 1e-4
        q_var = ((q_pn[0] * torch.arange(1, J + 1).square()).sum()
                 - (q_pn[0] * torch.arange(1, J + 1)).sum().square())
        shifted_means = []
        with torch.no_grad():
            for delta in (-eps, eps):
                m.lam.add_(delta)
                _, shifted_pn = m.log_Z(ix, drop_empty=True, return_size=True)
                shifted_means.append(
                    (shifted_pn[0] * torch.arange(1, J + 1)).sum())
                m.lam.sub_(delta)
        finite_diff = (shifted_means[1] - shifted_means[0]) / (2.0 * eps)
        self.assertLess(abs(float(finite_diff - q_var)), 5e-5)

    def test_exact_revealed_set_conditioning_matches_enumeration(self):
        """Conditioning on R must not depend on an arbitrary finite utility pin."""
        torch.manual_seed(127)
        # Include one declared category with no remaining item.  Completion evaluation
        # retains such empty rows after revealed products are removed.  Their polynomial
        # is exactly (1,0,...), so they must remain on the stable native log path rather
        # than forcing the numerically weaker linear convolution.
        J, C, Kz = 6, 3, 4
        category = torch.tensor([0, 0, 0, 1, 1, 1])
        full_ix = RaggedIndex(torch.arange(J), category,
                              torch.zeros(C, dtype=torch.long), torch.arange(C), 1)
        m = RaggedModel(J, 1, C, K=2, Kz=Kz, nmax=J, R=J,
                        S=1, Kp=2, Kt=2, Ks=2, seed=19)
        m.house, m.ctx = torch.tensor([0]), None
        with torch.no_grad():
            m.cat_of.copy_(category)
            m.lam.copy_(torch.tensor([-1.2, -0.7, -1.4, -0.8, -1.0, -0.5]))
            m.phi.normal_()
            m.phi.mul_(0.35 / m.phi.norm(dim=1, keepdim=True))
            m.rho_c.copy_(torch.tensor([-0.14, 0.09, -0.20]))
            n = torch.arange(1, J + 1, dtype=m.lam.dtype)
            m.rho_0_free.copy_(0.04 * n.square() + 0.03 * n)

        # Directly enumerate the declared joint law, retaining only supersets of R.
        fixed = torch.tensor([0, 3])
        remaining = torch.tensor([1, 2, 4, 5])
        mask = torch.arange(1, 1 << J)
        X = ((mask[:, None] >> torch.arange(J)) & 1).to(m.lam.dtype)
        keep = X[:, fixed].bool().all(1)
        X = X[keep]
        n = X.sum(1).to(torch.long)
        phi_sum = X @ m.phi
        pair = 0.5 * (phi_sum.square().sum(1) - X @ m.phi.square().sum(1))
        nc = torch.stack([X[:, category == c].sum(1) for c in range(C)], 1)
        energy = (X @ m.b_flat(full_ix) + pair
                  - (m.rho_c * m.pair_feature(nc)).sum(1) - m.rho_0()[n])
        probability = torch.softmax(energy, 0)
        exact_pi = (probability[:, None] * X[:, remaining]).sum(0)

        # Equivalent conditional normaliser over T = S \\ R.
        cond_ix = RaggedIndex(remaining, category[remaining],
                              torch.zeros(C, dtype=torch.long), torch.arange(C), 1)
        base_cat = torch.tensor([[1, 1, 0]], dtype=torch.long)
        base_size = torch.tensor([2], dtype=torch.long)
        base_phi = m.phi[fixed].sum(0)
        b0 = (m.b_flat(full_ix)[remaining]
              + m.phi[remaining] @ base_phi).detach().requires_grad_(True)
        set_quad(m, qmc_n=4096, qmc_seed=43, qmc_reps=4, Kz=Kz,
                 probe=Kz, steps=5, chunk=128)
        m._esp_native = True
        m._poly_degree_native = True
        m._b_override = b0
        m._condition_cat_count = base_cat
        m._condition_size = base_size
        try:
            cache = sparse_prepare(m, cond_ix)
            self.assertTrue(cache["inactive_identity"])
            self.assertTrue(cache["const_identity"])
            conditional_lz = m.log_Z(cond_ix, drop_empty=False)
            got_pi = torch.autograd.grad(conditional_lz.sum(), b0)[0]
        finally:
            m._b_override = None
            m._condition_cat_count = None
            m._condition_size = None

        self.assertLess(float((got_pi - exact_pi).abs().max()), 8e-4)
        exact_extra_n = ((probability[:, None] * X[:, remaining]).sum(0)).sum()
        self.assertLess(abs(float(got_pi.sum() - exact_extra_n)), 8e-4)

    def test_exact_additive_normalizer_matches_constant_qmc(self):
        torch.manual_seed(41)
        m, ix = one_row_model(30, 6, nmax=12)
        with torch.no_grad():
            m.lam.normal_(-2.0, 0.25)
            m.phi.zero_()
            m.rho_c.fill_(-0.12)
            m.rho_0_free.copy_(0.02 * torch.arange(1, 13).square())
        set_quad(m, qmc_n=32, qmc_seed=5, qmc_reps=4, Kz=6,
                 probe=-1, steps=2, chunk=8)
        with torch.no_grad():
            q_lz, q_ess, q_pn = m.log_Z(
                ix, drop_empty=True, return_ess=True, return_size=True)
            m._exact_additive = True
            a_lz, a_ess, a_pn = m.log_Z(
                ix, drop_empty=True, return_ess=True, return_size=True)
        self.assertLess(float((a_lz - q_lz).abs().max()), 2e-13)
        self.assertLess(float((a_pn - q_pn).abs().max()), 2e-13)
        self.assertEqual(float(a_ess.min()), 1.0)
        self.assertEqual(float(m._last_qmc_logz_se.max()), 0.0)

        m.zero_grad(set_to_none=True)
        m.log_Z(ix, drop_empty=True, return_size=True)[0].sum().backward()
        self.assertTrue(bool(torch.isfinite(m.lam.grad).all()))
        self.assertTrue(bool(torch.isfinite(m.rho_0_free.grad).all()))

    def test_fused_polynomial_backward_matches_reference(self):
        torch.manual_seed(2)
        # Include a broadcast leading axis: A_const has this shape in the real kernel.
        a = torch.rand(3, 2, 7, requires_grad=True)
        b = torch.rand(1, 2, 5, requires_grad=True)
        ref = _poly_mul_trunc_eager(a, b, 8)
        got = _poly_mul_trunc(a, b, 8)
        self.assertEqual(float((got - ref).abs().max().detach()), 0.0)
        probe = torch.randn_like(got)
        ga = torch.autograd.grad((got * probe).sum(), (a, b), retain_graph=True)
        ra = torch.autograd.grad((ref * probe).sum(), (a, b))
        for x, y in zip(ga, ra):
            self.assertLess(float((x - y).abs().max()), 2e-14)

    def test_full_support_attractive_rho_stays_in_log_domain(self):
        """R=120 must not impose the exp(-rho*C(n,2)) float64 boundary."""
        J, Kz = 120, 2
        m, ix = one_row_model(J, Kz, nmax=J)
        m._esp_native = True
        m._poly_degree_native = True
        with torch.no_grad():
            m.lam.fill_(-2.0)
            m.phi.zero_()
            m.phi[:, 0] = 0.01       # all products use the fused active path
            m.rho_c.fill_(-0.20)     # exp(0.20*C(120,2)) would overflow
            m.rho_0_free.zero_()

        z = torch.zeros(1, 1, Kz)
        got = log_f_sparse(m, z, ix, sparse_prepare(m, ix), drop_empty=True).sum()
        got_grad = torch.autograd.grad(got, m.rho_c)[0]

        n = torch.arange(1, J + 1, dtype=m.lam.dtype)
        log_choose = (torch.lgamma(torch.tensor(J + 1.0))
                      - torch.lgamma(n + 1.0)
                      - torch.lgamma(torch.tensor(J + 1.0) - n))
        bt = -2.0 - 0.5 * 0.01 ** 2
        pair = n * (n - 1.0) / 2.0
        terms = log_choose + n * bt + 0.20 * pair
        expected = torch.logsumexp(terms, dim=0)
        expected_grad = -(torch.softmax(terms, dim=0) * pair).sum()

        self.assertTrue(bool(torch.isfinite(got)))
        self.assertTrue(bool(torch.isfinite(got_grad).all()))
        self.assertLess(abs(float(got - expected)), 2e-11)
        self.assertLess(abs(float(got_grad[0] - expected_grad)), 2e-9)

    def test_degree_tilt_handles_incompatible_attractive_category_maxima(self):
        """Several attractive categories must not underflow each other's low degrees."""
        per_category, categories, Kz, nmax = 120, 2, 2, 120
        J = per_category * categories
        item = torch.arange(J)
        category = torch.arange(categories).repeat_interleave(per_category)
        ix = RaggedIndex(item, category, torch.zeros(categories, dtype=torch.long),
                         torch.arange(categories), 1)
        m = RaggedModel(J, 1, categories, K=2, Kz=Kz, nmax=nmax, R=nmax,
                        S=1, Kp=2, Kt=2, Ks=2)
        m.house, m.ctx = torch.tensor([0]), None
        m._esp_native = True
        m._poly_degree_native = True
        with torch.no_grad():
            m.cat_of.copy_(category)
            m.lam.fill_(-2.0)
            m.phi.zero_()
            m.phi[:, 0] = 0.01
            m.rho_c.copy_(torch.tensor([-0.20, -0.18]))
            # Counter the quadratic attraction enough to keep many global degrees finite
            # and relevant; this does not cancel how the degree splits across categories.
            n = torch.arange(1, nmax + 1, dtype=m.lam.dtype)
            m.rho_0_free.copy_(0.10 * n * (n - 1.0) / 2.0 + 0.02 * n.square())

        z = torch.zeros(1, 1, Kz)
        got = log_f_sparse(m, z, ix, sparse_prepare(m, ix), drop_empty=True).sum()
        got_grad = torch.autograd.grad(got, m.rho_c)[0]

        # Closed-form per-size reference: split n between two equal-weight categories.
        bt = -2.0 - 0.5 * 0.01 ** 2
        size_terms = []
        for total in range(1, nmax + 1):
            split_terms = []
            lo, hi = max(0, total - per_category), min(per_category, total)
            for left in range(lo, hi + 1):
                right = total - left
                log_choose = (torch.lgamma(torch.tensor(per_category + 1.0))
                              - torch.lgamma(torch.tensor(left + 1.0))
                              - torch.lgamma(torch.tensor(per_category - left + 1.0))
                              + torch.lgamma(torch.tensor(per_category + 1.0))
                              - torch.lgamma(torch.tensor(right + 1.0))
                              - torch.lgamma(torch.tensor(per_category - right + 1.0)))
                split_terms.append(
                    log_choose + total * bt
                    + 0.20 * left * (left - 1.0) / 2.0
                    + 0.18 * right * (right - 1.0) / 2.0)
            size_terms.append(torch.logsumexp(torch.stack(split_terms), 0)
                              - m.rho_0()[total].detach())
        expected = torch.logsumexp(torch.stack(size_terms), 0)

        self.assertTrue(bool(torch.isfinite(got)))
        self.assertTrue(bool(torch.isfinite(got_grad).all()))
        self.assertLess(abs(float(got - expected)), 2e-10)

    def test_esp_covers_rows_above_256(self):
        for n in (300, 1774):
            w = torch.full((1, n), 1.0 / n)
            got = esp_bucketed(w, torch.zeros(n, dtype=torch.long), 1, 4,
                               torch.tensor([n]), torch.arange(n))[0, 0]
            want = torch.tensor([
                1.0,
                1.0,
                (n - 1) / (2 * n),
                (n - 1) * (n - 2) / (6 * n ** 2),
                (n - 1) * (n - 2) * (n - 3) / (24 * n ** 3),
            ])
            self.assertLess(float((got - want).abs().max()), 1e-11)

    def test_sparse_kernel_and_chunked_size_law(self):
        torch.manual_seed(3)
        m, ix = one_row_model(300, 6, nmax=8)
        with torch.no_grad():
            m.lam.normal_(-2.5, 0.2)
            m.phi.normal_(0.0, 0.04)
            m.rho_c.fill_(-0.15)
        z = torch.randn(1, 5, m.Kz)
        with torch.no_grad():
            dense = log_f_ragged(m, z, ix, True)
            sparse = log_f_sparse(m, z, ix, sparse_prepare(m, ix), True)
        self.assertLess(float((dense - sparse).abs().max().detach()), 1e-11)

        set_quad(m, qmc_n=32, qmc_seed=11, qmc_reps=4, Kz=m.Kz,
                 probe=4, steps=2, chunk=0)
        with torch.no_grad():
            whole = m.log_Z(ix, drop_empty=True, return_ess=True, return_size=True)
        m.quad_chunk = 7
        with torch.no_grad():
            chunked = m.log_Z(ix, drop_empty=True, return_ess=True, return_size=True)
        self.assertLess(float((whole[0] - chunked[0]).abs().max()), 1e-12)
        self.assertLess(float((whole[2] - chunked[2]).abs().max()), 1e-12)
        self.assertAlmostEqual(float(chunked[2].sum()), 1.0, places=12)

        m.zero_grad(set_to_none=True)
        m.log_Z(ix, drop_empty=True, return_size=True)[0].sum().backward()
        self.assertTrue(bool(torch.isfinite(m.phi.grad).all()))

    def test_rqmc_matches_exact_subset_normalizer(self):
        torch.manual_seed(7)
        J, Kz = 10, 8
        m, ix = one_row_model(J, Kz)
        with torch.no_grad():
            m.lam.copy_(torch.linspace(-1.7, -0.7, J))
            m.phi.normal_()
            m.phi.mul_(0.55 / m.phi.norm(dim=1, keepdim=True))
            m.rho_c.fill_(0.08)
            n = torch.arange(1, J + 1)
            m.rho_0_free.copy_(0.025 * n ** 2)

        b, rho0 = m.b_flat(ix).detach(), m.rho_0().detach()
        phi, rho_c = m.phi.detach(), m.rho_c.detach()
        terms = []
        for mask in range(1, 1 << J):
            sel = torch.tensor([j for j in range(J) if (mask >> j) & 1])
            p, n = phi[sel], len(sel)
            pair = 0.5 * ((p.sum(0) ** 2).sum() - (p ** 2).sum())
            terms.append(b[sel].sum() + pair
                         - rho_c[0] * n * (n - 1) / 2 - rho0[n])
        exact = torch.logsumexp(torch.stack(terms), dim=0)

        set_quad(m, qmc_n=1024, qmc_seed=9, qmc_reps=4, Kz=Kz,
                 probe=Kz, steps=6, chunk=64)
        with torch.no_grad():
            estimate, _ = m.log_Z(ix, drop_empty=True, return_ess=True)
        self.assertLess(abs(float(estimate - exact)), 3e-3)
        self.assertIsNotNone(m._last_qmc_logz_se)
        self.assertTrue(math.isfinite(float(m._last_qmc_logz_se[0])))

    def test_remote_scaling_identity_frame_and_operator_projection(self):
        # A z-independent scale overflows here: twenty aligned degree-20 weights carry
        # exp(0.96*40) each, so their product is beyond float64.  Per-node rescaling keeps
        # the mathematically finite log polynomial representable.
        m, ix = one_row_model(20, 4)
        with torch.no_grad():
            m.lam.fill_(-1.0)
            m.phi.zero_()
            m.phi[:, 0] = 0.96
            n = torch.arange(1, 21, dtype=m.lam.dtype)
            m.rho_0_free.copy_(0.01 * n.square())
        z = torch.zeros(1, 1, 4)
        z[..., 0] = 40.0
        with torch.no_grad():
            dense = log_f_ragged(m, z, ix, True)
            sparse = log_f_sparse(m, z, ix, sparse_prepare(m, ix), True)
        self.assertTrue(bool(torch.isfinite(sparse).all()))
        self.assertLess(float((dense - sparse).abs().max()), 1e-10)

        # The operator projection caps the catalogue accumulation while retaining all rows.
        torch.manual_seed(19)
        with torch.no_grad():
            m.phi.normal_(0.0, 0.8)
        m.project(phi_max=10.0, op_max=2.0)
        lam = torch.linalg.eigvalsh(m.phi.detach().T @ m.phi.detach())[-1]
        self.assertLessEqual(float(lam), 2.0 + 2e-12)
        self.assertEqual(int((m.phi.norm(dim=1) > 0).sum()), 20)

        set_quad(m, qmc_n=32, qmc_seed=3, qmc_reps=4, Kz=4,
                 probe=-1, steps=2, chunk=32)
        with torch.no_grad():
            cache = sparse_prepare(m, ix)
            _zh, sd, _Q = m._adaptive_frame(ix, True, 2, cache=cache)
        self.assertEqual(float((sd - 1.0).abs().max()), 0.0)

    def test_projection_is_stable_at_repeated_operator_cap(self):
        # Whitening plus the operator cap deliberately produces a repeated singular
        # spectrum.  The tall-matrix SVD used here before run155 could fail to converge on
        # the next update even though the matrix was finite.  Repeated projections must be
        # finite, idempotent and preserve the cap.
        torch.manual_seed(2026)
        m = RaggedModel(545, 2, 1, K=2, Kz=32, nmax=4, R=4,
                        phi_init=0.03).double()
        with torch.no_grad():
            x = torch.randn(545, 32, dtype=torch.float64)
            x -= x.mean(0, keepdim=True)
            q, _ = torch.linalg.qr(x, mode="reduced")
            m.phi.copy_(math.sqrt(2.0) * q)
            for _ in range(4):
                m.project(phi_max=0.96, centre=True, whiten=0.5, op_max=2.0)
            gram = m.phi.T @ m.phi
        self.assertTrue(bool(torch.isfinite(m.phi).all()))
        self.assertLessEqual(float(torch.linalg.eigvalsh(gram)[-1]), 2.0 + 2e-12)
        # Centre changes the deliberately uncentred input on the first pass; subsequent
        # projections should be numerically idempotent at the repeated cap.
        stable = m.phi.detach().clone()
        m.project(phi_max=0.96, centre=True, whiten=0.5, op_max=2.0)
        self.assertLess(float((m.phi.detach() - stable).abs().max()), 2e-12)

    def test_size_stratified_rule_recovers_invisible_remote_basin(self):
        # Construct the exact pathology seen in the failed checkpoint.  The partition has
        # two unit-covariance Gaussian basins, at sizes 2 and 18, whose centres are eight
        # standard deviations apart.  At z=0 the large-size basin is down by about 40 nats,
        # so a zero-start total-mode iteration and even 1024 local nodes never see it.
        J, Kz, p = 20, 4, 0.5
        m, ix = one_row_model(J, Kz)
        desired = torch.full((J + 1,), -20.0)
        desired[2], desired[18] = 0.0, -0.5
        with torch.no_grad():
            m.lam.zero_()
            m.phi.zero_()
            m.phi[:, 0] = p
            m.rho_c.zero_()
            rho = []
            for n in range(1, J + 1):
                base = (math.lgamma(J + 1) - math.lgamma(n + 1)
                        - math.lgamma(J - n + 1) + 0.5 * p * p * n * (n - 1))
                rho.append(base - float(desired[n]))
            m.rho_0_free.copy_(torch.tensor(rho))
        exact = torch.logsumexp(desired[1:], dim=0)

        set_quad(m, qmc_n=1024, qmc_seed=9, qmc_reps=4, Kz=Kz,
                 probe=-1, steps=2, chunk=64)
        with torch.no_grad():
            local = m.log_Z(ix, drop_empty=True)
        self.assertGreater(abs(float(local - exact)), 0.4)

        # The screen takes only three vectorised passes and the integration still uses 32
        # nodes TOTAL (four per mode per replicate), not 32 nodes per mode.
        set_quad(m, qmc_n=32, qmc_seed=9, qmc_reps=4, Kz=Kz,
                 probe=-1, steps=2, chunk=32, size_bands=1, size_steps=3,
                 mode_logtol=4.0, mode_sep=1.0, mix_n=32)
        with torch.no_grad():
            lz, ess, pn = m.log_Z(
                ix, drop_empty=True, return_ess=True, return_size=True)
        self.assertLess(abs(float(lz - exact)), 1e-5)
        self.assertGreater(float(ess), 0.9)
        self.assertEqual(int(m._last_qmc_mode_count[0]), 2)
        self.assertAlmostEqual(float(m._last_qmc_mode_sep[0]), 8.0, places=5)
        expected_n = float((2.0 + 18.0 * math.exp(-0.5)) / (1.0 + math.exp(-0.5)))
        got_n = float((pn * torch.arange(1, J + 1)).sum())
        self.assertAlmostEqual(got_n, expected_n, places=5)

        m.zero_grad(set_to_none=True)
        m.log_Z(ix, drop_empty=True, return_size=True)[0].sum().backward()
        self.assertTrue(bool(torch.isfinite(m.phi.grad).all()))

    def test_size_rule_keeps_the_active_sobol_rotation(self):
        # Regression for run109's broad-shell failure.  The ordinary adaptive rule rotates
        # a finite Sobol block into the Phi'Phi eigenframe, but the first multimode branch
        # accidentally used the raw coordinates.  The Gaussian law is rotation invariant;
        # 32 deterministic Sobol points are not, and one raw scramble dominated log Z.
        torch.manual_seed(29)
        m, ix = one_row_model(24, 4, nmax=12)
        with torch.no_grad():
            m.lam.normal_(-2.0, 0.1)
            m.phi.normal_(0.0, 0.03)
            # Make the leading direction visibly non-axis-aligned.
            v = torch.tensor([0.5, -0.5, 0.5, -0.5])
            m.phi.add_(torch.linspace(-0.08, 0.08, 24)[:, None] * v[None, :])
        set_quad(m, qmc_n=32, qmc_seed=0, qmc_reps=4, Kz=4,
                 probe=-1, steps=2, chunk=32, size_bands=1, size_steps=2,
                 mode_logtol=8.0, mode_sep=100.0, mix_n=64)
        with torch.no_grad():
            cache = sparse_prepare(m, ix)
            z, base, top = m._size_multimode_proposal(ix, True, cache)
            gram = m.phi.T @ m.phi
            ev, Q = torch.linalg.eigh(gram)
            Q = Q[:, torch.argsort(ev, descending=True)]
            x, w = m.quad_a
            recovered = (z - top[:, None, :]) @ Q
            expected_base = (-0.5 * z.square().sum(-1)
                             + 0.5 * x.square().sum(-1)[None, :]
                             + w.log()[None, :])
        self.assertLess(float((recovered - x[None, :, :]).abs().max()), 2e-14)
        self.assertGreater(float((z - top[:, None, :] - x[None, :, :]).abs().max()), 0.1)
        self.assertLess(float((base - expected_base).abs().max()), 2e-14)


if __name__ == "__main__":
    unittest.main()
