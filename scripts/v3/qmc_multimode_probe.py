"""Frozen-checkpoint experiment for a multimode RQMC normaliser.

The Hubbard--Stratonovich target has the exact representation

    N(z; 0, I) f(z) = sum_S exp(E(S)) N(z; mu_S, I),
    mu_S = sum_{j in S} phi_j.

It is therefore a (very large) unit-covariance Gaussian mixture.  This script tests a
small, deterministic-mixture importance rule against the single local proposal used by
``RaggedModel._log_Z_adaptive``.  It never changes model parameters.  Proposal centres are
detached, and the balance denominator makes the estimator valid even when components
overlap.

This is an experiment, not a training entry point.  Run from the repository root:

    V3_AFFINITY=1 python scripts/v3/qmc_multimode_probe.py \
        --checkpoint out/v3_run100_best.pt --trips 15194
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import build
from features import Features
from fit import Batcher
from ragged import RaggedModel, log_f_sparse, set_quad, sparse_prepare


torch.set_default_dtype(torch.float64)

_NEG = -1.0e100


def log_esp_bucketed(logw, ix, R):
    """Log-domain elementary symmetric polynomials, used as a numerical reference.

    Linear-space ESPs are faster near the ordinary mode.  They cannot represent a remote
    basin where degree-120 coefficients differ by thousands of log units.  This recurrence
    is subtraction-free and rescales every degree implicitly by storing its logarithm.
    """
    D = logw.shape[0]
    out = torch.full((D, ix.n_rows, R + 1), _NEG, dtype=logw.dtype,
                     device=logw.device)
    out[..., 0] = 0.0
    max_size = int(ix.row_size.max()) if ix.row_size.numel() else 0
    limits = [8, 32, 96, 256]
    while limits[-1] < max_size:
        limits.append(min(max_size, 2 * limits[-1]))
    lo = 0
    for hi in limits:
        row_mask = (ix.row_size > lo) & (ix.row_size <= hi)
        lo = hi
        if not bool(row_mask.any()):
            continue
        rows = torch.nonzero(row_mask, as_tuple=True)[0]
        loc = torch.full((ix.n_rows,), -1, dtype=torch.long, device=logw.device)
        loc[rows] = torch.arange(len(rows), device=logw.device)
        item_mask = row_mask[ix.row_of]
        flat = loc[ix.row_of[item_mask]] * hi + ix.item_pos[item_mask]
        P = torch.full((D, len(rows) * hi), _NEG, dtype=logw.dtype,
                       device=logw.device)
        P = P.index_copy(1, flat, logw[:, item_mask]).view(D, len(rows), hi)
        E = torch.full((D, len(rows), R + 1), _NEG, dtype=logw.dtype,
                       device=logw.device)
        E[..., 0] = 0.0
        for i in range(hi):
            update = P[..., i].unsqueeze(-1) + E[..., :-1]
            E = torch.cat([E[..., :1], torch.logaddexp(E[..., 1:], update)], dim=-1)
        out = out.index_copy(1, rows, E)
    return out


def log_poly_mul(A, G, nmax):
    degree = min(nmax, A.shape[-1] + G.shape[-1] - 2)
    terms = []
    for n in range(degree + 1):
        lo = max(0, n - (G.shape[-1] - 1))
        hi = min(A.shape[-1] - 1, n)
        av = A[..., lo:hi + 1]
        gv = G[..., n - hi:n - lo + 1].flip(-1)
        terms.append(torch.logsumexp(av + gv, dim=-1))
    return torch.stack(terms, dim=-1)


def log_poly_tree(P, nmax):
    while P.shape[-2] > 1:
        if P.shape[-2] % 2:
            ident = torch.full(P.shape[:-2] + (1, P.shape[-1]), _NEG,
                               dtype=P.dtype, device=P.device)
            ident[..., 0, 0] = 0.0
            P = torch.cat([P, ident], dim=-2)
        P = log_poly_mul(P[..., 0::2, :], P[..., 1::2, :], nmax)
    return P[..., 0, :]


def log_f_logspace(model, z, ix, drop_empty=True, return_terms=False):
    """Stable definition-level implementation of log f for theory/reference tests."""
    phi_i = model.phi[ix.item]
    bt = model.b_flat(ix) - 0.5 * phi_i.square().sum(-1)
    proj = (z[ix.item_trip] * phi_i.unsqueeze(1)).sum(-1)
    logw = (bt.unsqueeze(1) + proj).transpose(0, 1)                 # [D,T]
    le = log_esp_bucketed(logw, ix, model.R)
    r = torch.arange(model.R + 1, dtype=logw.dtype, device=logw.device)
    lG = le - model.rho_c[ix.row_cat].unsqueeze(0).unsqueeze(-1) * r * (r - 1) / 2
    Gp = torch.full((z.shape[1], ix.B * ix.Cpad, model.R + 1), _NEG,
                    dtype=logw.dtype, device=logw.device)
    Gp[..., 0] = 0.0
    Gp = Gp.index_copy(1, ix.flat_slot, lG).view(
        z.shape[1], ix.B, ix.Cpad, model.R + 1)
    lA = log_poly_tree(Gp, model.nmax)
    lg = lA - model.rho_0()[:lA.shape[-1]]
    if drop_empty:
        lg = lg[..., 1:]
    if return_terms:
        return lg
    return torch.logsumexp(lg, dim=-1).transpose(0, 1)


def tail_partition_upper_bound(model, ix, n_tail=25):
    """Rigorous upper bound on the contribution of every basket of size >= n_tail.

    Let lambda be the largest eigenvalue of Phi'Phi.  For an indicator x with |x|=n,
    ||Phi'x||^2 <= lambda ||x||^2 = lambda*n.  Hence the pair interaction is at most
    lambda*n/2 after dropping its non-positive diagonal correction.  Summing the remaining
    linear/category model exactly gives an upper bound over *all* subsets, not a mode scan.
    """
    D = 1
    b = model.b_flat(ix).detach().unsqueeze(0)
    le = log_esp_bucketed(b, ix, model.R)
    r = torch.arange(model.R + 1, dtype=b.dtype)
    lG = le - model.rho_c.detach()[ix.row_cat].unsqueeze(0).unsqueeze(-1) * r * (r - 1) / 2
    Gp = torch.full((D, ix.B * ix.Cpad, model.R + 1), _NEG, dtype=b.dtype)
    Gp[..., 0] = 0.0
    Gp = Gp.index_copy(1, ix.flat_slot, lG).view(D, ix.B, ix.Cpad, model.R + 1)
    lA = log_poly_tree(Gp, model.nmax)[0]
    lam = torch.linalg.eigvalsh(model.phi.detach().T @ model.phi.detach())[-1]
    n = torch.arange(lA.shape[-1], dtype=b.dtype)
    ub = lA - model.rho_0().detach()[:lA.shape[-1]] + 0.5 * lam * n
    lo = max(1, int(n_tail))
    return torch.logsumexp(ub[:, lo:], dim=-1), ub, lam


def resolve_checkpoint_R(metadata, config):
    """Declared category-count support, with self-describing checkpoint data first."""
    return int(metadata.get("R", config.get("R", config.get("r", 23))))


def load_problem(checkpoint: str, trips: list[int] | None, batch: int, phi_scale: float = 1.0):
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = blob.get("data", {}) if isinstance(blob, dict) else {}
    if metadata.get("affinity") == "1":
        os.environ["V3_AFFINITY"] = "1"
    D = build()
    J, N, C, S = (int(D[k]) for k in ("n_item", "n_user", "n_cat", "n_store"))
    if metadata:
        if int(metadata.get("n_item", J)) != J or int(metadata.get("n_cat", C)) != C:
            raise RuntimeError("checkpoint/data catalogue or category partition mismatch")
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    K = int(state["alpha"].shape[1])
    Kz = int(state["phi"].shape[1])
    Kp = int(state["beta"].shape[1])
    nmax = int(state["rho_0_free"].shape[0])

    cfg_path = Path(checkpoint).with_suffix(".json")
    # Checkpoints are named v3_run*.pt and their configurations v3_run*.json.  A best
    # suffix has no separate configuration, so remove it if necessary.
    if not cfg_path.exists():
        cfg_path = Path(str(cfg_path).replace("_best.json", ".json"))
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    # The sibling ``*_best.json`` file is an evaluation summary containing only iteration,
    # score and epoch.  Treating its missing R as the historical default silently rebuilt
    # complete-support R=120 checkpoints at R=23 in every frozen audit.  Checkpoint metadata
    # is self-describing and authoritative; a run configuration is only a legacy fallback.
    R = resolve_checkpoint_R(metadata, cfg)
    m = RaggedModel(J, N, C, K=K, Kz=Kz, nmax=nmax, R=R, S=S, Kp=Kp)
    qmeta = blob.get("quad", {}) if isinstance(blob, dict) else {}
    m._poly_degree_native = bool(qmeta.get("poly_degree_native", False))
    m._esp_native = bool(qmeta.get("esp_native", False))
    if metadata and (int(metadata.get("nmax", nmax)) != nmax
                     or int(metadata.get("rho_pair_cap", nmax)) != m.rho_pair_cap):
        raise RuntimeError("checkpoint support metadata is inconsistent with its tensors")
    missing, unexpected = m.load_state_dict(state, strict=False)
    missing = [x for x in missing if x != "cat_of"]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    if phi_scale != 1.0:
        with torch.no_grad():
            m.phi.mul_(phi_scale)
    with torch.no_grad():
        cat_of = torch.zeros(J, dtype=torch.long)
        cat_of[torch.as_tensor(D["line_item"], dtype=torch.long)] = \
            torch.as_tensor(D["line_cat"], dtype=torch.long)
        m.cat_of.copy_(cat_of)
    m.eval()

    tr = np.flatnonzero(D["trip_split"] == 0)
    if trips is None:
        rng = np.random.default_rng()
        if isinstance(blob, dict) and blob.get("rng_np") is not None:
            rng.bit_generator.state = blob["rng_np"]
        chosen = tr[rng.choice(len(tr), size=batch, replace=False)].tolist()
    else:
        chosen = trips
    F = Features(J, S, 712)
    bx = Batcher(D, F, nmax)
    ix, ctx, *_rest = bx.make(np.asarray(chosen, dtype=np.int64))
    house = _rest[1]
    m.house, m.ctx = house, ctx
    return m, ix, chosen


def active_frame(model: RaggedModel, ix):
    """Global Phi frame ordered by catalogue energy, matching the production rule."""
    with torch.no_grad():
        present = torch.unique(ix.item)
        p = model.phi.index_select(0, present)
        gram = p.T @ p
        evals, Q = torch.linalg.eigh(gram)
        order = torch.argsort(evals, descending=True)
        return evals[order], Q[:, order]


def objective(model, ix, cache, z, stable=False):
    lf = (log_f_logspace(model, z, ix, True) if stable else
          log_f_sparse(model, z, ix, cache, True, detach_params=True))
    return lf - 0.5 * z.square().sum(-1)


def refine_modes(model, ix, cache, seeds, steps=6, damping=0.5, stable=False):
    """Damped fixed-point/ascent iteration on F(z) = log f(z) - ||z||^2/2."""
    z = seeds.detach().clone()
    for _ in range(steps):
        zz = z.detach().requires_grad_(True)
        with torch.enable_grad():
            val = objective(model, ix, cache, zz, stable=stable)
            grad = torch.autograd.grad(val.sum(), zz)[0]
        # The stationary equation is z = grad log f(z).  This update is the average of
        # the old point and that fixed-point image, equivalently z + 0.5 grad F.
        z = (zz + damping * grad).detach()
    with torch.no_grad():
        val = objective(model, ix, cache, z, stable=stable)
    return z, val


def discover_modes(model, ix, cache, top_dirs=4, radii=(8.0, 16.0, 24.0, 32.0),
                   steps=6, max_modes=3, separation=3.0, stable=False,
                   random_dirs=0, seed=0):
    """Find distinct basins from signed leading-Phi directions and a local seed.

    The leading directions are not asserted to be a proof of global coverage.  The probe
    deliberately reports the best boundary score and is later compared with wider scans;
    this function is the cheap candidate whose coverage must earn its way into training.
    """
    B, Kz = ix.B, model.Kz
    evals, Q = active_frame(model, ix)
    seed_rows = [torch.zeros(Kz)]
    for k in range(min(top_dirs, Kz)):
        for radius in radii:
            seed_rows.extend([radius * Q[:, k], -radius * Q[:, k]])
    if random_dirs > 0:
        eng = torch.quasirandom.SobolEngine(Kz, scramble=True, seed=seed + 7919)
        u = eng.draw(random_dirs).double().clamp(1e-12, 1 - 1e-12)
        dirs = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        # The largest radius is sufficient for basin discovery; refinement moves to the
        # stationary point.  Axis starts above retain the radial sensitivity check.
        seed_rows.extend([radii[-1] * d for d in dirs])
    base = torch.stack(seed_rows).to(dtype=model.lam.dtype)
    seeds = base.unsqueeze(0).expand(B, -1, -1).contiguous()
    modes, scores = refine_modes(model, ix, cache, seeds, steps=steps, stable=stable)

    picked = torch.empty(B, max_modes, Kz, dtype=modes.dtype)
    picked_scores = torch.empty(B, max_modes, dtype=modes.dtype)
    picked_seed = torch.empty(B, max_modes, dtype=torch.long)
    for b in range(B):
        # Always retain the zero-start basin.  It remains an essential defensive component
        # even when a remote basin has much larger pointwise density.
        ids = [0]
        for j in torch.argsort(scores[b], descending=True).tolist():
            if j == 0:
                continue
            if all(float((modes[b, j] - modes[b, k]).norm()) >= separation for k in ids):
                ids.append(j)
            if len(ids) == max_modes:
                break
        while len(ids) < max_modes:
            ids.append(ids[-1])
        picked[b] = modes[b, ids]
        picked_scores[b] = scores[b, ids]
        picked_seed[b] = torch.as_tensor(ids)
    return picked, picked_scores, picked_seed, scores, evals


def map_subset_at_z(model, ix, bslot, z, trip):
    """Exact conditional MAP subset at fixed z by max-product category DP.

    Returns its Phi sum, exact model energy, and assortment-slot indices.  This is the
    max-product analogue of the sum-product ESP/category convolution used by log f.
    """
    rows = torch.nonzero(ix.row_trip == trip, as_tuple=True)[0].tolist()
    row_options = []
    for row in rows:
        slots = torch.nonzero(ix.row_of == row, as_tuple=True)[0]
        ph = model.phi[ix.item[slots]]
        score = (bslot[slots] - 0.5 * ph.square().sum(-1) + ph @ z)
        k = min(model.R, len(slots))
        vals, order = torch.topk(score, k=k, largest=True, sorted=True)
        cumulative = torch.cat([torch.zeros(1, dtype=score.dtype), vals.cumsum(0)])
        r = torch.arange(k + 1, dtype=score.dtype)
        cat = int(ix.row_cat[row])
        option = cumulative - model.rho_c[cat] * r * (r - 1) / 2
        row_options.append((option, slots[order]))

    dp = torch.full((model.nmax + 1,), _NEG, dtype=bslot.dtype)
    dp[0] = 0.0
    back = []
    for option, _ in row_options:
        new = torch.full_like(dp, _NEG)
        choice = torch.zeros(model.nmax + 1, dtype=torch.long)
        for r in range(len(option)):
            cand = dp[:model.nmax + 1 - r] + option[r]
            cur = new[r:]
            take = cand > cur
            new[r:] = torch.where(take, cand, cur)
            choice[r:] = torch.where(take, torch.full_like(choice[r:], r), choice[r:])
        dp = new
        back.append(choice)

    total = dp - model.rho_0()[:model.nmax + 1]
    n = int(torch.argmax(total[1:])) + 1
    selected = []
    left = n
    for i in range(len(row_options) - 1, -1, -1):
        r = int(back[i][left])
        if r:
            selected.append(row_options[i][1][:r])
        left -= r
    slots = torch.cat(selected) if selected else torch.empty(0, dtype=torch.long)
    ph = model.phi[ix.item[slots]]
    mu = ph.sum(0)
    cats = ix.row_cat[ix.row_of[slots]]
    counts = torch.bincount(cats, minlength=model.C).to(bslot.dtype)
    energy = (bslot[slots].sum()
              + 0.5 * (mu.square().sum() - ph.square().sum())
              - (model.rho_c * counts * (counts - 1) / 2).sum()
              - model.rho_0()[len(slots)])
    return mu.detach(), energy.detach(), slots


def map_coordinate_ascent(model, ix, starts, steps=20):
    """Monotone alternating maximisation over a discrete subset and its Gaussian mean."""
    B, D, _ = starts.shape
    bslot = model.b_flat(ix).detach()
    centres = torch.empty_like(starts)
    energies = torch.empty(B, D, dtype=starts.dtype)
    sizes = torch.empty(B, D, dtype=torch.long)
    for b in range(B):
        for d in range(D):
            z = starts[b, d].detach()
            previous = None
            for _ in range(steps):
                mu, energy, slots = map_subset_at_z(model, ix, bslot, z, b)
                key = tuple(slots.tolist())
                z = mu
                if key == previous:
                    break
                previous = key
            centres[b, d], energies[b, d], sizes[b, d] = z, energy, len(slots)
    return centres, energies, sizes


def normal_sobol_blocks(dim, reps, components, per_component, seed):
    if per_component < 2 or per_component & (per_component - 1):
        raise ValueError("per-component nodes must be a power of two >= 2")
    out = torch.empty(reps, components, per_component, dim)
    for r in range(reps):
        for c in range(components):
            eng = torch.quasirandom.SobolEngine(
                dim, scramble=True, seed=seed + 104729 * r + 13007 * c)
            u = eng.draw(per_component).double().clamp(1e-12, 1 - 1e-12)
            out[r, c] = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
    return out


def laplace_frames(model, ix, cache, centres, stable=True, eps=0.05):
    """Detached eigendecomposition of -H_F at each proposed mode.

    The log-domain reference contains sentinel states for unreachable polynomial degrees.
    Its value and first derivative are finite, but differentiating those sentinels a second
    time produces 0/0 internally.  Symmetric differences of the verified first derivative
    measure the same Hessian without traversing those dead second-derivative paths.
    """
    B, M, Kz = centres.shape
    eye = torch.eye(Kz, dtype=centres.dtype, device=centres.device)
    Q = eye.view(1, 1, Kz, Kz).expand(B, M, -1, -1).clone()
    sd = torch.ones(B, M, Kz, dtype=centres.dtype, device=centres.device)
    raw = torch.empty_like(sd)
    probes = torch.cat([eye, -eye], dim=0) * eps
    for m in range(M):
        z = centres[:, m:m + 1, :] + probes.unsqueeze(0)
        zz = z.detach().requires_grad_(True)
        with torch.enable_grad():
            val = objective(model, ix, cache, zz, stable=stable)
            grad = torch.autograd.grad(val.sum(), zz)[0].detach()
        # grad[:, k, j] is derivative j at z + eps*e_k, hence columns k of H
        # are the transposed rows of this finite difference.
        H = ((grad[:, :Kz] - grad[:, Kz:]) / (2.0 * eps)).transpose(-1, -2)
        H = 0.5 * (H + H.transpose(-1, -2))
        for b in range(B):
            ev, q = torch.linalg.eigh(-H[b])
            raw[b, m] = ev
            # Positive curvature means the point is not a local maximum.  Retaining a
            # broad but finite defensive proposal is safer than pretending its inverse is
            # a covariance.
            good = ev.clamp(1.0 / 64.0, 16.0)
            Q[b, m], sd[b, m] = q, good.rsqrt()
    return Q.detach(), sd.detach(), raw.detach()


def directional_frames(model, ix, cache, centres, probes=8, stable=False, eps=0.35):
    """Cheap diagonal curvature in the catalogue's global ``Phi'Phi`` frame.

    A full Hessian needs 2*Kz first-gradient evaluations.  Symmetric objective values
    along the leading catalogue directions need no second derivative and expose the
    radial broadening relevant to the high-dimensional-shell failure.
    """
    B, M, Kz = centres.shape
    _evals, q0 = active_frame(model, ix)
    P = min(int(probes), Kz)
    directions = q0[:, :P].T * eps
    z = torch.cat([centres[:, :, None, :],
                   centres[:, :, None, :] + directions[None, None, :, :],
                   centres[:, :, None, :] - directions[None, None, :, :]], dim=2)
    zflat = z.reshape(B, 1 + 2 * P, Kz) if M == 1 else z.reshape(B, M * (1 + 2 * P), Kz)
    with torch.no_grad():
        val = objective(model, ix, cache, zflat, stable=stable)
    val = val.reshape(B, M, 1 + 2 * P)
    raw = (2.0 * val[:, :, :1] - val[:, :, 1:1 + P]
           - val[:, :, 1 + P:]) / (eps * eps)
    sd = torch.ones(B, M, Kz, dtype=centres.dtype, device=centres.device)
    sd[:, :, :P] = raw.clamp(1.0 / 64.0, 16.0).rsqrt()
    Q = q0.view(1, 1, Kz, Kz).expand(B, M, -1, -1).contiguous()
    return Q.detach(), sd.detach(), raw.detach()


def subspace_laplace_frames(model, ix, cache, centres, rank=8, iterations=1,
                            stable=False, eps=0.05):
    """Low-rank local Laplace frame from exact first-gradient Hessian products.

    For the latent log target F, ``Cov(mu|z) v = (H_F + I) v``.  A central difference
    of the verified first gradient gives this product without using the ESP kernel's
    unsupported second-order autograd.  Block subspace iteration then finds the locally
    broad directions; Rayleigh--Ritz supplies their curvature.  Unresolved orthogonal
    directions retain the exact mixture component's unit variance.
    """
    if rank < 1 or rank > model.Kz or iterations < 0:
        raise ValueError("subspace rank must be in [1,Kz] and iterations non-negative")
    B, M, Kz = centres.shape
    _evals, q0 = active_frame(model, ix)
    basis = q0[:, :rank].T.view(1, 1, rank, Kz).expand(B, M, -1, -1).clone()

    def covariance_products(directions):
        # directions [B,M,P,K].  Flatten modes and signed probes into the node axis used
        # by the production sparse kernel, then finite-difference its exact first gradient.
        P = directions.shape[2]
        plus = centres[:, :, None, :] + eps * directions
        minus = centres[:, :, None, :] - eps * directions
        z = torch.cat([plus, minus], dim=2).reshape(B, M * 2 * P, Kz)
        zz = z.detach().requires_grad_(True)
        with torch.enable_grad():
            val = objective(model, ix, cache, zz, stable=stable)
            grad = torch.autograd.grad(val.sum(), zz)[0].detach()
        grad = grad.reshape(B, M, 2 * P, Kz)
        hv = (grad[:, :, :P] - grad[:, :, P:]) / (2.0 * eps)
        return hv + directions                 # (H_F + I)v = Cov(mu|z)v

    for _ in range(iterations):
        cb = covariance_products(basis)
        q, _r = torch.linalg.qr(cb.transpose(-1, -2), mode="reduced")
        basis = q.transpose(-1, -2)

    cb = covariance_products(basis)
    projected = torch.matmul(basis, cb.transpose(-1, -2))
    projected = 0.5 * (projected + projected.transpose(-1, -2))
    lam, rotate = torch.linalg.eigh(projected)
    order = torch.argsort(lam, dim=-1, descending=True)
    lam = torch.gather(lam, -1, order).clamp_min(0.0)
    rotate = torch.gather(rotate, -1, order.unsqueeze(-2).expand_as(rotate))
    leading = torch.matmul(basis.transpose(-1, -2), rotate)          # [B,M,K,P]

    # Complete the local eigenvectors to an orthonormal K-dimensional proposal frame.
    Q = torch.empty(B, M, Kz, Kz, dtype=centres.dtype, device=centres.device)
    for b in range(B):
        for m in range(M):
            seeded = torch.cat([leading[b, m], q0], dim=1)
            Q[b, m] = torch.linalg.qr(seeded, mode="reduced")[0]
    curvature = (1.0 - lam).clamp(1.0 / 64.0, 1.0)
    sd = torch.ones(B, M, Kz, dtype=centres.dtype, device=centres.device)
    sd[:, :, :rank] = curvature.rsqrt()
    return Q.detach(), sd.detach(), curvature.detach()


def multimode_rqmc(model, ix, cache, centres, total_nodes=128, reps=4, seed=0,
                   return_size=True, stable=False, frames=None):
    """Equal-allocation deterministic-mixture RQMC with the balance heuristic.

    For q(z)=M^-1 sum_m q_m(z), equal allocation gives

        (1/M) sum_m E_{q_m}[h(z)/q(z)] = integral h(z) dz.

    Thus overlap or duplicate components cannot bias the estimator.  Random scrambling
    supplies independent replicate estimates; fixed seeds make the training rule
    deterministic across optimisation steps.
    """
    B, M, Kz = centres.shape
    if total_nodes % (reps * M):
        raise ValueError("total_nodes must be divisible by reps * number of modes")
    per = total_nodes // (reps * M)
    x = normal_sobol_blocks(Kz, reps, M, per, seed).to(centres)
    if frames is None:
        Q = torch.eye(Kz, dtype=centres.dtype, device=centres.device).view(
            1, 1, Kz, Kz).expand(B, M, -1, -1)
        sd = torch.ones(B, M, Kz, dtype=centres.dtype, device=centres.device)
    else:
        Q, sd = frames
    scaled = x.unsqueeze(0) * sd[:, None, :, None, :]
    delta = torch.einsum("brmlk,bmdk->brmld", scaled, Q)
    z = centres[:, None, :, None, :] + delta
    zflat = z.reshape(B, reps * M * per, Kz)

    # Exact density of the proposal mixture.  The target's Gaussian-mixture identity says
    # unit covariance is model-implied, not an arbitrary tuning parameter.
    diff = zflat[:, :, None, :] - centres[:, None, :, :]
    eigcoord = torch.einsum("bdmj,bmjk->bdmk", diff, Q)
    comp_lq = (-0.5 * (eigcoord / sd[:, None, :, :]).square().sum(-1)
               - sd.log().sum(-1)[:, None, :]
               - 0.5 * Kz * math.log(2.0 * math.pi))
    logq = torch.logsumexp(comp_lq, dim=-1) - math.log(M)
    base = -0.5 * zflat.square().sum(-1) - 0.5 * Kz * math.log(2.0 * math.pi)

    if return_size:
        lg = (log_f_logspace(model, zflat, ix, True, return_terms=True) if stable else
              log_f_sparse(model, zflat, ix, cache, True, return_terms=True))
        # lg [nodes,B,n], while proposal tensors are [B,nodes].
        joint = lg.permute(1, 0, 2) + (base - logq).unsqueeze(-1)
        lw = torch.logsumexp(joint, dim=-1)
    else:
        lf = (log_f_logspace(model, zflat, ix, True) if stable else
              log_f_sparse(model, zflat, ix, cache, True))
        lw = base + lf - logq
        joint = None

    rep_lz = torch.logsumexp(lw.view(B, reps, M * per), dim=-1) - math.log(M * per)
    lz = torch.logsumexp(rep_lz, dim=-1) - math.log(reps)
    se = rep_lz.std(dim=-1, unbiased=True) / math.sqrt(reps)
    ess = (torch.exp(2 * torch.logsumexp(lw, dim=-1)
                     - torch.logsumexp(2 * lw, dim=-1)) / total_nodes).clamp(max=1.0)
    pn = None
    if joint is not None:
        logmass = torch.logsumexp(joint, dim=1) - math.log(total_nodes)
        pn = torch.softmax(logmass, dim=-1)
    return lz, se, ess, pn, rep_lz


def local_rule(model, ix, n, seed=0, probe=None):
    set_quad(model, qmc_n=n, qmc_seed=seed, qmc_reps=4, Kz=model.Kz,
             probe=model.Kz if probe is None else probe, steps=2, chunk=32)
    t0 = time.perf_counter()
    with torch.no_grad():
        lz, ess, pn = model.log_Z(ix, drop_empty=True, return_ess=True, return_size=True)
    return lz, model._last_qmc_logz_se.clone(), ess, pn, time.perf_counter() - t0


def moments(pn):
    n = torch.arange(1, pn.shape[-1] + 1, dtype=pn.dtype)
    e = (pn * n).sum(-1)
    v = (pn * n.square()).sum(-1) - e.square()
    return e, v


def main(args):
    torch.set_flush_denormal(True)
    m, ix, trips = load_problem(args.checkpoint, args.trips, args.batch, args.phi_scale)
    print(f"checkpoint={args.checkpoint} trips={trips} B={ix.B} Kz={m.Kz} slots={len(ix.item):,}",
          flush=True)
    t0 = time.perf_counter()
    cache = sparse_prepare(m, ix)
    print(f"cache {time.perf_counter()-t0:.3f}s", flush=True)
    if args.tail_bound > 0:
        t0 = time.perf_counter()
        tub, _ubn, gram_lam = tail_partition_upper_bound(m, ix, args.tail_bound)
        print(f"rigorous n>={args.tail_bound} log-partition upper bound {tub.tolist()} "
              f"at lambda_max(Phi'Phi)={float(gram_lam):.6f} "
              f"({time.perf_counter()-t0:.3f}s)", flush=True)

    radii = tuple(float(x) for x in args.radii.split(",") if x)
    t0 = time.perf_counter()
    centres, cscores, cseed, all_scores, evals = discover_modes(
        m, ix, cache, top_dirs=args.top_dirs, radii=radii, steps=args.mode_steps,
        max_modes=args.modes, separation=args.separation, stable=args.stable_modes,
        random_dirs=args.random_dirs, seed=args.seed)
    discover_s = time.perf_counter() - t0
    print(f"discover {discover_s:.3f}s; leading Phi-gram eigenvalues "
          f"{evals[:min(6,len(evals))].tolist()}", flush=True)
    zz = centres.detach().requires_grad_(True)
    with torch.enable_grad():
        vv = objective(m, ix, cache, zz, stable=args.stable_modes)
        gg = torch.autograd.grad(vv.sum(), zz)[0].detach()
    for b, trip in enumerate(trips):
        desc = [dict(norm=round(float(centres[b, j].norm()), 4),
                     F=round(float(cscores[b, j]), 4), seed=int(cseed[b, j]))
                     | {"grad": round(float(gg[b, j].norm()), 4)}
                for j in range(args.modes)]
        print(f"trip {trip} modes {desc}; scan_F=[{float(all_scores[b].min()):.3f},"
              f"{float(all_scores[b].max()):.3f}]", flush=True)

    if args.map_starts > 0:
        eng = torch.quasirandom.SobolEngine(m.Kz, scramble=True, seed=args.seed + 15485863)
        u = eng.draw(args.map_starts).double().clamp(1e-12, 1 - 1e-12)
        dirs = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        random_start = (args.map_radius * dirs).unsqueeze(0).expand(ix.B, -1, -1)
        map_start = torch.cat([centres, random_start], dim=1)
        t0 = time.perf_counter()
        map_centres, map_energy, map_sizes = map_coordinate_ascent(
            m, ix, map_start, steps=args.map_steps)
        print(f"MAP certificate {time.perf_counter()-t0:.3f}s", flush=True)
        for b, trip in enumerate(trips):
            order = torch.argsort(map_energy[b], descending=True)
            seen, desc = [], []
            for j in order.tolist():
                if any(float((map_centres[b, j] - q).norm()) < args.separation
                       for q in seen):
                    continue
                seen.append(map_centres[b, j])
                desc.append(dict(E=round(float(map_energy[b, j]), 4),
                                 norm=round(float(map_centres[b, j].norm()), 4),
                                 size=int(map_sizes[b, j])))
                if len(desc) == min(8, args.map_starts + args.modes):
                    break
            print(f"trip {trip} MAP lower bounds {desc}", flush=True)
        if args.use_map_centres:
            new_centres = centres.clone()
            for b in range(ix.B):
                chosen = [centres[b, 0]]
                for j in torch.argsort(map_energy[b], descending=True).tolist():
                    q = map_centres[b, j]
                    if all(float((q - old).norm()) >= args.separation for old in chosen):
                        chosen.append(q)
                    if len(chosen) == args.modes:
                        break
                while len(chosen) < args.modes:
                    chosen.append(chosen[-1])
                new_centres[b] = torch.stack(chosen)
            centres = new_centres
            print("using local plus highest-energy MAP component means as proposal centres",
                  flush=True)

    results = {}
    for n in args.local_nodes:
        lz, se, ess, pn, sec = local_rule(
            m, ix, n, seed=args.seed, probe=args.local_probe)
        en, var = moments(pn)
        results[f"local{n}"] = dict(lz=lz, se=se, ess=ess, en=en, var=var, sec=sec)
        print(f"local N={n:5d} {sec:7.3f}s  "
              f"logZ={lz.tolist()}  se={se.tolist()}  E[n]={en.tolist()}  "
              f"Var(n)={var.tolist()}  ESS={ess.tolist()}", flush=True)

    frames = None
    if args.laplace:
        t0 = time.perf_counter()
        Q, sd, raw = laplace_frames(m, ix, cache, centres, stable=args.stable_modes)
        frames = (Q, sd)
        print(f"Laplace frames {time.perf_counter()-t0:.3f}s; "
              f"curvature min/max={float(raw.min()):.6g}/{float(raw.max()):.6g}; "
              f"sd min/max={float(sd.min()):.6g}/{float(sd.max()):.6g}", flush=True)
    elif args.subspace > 0:
        t0 = time.perf_counter()
        Q, sd, raw = subspace_laplace_frames(
            m, ix, cache, centres, rank=args.subspace,
            iterations=args.subspace_iters, stable=args.stable_modes,
            eps=args.subspace_eps)
        frames = (Q, sd)
        print(f"Subspace Laplace frames (rank {args.subspace}, "
              f"iterations {args.subspace_iters}) {time.perf_counter()-t0:.3f}s; "
              f"curvature min/max={float(raw.min()):.6g}/{float(raw.max()):.6g}; "
              f"sd min/max={float(sd.min()):.6g}/{float(sd.max()):.6g}", flush=True)
    elif args.directional > 0:
        t0 = time.perf_counter()
        Q, sd, raw = directional_frames(
            m, ix, cache, centres, probes=args.directional,
            stable=args.stable_modes, eps=args.directional_eps)
        frames = (Q, sd)
        print(f"Directional frames ({args.directional} probes) "
              f"{time.perf_counter()-t0:.3f}s; "
              f"curvature min/max={float(raw.min()):.6g}/{float(raw.max()):.6g}; "
              f"sd min/max={float(sd.min()):.6g}/{float(sd.max()):.6g}", flush=True)

    for n in args.mix_nodes:
        t0 = time.perf_counter()
        with torch.no_grad():
            lz, se, ess, pn, replz = multimode_rqmc(
                m, ix, cache, centres, total_nodes=n, reps=args.reps, seed=args.seed,
                stable=args.stable_integral, frames=frames)
        sec = time.perf_counter() - t0
        en, var = moments(pn)
        results[f"mix{n}"] = dict(lz=lz, se=se, ess=ess, en=en, var=var, sec=sec)
        print(f"mix   N={n:5d} {sec:7.3f}s  "
              f"logZ={lz.tolist()}  se={se.tolist()}  E[n]={en.tolist()}  "
              f"Var(n)={var.tolist()}  ESS={ess.tolist()} reps={replz.tolist()}", flush=True)

    # Last requested mixture count is the reference for a compact error table.  This is a
    # convergence reference, not a claim of truth; coverage is checked separately by wider
    # direction/radius scans.
    ref = results[f"mix{args.mix_nodes[-1]}"]["lz"]
    print("errors vs largest multimode rule:", flush=True)
    for name, value in results.items():
        err = value["lz"] - ref
        print(f"  {name:>12s}: mean_abs={float(err.abs().mean()):.6f} "
              f"max_abs={float(err.abs().max()):.6f}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="out/v3_run100_best.pt")
    p.add_argument("--phi-scale", type=float, default=1.0,
                   help="frozen diagnostic rescaling only; never writes the checkpoint")
    p.add_argument("--tail-bound", type=int, default=25,
                   help="report rigorous upper bound on partition mass at/above this size")
    p.add_argument("--trips", type=int, nargs="*")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--top-dirs", type=int, default=4)
    p.add_argument("--random-dirs", type=int, default=0)
    p.add_argument("--radii", default="8,16,24,32")
    p.add_argument("--mode-steps", type=int, default=6)
    p.add_argument("--modes", type=int, default=2)
    p.add_argument("--separation", type=float, default=3.0)
    p.add_argument("--stable-modes", action="store_true",
                   help="locate modes with the slower log-domain reference kernel")
    p.add_argument("--stable-integral", action="store_true",
                   help="evaluate mixture nodes with the log-domain reference kernel")
    p.add_argument("--laplace", action="store_true",
                   help="use the measured Hessian covariance at every discovered mode")
    p.add_argument("--directional", type=int, default=0,
                   help="use this many cheap Phi-frame directional curvature probes")
    p.add_argument("--directional-eps", type=float, default=0.35)
    p.add_argument("--subspace", type=int, default=0,
                   help="rank of a local covariance Hessian-vector sketch; 0 disables")
    p.add_argument("--subspace-iters", type=int, default=1,
                   help="block subspace iterations before the Rayleigh--Ritz solve")
    p.add_argument("--subspace-eps", type=float, default=0.05)
    p.add_argument("--map-starts", type=int, default=0,
                   help="number of spherical starts for the exact MAP lower-bound check")
    p.add_argument("--map-radius", type=float, default=32.0)
    p.add_argument("--map-steps", type=int, default=20)
    p.add_argument("--use-map-centres", action="store_true")
    p.add_argument("--reps", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260820)
    p.add_argument("--local-nodes", type=int, nargs="+", default=[32, 128, 512])
    p.add_argument("--local-probe", type=int, default=None,
                   help="override local rule probes; -1 selects its identity frame")
    p.add_argument("--mix-nodes", type=int, nargs="+", default=[64, 128, 512, 2048])
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
