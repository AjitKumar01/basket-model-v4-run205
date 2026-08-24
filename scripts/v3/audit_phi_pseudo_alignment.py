"""Read-only alignment audit for a version-4 conditional Phi direction.

This does not optimize parameters or write a checkpoint.  It compares the exact
single-site pseudolikelihood Phi score with two independent high-node estimates of the
unchanged full joint-likelihood Phi score on a frozen training minibatch.  A candidate is
rejected early if the independent joint references do not agree or if the conditional
direction fails the predeclared ascent-direction gates.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from data import build
from features import Features
from fit import Batcher
from qmc_multimode_probe import load_problem
from ragged import set_quad


torch.set_default_dtype(torch.float64)

RUN184_BATCH90 = [
    35555, 76873, 132289, 122952, 20290, 86321, 136514, 16685,
    94155, 103420, 58866, 11755, 59672, 118153, 59721, 4909,
    132027, 21295, 69500, 103778, 8256, 20555, 33836, 108623,
]


def vector_metrics(candidate, reference):
    a, b = candidate.reshape(-1), reference.reshape(-1)
    aa, bb = float(a @ a), float(b @ b)
    ab = float(a @ b)
    na, nb = math.sqrt(max(aa, 0.0)), math.sqrt(max(bb, 0.0))
    cosine = ab / max(na * nb, 1e-300)
    scale = ab / max(aa, 1e-300)
    scaled_residual = float((scale * a - b).norm()) / max(nb, 1e-300)
    return {
        "cosine": cosine,
        "candidate_norm": na,
        "reference_norm": nb,
        "best_positive_scale": max(scale, 0.0),
        "unconstrained_scale": scale,
        "scaled_residual": scaled_residual,
    }


def joint_phi_score(model, ix, line_item, line_trip, line_cat, line_ctx,
                    nodes, seed, antithetic):
    # Full finite-difference local covariance is deliberately reference-only.  Its cost is
    # acceptable for this frozen audit and was already validated against subset
    # enumeration; production training is prohibited from enabling the rejected sketch.
    set_quad(model, qmc_n=nodes, qmc_seed=seed, qmc_reps=4, Kz=model.Kz,
             probe=-1, steps=2, chunk=32, size_bands=1, size_steps=3,
             mode_logtol=4.0, mode_sep=1.0, mix_n=2 * nodes,
             antithetic=antithetic)
    model.quad_subspace_rank = model.Kz
    model.quad_subspace_iters = 0
    model.quad_subspace_eps = 0.05
    model.zero_grad(set_to_none=True)
    started = time.perf_counter()
    observed_energy = model.energy(
        line_item, line_trip, line_cat, ix.B, line_ctx).sum()
    logz = model.log_Z(ix, drop_empty=True).sum()
    score = torch.autograd.grad(observed_energy - logz, model.phi)[0].detach()
    seconds = time.perf_counter() - started
    model.quad_subspace_rank = 0
    return score, float(observed_energy.detach() - logz.detach()), seconds


def pseudo_phi_score(model, ix, line_item, line_trip, line_cat, line_ctx):
    model.zero_grad(set_to_none=True)
    started = time.perf_counter()
    objective = model.pseudo_loglik(
        ix, line_item, line_trip, line_cat, ix.B, line_ctx=line_ctx).sum()
    score = torch.autograd.grad(objective, model.phi)[0].detach()
    return score, float(objective.detach()), time.perf_counter() - started


def stiefel_tangent(phi, score, spectral_mass=2.0):
    """Orthogonal score projection for Phi'Phi = spectral_mass I."""
    centred = score - score.mean(0, keepdim=True)
    cross = phi.T @ centred
    symmetric = 0.5 * (cross + cross.T)
    return centred - phi @ (symmetric / spectral_mass)


def actual_projected_direction(model, score, epsilon):
    """Directional derivative through fit.py's centre/whiten/operator projection."""
    base = model.phi.detach().clone()
    unit = score / score.norm().clamp_min(1e-300)
    with torch.no_grad():
        model.phi.copy_(base)
        model.project(0.96, centre=True, whiten=0.5, op_max=2.0)
        projected_base = model.phi.clone()
        model.phi.copy_(projected_base + float(epsilon) * unit)
        model.project(0.96, centre=True, whiten=0.5, op_max=2.0)
        direction = (model.phi - projected_base) / float(epsilon)
        model.phi.copy_(base)
    return direction


def main(args):
    torch.set_flush_denormal(True)
    trips = args.trips or RUN184_BATCH90
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_iteration = int(blob.get("iter", -1)) if isinstance(blob, dict) else -1
    checkpoint_data = blob.get("data", {}) if isinstance(blob, dict) else {}
    if checkpoint_data.get("affinity") == "1":
        os.environ["V3_AFFINITY"] = "1"
    model, _ix0, _ = load_problem(args.checkpoint, [int(trips[0])], 1)
    if bool(model.factored_size_enabled):
        raise RuntimeError("checkpoint uses the retired factored-size model")
    if int(checkpoint_data.get("n_cat", model.C)) != model.C:
        raise RuntimeError("checkpoint/data category partition mismatch")
    with torch.no_grad():
        historical_phi = model.phi.detach().clone()
        historical_gram = torch.linalg.eigvalsh(historical_phi.T @ historical_phi)
        if args.apply_production_initial_projection:
            model.project(0.96, centre=True, whiten=0.5, op_max=2.0)
        initial_projection_jump = float((model.phi - historical_phi).norm())
    data = build()
    training = set(np.flatnonzero(data["trip_split"] == 0).tolist())
    if any(int(t) not in training for t in trips):
        raise RuntimeError("every frozen audit trip must belong to the training split")
    features = Features(int(data["n_item"]), int(data["n_store"]), 712)
    batcher = Batcher(data, features, model.nmax)
    ix, ctx, line_ctx, house, line_item, line_trip, line_cat, _units = \
        batcher.make(np.asarray(trips, dtype=np.int64))
    model.house, model.ctx = house, ctx

    pseudo, pseudo_value, pseudo_seconds = pseudo_phi_score(
        model, ix, line_item, line_trip, line_cat, line_ctx)
    references, reference_values, reference_seconds = [], [], []
    for seed in args.reference_seeds:
        score, value, seconds = joint_phi_score(
            model, ix, line_item, line_trip, line_cat, line_ctx,
            args.reference_nodes, int(seed), bool(args.antithetic))
        references.append(score)
        reference_values.append(value)
        reference_seconds.append(seconds)
        print(f"joint reference seed {seed}: value {value:.6f}, {seconds:.2f}s",
              flush=True)
    reference = torch.stack(references).mean(0)
    reference_agreement = (vector_metrics(references[0], references[1])
                           if len(references) == 2 else None)
    alignment = vector_metrics(pseudo, reference)
    candidate, candidate_value, candidate_seconds = joint_phi_score(
        model, ix, line_item, line_trip, line_cat, line_ctx,
        args.candidate_nodes, int(args.candidate_seed), bool(args.antithetic))
    candidate_alignment = vector_metrics(candidate, reference)
    candidate_value_gap_per_trip = abs(
        candidate_value - float(np.mean(reference_values))) / len(trips)
    phi = model.phi.detach()
    gram_eigenvalues = torch.linalg.eigvalsh(phi.T @ phi)
    tangent_references = [stiefel_tangent(phi, score) for score in references]
    tangent_reference = torch.stack(tangent_references).mean(0)
    tangent_pseudo = stiefel_tangent(phi, pseudo)
    tangent_reference_agreement = (
        vector_metrics(tangent_references[0], tangent_references[1])
        if len(tangent_references) == 2 else None)
    tangent_alignment = vector_metrics(tangent_pseudo, tangent_reference)
    projected_directions = {
        str(epsilon): {
            "retained_norm_fraction": float(actual_projected_direction(
                model, pseudo, epsilon).norm()),
            "alignment_with_joint_score": vector_metrics(
                actual_projected_direction(model, pseudo, epsilon), reference),
        }
        for epsilon in args.projection_eps
    }

    # Gates are fixed before looking at the result.  Reference agreement ensures the test
    # does not reject/accept based on QMC noise.  A cosine of 0.8 means at most 36.9 degrees
    # from the audited joint ascent direction; residual <=0.6 is its equivalent one-scalar
    # approximation gate.  Every gate must pass; otherwise no pilot is authorized.
    gates = {
        "reference_cosine_at_least_0.99": (
            reference_agreement is not None
            and reference_agreement["cosine"] >= 0.99),
        "reference_value_gap_at_most_0.02_nat_per_trip": (
            max(reference_values) - min(reference_values)
            <= 0.02 * len(trips)),
        "pseudo_joint_cosine_at_least_0.8": alignment["cosine"] >= 0.8,
        "positive_projection": alignment["unconstrained_scale"] > 0.0,
        "scaled_residual_at_most_0.6": alignment["scaled_residual"] <= 0.6,
        "tangent_reference_cosine_at_least_0.99": (
            tangent_reference_agreement is not None
            and tangent_reference_agreement["cosine"] >= 0.99),
        "tangent_pseudo_joint_cosine_at_least_0.8": (
            tangent_alignment["cosine"] >= 0.8),
        "tangent_positive_projection": tangent_alignment["unconstrained_scale"] > 0.0,
    }
    accepted = all(gates.values())
    candidate_gates = {
        "candidate_joint_cosine_at_least_0.99": (
            candidate_alignment["cosine"] >= 0.99),
        "candidate_scaled_residual_at_most_0.15": (
            candidate_alignment["scaled_residual"] <= 0.15),
        "candidate_value_gap_at_most_0.02_nat_per_trip": (
            candidate_value_gap_per_trip <= 0.02),
    }
    output = {
        "purpose": "read-only direction audit; no optimizer update",
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": checkpoint_iteration,
        "trips": [int(t) for t in trips],
        "reference": {
            "nodes": args.reference_nodes,
            "seeds": [int(x) for x in args.reference_seeds],
            "joint_loglik_sum": reference_values,
            "seconds": reference_seconds,
            "independent_agreement": reference_agreement,
        },
        "pseudolikelihood": {
            "value_sum": pseudo_value,
            "seconds": pseudo_seconds,
            "alignment_with_mean_joint_score": alignment,
        },
        "candidate_joint_rule": {
            "nodes": int(args.candidate_nodes),
            "seed": int(args.candidate_seed),
            "joint_loglik_sum": candidate_value,
            "seconds": candidate_seconds,
            "value_gap_nat_per_trip": candidate_value_gap_per_trip,
            "alignment_with_mean_reference_score": candidate_alignment,
            "gates": candidate_gates,
            "accepted": all(candidate_gates.values()),
        },
        "constraint_geometry": {
            "production_initial_projection_applied": bool(
                args.apply_production_initial_projection),
            "historical_gram_eigenvalue_min": float(historical_gram.min()),
            "historical_gram_eigenvalue_max": float(historical_gram.max()),
            "gram_eigenvalue_min": float(gram_eigenvalues.min()),
            "gram_eigenvalue_max": float(gram_eigenvalues.max()),
            "operator_ceiling": 2.0,
            "historical_iteration_zero_projection_jump": initial_projection_jump,
            "joint_tangent_fraction": float(
                tangent_reference.norm() / reference.norm().clamp_min(1e-300)),
            "pseudo_tangent_fraction": float(
                tangent_pseudo.norm() / pseudo.norm().clamp_min(1e-300)),
            "tangent_reference_agreement": tangent_reference_agreement,
            "tangent_alignment": tangent_alignment,
            "actual_projection_finite_differences": projected_directions,
        },
        "gates": gates,
        "accepted_for_further_frozen_batches": accepted,
        "training_authorized": False,
    }
    print(json.dumps(output, indent=2), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
        print(f"wrote {args.output}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=(
        "out/v3_run172_v4_moment_ipf_fresh1000_best.pt"))
    parser.add_argument("--trips", nargs="*", type=int)
    parser.add_argument("--reference-nodes", type=int, default=512)
    parser.add_argument("--reference-seeds", nargs=2, type=int,
                        default=[17000069, 19000081])
    parser.add_argument("--candidate-nodes", type=int, default=64)
    parser.add_argument("--candidate-seed", type=int, default=0)
    parser.add_argument("--antithetic", type=int, default=1)
    parser.add_argument("--projection-eps", nargs="*", type=float,
                        default=[1e-5, 1e-4, 1e-3])
    parser.add_argument("--apply-production-initial-projection", type=int, default=1)
    parser.add_argument("--output", default="")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
