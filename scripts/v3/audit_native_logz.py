"""Read-only complete-log-Z gate for the native degree-aware category kernel."""
from __future__ import annotations

import argparse
import contextlib
import json
import time

import numpy as np
import torch

from audit_phi_pseudo_alignment import RUN184_BATCH90
from qmc_multimode_probe import load_problem
from ragged import set_quad


torch.set_default_dtype(torch.float64)


def evaluate(model, ix, native, profile=False, esp_native=None):
    model._poly_degree_native = bool(native)
    model._esp_native = bool(native if esp_native is None else esp_native)
    names, parameters = zip(*[(name, value) for name, value in model.named_parameters()
                              if value.requires_grad])
    profiler = (torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True)
        if profile else contextlib.nullcontext())
    started = time.perf_counter()
    with profiler:
        logz, pn = model.log_Z(ix, drop_empty=True, return_size=True)
        size_axis = torch.arange(1, pn.shape[1] + 1, dtype=pn.dtype, device=pn.device)
        # The small size-moment term forces the returned size graph through the same reverse
        # path used by version-4 calibration penalties; it is identical for both kernels.
        objective = logz.sum() + 0.01 * (pn * size_axis).sum()
        raw_gradient = torch.autograd.grad(objective, parameters, allow_unused=True)
    seconds = time.perf_counter() - started
    gradients = {name: value.detach() for name, value in zip(names, raw_gradient)
                 if value is not None}
    profile_table = (profiler.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=40) if profile else None)
    return dict(logz=logz.detach(), pn=pn.detach(), gradients=gradients,
                seconds=seconds, profile_table=profile_table)


def main(args):
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    trips = RUN184_BATCH90[:args.batch]
    model, ix, _ = load_problem(args.checkpoint, trips, args.batch)
    active_products = int(model.phi.shape[0])
    if args.phi_mask:
        mask = torch.as_tensor(np.load(args.phi_mask), dtype=torch.bool)
        if mask.shape != (model.phi.shape[0],):
            raise ValueError("phi mask does not match the checkpoint catalogue")
        with torch.no_grad():
            model.phi[~mask] = 0.0
        active_products = int(mask.sum())
    reference_chunk = args.reference_chunk if args.reference_chunk > 0 else args.chunk
    set_quad(model, qmc_n=args.nodes, qmc_seed=args.qmc_seed,
             qmc_reps=4, Kz=model.Kz, probe=-1, steps=2, chunk=args.chunk,
             size_bands=1, size_steps=3, mode_logtol=4.0, mode_sep=1.0,
             mix_n=2 * args.nodes, antithetic=True)

    if args.reference_chunk > 0:
        model.quad_chunk = reference_chunk
        dense = evaluate(model, ix, True)
        model.quad_chunk = args.chunk
    elif args.reference_old_native:
        dense = evaluate(model, ix, True, esp_native=False)
    else:
        dense = None if args.native_only else evaluate(model, ix, False)
    native = evaluate(model, ix, True, profile=args.profile)
    if native["profile_table"] is not None:
        print(native["profile_table"])
    shared = ([] if dense is None else
              sorted(set(dense["gradients"]) & set(native["gradients"])))
    rows = {}
    max_relative = 0.0
    max_absolute = 0.0
    for name in shared:
        expected, actual = dense["gradients"][name], native["gradients"][name]
        absolute = float((actual - expected).abs().max())
        relative = float((actual - expected).norm()
                         / expected.norm().clamp_min(1e-300))
        rows[name] = {"max_absolute": absolute, "relative_l2": relative}
        max_absolute = max(max_absolute, absolute)
        max_relative = max(max_relative, relative)

    result = {
        "purpose": ("complete original-version4 chunk-equivalence gate; no training"
                    if args.reference_chunk > 0 else
                    "complete original-version4 log-Z native-kernel gate; no training"),
        "checkpoint": args.checkpoint,
        "trips": trips,
        "batch": args.batch,
        "nodes": args.nodes,
        "threads": torch.get_num_threads(),
        "chunk": args.chunk,
        "active_interaction_products": active_products,
        "reference_chunk": (reference_chunk if args.reference_chunk > 0 else None),
        "dense_seconds": None if dense is None else dense["seconds"],
        "native_seconds": native["seconds"],
        "speedup": None if dense is None else dense["seconds"] / native["seconds"],
        "max_abs_logz_error": (None if dense is None else
                                float((native["logz"] - dense["logz"]).abs().max())),
        "max_abs_size_probability_error": (None if dense is None else
                                             float((native["pn"] - dense["pn"]).abs().max())),
        "max_parameter_gradient_absolute_error": max_absolute,
        "max_parameter_gradient_relative_l2": max_relative,
        "gradient_blocks": rows,
    }
    minimum_speedup = 1.0 if args.reference_chunk > 0 else 1.5
    result["passes_full_gate"] = None if dense is None else bool(
        result["speedup"] >= minimum_speedup
        and result["max_abs_logz_error"] <= 1e-10
        and result["max_abs_size_probability_error"] <= 1e-10
        and result["max_parameter_gradient_relative_l2"] <= 1e-9)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=(
        "out/v3_run172_v4_moment_ipf_fresh1000_best.pt"))
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--qmc-seed", type=int, default=19000057)
    parser.add_argument("--threads", type=int, default=0,
                        help="intra-op CPU threads; zero retains the PyTorch default")
    parser.add_argument("--native-only", action="store_true",
                        help="timing-only mode after the dense/native exactness gate passed")
    parser.add_argument("--reference-chunk", type=int, default=0,
                        help="compare native output/gradient with this node chunk")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--phi-mask", default="",
                        help="read-only timing ablation with inactive Phi rows zeroed")
    parser.add_argument("--reference-old-native", action="store_true",
                        help="compare against the run191 native-category/old-ESP path")
    main(parser.parse_args())
