"""Frozen-checkpoint audit of the original version-4 full-joint QMC normalizer.

This script never changes model parameters.  It compares production-size RQMC rules on
one fixed validation manifest against an independent high-node rule, reporting log-Z and
size-moment gaps, replicate SE, mode-selection rate and runtime.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from data import build
from features import Features
from fit import Batcher
from ragged import RaggedModel, set_quad


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "out"))


def load_model(path, data, R):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    J, N, C, S = (int(data[k]) for k in ("n_item", "n_user", "n_cat", "n_store"))
    model = RaggedModel(
        J, N, C, K=int(state["alpha"].shape[1]), Kz=int(state["phi"].shape[1]),
        nmax=int(state["rho_0_free"].shape[0]), R=R, S=S,
        Kp=int(state["beta"].shape[1])).double()
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [key for key in missing if key != "cat_of"]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    if bool(model.factored_size_enabled):
        raise RuntimeError("checkpoint uses the retired factored-size model")
    if model.rho_pair_cap != model.nmax:
        raise RuntimeError("category interaction is not quadratic on complete support")
    quad = blob.get("quad", {}) if isinstance(blob, dict) else {}
    model._poly_degree_native = bool(quad.get("poly_degree_native", False))
    model._esp_native = bool(quad.get("esp_native", False))
    model.eval()
    return model, blob


@torch.no_grad()
def evaluate(model, batcher, trips, config, chunk):
    # Match production's paired normal nodes.  This is a proposal/integration setting,
    # not a model change, and must agree when comparing node counts on a checkpoint.
    set_quad(
        model, qmc_n=config["nodes"], qmc_seed=config["seed"],
        qmc_reps=config["reps"], Kz=model.Kz, probe=config["probe"],
        steps=config["steps"], chunk=config["chunk"],
        size_bands=config["size_bands"],
        size_steps=config["size_steps"], mode_logtol=config["mode_logtol"],
        mode_sep=config["mode_sep"], mix_n=2 * config["nodes"],
        antithetic=config.get("antithetic", True))
    lz_all, en_all, se_all, mode_all, ess_all = [], [], [], [], []
    started = time.perf_counter()
    for start in range(0, len(trips), chunk):
        sub = trips[start:start + chunk]
        ix, ctx, _lctx, house, *_ = batcher.make(sub)
        model.house, model.ctx = house, ctx
        lz, ess, pn = model.log_Z(
            ix, drop_empty=True, return_ess=True, return_size=True)
        n = torch.arange(1, pn.shape[1] + 1, dtype=pn.dtype)
        lz_all.append(lz.cpu())
        en_all.append((pn * n).sum(1).cpu())
        se_all.append(model._last_qmc_logz_se.cpu())
        mode_count = model._last_qmc_mode_count
        mode_all.append((torch.ones(ix.B, dtype=torch.long)
                         if mode_count is None else mode_count.cpu()))
        ess_all.append(ess.cpu())
    return {
        "logz": torch.cat(lz_all).numpy(),
        "en": torch.cat(en_all).numpy(),
        "se": torch.cat(se_all).numpy(),
        "modes": torch.cat(mode_all).numpy(),
        "ess": torch.cat(ess_all).numpy(),
        "seconds": time.perf_counter() - started,
    }


def summary(raw, reference=None):
    result = {
        "seconds": float(raw["seconds"]),
        "rqmc_se_mean": float(raw["se"].mean()),
        "rqmc_se_max": float(raw["se"].max()),
        "ess_mean": float(raw["ess"].mean()),
        "ess_min": float(raw["ess"].min()),
        "two_mode_rate": float((raw["modes"] == 2).mean()),
        "mean_expected_size": float(raw["en"].mean()),
    }
    if reference is not None:
        dz = raw["logz"] - reference["logz"]
        dn = raw["en"] - reference["en"]
        result.update(
            logz_mean_abs_gap=float(np.abs(dz).mean()),
            logz_max_abs_gap=float(np.abs(dz).max()),
            en_mean_abs_gap=float(np.abs(dn).mean()),
            en_max_abs_gap=float(np.abs(dn).max()),
        )
    return result


def main(args):
    torch.set_default_dtype(torch.float64)
    torch.set_flush_denormal(True)
    data = build()
    J, S = int(data["n_item"]), int(data["n_store"])
    features = Features(J, S, 712)
    model, blob = load_model(args.checkpoint, data, args.R)
    batcher = Batcher(data, features, model.nmax)
    valid = np.flatnonzero(data["trip_split"] == 1)
    valid = valid[np.random.default_rng(args.manifest_seed).permutation(len(valid))]
    trips = valid[:args.trips]
    common = dict(reps=4, probe=-1, steps=args.steps, chunk=32,
                  size_steps=args.size_steps, mode_sep=1.0, mode_logtol=4.0,
                  antithetic=True)
    configs = {
        "n64_production": common | dict(nodes=64, seed=0,
                                         size_bands=args.size_bands),
        "n128_production": common | dict(nodes=128, seed=0,
                                          size_bands=args.size_bands),
        "n256_production": common | dict(nodes=256, seed=0,
                                          size_bands=args.size_bands),
        "reference": common | dict(nodes=args.reference_nodes, seed=1_000_003,
                                    size_bands=args.reference_size_bands),
    }
    raw = {}
    for name, config in configs.items():
        print(f"[qmc-audit] {name}: {config}", flush=True)
        raw[name] = evaluate(model, batcher, trips, config, args.chunk)
        row = summary(raw[name])
        print(f"[qmc-audit]   {row}", flush=True)
    reference = raw["reference"]
    result = {
        "schema": 1,
        "checkpoint": os.path.basename(args.checkpoint),
        "checkpoint_iter": int(blob.get("iter", -1)) if isinstance(blob, dict) else None,
        "law": "original-version4-joint",
        "catalogue": J,
        "Kz": model.Kz,
        "R": model.R,
        "nmax": model.nmax,
        "rho_pair_cap": model.rho_pair_cap,
        "manifest_seed": args.manifest_seed,
        "trip_ids": [int(x) for x in trips],
        "configs": configs,
        "results": {name: summary(value, None if name == "reference" else reference)
                    for name, value in raw.items()},
    }
    path = os.path.join(OUT, args.output + ".json")
    with open(path, "w") as stream:
        json.dump(result, stream, indent=2)
    print(f"[qmc-audit] wrote {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(
        OUT, "v3_run142_original_fresh_nolamcap.pt"))
    parser.add_argument("--trips", type=int, default=48)
    parser.add_argument("--chunk", type=int, default=24)
    parser.add_argument("--manifest-seed", type=int, default=12345)
    parser.add_argument("--reference-nodes", type=int, default=256)
    parser.add_argument("--size-bands", type=int, choices=(0, 1), default=1)
    parser.add_argument("--reference-size-bands", type=int, choices=(0, 1), default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--size-steps", type=int, default=3)
    parser.add_argument("--R", type=int, default=120)
    parser.add_argument("--output", default="v3_run142_original_qmc_audit")
    main(parser.parse_args())
