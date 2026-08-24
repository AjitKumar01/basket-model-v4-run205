"""Evaluate complete-the-basket MRR and truncated MRR@K on fit.py's fixed holdouts.

MRR@K is zero when the held-out item's rank exceeds K.  Recall@K is reported beside it so
the two commonly-confused quantities remain explicit.
"""
import argparse
import os

import numpy as np
import torch

import ragged
from data import build
from diagnose_bucket_coverage import legacy_esp_bucketed
from evalall import load_any
from features import Features
from fit import Batcher, popularity_logits, rec_eval
from ragged import RaggedModel


def popularity_ranks(D, trips, seed=0):
    """Exposure-corrected popularity on the exact rec_eval holdouts/candidate sets."""
    score = popularity_logits(D, np.flatnonzero(D["trip_split"] == 0)).numpy()
    rng = np.random.default_rng(seed)
    ranks = []
    C = int(D["n_cat"])
    for trip in trips:
        lo, hi = int(D["line_ptr"][trip]), int(D["line_ptr"][trip + 1])
        basket = D["line_item"][lo:hi]
        if len(basket) < 2:
            continue
        hidden_pos = int(rng.integers(len(basket)))
        hidden = int(basket[hidden_pos])
        rest = np.delete(basket, hidden_pos)
        store = int(D["trip_store"][trip])
        alo = int(D["store_cat_ptr"][store * C])
        ahi = int(D["store_cat_ptr"][(store + 1) * C])
        candidates = D["store_items"][alo:ahi]
        candidates = candidates[~np.isin(candidates, rest)]
        if not np.any(candidates == hidden):
            continue
        ranks.append(1 + int(np.sum(score[candidates] > score[hidden])))
    return np.asarray(ranks, dtype=float)


def main(a):
    torch.set_default_dtype(torch.float64)
    D = build()
    J, N, C, S = (int(D[k]) for k in ("n_item", "n_user", "n_cat", "n_store"))
    path = a.ckpt if os.path.isabs(a.ckpt) else os.path.join("..", "..", "out", a.ckpt)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    q = blob.get("quad") or {}
    kz = int(q.get("Kz", blob["model"]["phi"].shape[1]))
    model = RaggedModel(J=J, N=N, C=C, K=32, Kz=kz, nmax=a.nmax, R=a.R, S=S, Kp=8)
    meta = load_any(path, model, J, D)
    if a.qmc_n > 0:
        ragged.set_quad(
            model, qmc_n=a.qmc_n, qmc_seed=a.qmc_seed,
            probe=int(q.get("probe", 8)), steps=int(q.get("steps", 2)),
            chunk=int(q.get("chunk", 0)), qmc_reps=int(q.get("reps", 1)),
            size_bands=int(q.get("size_bands", 0)),
            size_steps=int(q.get("size_steps", 2)),
            mode_logtol=float(q.get("mode_logtol", 8.0)),
            mode_sep=float(q.get("mode_sep", 1.0)),
            mix_n=2 * a.qmc_n)
    model.double().eval()
    batcher = Batcher(D, Features(J, S, 712), a.nmax)
    if a.legacy_buckets:
        ragged.esp_bucketed = legacy_esp_bucketed

    split_code = {"validation": 1, "test": 2}[a.split]
    trips = np.flatnonzero(D["trip_split"] == split_code)
    lp, lc = D["line_ptr"], D["line_cat"]
    keep = []
    for t in trips:
        lo, hi = int(lp[t]), int(lp[t + 1])
        if hi - lo <= a.nmax and (hi <= lo or np.bincount(lc[lo:hi]).max() <= a.R):
            keep.append(t)
    trips = np.asarray(keep, dtype=int)
    trips = trips[np.random.default_rng(12345).permutation(len(trips))][:a.n_trips]
    ranks = rec_eval(model, batcher, trips, seed=0, chunk=a.chunk, return_ranks=True,
                     conditioned=not a.unconditioned, pin_strength=a.pin_strength,
                     legacy_pin=a.legacy_pin)
    print(f"checkpoint: {os.path.basename(path)} ({meta})")
    print(f"split: {a.split}")
    if a.unconditioned:
        score_name = "pi unconditioned"
    elif a.legacy_pin:
        score_name = f"pi | rest (+{a.pin_strength:g} legacy pin)"
    else:
        score_name = "pi | rest (exact conditional law)"
    print(f"score: {score_name}")
    print(f"cases: {len(ranks)}")
    reciprocal = 1.0 / ranks
    mrr_se = reciprocal.std(ddof=1) / np.sqrt(len(reciprocal))
    print(f"MRR: {reciprocal.mean():.6f}  SE: {mrr_se:.6f}  "
          f"normal-95%: [{max(0.0, reciprocal.mean()-1.96*mrr_se):.6f}, "
          f"{min(1.0, reciprocal.mean()+1.96*mrr_se):.6f}]")
    print(f"median rank: {np.median(ranks):.0f}")
    for k in a.cutoffs:
        hit = ranks <= k
        print(f"MRR@{k}: {np.mean(np.where(hit, 1.0 / ranks, 0.0)):.6f}  "
              f"R@{k}: {np.mean(hit):.6f}")
    if not a.no_popularity:
        pop = popularity_ranks(D, trips, seed=0)
        if len(pop) != len(ranks):
            raise RuntimeError("model and popularity did not retain the same holdout cases")
        pop_rr = 1.0 / pop
        difference = reciprocal - pop_rr
        difference_se = difference.std(ddof=1) / np.sqrt(len(difference))
        print(f"popularity MRR: {pop_rr.mean():.6f}  median rank: {np.median(pop):.0f}")
        print(f"paired MRR gain: {difference.mean():+.6f}  SE: {difference_se:.6f}  "
              f"normal-95%: [{difference.mean()-1.96*difference_se:+.6f}, "
              f"{difference.mean()+1.96*difference_se:+.6f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="v3_run112_blockscaled_safe_bestmrr.pt")
    parser.add_argument("--split", choices=["validation", "test"],
                        default="validation")
    parser.add_argument("--n-trips", type=int, default=192)
    parser.add_argument("--chunk", type=int, default=24)
    parser.add_argument("--nmax", type=int, default=120)
    parser.add_argument("--R", type=int, default=23)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--legacy-buckets", action="store_true")
    parser.add_argument("--unconditioned", action="store_true",
                        help="rank on marginal incidence without the +6 rest-of-cart pin")
    parser.add_argument("--pin-strength", type=float, default=6.0,
                        help="finite utility shift used by the legacy conditioning audit")
    parser.add_argument("--legacy-pin", action="store_true",
                        help="audit the superseded finite-utility conditioning approximation")
    parser.add_argument("--qmc-n", type=int, default=0,
                        help="override checkpoint QMC nodes for a gradient-convergence audit")
    parser.add_argument("--qmc-seed", type=int, default=0)
    parser.add_argument("--no-popularity", action="store_true")
    main(parser.parse_args())
