"""Measure the likelihood effect of the historical 256-item ESP bucket cutoff.

The old ragged kernel silently left every category larger than 256 at the identity
polynomial.  Affinity data has a 1,774-product residual category, so this compares that
historical normalizer with the coverage-complete kernel on exactly fit.py's validation
sample.  This is a diagnostic only; training and evaluation must use the complete kernel.
"""
import argparse
import os

import numpy as np
import torch

import ragged
from data import build
from evalall import load_any
from features import Features
from fit import Batcher


def legacy_esp_bucketed(w, row_of, n_rows, R, row_size, item_pos,
                        buckets=(8, 32, 96, 256), parallel=False):
    """The pre-fix implementation: rows above the last bucket remain (1, 0, ...)."""
    lead = w.shape[:-1]
    out = torch.zeros(lead + (n_rows, R + 1), dtype=w.dtype, device=w.device)
    out[..., 0] = 1.0
    lo = 0
    for hi in buckets:
        sel_r = (row_size > lo) & (row_size <= hi)
        lo = hi
        if not bool(sel_r.any()):
            continue
        ridx = torch.nonzero(sel_r, as_tuple=True)[0]
        loc = torch.full((n_rows,), -1, dtype=torch.long, device=w.device)
        loc[ridx] = torch.arange(len(ridx), device=w.device)
        sel_i = sel_r[row_of]
        wi = w[..., sel_i]
        flat = loc[row_of[sel_i]] * hi + item_pos[sel_i]
        P = torch.zeros(lead + (len(ridx) * hi,), dtype=w.dtype, device=w.device)
        P = P.index_copy(-1, flat, wi).view(lead + (len(ridx), hi))
        E = torch.zeros(lead + (len(ridx), R + 1), dtype=w.dtype, device=w.device)
        E[..., 0] = 1.0
        for i in range(hi):
            x = P[..., i].unsqueeze(-1)
            E = E + x * torch.nn.functional.pad(E[..., :-1], (1, 0))
        out = out.index_copy(-2, ridx, E)
    return out


def main(a):
    torch.set_default_dtype(torch.float64)
    D = build()
    J, N, C, S = (int(D[k]) for k in ("n_item", "n_user", "n_cat", "n_store"))
    path = a.ckpt if os.path.isabs(a.ckpt) else os.path.join("..", "..", "out", a.ckpt)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    q = blob.get("quad") or {}
    kz = int(q.get("Kz", blob["model"]["phi"].shape[1]))
    model = ragged.RaggedModel(J=J, N=N, C=C, K=32, Kz=kz,
                               nmax=a.nmax, R=a.R, S=S, Kp=8)
    meta = load_any(path, model, J, D)
    model.double().eval()
    batcher = Batcher(D, Features(J, S, 712), a.nmax)

    trips = np.flatnonzero(D["trip_split"] == 1)
    lp, lc = D["line_ptr"], D["line_cat"]
    keep = []
    for t in trips:
        lo, hi = int(lp[t]), int(lp[t + 1])
        if hi - lo <= a.nmax and (hi <= lo or np.bincount(lc[lo:hi]).max() <= a.R):
            keep.append(t)
    trips = np.asarray(keep, dtype=int)
    trips = trips[np.random.default_rng(12345).permutation(len(trips))][:a.n_trips]

    if a.legacy:
        ragged.esp_bucketed = legacy_esp_bucketed
    energy, logz, lines = 0.0, 0.0, 0
    for k in range(0, len(trips), a.chunk):
        sub = trips[k:k + a.chunk]
        ix, ctx, lctx, hh, li, lt, lc, _ = batcher.make(sub)
        model.house, model.ctx = hh, ctx
        with torch.no_grad():
            energy += float(model.energy(li, lt, lc, ix.B, lctx).sum())
            logz += float(model.log_Z(ix, drop_empty=True).sum())
        lines += len(li)
    n = len(trips)
    print(f"checkpoint: {os.path.basename(path)} ({meta})")
    print(f"kernel: {'legacy cutoff at 256' if a.legacy else 'complete coverage'}")
    print(f"cases: {n}, lines: {lines}")
    print(f"energy/basket: {energy / n:.6f}")
    print(f"logZ/basket: {logz / n:.6f}")
    print(f"set loglik/basket: {(energy - logz) / n:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--n-trips", type=int, default=384)
    parser.add_argument("--chunk", type=int, default=48)
    parser.add_argument("--nmax", type=int, default=120)
    parser.add_argument("--R", type=int, default=23)
    main(parser.parse_args())
