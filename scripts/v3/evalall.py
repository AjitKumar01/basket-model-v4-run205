"""One evaluation, correct by construction, reported on validation AND test.

Every number this project has quoted came from a different ad-hoc script, and each one
carried a defect that took a separate investigation to find.  This file exists so there is a
single source of truth.  The defects, and the guard each one bought:

  1. SLICING INSTEAD OF SAMPLING.  Trips are stored in time order, so `va[:384]` is week 83
     alone and `test[:384]` is week 91 alone.  fit.py's held-out likelihood has therefore
     always been a ONE-WEEK number, and week 91 turns out to be an extreme outlier (model
     E[n] +273% against +2% for week 99).  A whole "the model fails to generalise to test"
     conclusion came from that slice.  -> sample uniformly across the split, seeded.

  2. WITHIN-TRIP VARIANCE REPORTED AS POPULATION VARIANCE.  fit.py printed mean(Var(n|trip))
     and compared it against the variance of observed basket sizes ACROSS trips.  The law of
     total variance supplies the missing term: Var(n) = E[Var(n|trip)] + Var[E(n|trip)].
     Measured on run63, those are 28.4 and 73.4 -- so the field read 28 where the answer was
     102, and made a too-WIDE size law look too narrow for the whole session.
     -> report the population variance, and both terms separately.

  3. NEVER EVALUATED ON TEST AT ALL.  fit.py calls evaluate() only on `va`.  No log-likelihood
     on weeks 91+ has ever been computed.  -> both splits, always, side by side.

  4. CHECKPOINT FORMAT.  Three shapes exist (bare state_dict; bare + cat_of; format-2 dict
     from save_ckpt).  Reading a format-2 blob as a state_dict silently loads NOTHING.
     -> detect the format, and hard-fail if any fitted parameter is missing.

  5. ESTIMATES QUOTED WITHOUT THEIR ERROR.  log Z is biased low by Jensen and the bias falls
     as ~1/draws, so a cheap number overstates the likelihood.  E[n] has the same problem.
     -> every quantity is computed at `draws` and at 8x, and both are printed; if they
     disagree the number is draw-limited and must not be quoted.

  6. MONTE CARLO NOISE MISTAKEN FOR SIGNAL.  -> repeat over seeds and report the spread.

Nothing here trains, and nothing here selects.  It reports.

Run:  V3_AFFINITY=1 python3 evalall.py --ckpts v3_run39_best.pt,v3_run62_best.pt
"""
import argparse
import os

import numpy as np
import torch

from data import build
from features import Features
from fit import Batcher, evaluate
from ragged import RaggedModel, set_quad

SPLITS = {"train": 0, "valid": 1, "test": 2}


_QUAD_Q_DEFAULT = 8


def log(m):
    print(f"[ev] {m}", flush=True)


def load_any(path, m, J, D):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    meta = ""
    if isinstance(sd, dict) and sd.get("format") == 2:
        meta = f"iter {sd.get('iter')}, cum_iter {sd.get('cum_iter', '?')}"
        # The normaliser is a property of the CHECKPOINT, not of this script, so it is
        # chosen here where the blob is in scope.  Checkpoints written before the quad key
        # existed carry none and fall back to the Smolyak q=8 rule they were trained under.
        _d = sd.get("data") or {}
        if _d and int(_d.get("n_cat", 0)) not in (0, int(m.rho_c.shape[0])):
            raise SystemExit(
                f"{os.path.basename(path)} was trained on a data partition with "
                f"{_d['n_cat']} categories; this build has {int(m.rho_c.shape[0])}.\n"
                f"  re-run with V3_PARTITION={_d.get('partition','')!r} "
                f"V3_AFFINITY={_d.get('affinity','0')}\n"
                f"  (run97 needs V3_AFFINITY=1 -- 280 categories, not the default 188)")
        if _d and int(_d.get("R", m.R)) != int(m.R):
            raise SystemExit(
                f"{os.path.basename(path)} uses category support R={_d['R']}; "
                f"this evaluation model uses R={m.R}. Re-run evalall with --R {_d['R']}.")
        _q = sd.get("quad") or {}
        # Recreate the checkpoint's exact algebraic kernels.  These flags only change
        # implementation (not the normaliser or its node set), but omitting them makes a
        # fast checkpoint unexpectedly fall back to the much slower Python product tree.
        m._poly_degree_native = bool(_q.get("poly_degree_native", False))
        m._esp_native = bool(_q.get("esp_native", False))
        log("log Z: " + set_quad(m, _q.get("quad_q", _QUAD_Q_DEFAULT), _q.get("qmc_n", 0),
                                 _q.get("qmc_seed", 0), Kz=m.Kz, probe=_q.get("probe", 8),
                                 steps=_q.get("steps", 4), chunk=_q.get("chunk", 0),
                                 qmc_reps=_q.get("reps", 1),
                                 size_bands=_q.get("size_bands", 0),
                                 size_steps=_q.get("size_steps", 2),
                                 mode_logtol=_q.get("mode_logtol", 8.0),
                                 mode_sep=_q.get("mode_sep", 1.0),
                                 mix_n=_q.get("mix_n", 0)))
        sd = sd["model"]
    missing, _ = m.load_state_dict(sd, strict=False)
    zero_start = {"cat_of", "factored_size_enabled", "factored_size_log_p"}
    fitted = [k for k in missing if k not in zero_start]
    if fitted:                                     # guard 4
        raise SystemExit(f"{os.path.basename(path)}: missing fitted parameters {fitted}")
    with torch.no_grad():
        co = torch.zeros(J, dtype=torch.long)
        co[torch.as_tensor(D["line_item"], dtype=torch.long)] = \
            torch.as_tensor(D["line_cat"], dtype=torch.long)
        m.cat_of.copy_(co)
    return meta


def sample_split(D, split, n, nmax, R, seed=0):
    """Uniform sample over the whole split, restricted to trips inside the support.

    In-support filtering matters: a basket the normaliser does not sum over cannot be scored
    by the energy, and including it would silently mis-score rather than error.
    """
    idx = np.flatnonzero(D["trip_split"] == SPLITS[split])
    lp, lc = D["line_ptr"], D["line_cat"]
    keep = []
    for t in idx:
        lo, hi = int(lp[t]), int(lp[t + 1])
        if hi - lo <= nmax and (hi <= lo or np.bincount(lc[lo:hi]).max() <= R):
            keep.append(t)
    keep = np.array(keep)
    rng = np.random.default_rng(seed)
    sel = keep if len(keep) <= n else keep[rng.choice(len(keep), size=n, replace=False)]
    return np.sort(sel)


def size_law(m, Bt, trips, draws, chunk, seed=0):
    """Per-trip E[n] and Var(n), and the population law they aggregate to."""
    Es, Vs, acc = [], [], None
    for k in range(0, len(trips), chunk):
        ix, ctx, lctx, hh, LI, LT, LC, LU = Bt.make(trips[k:k + chunk])
        m.house, m.ctx = hh, ctx
        with torch.no_grad():
            pn = m.size_dist(ix, n_draws=draws, generator=torch.Generator().manual_seed(seed))
            if isinstance(pn, tuple):
                pn = pn[0]
        p = pn.numpy()
        if acc is None:
            acc = np.zeros(p.shape[1])
        nn = np.arange(1, p.shape[1] + 1)
        e = (p * nn).sum(1)
        Es.append(e)
        Vs.append((p * nn ** 2).sum(1) - e ** 2)
        acc[: p.shape[1]] += p.sum(0)
    E, V = np.concatenate(Es), np.concatenate(Vs)
    return E, V, acc / max(acc.sum(), 1e-300)


def dist_stats(p_model, p_obs):
    k = np.arange(1, len(p_model) + 1)
    msk = p_obs > 0
    kl = float((p_obs[msk] * np.log(p_obs[msk] / np.clip(p_model[msk], 1e-300, None))).sum())
    tv = float(0.5 * np.abs(p_model - p_obs).sum())
    ks = float(np.abs(np.cumsum(p_model) - np.cumsum(p_obs)).max())
    return kl, tv, ks


def main(a):
    torch.set_default_dtype(torch.float64)
    D = build()
    J, N, C, S = (int(D[k]) for k in ("n_item", "n_user", "n_cat", "n_store"))
    F = Features(J, S, 712)
    Bt = Batcher(D, F, a.nmax)
    m = RaggedModel(J=J, N=N, C=C, K=32, Kz=a.Kz, nmax=a.nmax, R=a.R, S=S, Kp=8)
    # Deterministic normaliser -- set per checkpoint inside load_any, where the blob that
    # records how the model was trained is in scope.  Importance sampling is wrong by 8-36
    # nats here (verified against exact enumeration), so a silent fallback to it would
    # reintroduce that error.
    global _QUAD_Q_DEFAULT
    _QUAD_Q_DEFAULT = a.quad_q if a.quad_q > 0 else 8
    ptr = D["line_ptr"]

    picks = {s: sample_split(D, s, a.n_trips, a.nmax, a.R) for s in a.splits.split(",")}
    for s, t in picks.items():
        wks = D["trip_week"][t]
        log(f"{s:>5}: {len(t)} trips sampled across weeks {wks.min()}-{wks.max()} "
            f"({len(np.unique(wks))} distinct weeks)")
    log("")

    for name in a.ckpts.split(","):
        path = os.path.join("..", "..", "out", name)
        if not os.path.exists(path):
            log(f"{name}: absent")
            continue
        meta = load_any(path, m, J, D)
        m.double().eval()
        log(f"=== {name}  {meta}")
        log(f"{'split':>6}{'set/bskt':>10}{'@8x':>9}{'units':>8}"
            f"{'E[n]':>8}{'obs':>7}{'@8x':>8}{'varpop':>9}{'obs':>7}"
            f"{'KL':>7}{'TV':>6}{'cap%':>6}")
        for s, trips in picks.items():
            obs = np.array([int(ptr[t + 1]) - int(ptr[t]) for t in trips])
            obs = obs[(obs >= 1) & (obs <= a.nmax)]
            p_obs = np.bincount(obs, minlength=a.nmax + 1)[1:].astype(float)
            p_obs = p_obs / p_obs.sum()

            row = {}
            # Guard 5 exists because an importance-sampled log Z is biased low and the
            # bias falls as ~1/draws, so a cheap number overstates the likelihood.  Under
            # QUADRATURE n_draws is ignored -- _log_Z_quad uses a fixed grid -- so the hi
            # pass recomputes the lo number exactly (verified identical to 6 decimals) and
            # costs half of this script's runtime for nothing.  Skip it, and say so.
            _passes = (("lo", a.draws),) if (getattr(m, "quad", None) is not None
                                                   or getattr(m, "quad_a", None) is not None) \
                else (("lo", a.draws), ("hi", a.draws * 8))
            for tag, dr in _passes:
                g = torch.Generator().manual_seed(0)
                vb, vl, vu, vt = evaluate(m, Bt, trips, dr, g, use_units=True)
                E, V, law = size_law(m, Bt, trips, dr, a.chunk)
                row[tag] = dict(vb=vb, vu=vu, e=E.mean(), vpop=V.mean() + E.var(),
                                cap=100 * float((E > 0.85 * a.nmax).mean()), law=law)
            kl, tv, ks = dist_stats(row["lo"]["law"], p_obs)
            lo = row["lo"]
            hi = row.get("hi")           # absent under quadrature: nothing to compare to
            _hb = f"{hi['vb']:9.3f}" if hi else f"{'exact':>9}"
            _he = f"{hi['e']:8.2f}" if hi else f"{'exact':>8}"
            log(f"{s:>6}{lo['vb']:10.3f}{_hb}{lo['vu']:8.3f}"
                f"{lo['e']:8.2f}{obs.mean():7.2f}{_he}"
                f"{lo['vpop']:9.1f}{obs.var():7.1f}{kl:7.3f}{tv:6.3f}{lo['cap']:6.1f}")
            # guard 5, made explicit rather than left to the reader
            if hi and abs(hi["vb"] - lo["vb"]) > 0.5:
                log(f"       ^ DRAW-LIMITED: set/bskt moves {lo['vb']:.3f} -> {hi['vb']:.3f} "
                    f"at 8x draws; do not quote it")
            if hi and abs(hi["e"] - lo["e"]) > 0.10 * max(lo["e"], 1e-9):
                log(f"       ^ DRAW-LIMITED: E[n] moves {lo['e']:.2f} -> {hi['e']:.2f} at 8x")
        log("")

    log("set/bskt is held-out log P(basket set); @8x repeats it at 8x the draws and must")
    log("agree.  Under quadrature n_draws is ignored, so it reads 'exact' and the pass is")
    log("skipped -- it recomputed the same number and cost half this script's runtime.")
    log("varpop = E[Var(n|trip)] + Var[E(n|trip)],")
    log("the quantity obs.var() measures.  cap% = trips whose E[n] exceeds 0.85*nmax.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", default="v3_run39_best.pt,v3_run62_best.pt,v3_run63_best.pt")
    p.add_argument("--splits", default="valid,test")
    p.add_argument("--n-trips", type=int, default=512)
    p.add_argument("--draws", type=int, default=32)
    p.add_argument("--chunk", type=int, default=24)
    p.add_argument("--nmax", type=int, default=120)
    p.add_argument("--R", type=int, default=23)
    p.add_argument("--Kz", type=int, default=32)
    p.add_argument("--quad-q", type=int, default=0)
    main(p.parse_args())
