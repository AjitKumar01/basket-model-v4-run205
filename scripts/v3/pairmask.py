"""Choose the products that keep phi from CO-PURCHASE, not from their current norm.

run57 masked phi to the top 2% of products by norm and reached ||phi|| = 1.20 with the
normaliser converged -- and moved pair co-occurrence only 0.053 -> 0.065.  whichphi.py found
the reason: of the 200 most co-purchased real pairs, ZERO had both products retained, so
phi_j.phi_k was identically zero on every pair the evaluation scores.  The ceiling was 0.0%
before the first gradient step.

Top-k-by-norm fails for a structural reason.  A rare product has a strongly negative b_j, so
it barely enters log Z and growing its phi is nearly free; the rule therefore drifts toward
products where phi is UNPENALISED rather than where it is USEFUL.  Measured: the retained set
had median frequency rank 2138 of 5455, with 54% of it in the bottom half by frequency.

Which products actually participate in co-purchase is known from the data before any fitting,
so the mask can simply be built from it and checked in advance.  Products are ranked by the
co-purchase COUNT they carry -- the same quantity affinity.py partitions on, and the same one
the generation check scores -- and the top `budget` fraction is kept.

Counts, not lift.  Lift divides by marginal frequency and therefore promotes pairs of rare
products that co-occur twice; those cannot be estimated and are not what the evaluation asks
about.  affinity.py made this choice for the same reason and the partition it produced held
11.5% of co-purchase mass against the taxonomy's 4.1%.

The budget is bounded by what the normaliser can carry, measured in drawcurve.py at
||phi|| = 0.96: 1% of products costs 0.000 nats at 16 draws, 5% costs 1.888, 20% costs 23.9.
So 5% is the outer limit and needs more than 16 draws; 2-3% is the comfortable range.

Writes basket_input/v3_phimask_<pct>.npy, a bool array over products.

Run:  V3_AFFINITY=1 python3 pairmask.py --budget 0.03
"""
import argparse
import os
from collections import Counter

import numpy as np

from data import build

HERE = os.path.dirname(os.path.abspath(__file__))
BI = os.path.join(HERE, "..", "..", "basket_input")


def log(m):
    print(f"[pm] {m}", flush=True)


def pair_counts(D, max_basket):
    """Co-purchase count per unordered pair, over training trips only.

    Training trips only: the mask is a modelling choice fitted to data, so building it from
    validation or test baskets would leak the very co-occurrence the held-out numbers are
    meant to test.
    """
    ptr, li = D["line_ptr"], D["line_item"]
    split = D["trip_split"]
    cnt = Counter()
    n_used = 0
    for t in range(len(ptr) - 1):
        if split[t] != 0:
            continue
        s, e = int(ptr[t]), int(ptr[t + 1])
        items = np.unique(li[s:e])
        # A 40-line basket contributes 780 pairs and would dominate the counts while being
        # 0.1% of trips; the cap keeps the ranking about ordinary shopping.
        if len(items) < 2 or len(items) > max_basket:
            continue
        n_used += 1
        for x in range(len(items)):
            for y in range(x + 1, len(items)):
                cnt[(int(items[x]), int(items[y]))] += 1
    return cnt, n_used


def main(a):
    D = build()
    J = int(D["n_item"])
    cnt, n_used = pair_counts(D, a.max_basket)
    log(f"{n_used:,} training baskets, {len(cnt):,} distinct co-purchased pairs")

    # mass each product carries, summed over every pair it appears in
    mass = np.zeros(J, np.int64)
    for (x, y), c in cnt.items():
        mass[x] += c
        mass[y] += c
    # An explicit product count, because the budget-as-fraction knob cannot express the
    # scale that matters.  c = max_u sum_j max(phi_j'u,0) grows with the NUMBER of products
    # carrying phi, and log Z ~ c^2/2 must stay near sqrt(Kz) for the normaliser to be
    # computable at all.  Measured on run84's phi, rescaled to c = sqrt(Kz) = 5.66:
    #     K products     8      16      32      64     128     272
    #     phi'phi     3.474   1.844   0.678   0.213   0.063   0.018
    # against the 0.920 that a grocery complement lift of 2.5 needs.  The crossover is
    # around 24 products; at the 272 this file used to emit, the computable ceiling on
    # phi'phi is 2% of what the data asks for.
    k = int(a.k) if a.k else max(1, int(a.budget * J))
    keep = np.argsort(-mass)[:k]
    mask = np.zeros(J, bool)
    mask[keep] = True
    log(f"budget {a.budget:.1%} -> {k} products, carrying "
        f"{100.0*mass[keep].sum()/max(mass.sum(),1):.1f}% of all co-purchase mass")

    # ---- the check run57 needed and did not have: coverage BEFORE fitting --------------
    log("")
    log("pair coverage (phi_j.phi_k is identically zero unless BOTH are kept):")
    for n_top in (200, 1000, 5000):
        top = [p for p, _ in cnt.most_common(n_top)]
        both = sum(1 for (x, y) in top if mask[x] and mask[y])
        log(f"  top {n_top:5d} pairs: {both:5d} modellable ({100.0*both/len(top):5.1f}%)")
    tot = sum(cnt.values())
    kept_mass = sum(c for (x, y), c in cnt.items() if mask[x] and mask[y])
    log(f"  co-purchase MASS on modellable pairs: {100.0*kept_mass/max(tot,1):.1f}%")

    freq = np.bincount(D["line_item"], minlength=J)
    order = np.argsort(-freq)
    rank = np.empty(J, np.int64)
    rank[order] = np.arange(J)
    log("")
    log(f"frequency rank of the mask: median {int(np.median(rank[keep]))} of {J} "
        f"(run57's norm-based mask was 2138)")

    out = os.path.join(BI, f"v3_phimask_k{k}.npy" if a.k
                       else f"v3_phimask_{int(round(a.budget*100)):02d}.npy")
    np.save(out, mask)
    log(f"wrote {os.path.basename(out)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=float, default=0.03)
    p.add_argument("--max-basket", type=int, default=40)
    p.add_argument("--k", type=int, default=0, help="explicit product count; overrides --budget")
    main(p.parse_args())
