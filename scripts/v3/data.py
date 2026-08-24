"""
Data engineering for the version-3 fit on dunnhumby.

WHAT THE MODEL NEEDS THAT THE EXISTING PIPELINE DOES NOT PROVIDE.  Version 3's normaliser
sums over every subset of the STORE'S ASSORTMENT, so a trip's cost is set by how many
products that store carries and how they split across categories.  Two consequences shape
this file:

  * The assortment is ragged.  Categories at the median store hold 9 products; the largest
    holds 182.  Padding every category to the maximum would waste roughly 20x the work --
    that mistake cost an earlier branch 17.8 hours per fit against a projected 1.  So items
    are kept in one flat array with a row index, and only the CATEGORY axis is padded,
    because it is short (at most 188) and a scan over it is unavoidable anyway.

  * The observed basket must be expressed in assortment-local coordinates.  A purchased
    product is a position within its (store, category) block, not a global product id,
    because that is what the elementary symmetric polynomials index.

WHAT IS BUILT, once, and cached to basket_input/v3_index.npz:

    store_cat_ptr   [S, C+1]   for each store, where each category's block starts and ends
    store_items     [T]        product ids, grouped by (store, category)
    item_slot       [S, J]     position of a product inside its (store, category) block,
                               or -1 if the store does not carry it

and, per trip, the arrays a batch needs: household, store, day, week, and the purchased
products with their slots.

INTEGRITY.  Every purchased product must lie in its store's assortment -- if it does not,
the likelihood is evaluated on a support that excludes the observed basket and is silently
wrong.  `build` asserts this rather than trusting it, and reports the repair if the
assortment definition and the transaction log disagree.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
BI = os.path.join(ROOT, "basket_input")
CACHE = os.path.join(BI, "v3_index.npz")
CACHE_AFF = os.path.join(BI, "v3_index_affinity.npz")
_P = os.environ.get("V3_PARTITION", "")
CACHE_PART = os.path.join(BI, "v3_index_" + os.path.splitext(_P)[0] + ".npz") if _P else None


def log(m):
    print(f"[dat] {m}", flush=True)


def build(min_lines_per_store_cat=1, force=False):
    global CACHE
    if CACHE_PART is not None:
        CACHE = CACHE_PART          # a partition change rebuilds the whole ragged index
    elif os.environ.get("V3_AFFINITY", "0") == "1":
        CACHE = CACHE_AFF
    """Assortment index plus the trip table.  Cached; pass force=True to rebuild."""
    if os.path.exists(CACHE) and not force:
        z = np.load(CACHE, allow_pickle=True)
        log(f"loaded cache {CACHE}")
        return {k: z[k] for k in z.files}

    meta = json.load(open(os.path.join(BI, "meta.json")))
    J, S, C = meta["n_items"], meta["n_stores"], meta["n_commodities"]
    b = pd.read_parquet(os.path.join(BI, "baskets.parquet"))
    # The category partition is a modelling choice, not a fact about the data.  rho_c is
    # the model's EXACT dependence mechanism -- the convolution computes it with no draws
    # and no stability condition -- while phi_j'phi_k needs the Gaussian integral and
    # section 14's lambda_max < 1.  Real baskets need a 2.5x lift on common pairs; the phi
    # route needs lambda_max ~ 9 and the normaliser collapses there, while rho_c already
    # reaches 4.48x.  So the partition is rebuilt around what shoppers buy together instead
    # of how a merchandiser filed things, letting the exact term carry the complementarity.
    # V3_PARTITION names any parquet of (item_id, cat_id), so a partition can be swapped
    # without a code change.  It defines the ragged row structure, so it also sets C and the
    # cache file -- checkpoints across partitions are not comparable.
    _part = os.environ.get("V3_PARTITION", "")
    _aff = os.path.join(BI, "items_affinity.parquet")
    if _part and os.path.exists(os.path.join(BI, _part)):
        it = pd.read_parquet(os.path.join(BI, _part))[["item_id", "cat_id"]]
        C = int(it.cat_id.max()) + 1
        _sz = it.groupby("cat_id").size()
        log(f"category partition: {_part}, C = {C}, group size median "
            f"{int(_sz.median())} max {int(_sz.max())}")
    elif os.environ.get("V3_AFFINITY", "0") == "1" and os.path.exists(_aff):
        it = pd.read_parquet(_aff)[["item_id", "cat_id"]]
        C = int(it.cat_id.max()) + 1        # the partition sets C, not meta.json
        log(f"category partition: AFFINITY groups from {os.path.basename(_aff)}, C = {C}")
    else:
        it = pd.read_parquet(os.path.join(BI, "items.parquet"))[["item_id", "cat_id"]]
    b = b.merge(it, on="item_id", how="left")
    log(f"{len(b):,} purchase rows, {J:,} products, {S} stores, {C} categories")

    # ---- assortment ---------------------------------------------------------------------
    # The inherited definition -- "store s carries product j if s sold j in training" -- is
    # not merely selection on the outcome, it is MEASURABLY WRONG as a description of
    # availability: 17.4% of test purchase lines, touching 51.7% of test trips, are on
    # products the store never sold in training.  Those baskets would lie outside their own
    # support and have likelihood zero.  Repairing that by adding the held-out purchases
    # leaks test data into the support, so that is not an option either.
    #
    # The definition used here is training-only and strictly coarser:
    #
    #     j is available at store s  <=>  s sold something from category c(j) in training,
    #                                     and j was sold anywhere in the chain in training.
    #
    # Store-specific at the category level, chain-level within a category.  It expands the
    # support, which makes the likelihood harder rather than easier, so it is conservative
    # for any model evaluated on it.
    tr = b[b.split == "train"]
    chain_items = np.sort(tr.item_id.unique())
    sc = tr.groupby(["store_id", "cat_id"]).size().reset_index(name="n")
    log(f"products sold anywhere in training: {len(chain_items):,} of {J:,}")
    log(f"(store, category) blocks stocked in training: {len(sc):,} of {S * C:,}")
    ci = it[it.item_id.isin(set(chain_items.tolist()))]
    cat_items = {int(c): np.sort(g.item_id.unique()) for c, g in ci.groupby("cat_id")}
    rows_s, rows_i = [], []
    for st, c in zip(sc.store_id.to_numpy(), sc.cat_id.to_numpy()):
        v = cat_items.get(int(c))
        if v is None or len(v) == 0:
            continue
        rows_s.append(np.full(len(v), st, np.int64))
        rows_i.append(v)
    pair = pd.DataFrame({"store_id": np.concatenate(rows_s),
                         "item_id": np.concatenate(rows_i)})
    log(f"assortment pairs (store, product): {len(pair):,} = "
        f"{len(pair) / (S * J):.1%} of the grid")

    # ---- INTEGRITY: a check, not a repair -------------------------------------------------
    carried = set(zip(pair.store_id.to_numpy().tolist(), pair.item_id.to_numpy().tolist()))
    for sp in ("train", "validation", "test"):
        d = b[b.split == sp]
        bad = np.fromiter((( int(s_), int(i_)) not in carried
                           for s_, i_ in zip(d.store_id.to_numpy(), d.item_id.to_numpy())),
                          bool, len(d))
        nt = d.BASKET_ID.nunique()
        bt = d[bad].BASKET_ID.nunique() if bad.any() else 0
        log(f"  {sp:11s} lines outside the support: {int(bad.sum()):,} "
            f"({bad.mean():.2%}); trips touched {bt:,}/{nt:,}")
        if sp == "train" and bad.any():
            raise SystemExit("training basket outside its own support")
    drop = np.fromiter(((int(s_), int(i_)) not in carried
                        for s_, i_ in zip(b.store_id.to_numpy(), b.item_id.to_numpy())),
                       bool, len(b))
    if drop.any():
        log(f"  dropping {int(drop.sum()):,} held-out lines still outside the support; "
            f"reported, not hidden")
        b = b[~drop]

    pair = pair.merge(it, on="item_id", how="left").sort_values(
        ["store_id", "cat_id", "item_id"], kind="mergesort")

    # ---- flat, grouped by (store, category) --------------------------------------------
    store_items = pair.item_id.to_numpy(np.int32)
    key = pair.store_id.to_numpy(np.int64) * C + pair.cat_id.to_numpy(np.int64)
    counts = np.bincount(key, minlength=S * C)
    ptr = np.zeros(S * C + 1, np.int64)
    np.cumsum(counts, out=ptr[1:])
    item_slot = np.full((S, J), -1, np.int32)
    starts = ptr[:-1].reshape(S, C)
    for r in range(len(pair)):
        k = key[r]
        item_slot[k // C, store_items[r]] = r - ptr[k]
    log(f"assortment: median {int(np.median(counts[counts > 0]))} products per "
        f"(store, category); max {counts.max()}; "
        f"median {int(np.median(np.add.reduceat(counts, np.arange(0, S * C, C))))} "
        f"products per store")

    # ---- trips --------------------------------------------------------------------------
    trips = (b.groupby("BASKET_ID")
               .agg(user=("user_id", "first"), store=("store_id", "first"),
                    day=("DAY", "first"), week=("WEEK_NO", "first"),
                    split=("split", "first"), n=("item_id", "size"))
               .reset_index().sort_values("BASKET_ID", kind="mergesort"))
    tid = {v: i for i, v in enumerate(trips.BASKET_ID.to_numpy())}
    b = b.assign(trip=b.BASKET_ID.map(tid))
    b = b.sort_values(["trip", "cat_id", "item_id"], kind="mergesort")
    line_trip = b.trip.to_numpy(np.int32)
    line_item = b.item_id.to_numpy(np.int32)
    line_cat = b.cat_id.to_numpy(np.int32)
    line_units = b.units.to_numpy(np.int16)
    line_slot = item_slot[b.store_id.to_numpy(np.int64), line_item]
    assert (line_slot >= 0).all(), "a purchased product has no slot in its store block"
    tptr = np.zeros(len(trips) + 1, np.int64)
    np.cumsum(np.bincount(line_trip, minlength=len(trips)), out=tptr[1:])

    split_code = trips.split.map({"train": 0, "validation": 1, "test": 2}).to_numpy(np.int8)
    log(f"trips: {len(trips):,}  ({(split_code==0).sum():,} train / "
        f"{(split_code==1).sum():,} val / {(split_code==2).sum():,} test)")
    log(f"basket size: mean {trips.n.mean():.3f}  var {trips.n.var():.3f}  "
        f"dispersion {trips.n.var()/trips.n.mean():.3f}  max {trips.n.max()}")

    out = dict(store_cat_ptr=ptr.reshape(-1), store_items=store_items,
               item_slot=item_slot, n_store=np.int64(S), n_cat=np.int64(C),
               n_item=np.int64(J), n_user=np.int64(meta["n_users"]),
               trip_user=trips.user.to_numpy(np.int32),
               trip_store=trips.store.to_numpy(np.int32),
               trip_day=trips.day.to_numpy(np.int32),
               trip_week=trips.week.to_numpy(np.int32),
               trip_split=split_code, trip_nlines=trips.n.to_numpy(np.int32),
               line_ptr=tptr, line_item=line_item, line_cat=line_cat,
               line_slot=line_slot.astype(np.int32), line_units=line_units)
    np.savez_compressed(CACHE, **out)
    log(f"wrote {CACHE}")
    return out


def batch_index(D, trips, nmax, R):
    """Ragged index for one batch of trips.

    Returns the arrays the model needs, with items flat and only the category axis padded:

        item_id   [T]      product id of every assortment slot in the batch
        row_of    [T]      which (trip, category) row each slot belongs to
        row_trip  [n_rows] which trip each row belongs to
        row_cat   [n_rows] which category
        row_k     [n_rows] how many of that category the observed basket holds
        sel       [T]      1 where the slot is in the observed basket
    """
    S_ptr = D["store_cat_ptr"]
    C = int(D["n_cat"])
    items, row_of, row_trip, row_cat = [], [], [], []
    nrow = 0
    for bi, t in enumerate(trips):
        s = int(D["trip_store"][t])
        base = s * C
        for c in range(C):
            lo, hi = int(S_ptr[base + c]), int(S_ptr[base + c + 1])
            if hi <= lo:
                continue
            items.append(D["store_items"][lo:hi])
            row_of.append(np.full(hi - lo, nrow, np.int64))
            row_trip.append(bi)
            row_cat.append(c)
            nrow += 1
    item_id = np.concatenate(items)
    row_of = np.concatenate(row_of)
    return dict(item_id=item_id, row_of=row_of,
                row_trip=np.array(row_trip, np.int64),
                row_cat=np.array(row_cat, np.int64), n_rows=nrow)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    D = build(force=a.force)
    log("")
    log("sanity: assortment size per trip, and rows per trip")
    ptr = D["store_cat_ptr"].reshape(int(D["n_store"]), int(D["n_cat"]) + 1) \
        if D["store_cat_ptr"].size == int(D["n_store"]) * (int(D["n_cat"]) + 1) else None
    S, C = int(D["n_store"]), int(D["n_cat"])
    full = D["store_cat_ptr"]
    per_store = np.array([full[(s + 1) * C] - full[s * C] for s in range(S)])
    rows_per_store = np.array([
        int((np.diff(full[s * C:(s + 1) * C + 1]) > 0).sum()) for s in range(S)])
    log(f"  products per store: median {int(np.median(per_store))}  "
        f"min {per_store.min()}  max {per_store.max()}")
    log(f"  non-empty categories per store: median {int(np.median(rows_per_store))}  "
        f"max {rows_per_store.max()}")
