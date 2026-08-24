"""
Conditioning features for the dunnhumby fit, gathered at assortment slots.

The model scores every product a store carries, not just the purchased ones, so every
feature has to be available at ~5,420 slots per trip rather than at ~8 purchase lines.  That
is the whole difficulty: these are lookups into (product x day), (product x store x week)
and (household x product) panels, and they have to be gathered, not joined.

WHAT IS WIRED HERE, and what each costs to evaluate:

    price        Delta log p_jst = Delta log p_jt + Delta^s_jsw
                 dense (5455 x 712) plus a sparse store deviation, 244,880 non-zero cells
    promotion    display and mailer indicators, sparse over (product, store, week)
    seasonality  mu_j' delta_w
    store        zeta_j' xi_s

WHAT IS NOT WIRED YET, and why it is a separate job rather than an oversight:

    recency      needs the strictly-before lookup in state.npz, which is a searchsorted into
                 a 1.2M-element key array per (household, sub-commodity).  Version 2 pays
                 this per purchase line; at assortment scale it is per slot, ~700x more, and
                 wants a different data structure.
    coupon       the eligibility panel of section 21 is built from campaign_table x coupon x
                 window and does not exist as an array yet.
    days-of-supply  needs the recency machinery above.

The first fit is therefore on price, promotion, seasonality and store.  That is stated
plainly rather than left to be inferred from the parameter list: any elasticity it produces
is missing the persistence terms that version 2 measures as its largest single ablation, and
is not comparable to one that has them.
"""
import os

import math

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BI = os.path.join(HERE, "..", "..", "basket_input")


def log(m):
    print(f"[fea] {m}", flush=True)


class Features:
    """Panels held once, gathered per batch."""

    def __init__(self, n_item, n_store, n_day, device=None):
        self.J, self.S, self.D = n_item, n_store, n_day
        self.dev = torch.from_numpy(
            np.load(os.path.join(BI, "log_price_dev.npy")).astype(np.float32))
        log(f"price deviation panel {tuple(self.dev.shape)}, "
            f"|mean| {float(self.dev.abs().mean()):.4f}")

        # Key conventions are NOT guessed.  The promotion panel is built by
        # pipeline/23_promo_data.py as (item * n_stores + store) * 128 + WEEK_NO with the
        # RAW week, and store_price.npz stores item/store/week directly over weeks 1..102.
        # An earlier draft of this file guessed a different multiplier and a modulo; the
        # lookup would have silently returned zeros for almost every cell.
        sp = np.load(os.path.join(BI, "store_price.npz"))
        key = ((sp["item"].astype(np.int64) * self.S + sp["store"].astype(np.int64)) * 128
               + sp["week"].astype(np.int64))
        o = np.argsort(key)
        self.sp_key = torch.from_numpy(key[o])
        self.sp_val = torch.from_numpy(sp["dev"][o].astype(np.float32))
        log(f"store price deviations: {len(key):,} cells "
            f"({len(key) / (n_item * n_store * 52):.2%} of the grid)")

        st = np.load(os.path.join(BI, "state.npz"))
        self.st_keys = torch.from_numpy(st["keys"])
        self.item_sub = torch.from_numpy(st["item_sub"].astype(np.int64))
        self.sub_gap = torch.from_numpy(st["sub_gap"])
        self.n_sub = int(self.item_sub.max()) + 1
        log(f"recency: {len(self.st_keys):,} purchase events over {self.n_sub} "
            f"sub-commodities; median gap {float(self.sub_gap.median()):.1f} days")

        pr = np.load(os.path.join(BI, "promo.npz"))
        o = np.argsort(pr["keys"])
        self.pk = torch.from_numpy(pr["keys"][o])
        self.pd_ = torch.from_numpy(pr["disp"][o].astype(np.float32))
        self.pm = torch.from_numpy(pr["mail"][o].astype(np.float32))
        log(f"promotion cells: {len(self.pk):,}; "
            f"display rate {float(self.pd_.mean()):.3f}, mailer {float(self.pm.mean()):.3f}")

    def recency(self, item, user, day):
        """The four recency functions of the specification, at every assortment slot.

        Purchase events are stored as sorted keys (user * n_sub + sub) * 1024 + DAY, so a
        strictly-before lookup is one searchsorted.  The previous key belongs to the same
        (household, sub-commodity) group only if its group field matches, which is what
        separates "no earlier purchase" from "the previous event is someone else's".

        This is the block version 2 measures as its largest single ablation, and it was the
        largest thing missing here.
        """
        sub = self.item_sub[item]
        group = user.to(torch.int64) * self.n_sub + sub
        key = group * 1024 + day
        idx = torch.searchsorted(self.st_keys, key)
        prev = (idx - 1).clamp(0, len(self.st_keys) - 1)
        pk = self.st_keys[prev]
        same = (idx > 0) & (torch.div(pk, 1024, rounding_mode="floor") == group)
        since = torch.where(same, (day - pk % 1024).double(),
                            torch.zeros(1, dtype=torch.float64))
        gap = self.sub_gap[sub].double().clamp_min(1.0)
        z = torch.zeros(1, dtype=torch.float64)
        return torch.stack([
            (~same).double(),
            torch.where(same, torch.exp(-since / 7.0), z),
            torch.where(same, torch.exp(-since / gap), z),
            torch.where(same, torch.log1p(since) / math.log(100.0), z)], dim=-1)

    @staticmethod
    def _lookup(keys, vals, q):
        """Sparse gather: value where the key exists, zero where it does not."""
        i = torch.searchsorted(keys, q)
        i = i.clamp(max=len(keys) - 1)
        hit = keys[i] == q
        return torch.where(hit, vals[i], torch.zeros((), dtype=vals.dtype))

    def gather(self, item, store, day, week):
        """Features at every assortment slot.  item/store/day/week are [T] longs."""
        q = (item * self.S + store) * 128 + week          # one convention for both panels
        dlp = self.dev[item, day] + self._lookup(self.sp_key, self.sp_val, q)
        disp = self._lookup(self.pk, self.pd_, q)
        mail = self._lookup(self.pk, self.pm, q)
        return dlp, disp, mail


def selftest(F, n_item, n_store):
    """Check the lookups hit, rather than trusting that they do.  A sparse gather that
    misses returns zero, which is indistinguishable from a real zero unless the hit rate is
    measured -- so it is measured."""
    pr = np.load(os.path.join(BI, "promo.npz"))
    k = torch.from_numpy(pr["keys"][:20000])
    item = torch.div(torch.div(k, 128, rounding_mode="floor"), n_store,
                     rounding_mode="floor")
    store = torch.div(k, 128, rounding_mode="floor") % n_store
    week = k % 128
    d, m = F._lookup(F.pk, F.pd_, k), F._lookup(F.pk, F.pm, k)
    exact = (d + m > 0).double().mean()
    log(f"promotion lookup on known-promoted cells: hit rate {float(exact):.4f} "
        f"(must be 1.0000)")
    day = (week.clamp(1, 102) - 1) * 7
    dlp, dd, mm = F.gather(item.long(), store.long(), day.long(), week.long())
    log(f"gather at those cells: display {float((dd > 0).double().mean()):.4f}, "
        f"mailer {float((mm > 0).double().mean()):.4f}, "
        f"|Delta log p| mean {float(dlp.abs().mean()):.4f}")
    return float(exact)
