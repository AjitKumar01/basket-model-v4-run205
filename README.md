# Basket Model v4 — run205/run207 minimal branch

This branch contains only the theory, production code, estimator audits, tests, and frozen
launchers used by the fresh `run205_sparse400_q64_full` lineage and its exact
`run207_sparse400_q128_bounded_full` continuation.  Historical models, baselines, figures,
and bulk outputs are deliberately absent.

Start with:

- [`paper/RUN205_THEORY.md`](paper/RUN205_THEORY.md) for the model and estimator derivation;
- [`paper/RUN205_CODE_ARCHITECTURE.md`](paper/RUN205_CODE_ARCHITECTURE.md) for the exact
  theory-to-code map; and
- [`paper/version4.html`](paper/version4.html) for the original foundational specification.

## External inputs

The private/regenerable `basket_input/` directory and runtime `out/` directory are not
committed.  The required files and their roles are listed in the architecture document.
In the original workspace they already exist.

## Build and verify

```bash
python -m pip install -r requirements.txt

python scripts/v3/setup_poly_degree_native.py build_ext \
  --build-lib out/poly_degree_native_build/lib \
  --build-temp out/poly_degree_native_build/temp --force

python -m unittest scripts.v3.test_qmc scripts.v3.test_initialization
```

## Rebuild the interaction support

This is optional when `basket_input/v3_phimask_k400.npy` already exists.

```bash
env V3_AFFINITY=1 python scripts/v3/pairmask.py --k 400 --max-basket 40
```

The size-40 rule affects only the training-only support-ranking statistic.  No basket is
removed from likelihood training or evaluation.

## Train or continue

```bash
zsh scripts/v3/run205_sparse400_q64_full.zsh
```

Run205's best atomic checkpoint is iteration 600.  Its corrected, faster continuation uses
the bounded sparse-polynomial adjoint and Q128 training rule:

```bash
zsh scripts/v3/run207_sparse400_q128_bounded_full.zsh
```

Live log:

```bash
tail -f out/v3_run205_sparse400_q64_full.log
tail -f out/v3_run207_sparse400_q128_bounded_full.log
```

## Evaluate exact conditional recommendation

```bash
env V3_AFFINITY=1 python scripts/v3/eval_mrr_cutoffs.py \
  --ckpt "$PWD/out/v3_run205_sparse400_q64_full_best.pt" \
  --split validation --n-trips 384 --chunk 24 --nmax 120 --R 120 \
  --qmc-n 128 --cutoffs 5 10 20
```

The run guard rejects model-changing factored-size variants, incomplete support, the wrong
category partition, insufficient interaction rank, non-QMC normalizers, and non-fresh
lineage.  The launcher is the experiment configuration; do not silently substitute defaults.
