# run205/run207 code architecture and its correspondence with theory

## 1. Purpose

This document explains the smallest code path needed to reproduce
the fresh `run205_sparse400_q64_full` lineage and its exact
`run207_sparse400_q128_bounded_full` continuation.  It is organized in execution order and explicitly matches
each component to the equations in `RUN205_THEORY.md`.

The production path is intentionally small:

```text
run205_sparse400_q64_full.zsh
        |
        v
      fit.py -------------------- initialization, optimizer, guards, checkpoints
        |
        +---- data.py ----------- trip/store assortment index
        +---- features.py ------- price, promotion, week, store, recency panels
        +---- ragged.py --------- model, exact subset algebra, RQMC, sampler
                 |
                 +---- poly_degree_native.py
                            |
                            +---- poly_degree_native.cpp
```

Evaluation calls the same model and normalizer through `eval_mrr_cutoffs.py`.  Audit scripts
change node counts or compare gradients, but do not implement another statistical model.

## 2. Files retained in the minimal branch

### Foundational and explanatory files

| File | Reason it is retained |
|---|---|
| `paper/version4.html` | Original version-4 model statement. Some old run97 computational details are superseded, but the joint law is authoritative. |
| `paper/RUN205_THEORY.md` | Current end-to-end derivation using the accepted estimator. |
| `paper/RUN205_CODE_ARCHITECTURE.md` | This theory-to-code map. |
| `README.md` | Minimal build, data, train, evaluate, and log instructions. |

### Production training files

| File | Role |
|---|---|
| `scripts/v3/run205_sparse400_q64_full.zsh` | Frozen command line for the full fresh run. |
| `scripts/v3/fit.py` | Builds batches, initializes parameters, trains, validates, guards, and saves. |
| `scripts/v3/ragged.py` | Defines `RaggedModel`, exact polynomial subset sum, RQMC normalizer, incidence, and generation. |
| `scripts/v3/data.py` | Loads/builds temporal splits and store-specific ragged assortments. |
| `scripts/v3/features.py` | Loads and gathers per-assortment price/promotion/context features. |
| `scripts/v3/pairmask.py` | Builds the training-only 400-row interaction support. |
| `scripts/v3/poly_degree_native.py` | PyTorch autograd wrappers for native exact polynomial kernels. |
| `scripts/v3/poly_degree_native.cpp` | CPU forward/reverse kernels used by the run. |
| `scripts/v3/setup_poly_degree_native.py` | Reproducible native extension build. |

### Evaluation and estimator-audit files

| File | Role |
|---|---|
| `scripts/v3/eval_mrr_cutoffs.py` | Correct exact-conditional MRR, MRR@K, recall@K, and popularity comparison. |
| `scripts/v3/evalall.py` | Checkpoint loading/provenance compatibility used by the evaluator. |
| `scripts/v3/diagnose_bucket_coverage.py` | Legacy-kernel comparison imported by the evaluator; inactive unless requested. |
| `scripts/v3/audit_original_qmc.py` | Frozen log-Z value/latency comparison across node counts. |
| `scripts/v3/audit_phi_pseudo_alignment.py` | Q64 versus independent Q512 interaction-gradient audit. |
| `scripts/v3/audit_native_logz.py` | Kernel-level latency and correctness profiler. |
| `scripts/v3/qmc_multimode_probe.py` | Independent proposal/mode diagnostics used by audits and tests. |
| `scripts/v3/test_qmc.py` | Estimator, conditional law, and polynomial regression tests. |
| `scripts/v3/test_initialization.py` | Initialization and objective-decomposition tests. |

No historical baselines, superseded factored-size experiments, old launchers, figures, or
bulk output files are needed by this run and are not retained in the minimal branch.

## 3. External data contract

The Dunnhumby-derived inputs and checkpoints are intentionally not committed.  They are
large, regenerable, and may be subject to data-distribution restrictions.  The launcher
expects this runtime layout:

```text
basket_input/
  v3_index_affinity.npz   # complete cached trip/assortment index, 280 groups
  log_price_dev.npy       # product x day price deviation
  store_price.npz         # sparse product/store/week price residuals
  promo.npz               # sparse display and mailer features
  state.npz               # recency history; loaded but psi is frozen at zero
  v3_beta_target.npz      # training-only weak price-pattern calibration
  v3_phimask_k400.npy     # Boolean interaction-row support

out/
  poly_degree_native_build/lib/v3_poly_degree_native*.so
  v3_run205_sparse400_q64_full.log
  v3_run205_sparse400_q64_full.pt
  v3_run205_sparse400_q64_full_best.pt
```

If `v3_index_affinity.npz` is absent, `data.py` can rebuild it from the lower-level parquet
inputs in `basket_input`, but those raw derived files are also external to the minimal
branch.

## 4. Entry point and frozen configuration

`run205_sparse400_q64_full.zsh` changes to the repository root, sets
`V3_AFFINITY=1`, invokes `fit.py`, and mirrors stdout/stderr into the run log.

The configuration that defines this run is:

```text
catalogue                 5,455 products
affinity/category rows    280
household/product rank    K=32
price rank                Kp=8
interaction rank          Kz=32
size support              1..120
category-count support    0..120, original choose(r,2)
interaction rows          fixed training-only k=400 support
batch                     24 trips
training RQMC             Q64, four scrambles, antithetic
two-mode RQMC             Q128 total
hard-trip retry           Q128 -> Q256 -> ... -> Q4096
checkpoint RQMC           Q256 with independent Q512 check
learning rate             .002; halve at 20,000 and 26,000
logs                      every 10 updates
recovery save/safety      every 100 updates
full validation           every 200 updates
```

`--require-version4 1` is an executable invariant check.  It rejects a run unless it is
fresh, uses the affinity-280 partition, has all 5,455 products, rank at least 32, support
covering every observed basket, a QMC normalizer, and the original non-factored joint law.
It permits only a product-row Gram support, not the old norm-driven moving top-k rule.

## 5. Data path: `data.py`

### 5.1 `build()`

With `V3_AFFINITY=1`, `build()` loads `v3_index_affinity.npz`.  Important arrays include:

- trip household, store, day, week, and split;
- a pointer from each trip to its observed product lines and quantities;
- store/category pointers and product lists defining the assortment;
- product-to-category assignments for the affinity-280 partition; and
- cached support/integrity metadata.

The normalizer must sum over the products that were actually available in that store.  It
cannot use a global catalogue and then score an observed basket on a different support.
`data.py` checks that every purchased item belongs to its trip's store assortment.

### 5.2 Temporal splits

The split is already encoded in the cache:

```text
0 = training
1 = validation
2 = test
```

Only training trips are used for popularity initialization, taste moments, interaction
support, IPF, and optimizer batches.  Validation is sampled with a fixed permutation rather
than taking a chronological prefix from one week.

## 6. Feature path: `features.py`

`Features` loads large feature panels once.  `Features.gather()` then maps them to the flat
assortment slots constructed for a minibatch.

| Theory term | Runtime source | Output used by `b_at` |
|---|---|---|
| price level/deviation | `log_price_dev.npy`, `store_price.npz` | `dlp`, `dlp_bar` |
| display | `promo.npz` | `disp` |
| mailer | `promo.npz` | `mail` |
| week | trip week | `week` |
| store | trip store | `store` |
| recency | `state.npz` | `rec` |

Run205 passes `--no-rec 1`, so `psi` is zero and frozen even though the feature is available.

## 7. Batch construction: `fit.Batcher`

`Batcher.make(trips)` walks each trip's store assortment and produces:

- `RaggedIndex`: every available product slot, its trip, category row, and position;
- slot-level context for the normalizer;
- purchased-line context for the observed energy;
- household IDs; and
- observed product, trip, category, and unit tensors.

The line and slot contexts deliberately come from the same feature functions.  This enforces
the theoretical requirement that `E(S)` and `Z` assign the same `b_j(x)` to a purchased
product.  Earlier separate paths violated this and gave the optimizer artificial reward.

## 8. Model parameters: `ragged.RaggedModel`

The constructor creates the parameter blocks appearing in the theory:

| Code parameter | Theory symbol | Shape in run205 |
|---|---|---:|
| `lam` | `lambda_j` | 5,455 |
| `theta`, `alpha` | household/product taste | 2,066x32, 5,455x32 |
| `gamma`, `beta` | price sensitivity factors | 2,066x8, 5,455x8 |
| `w_dsp`, `w_mlr` | promotion coefficients | 5,455 each |
| `mu`, `delta` | product/week factors | 5,455x8, 53x8 |
| `zeta`, `xi` | product/store factors | 5,455x4, 115x4 |
| `psi` | recency loading | 5,455x4, frozen zero |
| `phi` | Gram interaction rows | 5,455x32, 400 nonzero rows |
| `rho_c` | group-count interaction | 280 |
| `rho_0_free` | non-empty size potential | 120 |
| `a_q`, `gamma_q`, `beta_q`, `log_r` | conditional units model | quantity block |

The optional `factored_size_enabled` buffer remains false.  It exists only for rejected
historical ablations and is prohibited by the run guard.

## 9. One utility function: `RaggedModel.b_at`

`b_at(item, trip, context)` is the sole implementation of the additive item value.  Both
`b_flat()` for all assortment slots and `energy()` for observed lines call it.

The theory-to-code correspondence is literal:

```text
lambda                         self.lam[item]
household taste               self.theta_c()[house] * self.alpha[item]
price                         -softplus(gamma)*softplus(beta)*split_price
promotion                     w_dsp*disp + w_mlr*mail
week                          mu*delta_c[week]
store                         zeta*xi_c[store]
recency                       psi*rec, disabled in run205
```

`theta_c`, `delta_c`, and `xi_c` subtract detached means in the forward pass.  After Adam,
`project_context_gauges()` centers the raw parameters.  This removes unidentifiable constant
directions without changing any item utility.

## 10. Observed energy: `RaggedModel.energy`

`energy()` receives only purchased lines.  It computes:

1. `sum b_j` by `b_at`;
2. the Gram pair energy using the identity based on the sum of basket `phi` vectors;
3. `-rho_c choose(n_c,2)` from observed category counts; and
4. `-rho_0(n)` from observed total size.

This is the numerator of the version-4 probability.  It never approximates `log Z` and
never drops an observed trip within declared support.

## 11. Exact polynomial core: `ragged.py`

### 11.1 `RaggedIndex`

Stores flat assortment slots instead of padding each category to the largest category.  It
also stores row sizes, positions, trip/category row mappings, and a short padded category
axis used by the product tree.

### 11.2 ESP functions

`esp_bucketed()` groups category rows by their actual length.  The production native path
uses a balanced, subtraction-free ESP tree and its exact custom reverse pass.  Coefficients
above the number of products in a row are structurally zero.

### 11.3 Category polynomial product

Each row's ESP coefficients are tilted in log coordinates by
`-rho_c*choose(r,2)`.  `log_poly_tree_degree_native()` multiplies the category polynomials,
truncating every intermediate result at degree 120 and using each row's achievable degree.
The log-coordinate forward and bounded probability adjoint avoid overflow and `0*inf`
gradients when an attractive category has a tiny high-degree coefficient.

### 11.4 Native boundary

`poly_degree_native.py` contains custom `torch.autograd.Function` wrappers.
`poly_degree_native.cpp` implements the CPU forward and reverse kernels.  The wrapper adds
the build output directory to `sys.path`; no installed package is assumed.

Build command:

```bash
python scripts/v3/setup_poly_degree_native.py build_ext \
  --build-lib out/poly_degree_native_build/lib \
  --build-temp out/poly_degree_native_build/temp --force
```

## 12. Sparse exact preparation: `sparse_prepare`

The binary support is represented by exactly zero rows of `model.phi`.
`sparse_prepare()` separates assortment slots into:

- active slots whose `phi_j` is nonzero and therefore depend on `z`; and
- inactive slots whose weight is independent of `z`.

It computes the inactive ESP and untouched-category product once.  For each RQMC node,
`log_f_sparse()` computes only active projections/ESPs and combines them with the cached
constant polynomial.  This is an algebraic factorization of the same generating polynomial,
not an approximation.

The fixed mask is applied before `rho_0` initialization and all six IPF passes.  This avoids
calibrating a dense model and then silently changing its interaction weights at step one.

## 13. RQMC construction: `set_quad` and Sobol helpers

`set_quad()` is the only function allowed to select the integrator.  For run205 it installs:

- `sobol_grid(32,64,replicates=4,antithetic=True)`;
- a 128-node two-component mixture grid;
- three vectorized size-mode iterations;
- a four-nat second-mode threshold and distance threshold one; and
- node chunks of 32 for memory-bounded autograd.

`sobol_grid()` uses a different Owen scramble seed for every replicate and maps uniform
Sobol coordinates through the Gaussian inverse CDF.  `sobol_mixture_grid()` preserves
replicate and mixture-component axes so the dispersion between complete mixture estimates
is a valid error diagnostic.

## 14. Proposal and normalizer path

The call chain is:

```text
RaggedModel.log_Z
  -> _log_Z_adaptive
      -> _log_Z_size_multimode
          -> sparse_prepare
          -> _size_multimode_proposal
              -> _size_multimode_centres
          -> log_f_sparse in chunks
```

`_size_multimode_centres()` runs fixed-point equations for the full size sum and coarse
size bands.  It discards impossible-to-recover tail modes using an explicit operator bound,
not an arbitrary basket-size cutoff.  It retains a second center only if it is separated and
has competitive Gaussian-adjusted mass.

`_size_multimode_proposal()` transforms one- or two-mode Sobol blocks and computes the exact
`log p(z)-log q(z)` correction.  `_log_Z_size_multimode()` combines this with `log f(z)` by
stable log-sum-exp.  If `return_size=True`, it preserves every size coefficient and returns
`P(n|x)` from the same node contributions.

## 15. Per-trip adaptive retries in `fit.py`

After the first minibatch normalizer call, `fit.py` examines `model._last_qmc_logz_se`.
Only indices above `0.015` nat are rebuilt as a smaller `RaggedIndex` and rescored at the
next node level.  `index_copy` replaces those trip graphs while leaving accepted trip graphs
untouched.

Counters in the log mean:

| Counter | Meaning |
|---|---|
| `skip` | optimizer update rejected entirely |
| `qretry` | individual trip-stage recomputations, not skipped steps |
| `qbad` | update rejected by QMC error/size guard |
| `gbad` | update rejected for a non-finite gradient |
| `rhold` | `rho_c` held because a trip consumed maximum estimator capacity |
| `drop` | trip excluded by the ESS floor from an otherwise retained update |
| `redo` | legacy higher-draw trip redo count |

## 16. Training loop in `fit.main`

### 16.1 Startup

1. Enable float64 and flush CPU denormals.
2. Load affinity-280 data and validate full support.
3. Build `RaggedModel` and native kernels.
4. Initialize `lambda`, household taste, random interactions, and `rho_0` using training
   data only.
5. Apply the 400-row support.
6. Run six size-IPF passes through the production normalizer.
7. Evaluate iteration zero and run a ten-update timing probe.

### 16.2 One optimizer update

1. Draw 24 training trips.
2. Build ragged assortment and observed-line views.
3. Evaluate `loglik() = energy() - log_Z(drop_empty=True)` and the size law.
4. Refine only high-SE trips.
5. Assemble joint data likelihood and declared calibration terms.
6. Reject the update if estimator checks fail or any gradient is non-finite.
7. Clip the total gradient norm and apply Adam.
8. Apply parameter-block step scales and all projections.
9. Reapply the fixed interaction mask.
10. Update telemetry and the staged learning-rate scheduler.

### 16.3 Checkpoint work

- Every 10 updates: minibatch telemetry only.
- Every 100: atomic recovery checkpoint plus independent QMC safety check.
- Every 200: fixed validation sample, size/sampler/elasticity checks, and best-likelihood
  checkpoint selection.

The normalizer safety guard compares independently seeded Q256 and Q512 rules.  Three
consecutive discrepancies above `0.02` nat are required for a fatal stop.  Replicate SE is
still printed but is not misinterpreted as observed bias.

## 17. Checkpoint format

`save_ckpt()` writes through a temporary file and atomically renames it.  A format-2
checkpoint contains:

- complete model state;
- optimizer and scheduler states;
- current and cumulative iteration;
- NumPy and Torch generator states;
- best likelihood and best iteration;
- current normalizer-strike state;
- model dimensions and statistical feature flags;
- data partition, catalogue, support, and category metadata; and
- quadrature rank, nodes, scrambles, proposal, and seed metadata.

This makes `--resume` a real continuation.  A warm start that loads only weights is a
different operation and is prohibited for run205's fresh lineage.

## 18. Recommendation path

`eval_mrr_cutoffs.py` loads the checkpoint into the same `RaggedModel`, replaces only the
evaluation node count if explicitly requested, and calls `fit.rec_eval()`.

`rec_eval()`:

1. chooses one hidden item with a fixed seed;
2. removes the revealed remainder from the candidate assortment;
3. shifts candidate utilities by their interaction with the revealed rows;
4. passes revealed category counts and size into the normalizer;
5. differentiates the exact conditional `log Z` with respect to candidate utilities; and
6. ranks the hidden item by conditional incidence.

The evaluator reports full MRR, truncated MRR@5/10/20, recall@5/10/20, median rank, normal
standard error, and an exposure-corrected popularity score on identical cases.

The minimal branch retains `evalall.py` only for its checkpoint loader and
`diagnose_bucket_coverage.py` only for an optional legacy-kernel audit imported by the
evaluator.  Neither changes the default score.

## 19. Generation path

`RaggedModel.sample()` reuses the same `sparse_prepare` cache and RQMC proposal as likelihood.
It importance-resamples one finite-rule latent node, then walks the log-coordinate size,
category, and item polynomials backward.  It returns actual product IDs; `sample_slots()`
maps them back to assortment slots for objectives that need the sampled context.

The sampler check at validation compares sampled basket size with analytic `E[n]` from
`return_size=True`.  Agreement is a cross-path test: it can catch a broken reverse sampler
even when the forward normalizer looks finite.

## 20. Verification files and accepted gates

`test_qmc.py` and `test_initialization.py` currently contain 33 passing tests.  They cover
Sobol construction, conditional laws, native/eager polynomial agreement, gradients,
support behavior, gauges, objective decomposition, and initialization invariants.

The empirical estimator audits retained in the minimal branch produced:

```text
Q64 vs independent Q512, 48 held-out trips
  mean absolute log-Z gap   0.000963 nat
  worst absolute gap        0.019737 nat

Q64 Phi gradient vs mean of two independent Q512 scores, 24 trips
  cosine                    0.991497
  scaled residual           0.130128
  likelihood gap/trip       0.000390 nat
```

The predeclared gradient gates were cosine at least 0.99, residual at most 0.15, and value
gap at most 0.02 nat; all passed.

## 21. Reproduction sequence

From the repository root with external inputs present:

```bash
# 1. Optionally rebuild the support from training data only.
env V3_AFFINITY=1 python scripts/v3/pairmask.py --k 400 --max-basket 40

# 2. Build the exact native polynomial kernels.
python scripts/v3/setup_poly_degree_native.py build_ext \
  --build-lib out/poly_degree_native_build/lib \
  --build-temp out/poly_degree_native_build/temp --force

# 3. Run unit/regression tests.
python -m unittest scripts.v3.test_qmc scripts.v3.test_initialization

# 4. Launch the fresh full experiment.
zsh scripts/v3/run205_sparse400_q64_full.zsh

# 5. Inspect live telemetry.
tail -f out/v3_run205_sparse400_q64_full.log

# 6. Evaluate the best checkpoint on exact conditional incidence.
env V3_AFFINITY=1 python scripts/v3/eval_mrr_cutoffs.py \
  --ckpt "$PWD/out/v3_run205_sparse400_q64_full_best.pt" \
  --split validation --n-trips 384 --chunk 24 --nmax 120 --R 120 \
  --qmc-n 128 --cutoffs 5 10 20
```

## 22. Where theory would break if code drifted

The following are hard red lines:

| Code drift | Theoretical consequence |
|---|---|
| replace `rho_0` with an empirical `P(n)` factor | changes joint law and removes basket-size price elasticity |
| truncate catalogue to interaction support | removes products from `Z` and changes all probabilities |
| use an arbitrary pairwise mask | can destroy PSD and invalidate Gaussian identity |
| saturate `choose(n_c,2)` below 120 | changes the declared category interaction |
| compute observed `b` differently from slot `b` | numerator and denominator cease to describe one probability |
| subtract one after exponentiating `log Z` | numerically unstable empty-basket conditioning |
| score recommendations by raw `b` | does not compute version-4 incidence |
| use external size law in rollout | sampler no longer draws from training likelihood |
| use replicate confidence radius as fatal bias | repeats run202's false abort |

The architecture is arranged so each shared mathematical quantity has one implementation:
`b_at` for utility, `set_quad` for integration choice, `log_f_sparse` for polynomial mass,
`log_Z` for normalization, and the same reverse polynomial for generation.  That one-path
discipline is the main defense against another estimator/model mismatch.

## 23. Corrected continuation and operational logs

`run207_sparse400_q128_bounded_full.zsh` resumes the atomic iteration-600 run205 checkpoint.
`fit.py` now explicitly skips size IPF on `--resume`; otherwise rho_0 would be changed after
restoring its Adam moments, so the job would not be an exact continuation.  Model weights,
optimizer, scheduler, NumPy RNG, Torch RNG, catalogue, rank, support, interaction mask, and
all objective weights are preserved.

The continuation changes only estimator execution:

| Setting | Fresh run205 | Corrected run207 |
|---|---:|---:|
| training nodes | Q64 | Q128 |
| hard-trip refinements | Q128--Q4096 | Q256--Q4096 |
| checkpoint rule | Q256 | Q512 |
| proposal solve | size-band mode solve | zero-centred, zero backward steps |
| sparse polynomial adjoint | raw linear coefficients | bounded log/probability coordinates |

The training log intentionally omits MRR when `--n-rec 0`; `MRR nan` was not information.
Recommendation remains a separate held-out task run by `eval_mrr_cutoffs.py`, where full
MRR, MRR@5/10/20, recall, eligibility counts, and the identical-trip popularity comparison
are all reported together.

Launch and inspect the corrected continuation with:

```bash
zsh scripts/v3/run207_sparse400_q128_bounded_full.zsh
tail -f out/v3_run207_sparse400_q128_bounded_full.log
```

The frozen estimator evidence is written to
`out/v3_run205_iter600_qmc_latency_audit.json`,
`out/v3_run205_iter600_zeromode_qmc_audit.json`, and
`out/v3_run205_iter600_phi_rqmc_efficiency.json`.  Generated audit outputs and checkpoints
are not committed to GitHub.
