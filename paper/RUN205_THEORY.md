# Basket Model v4: end-to-end theory for run205

## 1. What this document fixes in scope

This document describes the statistical model declared in `version4.html` and the
estimator used by `run205_sparse400_q64_full`.  The foundational probability law is not
changed.  In particular:

- a basket is a set, not a sequence;
- basket size is induced jointly by the item utilities, interactions, and `rho_0`;
- price can therefore change total basket size;
- product incidence is a derivative of the same joint normalizer;
- the interaction matrix remains a positive-semidefinite Gram matrix; and
- all 5,455 products and all non-empty basket sizes through 120 remain in support.

The changes from the old run97 implementation are computational: rank 32 instead of 4,
randomized quasi-Monte Carlo instead of Smolyak, stable log-coordinate polynomial kernels,
and a sparse set of product rows that are allowed to carry the Gram interaction.

## 2. The object being modelled

For a trip, let:

- `A_s` be the products sold by store `s`;
- `S` be the observed non-empty subset of `A_s`;
- `h` be the household;
- `t` be the date/week and other trip context;
- `n = |S|` be total basket size; and
- `n_c` be the number of chosen products in category or affinity group `c`.

The model assigns every possible non-empty basket a real-valued score called its energy.
High-energy baskets are more probable, but probability is obtained only after comparing a
basket against *every* other allowed basket.

## 3. The unchanged version-4 joint probability

For one trip context `x=(h,t,s,...)`,

```text
P(S | x, S non-empty) = exp(E(S,x)) / Z_+(x)

Z_+(x) = sum over T subset of A_s, 1 <= |T| <= 120 of exp(E(T,x)).
```

The energy is

```text
E(S,x)
  = sum_{j in S} b_j(x)
  + sum_{j<k, j,k in S} phi_j' phi_k
  - sum_c rho_c choose(n_c,2)
  - rho_0(n).
```

Each term has one job.

### 3.1 Item value

```text
b_j(x)
  = lambda_j
  + theta_h' alpha_j
  - softplus(gamma_h)' softplus(beta_j) * price_term_j(x)
  + w_dsp_j * display_j(x)
  + w_mlr_j * mailer_j(x)
  + mu_j' delta_week(x)
  + zeta_j' xi_store(x)
  + psi_j' recency_j(x).
```

- `lambda_j` is baseline popularity after correcting for assortment exposure.
- `theta_h'alpha_j` describes which households tend to buy which products.
- The price coefficient is non-negative before the leading minus sign.  Therefore making
  a product more expensive cannot increase its own item value by construction.
- Display, mailer, week, and store terms move utility with context.
- Recency exists in the model but is disabled in run205 because its raw feature distribution
  changes sharply between the temporal training and held-out periods.

The price feature is split into an assortment-wide mean and a product-specific deviation.
The common component affects how many products are bought; the deviation reallocates share
between products.  `price_kappa` scales only the deviation.

### 3.2 Product-pair interaction

The interaction between products `j` and `k` is

```text
W_jk = phi_j' phi_k.
```

Thus `W = Phi Phi'` is positive semidefinite and has rank at most `Kz=32`.  Aligned product
vectors attract each other.  This term describes reusable complementarity such as products
that tend to belong in the same shopping mission.

The PSD restriction is not incidental: it is exactly what makes the Gaussian identity in
Section 6 possible.  An arbitrary learned pair mask could destroy PSD and therefore break
the theorem, normalizer, and sampler.

### 3.3 Category-count interaction

```text
-rho_c choose(n_c,2)
```

adds the same increment for every unordered pair drawn from group `c`.

- `rho_c > 0` discourages multiple products from that group.
- `rho_c < 0` rewards them.

Unlike the Gram interaction, this term can have either sign and is evaluated exactly by the
category polynomial.  Run205 retains the original quadratic feature on the full declared
support; it is not saturated at an earlier implementation limit.

### 3.4 Basket-size potential

`rho_0(n)` is one free value for each size from 1 through 120.  It is part of the joint
energy, not an external empirical size distribution.  Consequently

```text
P(n | x) is proportional to exp(-rho_0(n)) * Z_n(x),
```

where `Z_n(x)` is the total unnormalized mass of size-`n` baskets.  Since `Z_n(x)` depends
on utilities and prices, basket size remains price-responsive.  This is the central feature
that would be lost by factoring in a context-free empirical size histogram.

## 4. Why the normalizer matters

The raw energy alone is not a probability.  A parameter can raise the observed basket's
energy while raising competing baskets even more.  Only

```text
log P(S|x) = E(S,x) - log Z_+(x)
```

judges both sides.

Direct summation is impossible because a store contains roughly five thousand products.
The number of subsets is astronomical.  The implementation therefore changes the algebra,
not the probability law.

## 5. Completing the square in the Gram interaction

Let

```text
v_S = sum_{j in S} phi_j.
```

Then

```text
sum_{j<k in S} phi_j'phi_k
  = 1/2 ||v_S||^2 - 1/2 sum_{j in S} ||phi_j||^2.
```

The second term is additive over products, so define

```text
tilde_b_j = b_j - 1/2 ||phi_j||^2.
```

The only remaining coupling is `exp(1/2 ||v_S||^2)`.

## 6. Hubbard--Stratonovich transformation

For `z ~ Normal(0,I)`,

```text
E_z exp(z'v) = exp(1/2 ||v||^2).
```

Applying this identity to `v_S` gives

```text
exp(E(S,x))
  = E_z [ exp(-rho_0(n) - sum_c rho_c choose(n_c,2))
          * product_{j in S} w_j(z) ],

w_j(z) = exp(tilde_b_j + phi_j'z).
```

Conditional on `z`, the cross-product Gram coupling has disappeared: each chosen product
contributes only its own positive weight `w_j(z)`.  The original sum over subsets becomes a
32-dimensional Gaussian expectation of a quantity that can be computed exactly.

## 7. Exact subset summation at a fixed Gaussian node

For products of category `c`, let `e_r(w_c)` be the elementary symmetric polynomial of
degree `r`.  It is the sum of weight products over every size-`r` subset in that category.
Define the category generating polynomial

```text
G_c(u;z)
  = sum_r exp(-rho_c choose(r,2)) * e_r(w_c(z)) * u^r.
```

Multiplying category polynomials combines their count choices:

```text
product_c G_c(u;z) = sum_n A_n(z) u^n.
```

The coefficient `A_n(z)` is therefore the exact mass of every way to select `n` products,
including all category-count interactions.  The non-empty integrand is

```text
f_+(z) = sum_{n=1}^{120} exp(-rho_0(n)) A_n(z),
```

and the normalizer is

```text
Z_+(x) = E_{z~N(0,I)}[f_+(z)].
```

There are two important exactness points:

1. Empty-basket conditioning is implemented by omitting the degree-zero coefficient.  It
   does not compute `exp(log Z)-1`, which would be numerically unstable near one.
2. The ESP and category convolution are exact recursions for a supplied `z`.  The only
   numerical integration error comes from the outer Gaussian expectation.

## 8. Why run205 uses randomized quasi-Monte Carlo

The old Smolyak rule grows combinatorially with dimension and forced the experiment down to
rank 4.  Scrambled Sobol points have a chosen node count independent of dimension, so the
model can retain rank 32.

The estimator uses the identity

```text
Z_+ = E_{z~q}[ f_+(z) Normal(z;0,I) / q(z) ].
```

The proposal `q` is constructed from the model but detached from differentiation:

1. Find a local center for the full size sum.
2. In parallel, find centers for coarse size bands `1-4`, `5-10`, `11-20`, `21-40`,
   `41-80`, and `81-120`.
3. Keep the dominant center and, only when sufficiently separated and within four nats,
   a second center.
4. Rotate Sobol coordinates into the interaction eigenspace.  Run205 deliberately uses
   unit proposal scales because the audited curvature probes cost more than the accuracy
   they add in this regime.
5. Transform four independently Owen-scrambled, antithetic Sobol blocks through the
   proposal and average positive importance contributions.

Run205 starts with 64 total nodes for a one-mode trip and 128 for a two-mode mixture.  Four
independent scrambles provide a replicate estimate of integration error.

### 8.1 Per-trip refinement

The estimator does not throw away an entire minibatch because one basket is hard.  If a
trip's replicate standard error exceeds `0.015` nat, only that trip is recomputed at 128
nodes.  If necessary it continues geometrically to 256, 512, and at most 4,096 nodes.  The
accepted graph for ordinary trips is retained.

This is why a log entry such as `qretry 158` does not mean 158 failed optimizer updates.  In
the 400-update pilot it meant 158 individual trip-stage refinements across 9,600 trip
positions, with zero skipped optimizer updates.

### 8.2 Independent checkpoint guard

Training error estimates and fatal convergence checks serve different purposes:

- replicate SE decides whether an individual trip needs more nodes now;
- an independent node-doubling discrepancy measures observed finite-rule error at a
  checkpoint.

The fatal guard uses the actual `N` versus `2N` discrepancy, not `max(2*SE, gap)`.  The old
maximum falsely aborted run202 even though its node-doubling gaps stayed near 0.006 nat.
Checkpoint evaluation uses 256 nodes and checks it against 512.

## 9. Sparse interactions without changing the model family

Let `g_j` be a binary row indicator and define

```text
phi*_j = g_j phi_j.
```

Then

```text
W = Phi* Phi*' = diag(g) Phi Phi' diag(g),
```

which is still PSD and rank at most 32.  A row with `g_j=0` loses only its Gram interaction.
The product still has its intercept, household taste, price response, promotions, context,
assortment exposure, and contribution to every term of `Z_+`.

Run205 fixes `sum_j g_j = 400`.  These rows cover 100% of the 200 most frequent and 95.5%
of the 1,000 most frequent training co-purchase pairs.

### 9.1 Robust treatment of unusually large baskets

The training size distribution has mean 7.80, 99th percentile 43, 99.5th percentile 50,
99.9th percentile 66, and maximum 117.  Validation and test contain a similar tail, so it
would be wrong to delete these baskets or declare them impossible.

However, a basket of size `n` contributes `choose(n,2)` raw pair votes.  One 117-line basket
contributes 6,786 pairs and can dominate a statistic whose only purpose is selecting where
to spend limited interaction capacity.  Therefore baskets above 40 lines are excluded only
from the training-only support-ranking statistic.  They remain fully present in likelihood,
validation, price response, and generation.  The rule removes no data point from the model.

### 9.2 Fixed support versus a genuinely trainable mask

In run205, the 400 retained `phi` rows are trainable but the binary support is fixed.  A
continuous gate would leave every row nonzero and restore dense computation.  A principled
future trainable support is an exact-likelihood active set:

1. optimize the current sparse support;
2. every `T` steps, pay for one dense diagnostic backward pass;
3. rank inactive rows by the norm of their exact joint-likelihood gradient;
4. exchange a small number with weak active rows and reset their Adam moments;
5. freeze support before the final learning-rate phase; and
6. finish with ordinary unpenalized fitting on the chosen support.

At `T=200`, a 4.9-second dense audit adds about 0.025 seconds per update, below 2% of the
measured sparse update time.  This method is not enabled in run205; it is stated separately
so a fixed screened support is not mislabeled as a learned binary mask.

## 10. Likelihood and quantity model

For one observed trip,

```text
set log likelihood = E(S,x) - log Z_+(x).
```

If a chosen product has quantity `q_j >= 1`, its excess units `q_j-1` follow a separate
negative-binomial factor.  Since the basket energy does not depend on units, the joint law
factorizes exactly:

```text
P(S, quantities | x)
  = P(S | x) * product_{j in S} P(q_j | j in S,x).
```

The log reports set, units, and total likelihood separately so the set model is not compared
unfairly with a baseline that does not model units.

## 11. Initialization and calibration

Run205 starts fresh.

1. `lambda` is initialized from training purchases divided by store-assortment exposure.
2. Household/product taste factors are initialized by a training-only smoothed log-share
   rank-31 SVD, with one dimension reserved for the exact common-offset gauge.
3. Active interaction rows start from small random rank-32 vectors; inactive rows are zero.
4. `rho_0` is initialized from the empirical training size law at reference contexts.
5. Six damped IPF passes adjust `rho_0` so the aggregate model size distribution begins
   near the training distribution.  This is initialization of the original `rho_0`, not a
   replacement size factor.

## 12. Training objective and safeguards

Adam minimizes the negative joint likelihood plus declared calibration terms:

- forward size-law cross-entropy;
- reverse size-law KL, which penalizes model mass in unsupported tails;
- a small per-trip expected-size error;
- aggregate price-elasticity calibration to `-0.121`; and
- a weak training-only calibration of the cross-product pattern of price sensitivity.

These terms do not alter the definition of `P(S|x)`, but they do make the fitted estimator a
regularized/calibrated maximum-likelihood procedure rather than unconstrained MLE.

After each accepted update:

- household, week, and store context factors are projected to their zero-mean gauges;
- `lambda` is centered and its common level transferred to `rho_0`, leaving utilities
  unchanged;
- `rho_c` is floored at `-0.92` for finite full-support arithmetic;
- interaction rows are capped at norm `0.96`, operator norm at `2`, and softly whitened;
- the fixed 400-row support is reapplied; and
- non-finite gradients or estimator failures are rejected before Adam mutates parameters.

The learning rates are:

```text
structural blocks: 0.002 through step 20,000
lambda:            0.001333... through step 20,000
taste factors:     0.0001 through step 20,000

all rates multiplied by 0.5 at 20,000 and again at 26,000.
```

## 13. Incidence probabilities and recommendation

Because `log Z` is a cumulant-generating function,

```text
pi_j = d log Z_+ / d b_j = P(j in S | x, S non-empty).
```

Also,

```text
sum_j pi_j = E[n | x],
d^2 log Z_+ / db_j db_k = Cov(1[j in S], 1[k in S]).
```

Recommendation must rank candidates by `pi_j`, not raw `b_j`.  `b_j` is only one input to
competition among thousands of products; `pi_j` includes size and interaction effects after
normalization.

For complete-the-basket evaluation, reveal the rest `R` of a basket and score a candidate
completion `T`.  The exact conditional law is another version-4 normalizer over unrevealed
products with shifted terms:

```text
b*_j = b_j + phi_j' sum_{k in R} phi_k,
category count starts at |R_c|,
total size starts at |R|.
```

Autograd through this conditional normalizer returns conditional incidences.  For cutoff
`K`, MRR@K assigns `1/rank` if the hidden item ranks at most `K`, otherwise zero.  Recall@K
is the fraction of hidden items inside the first `K`.

## 14. Price response

For a utility perturbation `b -> b + epsilon d`, exponential-family differentiation gives

```text
d E[n] / d epsilon = Cov(n, d'x).
```

For a uniform utility shift `delta`, `d'x=n`, hence

```text
d E[n] / d delta = Var(n).
```

This identity explains why basket-size variance, numerical stability, and price elasticity
must be audited together.  A model with an excessively heavy size tail amplifies a modest
utility drift into a large size response.  Conversely, replacing the joint size law with a
fixed empirical histogram would force this price-size response to zero and would no longer
be version 4.

## 15. Generation

The same polynomial factorization yields a top-down sampler:

1. sample a retained RQMC proposal node with probability proportional to its contribution
   to `Z_+`;
2. sample total size from `exp(-rho_0(n)) A_n(z)`;
3. sample the split of that size across categories by reversing the category polynomial;
4. within each category, sample exactly `r_c` products by reversing the ESP recursion; and
5. sample quantities from the negative-binomial factors.

Steps 2--5 are exact conditional on the finite RQMC representation of `z`.  There is no
Gibbs chain, burn-in, or mixing heuristic.  The sampler remains on sizes 1--120 and uses the
same row-sparse Gram model as likelihood.

## 16. Computational complexity

Let:

- `B` be minibatch trips;
- `Q` be RQMC nodes;
- `Kz=32`;
- `T_A` be active interaction slots across the trip assortments;
- `C_A` be categories touched by active rows; and
- `R=120` be declared polynomial support.

The node-dependent work is approximately

```text
O(B Q Kz T_A + B Q R T_A + category-product work on C_A),
```

plus a node-independent polynomial over inactive products.  Dense interaction rows have
`T_A` close to the full assortment size per trip; row sparsity makes this term proportional
to the 400-product support present in each assortment.

Measured at 400 equal updates:

| configuration | validation set log likelihood | wall time |
|---|---:|---:|
| dense rank-32 run202 | -55.228 | 36.7 min |
| 400-row rank-32 run204 | -55.2271 | 13.5 min |

The accepted end-to-end speedup is 2.72x with no observed likelihood loss at that checkpoint.

## 17. What the checks mean

Every checkpoint reports:

- held-out set, size, composition, units, and total likelihood;
- mean and worst RQMC replicate SE;
- independent node-doubling error;
- expected basket size and population variance;
- sampled versus analytic basket size;
- aggregate price elasticity;
- interaction norm, effective rank, and `lambda_max`; and
- counts of retries, skipped updates, non-finite gradients, dropped trips, and estimator
  holds.

No single metric licenses a checkpoint.  Likelihood can look good under an underestimated
normalizer, recommendation can look good while size is wrong, and a sampler can disagree
with the analytic law even when both are finite.  Run205 therefore requires the estimator,
size, sampler, elasticity, and data-retention checks together.

## 18. Accepted pilot evidence and current scope

The fresh run204 pilot is the acceptance experiment for run205:

- all 5,455 product utilities and sizes through 120 retained;
- rank 32;
- zero skipped or bad-gradient updates through 400;
- Q256-to-Q512 checkpoint discrepancy `0.0125` nat;
- validation likelihood `-55.2271`;
- full conditional MRR `0.03437` on 316 eligible cases;
- MRR@5/10/20 `0.02521/0.02906/0.03108`; and
- exposure-corrected popularity MRR `0.02653` on the same cases.

The 400-update MRR gain is positive but not statistically decisive.  It is evidence that the
speed path did not obviously destroy ranking, not a claim that the final model has converged
or beaten every baseline.

## 19. Boundaries of the result

- The 400-row binary support is screened, not dynamically learned.
- Interaction outside those rows is exactly zero, although all additive product effects
  remain.
- Recency is disabled pending a temporally stationary feature definition.
- Aggregate price elasticity is calibrated; product-specific causal elasticity is not
  identified by this alone.
- RQMC controls and measures finite-node error but does not make `log` of a finite unbiased
  normalizer estimate itself unbiased.
- The current full run must still demonstrate stable long-run convergence and statistically
  useful held-out recommendation.

Those limitations are explicit implementation/model-capacity boundaries.  None replaces the
version-4 joint law with a different size model or changes its foundational theorem.
