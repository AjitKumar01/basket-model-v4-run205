#!/bin/zsh
set -o pipefail

repo_root="${0:A:h:h:h}"
cd "$repo_root"

env V3_AFFINITY=1 python scripts/v3/fit.py \
  --label run205_sparse400_q64_full \
  --require-version4 1 \
  --K 32 --Kz 32 --Kp 8 --nmax 120 --R 120 \
  --iters 30000 --batch 24 --torch-threads 4 --draws 16 \
  --lr 0.002 --lam-lr-scale 0.6666667 --taste-lr-scale 0.05 \
  --taste-weight-decay 0 --cosine 0 --lr-milestones 20000,26000 --lr-gamma 0.5 \
  --eval-every 200 --log-every 10 --save-every 100 --safety-every 100 \
  --n-val 384 --eval-initial 1 --probe 10 --n-rec 0 \
  --init-popularity 1 --init-rho0 1 --size-ipf-steps 6 \
  --size-ipf-trips 384 --size-ipf-damp 0.7 \
  --moment-taste-init 1 --moment-taste-prior 100 --moment-taste-clip 3 \
  --moment-phi-init 0 --moment-pair-max-basket 40 \
  --units 1 --wd 0.00001 --no-rec 1 \
  --size-kl 1 --rkl-w 10 --rkl-eps 0.0001 --en-w 0.005 \
  --elast-w 20 --elast-target -0.121 \
  --beta-cal-w 0.1 --beta-target basket_input/v3_beta_target.npz \
  --phi-init 0.03 --phi-mask basket_input/v3_phimask_k400.npy \
  --phi-max 0.96 --phi-op-max 2 --phi-centre 1 --phi-whiten 0.5 \
  --rho-c-floor -0.92 --rho-c-step-scale 0.25 \
  --qmc-n 64 --qmc-reps 4 --qmc-refresh-every 1 --qmc-eval-n 256 \
  --qmc-step-se 0.015 --qmc-retry-n 128 --qmc-retry-max-n 4096 \
  --qmc-retry-probe -1 --qmc-en-max 2.5 \
  --qmc-size-bands 1 --qmc-size-steps 3 --qmc-mode-logtol 4 \
  --qmc-mode-sep 1 --qmc-mix-n 128 --quad-probe -1 --quad-chunk 32 \
  --antithetic 1 --esp-native 1 --poly-degree-native 1 \
  --pi-project-every 0 --lam-centre 1 \
  --ess-floor 0.3 --ess-floor-min 0.15 --min-keep 0.5 \
  --lz-gap 0.02 --lz-strikes 3 --clip 2 --seed 0 \
  2>&1 | tee out/v3_run205_sparse400_q64_full.log
