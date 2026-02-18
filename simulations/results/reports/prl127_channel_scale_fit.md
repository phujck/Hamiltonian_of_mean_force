# PRL127 Channel-Scale Fit (Symmetric Compressed Model)

Goal: test whether the apparent sign issue manifests as a relative amplitude distortion between `x` and `z` channels after composition with the system thermal factor.

## Setup
- Model: symmetric compressed finite model (`finite_hmf_v4_sym_state`)
- Mapping: `lambda_paper = g^2` (`lam2`)
- Reference: exact ED states from:
  - `simulations/results/data/prl127_signcheck_base_scan.csv` (`g <= 0.4`)
  - `simulations/results/data/prl127_signcheck_l8_base_scan.csv` (`g <= 8`)
- Fitted multipliers:
  - `sx` multiplies `Delta_x` channel.
  - `sz` multiplies `Delta_z` channel.

## Results

### Window `g <= 0.4`
- Coarse + refined fit:
  - Best refined around `sx ~ 0.125-0.150`, `sz ~ 0.75-0.80`.
  - Best row: `sx=0.125`, `sz=0.750`, `d_mean=0.030594`, `d_max=0.054368`.
- CSVs:
  - `simulations/results/data/prl127_channel_scale_fit_coarse.csv`
  - `simulations/results/data/prl127_channel_scale_fit_refined.csv`

### Full range `g <= 8`
- Focused fit:
  - Best row: `sx=0.1875`, `sz=0.500`, `d_mean=0.050854`, `d_max=0.060290`, `d_last=0.052065`.
- CSV:
  - `simulations/results/data/prl127_channel_scale_fit_l8.csv`

## Interpretation
1. This strongly supports your point: the mismatch is primarily relative channel amplitude (`x` vs `z`), not a pure global sign.
2. Best fits keep signs positive in this parameterization and suppress `x` more than `z`.
3. The symmetric compressed ansatz can be improved substantially by channel reweighting, but it still does not reach ordered-model accuracy.

## Likely algebraic implication
The equal-amplitude locking implied by the current compressed `Delta ~ K_eff (c s sigma_x - s^2 sigma_z)` is likely too restrictive. A missing combinatorial factor/sign in the reduction to that locked form can produce exactly this observed `(sx, sz)` distortion after noncommuting thermal composition.
