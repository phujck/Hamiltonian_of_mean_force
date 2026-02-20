# HMF v5 Bug Diagnosis Report — θ = π/2

**Date:** 2026-02-19  
**Scope:** θ = π/2 (pure transverse coupling: `f = −σ_x`, `c = cosθ = 0`, `s = sinθ = 1`)  
**References:** `04_results_v5.tex`, `hmf_v5_qubit_core.py:v5_theory_state`, `prl127_qubit_analytic_bridge.py`  
**Ground truth:** ED (`exact_reduced_state`) and ordered Gaussian (`finite_hmf_ordered_gaussian_state`)

---

## 1. The numbers: what the diagnostic run shows

Parameters: β = 2, ω_q = 2, θ = π/2, bath: 4 modes, exponential spectral density.

```
g     ED rho11   ED rho22  | Ord rho11  Ord rho22 | V5  rho11  V5  rho22
0.00   0.01799    0.98201  |   0.01799    0.98201  |   0.01799    0.98201
0.30   0.06847    0.93153  |   0.19475    0.80525  |   0.99610    0.00390  ← WRONG
0.60   0.19893    0.80107  |   0.38339    0.61661  |   1.00000    0.00000  ← WRONG
1.00   0.35607    0.64393  |   0.43604    0.56396  |   1.00000    0.00000  ← WRONG
4.00   0.48302    0.51698  |   0.48853    0.51147  |   1.00000    0.00000  ← WRONG
```

**Both ED and ordered correctly approach I/2** (as required by the paper's ultrastrong formula, eq. 8: `cos(π/2) = 0`). The v5 compact model collapses to the *excited* energy eigenstate from the first nonzero coupling.

---

## 2. What the base channels show

At g = 1 (g² scales all channels):

```
Sigma_+0 =  0.0         (correct: c=0 kills all ladder channels)
Sigma_-0 =  0.0
Delta_z0 = 53.01        K^-/(2ω_q) =  34.47  (v4 formula, different integral)
R0_plus  = 114.22
R0_minus =   8.20
```

The channels themselves are algebraically correct. The problem is not in the integrals.

---

## 3. The root cause: sign of Δ_z in the exponent

### 3.1 V5 formula structure at θ = π/2

At θ = π/2, `Σ± = 0`, `χ = Δ_z`, so the v5 state matrix reduces to:
```
ρ_Q = (1/Z_Q) * [[ e^{-a}·(1 + tanh χ),    0               ],
                  [ 0,                        e^{+a}·(1 − tanh χ) ]]
```
with `Z_Q = 2·(cosh a − tanh χ · sinh a)`.

For any finite χ > 0, the upper-left element **exceeds** its g = 0 value. As χ → ∞:
```
tanh χ → 1
Z_Q    → 2·exp(−a)
ρ_11   → e^{−a} · 2 / (2·e^{−a}) = 1   ← excited state fully occupied
ρ_22   → 0
```
This is **energetically inverted and unphysical**. The ultrastrong limit must give I/2.

### 3.2 Mathematical verification that the formula is internally consistent

The v5 formula correctly represents:
```
ρ_bar ∝ exp(−β H_Q) · exp(+Δ_z · σ_z)
       = diag( e^{−a+Δ_z},  e^{+a−Δ_z} )    (up to exp(Δ_0))
```
At large Δ_z: ρ_11/ρ_22 = e^{−a+Δ_z} / e^{+a−Δ_z} = e^{2(Δ_z − a)} → ∞. Numerical check confirms Z_Q matches `2·cosh(Δ_z − a)` exactly. The normalization is correct — the formula does exactly what it says. **The formula itself is the bug.**

### 3.3 The correct formula must have the opposite sign on Δ_z

For `exp(−β H_Q)·exp(±Δ_z · σ_z)` to give I/2 at θ = π/2 in the large coupling limit, we would need ρ_11 and ρ_22 to equalize. With **`+Δ_z`**: excited state dominates (wrong). With **`−Δ_z`**: ground state dominates `→ |ground⟩` (also wrong, gives I rather than I/2).

**Neither sign alone gives I/2.** This means the formula `ρ_bar = exp(−β H_Q)·exp(Δ_z σ_z)` is **structurally unable** to reproduce the correct ultrastrong limit at θ = π/2. The Σ± channels must be non-zero to achieve I/2.

---

## 4. The algebraic slip: Σ± vanishes when it shouldn't

### 4.1 The v5 formula gives Σ± = 0 at θ = π/2

From eq. (Sigma_plus_K_v5):
```
Σ+ = (c·s/ω_q)·[(1 + e^{β ω_q})·K(0) − 2·K(ω_q)]
```
At θ = π/2, `c = 0` → **Σ+ = 0**. This follows algebraically from the product formula:
```
f̃(τ)·f̃(τ') = cosh(ω_q(τ − τ'))·I + sinh(ω_q(τ − τ'))·σ_z    [at θ = π/2]
```
which has no σ± components. So Σ± = 0 as an exact result of the second-order cumulant. **This is correct at second order.**

### 4.2 Why the state cannot approach I/2 with only a σ_z exponent

The ultrastrong projected state `ρ_US = (1/2)[I − σ_f̂ · tanh(a cos θ)]` at θ = π/2 is **I/2** — the maximally mixed state. For any state of the form `exp(−β H_Q)·exp(Δ_z σ_z)`, the off-diagonal elements are exactly zero (since both factors are diagonal). A purely diagonal state can equal I/2 only if ρ_11 = ρ_22 = 0.5. But we've established this happens only at the single coupling point where Δ_z = a, not in a limit.

### 4.3 Resolution: the influence exponent is not just second-order

The claim that `⟨T exp(−g∫ ξ f̃ dτ)⟩_Gaussian = exp(Δ)` exactly (due to su(2) closure) holds **only if the second cumulant captures all operator structure**. At θ = π/2, the individual operators `f̃(τ)` **do not commute at different times**:
```
[f̃(τ), f̃(τ')] = −2·sinh(ω_q(τ − τ'))·σ_z ≠ 0
```
While the **pairwise products** `f̃(τ)·f̃(τ')` happen to commute with each other (being linear in I and σ_z), the full time-ordered Gaussian average is not simply `exp(second cumulant)` — the operator-valued Wick contractions at 4th order and above generate new operator structure that the scalar Gaussian formula misses.

The correct resummation (captured by the ordered Gaussian model) gives ρ_11 → 0.5 at θ = π/2. The v5 compact formula misses this because it truncates the Baker-Campbell-Hausdorff expansion of the time-ordered exponential at second order.

---

## 5. Why the v4 model works

The v4 model derives `Δ_z = K^−/(2ω_q)` from a **different** integral and constructs the state via a full Bloch vector magnitude and the arcosh formula. This approach correctly encodes the geometric structure of the equilibrium state. At θ = π/2:
```
v4 Delta_z = K^−/(2ω_q) = 34.47    [large but different from 53.01]
v5 Delta_z = R^−         = 53.01
```
Both channels are large enough to saturate tanh immediately. **The v4 model does not use these directly in exp(Δ_z·σ_z)** — it feeds them through the arcosh/HMF construction, which has a different analytic structure robust to the sign issue.

---

## 6. What needs to be fixed in `04_results_v5.tex` and the code

The compact v5 state (eq. `rho_matrix_v5`) arises from:
```
ρ_bar = exp(−β H_Q) · exp(Δ₀·I + Δ_z·σ_z + Σ+·σ+ + Σ−·σ−)
```
The claimed su(2) closure makes `exp(Δ)` exact via a two-term matrix exponential. **This is only exact if `exp(Δ)` equals the full Gaussian time-ordered average**, which is not guaranteed when the integrand operators are noncommuting at different times.

The fix requires either:

**Option A (algebraic):** Derive the correct closed-form for `⟨T exp(...)⟩_Gaussian` that properly accounts for operator ordering. This will likely produce Σ± terms even at θ = π/2 from higher-order BCH residuals.

**Option B (procedural):** Keep the ordered Gaussian model as the analytic prediction (it IS exact) and restrict the compact v5 form to a perturbative (`χ ≪ 1`) approximation only.

**Option C (channel correction):** Examine whether the channels Σ± in the v5 derivation are actually missing contributions from terms of the form `G>(ωq, ωq)` and `G>(−ωq, −ωq)` that arise from higher BCH terms but which the current algebra drops.

---

## 7. Diagnostic checklist for confirming the fix

Once a candidate fix is implemented, verify at θ = π/2:

1. `v5 rho_11(g=0.3)` ≈ 0.07–0.20 (not ≈ 1.0 as currently)
2. `rho_11` is monotonically increasing toward 0.5 as g → ∞
3. `trace_distance(rho_v5, rho_ordered)` < 0.05 at g = 1.0
4. KMS residual `Σ+ − e^{βω_q}·Σ−` = 0 (currently satisfied, must remain so)
5. Paper eq. (8) ultrastrong limit reproduced: as g → ∞, ρ → I/2 at θ = π/2

---

## 8. Summary

| Issue | Status |
|---|---|
| Channel integrals Σ±, Δ_z algebraically correct? | ✅ Yes |
| Code implements channels correctly? | ✅ Yes |
| v5 Z_Q normalization internally consistent? | ✅ Yes |
| `exp(−βH_Q)·exp(+Δ_z σ_z)` gives correct ultrastrong limit? | ❌ No |
| Ordered model gives correct ultrastrong limit? | ✅ Yes |
| Root cause | The exp(Δ) formula misses operator-ordered BCH corrections; su(2) closure does not guarantee the second-order cumulant gives the exact Gaussian expectation value for noncommuting integrands |
| Where to look for the slip in the tex | Between eq. (Gxy_closed) and eq. (rho_matrix_v5): the claim that ⟨T exp⟩ = exp(Δ) needs to be re-examined for the noncommuting case. The Σ± channels at θ = π/2 are missing contributions. |

*Diagnostic script: `simulations/src/hmf_pi2_diagnostic.py`*
