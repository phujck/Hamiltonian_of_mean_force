# Stable Exact Evaluator Check (Codex v16)

Comparison: naive Eq.(rho_matrix_v5) evaluator vs stable Mobius/log-domain evaluator.

- n_points: 54
- max |delta p00|: inf
- max |delta p11|: 7.754e-02
- max relative ratio diff: 9.999e-01
- naive non-finite count: 3
- stable non-finite count: 0

Stable form used:
$m_z=(u-\tanh a)/(1-u\tanh a)$, $u=\tanh(\chi)\Delta_z/\chi$, $p_{00}=(1+m_z)/2$.
Ratio from $\log R=-2a+\log(1+u)-\log(1-u)$.