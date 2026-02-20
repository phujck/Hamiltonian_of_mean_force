"""
Wide-range sweeps for Figure 2 [v50]:
  omega_q=1, n_modes=2, n_cut=32, omega in [0.0, 1.8]
  Log-spaced ranges for US recovery:
  g in [0.01, 100] (60 pts)
  beta in [0.1, 50] (50 pts)
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from hmf_component_normalized_compare_codex_v5 import _compact_components
from hmf_model_comparison_standalone_codex_v1 import (
    BenchmarkConfig as LiteConfig,
    RenormConfig,
    extract_density,
)
from hmf_v5_qubit_core import build_ed_context, exact_reduced_state
from prl127_qubit_benchmark import BenchmarkConfig as EDConfig

OUT_STEM = "hmf_fig2_sweeps_v50"

@dataclass(frozen=True)
class SweepDef:
    name: str
    param_name: str
    param_values: np.ndarray
    beta_fixed: float | None
    theta_fixed: float | None
    g_fixed: float | None

def _build_lite(beta: float, theta: float, n_modes: int) -> LiteConfig:
    return LiteConfig(
        beta=float(beta),
        omega_q=1.0,
        theta=float(theta),
        n_modes=int(n_modes),
        n_cut=1,
        omega_min=0.0,
        omega_max=1.8,
        q_strength=5.0,
        tau_c=0.5,
    )

def _build_ed(beta: float, theta: float) -> EDConfig:
    return EDConfig(
        beta=float(beta),
        omega_q=1.0,
        theta=float(theta),
        n_modes=2,
        n_cut=32, # High n_cut for strong g
        omega_min=0.0,
        omega_max=1.8,
        q_strength=5.0,
        tau_c=0.5,
        lambda_min=0.0,
        lambda_max=1.0,
        lambda_points=2,
        output_prefix=OUT_STEM,
    )

def run_sweeps() -> pd.DataFrame:
    ren = RenormConfig(scale=1.0, kappa=1.0)

    sweeps = [
        SweepDef(
            name="coupling",
            param_name="g",
            param_values=np.logspace(-1, 2, 60), # 0.1 to 100
            beta_fixed=1.0,
            theta_fixed=float(np.pi / 4.0),
            g_fixed=None,
        ),
        SweepDef(
            name="temperature",
            param_name="beta",
            param_values=np.logspace(-1, 1.7, 50), # 0.1 to ~50
            beta_fixed=None,
            theta_fixed=float(np.pi / 2.0),
            g_fixed=0.5,
        ),
    ]

    rows: list[dict[str, float | str]] = []
    total = sum(len(s.param_values) for s in sweeps)
    k = 0

    for sw in sweeps:
        for p in sw.param_values:
            beta = float(p) if sw.param_name == "beta" else float(sw.beta_fixed)
            theta = float(p) if sw.param_name == "theta" else float(sw.theta_fixed)
            g = float(p) if sw.param_name == "g" else float(sw.g_fixed)

            lite_disc = _build_lite(beta=beta, theta=theta, n_modes=2)
            disc_p00, disc_p11, disc_coh, _ = _compact_components(lite_disc, g, True, ren)

            lite_cont = _build_lite(beta=beta, theta=theta, n_modes=40)
            cont_p00, cont_p11, cont_coh, _ = _compact_components(lite_cont, g, True, ren)

            # ED is only stable up to a point
            # We skip ED for very large g if it's too slow or likely to fail
            # For 2-modes n_cut=20, let's try up to g=10
            do_ed = (g <= 10.0) 
            
            if do_ed:
                ed_cfg = _build_ed(beta=beta, theta=theta)
                ed_ctx = build_ed_context(ed_cfg)
                rho_ed = exact_reduced_state(ed_ctx, g)
                ed_p00, ed_p11, ed_coh = extract_density(rho_ed)
            else:
                ed_p00, ed_p11, ed_coh = np.nan, np.nan, np.nan

            rows.append({
                "sweep": sw.name, "param_name": sw.param_name, "param": float(p),
                "beta": beta, "theta": theta, "g": g,
                "ed_p00": float(ed_p00), "ed_p11": float(ed_p11), "ed_coh": float(ed_coh),
                "analytic_disc_p00": float(disc_p00), "analytic_disc_p11": float(disc_p11), "analytic_disc_coh": float(disc_coh),
                "analytic_cont_p00": float(cont_p00), "analytic_cont_p11": float(cont_p11), "analytic_cont_coh": float(cont_coh),
            })
            k += 1
            if k % 10 == 0: print(f"[PROGRESS] {k}/{total}")

    return pd.DataFrame.from_records(rows)

def main() -> None:
    print("Starting Wide-range sweeps (v50)...")
    df = run_sweeps()
    out_dir = Path(__file__).resolve().parent
    prod_data = Path(__file__).resolve().parents[1] / "production" / "data"
    prod_data.mkdir(exist_ok=True)
    scan_csv = out_dir / f"{OUT_STEM}_scan.csv"
    df.to_csv(scan_csv, index=False)
    df.to_csv(prod_data / "sweeps_v50.csv", index=False)
    print(f"Done. Saved to {scan_csv}")

if __name__ == "__main__":
    main()
