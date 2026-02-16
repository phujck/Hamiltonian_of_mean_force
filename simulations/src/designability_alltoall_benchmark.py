"""
End-to-end benchmark for all-to-all 2-local designability with free-spin stochastic simulation.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from designability_alltoall_ed import run_exact_ed
from designability_alltoall_models import (
    AllToAllModel,
    flatten_two_body_couplings,
    reconstruct_couplings,
    relative_l2_error,
    sample_model,
)
from designability_alltoall_stochastic import run_stochastic
from designability_alltoall_tn import run_tn_dmrg


@dataclass
class BenchmarkConfig:
    beta: float = 6.0
    n_values: Tuple[int, ...] = (6, 8, 10, 12, 16, 20)
    samples: int = 512
    tau_steps: int = 128
    seeds: int = 3
    field_scale: float = 0.35
    coupling_scale: float = 0.08
    tn_max_bond: int = 128
    tn_cutoff: float = 1e-8
    ed_max_n: int = 12
    tn_min_n: int = 12
    tn_observable_max_n: int = 12
    output_prefix: str = "designability_alltoall"
    simulation_zz_only: bool = True
    simulation_positive_zz: bool = True
    simulation_zz_floor: float = 0.01
    simulation_omega_x_scale: float = 0.25


def _parse_n_values(text: str) -> Tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not vals:
        raise ValueError("n-values cannot be empty")
    return vals


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _results_dirs() -> Tuple[Path, Path, Path]:
    root = _project_root()
    data_dir = root / "simulations" / "results" / "data"
    fig_dir = root / "simulations" / "results" / "figures"
    ms_fig_dir = root / "manuscript" / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    ms_fig_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, fig_dir, ms_fig_dir


def _to_row_base(method: str, n: int, beta: float, seed: int, recon_rel_l2: float) -> Dict[str, float]:
    row: Dict[str, float] = {
        "method": method,
        "N": int(n),
        "beta": float(beta),
        "seed": int(seed),
        "recon_rel_l2": float(recon_rel_l2),
    }
    for key in ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"):
        row[f"two_body_{key}"] = np.nan
        row[f"stderr_two_body_{key}"] = np.nan
    for key in ("mx", "my", "mz", "energy_density"):
        row[key] = np.nan
        row[f"stderr_{key}"] = np.nan
    row["runtime_s"] = np.nan
    row["peak_mem_mb"] = np.nan
    row["effective_samples"] = np.nan
    row["mean_weight"] = np.nan
    row["tn_converged"] = np.nan
    row["tn_max_bond_observed"] = np.nan
    return row


def _add_stochastic_row(rows: List[Dict[str, float]], model: AllToAllModel, seed: int, recon_rel_l2: float, cfg: BenchmarkConfig) -> None:
    result = run_stochastic(model=model, tau_steps=cfg.tau_steps, samples=cfg.samples, seed=seed + 17_000)
    row = _to_row_base(method="stochastic", n=model.n_sites, beta=model.beta, seed=seed, recon_rel_l2=recon_rel_l2)
    row["runtime_s"] = result.runtime_s
    row["peak_mem_mb"] = result.peak_mem_mb
    row["energy_density"] = result.energy_density
    row["stderr_energy_density"] = result.stderr_energy_density
    row["mx"] = result.mx
    row["my"] = result.my
    row["mz"] = result.mz
    row["stderr_mx"] = result.stderr_mx
    row["stderr_my"] = result.stderr_my
    row["stderr_mz"] = result.stderr_mz
    row["effective_samples"] = result.effective_samples
    row["mean_weight"] = result.mean_weight
    for key, val in result.two_body.items():
        row[f"two_body_{key}"] = val
    for key, val in result.stderr_two_body.items():
        row[f"stderr_two_body_{key}"] = val
    rows.append(row)


def _add_ed_row(rows: List[Dict[str, float]], model: AllToAllModel, seed: int, recon_rel_l2: float) -> None:
    result = run_exact_ed(model)
    row = _to_row_base(method="ed", n=model.n_sites, beta=model.beta, seed=seed, recon_rel_l2=recon_rel_l2)
    row["runtime_s"] = result.runtime_s
    row["peak_mem_mb"] = result.peak_mem_mb
    row["energy_density"] = result.energy_density
    row["mx"] = result.mx
    row["my"] = result.my
    row["mz"] = result.mz
    for key, val in result.two_body.items():
        row[f"two_body_{key}"] = val
    rows.append(row)


def _add_tn_row(rows: List[Dict[str, float]], model: AllToAllModel, seed: int, recon_rel_l2: float, cfg: BenchmarkConfig) -> None:
    result = run_tn_dmrg(
        model=model,
        tn_max_bond=cfg.tn_max_bond,
        cutoff=cfg.tn_cutoff,
        max_sweeps=8,
        compute_full_observables=(model.n_sites <= cfg.tn_observable_max_n),
    )
    row = _to_row_base(method="tn_dmrg", n=model.n_sites, beta=model.beta, seed=seed, recon_rel_l2=recon_rel_l2)
    row["runtime_s"] = result.runtime_s
    row["peak_mem_mb"] = result.peak_mem_mb
    row["energy_density"] = result.energy_density
    row["mx"] = result.mx
    row["my"] = result.my
    row["mz"] = result.mz
    row["tn_converged"] = float(result.converged)
    row["tn_max_bond_observed"] = float(result.max_bond_observed)
    for key, val in result.two_body.items():
        row[f"two_body_{key}"] = val
    rows.append(row)


def _self_tests() -> None:
    rng = np.random.default_rng(123)
    model = sample_model(n_sites=4, beta=2.0, rng=rng, field_scale=0.2, coupling_scale=0.05)
    target = flatten_two_body_couplings(model)
    reconstructed, rank, n_cols = reconstruct_couplings(model.n_sites, target)
    rel = relative_l2_error(target, reconstructed)
    if rank != n_cols:
        raise RuntimeError(f"Design matrix rank failure in self-test: rank={rank}, cols={n_cols}")
    if rel > 1e-12:
        raise RuntimeError(f"Design reconstruction self-test failed: rel={rel}")

    s1 = run_stochastic(model=model, tau_steps=12, samples=32, seed=999)
    s2 = run_stochastic(model=model, tau_steps=12, samples=32, seed=999)
    if abs(s1.energy_density - s2.energy_density) > 1e-12:
        raise RuntimeError("Stochastic reproducibility self-test failed")


def _project_simulation_sector(model: AllToAllModel, cfg: BenchmarkConfig) -> AllToAllModel:
    if not cfg.simulation_zz_only:
        return model

    projected = AllToAllModel(
        n_sites=model.n_sites,
        beta=model.beta,
        omega_x=model.omega_x.copy() * cfg.simulation_omega_x_scale,
        omega_z=model.omega_z.copy(),
        pair_terms=model.pair_terms,
        couplings=model.couplings.copy(),
    )
    for k, term in enumerate(projected.pair_terms):
        if term.a == 2 and term.b == 2:
            if cfg.simulation_positive_zz:
                projected.couplings[k] = abs(projected.couplings[k]) + cfg.simulation_zz_floor
        else:
            projected.couplings[k] = 0.0
    return projected


def _plot_reconstruction(recon_df: pd.DataFrame, fig_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    rank_summary = recon_df.groupby("N", as_index=False).agg(rank=("rank", "mean"), n_cols=("n_cols", "mean"), rel_l2=("recon_rel_l2", "mean"))
    axes[0].plot(rank_summary["N"], rank_summary["rank"], marker="o", label="rank(A)")
    axes[0].plot(rank_summary["N"], rank_summary["n_cols"], marker="s", linestyle="--", label="columns(A)")
    axes[0].set_xlabel("N")
    axes[0].set_ylabel("Dimension")
    axes[0].set_title("Design-map rank check")
    axes[0].legend()

    rel_vals = np.maximum(rank_summary["rel_l2"].to_numpy(dtype=float), 1e-16)
    axes[1].plot(rank_summary["N"], rel_vals, marker="o")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("N")
    axes[1].set_ylabel(r"Relative $\ell_2$ reconstruction error")
    axes[1].set_title("Inverse-map accuracy")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)


def _plot_ed_agreement(df: pd.DataFrame, fig_path: Path) -> pd.DataFrame:
    stoch = df[df["method"] == "stochastic"].copy()
    ed = df[df["method"] == "ed"].copy()
    overlap = pd.merge(
        stoch,
        ed,
        on=["N", "seed", "beta", "recon_rel_l2"],
        suffixes=("_stoch", "_ed"),
        how="inner",
    )
    if overlap.empty:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, "No stochastic/ED overlap", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=180)
        plt.close(fig)
        return overlap

    summary = overlap.groupby("N", as_index=False).agg(
        energy_stoch=("energy_density_stoch", "mean"),
        energy_stderr=("stderr_energy_density_stoch", "mean"),
        energy_ed=("energy_density_ed", "mean"),
        mz_stoch=("mz_stoch", "mean"),
        mz_stderr=("stderr_mz_stoch", "mean"),
        mz_ed=("mz_ed", "mean"),
        czz_stoch=("two_body_zz_stoch", "mean"),
        czz_stderr=("stderr_two_body_zz_stoch", "mean"),
        czz_ed=("two_body_zz_ed", "mean"),
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].errorbar(summary["N"], summary["energy_stoch"], yerr=summary["energy_stderr"], marker="o", label="stochastic")
    axes[0].plot(summary["N"], summary["energy_ed"], marker="s", linestyle="--", label="ED")
    axes[0].set_title("Energy density")
    axes[0].set_xlabel("N")
    axes[0].legend()

    axes[1].errorbar(summary["N"], summary["mz_stoch"], yerr=summary["mz_stderr"], marker="o", label="stochastic")
    axes[1].plot(summary["N"], summary["mz_ed"], marker="s", linestyle="--", label="ED")
    axes[1].set_title(r"$m_z$")
    axes[1].set_xlabel("N")

    axes[2].errorbar(summary["N"], summary["czz_stoch"], yerr=summary["czz_stderr"], marker="o", label="stochastic")
    axes[2].plot(summary["N"], summary["czz_ed"], marker="s", linestyle="--", label="ED")
    axes[2].set_title(r"$\overline{\langle Z_i Z_j \rangle}$")
    axes[2].set_xlabel("N")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    return overlap


def _plot_scaling(df: pd.DataFrame, fig_path: Path) -> None:
    scal = df[df["method"].isin(["stochastic", "tn_dmrg"])].copy()
    if scal.empty:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, "No scaling data", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=180)
        plt.close(fig)
        return

    summary = scal.groupby(["method", "N"], as_index=False).agg(
        runtime_s=("runtime_s", "mean"),
        peak_mem_mb=("peak_mem_mb", "mean"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method, grp in summary.groupby("method"):
        axes[0].plot(grp["N"], grp["runtime_s"], marker="o", label=method)
        axes[1].plot(grp["N"], grp["peak_mem_mb"], marker="o", label=method)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("N")
    axes[0].set_ylabel("Runtime [s]")
    axes[0].set_title("Scaling: runtime")
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("Peak memory [MB]")
    axes[1].set_title("Scaling: memory")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)


def run_benchmark(cfg: BenchmarkConfig, run_self_tests: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Path]]:
    if run_self_tests:
        _self_tests()

    data_dir, fig_dir, ms_fig_dir = _results_dirs()
    rows: List[Dict[str, float]] = []
    recon_rows: List[Dict[str, float]] = []

    for n in cfg.n_values:
        for seed in range(cfg.seeds):
            rng = np.random.default_rng(seed + 1_000 * n)
            model = sample_model(
                n_sites=n,
                beta=cfg.beta,
                rng=rng,
                field_scale=cfg.field_scale,
                coupling_scale=cfg.coupling_scale,
            )
            sim_model = _project_simulation_sector(model, cfg)

            target = flatten_two_body_couplings(model)
            reconstructed, rank, n_cols = reconstruct_couplings(n, target)
            recon_rel_l2 = relative_l2_error(target, reconstructed)
            recon_rows.append(
                {
                    "N": int(n),
                    "seed": int(seed),
                    "beta": float(cfg.beta),
                    "rank": int(rank),
                    "n_cols": int(n_cols),
                    "recon_rel_l2": float(recon_rel_l2),
                }
            )

            _add_stochastic_row(rows, model=sim_model, seed=seed, recon_rel_l2=recon_rel_l2, cfg=cfg)
            if n <= cfg.ed_max_n:
                _add_ed_row(rows, model=sim_model, seed=seed, recon_rel_l2=recon_rel_l2)
            if n >= cfg.tn_min_n:
                _add_tn_row(rows, model=sim_model, seed=seed, recon_rel_l2=recon_rel_l2, cfg=cfg)

    df = pd.DataFrame(rows).sort_values(["N", "seed", "method"]).reset_index(drop=True)
    recon_df = pd.DataFrame(recon_rows).sort_values(["N", "seed"]).reset_index(drop=True)

    metrics_csv = data_dir / f"{cfg.output_prefix}_metrics.csv"
    recon_csv = data_dir / f"{cfg.output_prefix}_reconstruction.csv"
    df.to_csv(metrics_csv, index=False)
    recon_df.to_csv(recon_csv, index=False)

    recon_fig = fig_dir / f"{cfg.output_prefix}_reconstruction.png"
    ed_fig = fig_dir / f"{cfg.output_prefix}_ed_agreement.png"
    scaling_fig = fig_dir / f"{cfg.output_prefix}_scaling.png"
    _plot_reconstruction(recon_df, recon_fig)
    overlap = _plot_ed_agreement(df, ed_fig)
    _plot_scaling(df, scaling_fig)

    overlap_csv = data_dir / f"{cfg.output_prefix}_ed_overlap.csv"
    overlap.to_csv(overlap_csv, index=False)

    manuscript_map = {
        "reconstruction": ms_fig_dir / "hmf_designability_alltoall_reconstruction.png",
        "ed_agreement": ms_fig_dir / "hmf_designability_alltoall_ed_agreement.png",
        "scaling": ms_fig_dir / "hmf_designability_alltoall_scaling.png",
    }
    shutil.copy2(recon_fig, manuscript_map["reconstruction"])
    shutil.copy2(ed_fig, manuscript_map["ed_agreement"])
    shutil.copy2(scaling_fig, manuscript_map["scaling"])

    paths = {
        "metrics_csv": metrics_csv,
        "recon_csv": recon_csv,
        "overlap_csv": overlap_csv,
        "reconstruction_fig": recon_fig,
        "ed_fig": ed_fig,
        "scaling_fig": scaling_fig,
        "ms_reconstruction_fig": manuscript_map["reconstruction"],
        "ms_ed_fig": manuscript_map["ed_agreement"],
        "ms_scaling_fig": manuscript_map["scaling"],
    }
    return df, recon_df, overlap, paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="All-to-all 2-local designability benchmark.")
    parser.add_argument("--beta", type=float, default=6.0)
    parser.add_argument("--n-values", type=str, default="6,8,10,12,16,20")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--tau-steps", type=int, default=128)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--field-scale", type=float, default=0.35)
    parser.add_argument("--coupling-scale", type=float, default=0.08)
    parser.add_argument("--tn-max-bond", type=int, default=128)
    parser.add_argument("--tn-cutoff", type=float, default=1e-8)
    parser.add_argument("--ed-max-n", type=int, default=12)
    parser.add_argument("--tn-min-n", type=int, default=12)
    parser.add_argument("--tn-observable-max-n", type=int, default=12)
    parser.add_argument("--output-prefix", type=str, default="designability_alltoall")
    parser.add_argument("--skip-self-tests", action="store_true")
    return parser


def run_from_args(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Path]]:
    cfg = BenchmarkConfig(
        beta=args.beta,
        n_values=_parse_n_values(args.n_values),
        samples=args.samples,
        tau_steps=args.tau_steps,
        seeds=args.seeds,
        field_scale=args.field_scale,
        coupling_scale=args.coupling_scale,
        tn_max_bond=args.tn_max_bond,
        tn_cutoff=args.tn_cutoff,
        ed_max_n=args.ed_max_n,
        tn_min_n=args.tn_min_n,
        tn_observable_max_n=args.tn_observable_max_n,
        output_prefix=args.output_prefix,
    )
    return run_benchmark(cfg=cfg, run_self_tests=not args.skip_self_tests)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    df, recon_df, overlap, paths = run_from_args(args)
    print("All-to-all designability benchmark complete.")
    print(f"Rows (metrics): {len(df)}")
    print(f"Rows (reconstruction): {len(recon_df)}")
    print(f"Rows (ED overlap): {len(overlap)}")
    for k, v in paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
