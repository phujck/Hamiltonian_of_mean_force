import argparse

from designability_alltoall_benchmark import build_parser as build_designability_parser
from designability_alltoall_benchmark import run_from_args as run_designability_from_args
from prl127_qubit_benchmark import build_parser as build_prl127_parser
from prl127_qubit_benchmark import run_from_args as run_prl127_from_args


def main() -> None:
    root = argparse.ArgumentParser(description="Hamiltonian of Mean Force simulation entrypoint.")
    root.add_argument(
        "--benchmark",
        choices=["prl127_qubit", "designability_alltoall"],
        default="prl127_qubit",
        help="Benchmark target to run.",
    )

    root_args, remaining = root.parse_known_args()

    if root_args.benchmark == "designability_alltoall":
        design_parser = build_designability_parser()
        design_args = design_parser.parse_args(remaining)
        _, _, _, paths = run_designability_from_args(design_args)
        print("All-to-all designability simulation finished.")
        for k, v in paths.items():
            print(f"{k}: {v}")
        return

    if root_args.benchmark == "prl127_qubit":
        prl_parser = build_prl127_parser()
        prl_args = prl_parser.parse_args(remaining)
        _, csv_path, fig_path = run_prl127_from_args(prl_args)
        print("Simulation finished.")
        print(f"CSV: {csv_path}")
        print(f"Figure: {fig_path}")
        return

    raise ValueError(f"Unknown benchmark target: {root_args.benchmark}")


if __name__ == "__main__":
    main()
