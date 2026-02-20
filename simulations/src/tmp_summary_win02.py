import pandas as pd
import numpy as np

df = pd.read_csv("c:/Users/gerar/VScodeProjects/Hamiltonian_of_mean_force/simulations/src/hmf_fig2_sweeps_v49_win02_scan.csv")

def _rmse(x):
    return np.sqrt(np.mean(np.square(x)))

summary_rows = []
for sweep, grp in df.groupby("sweep"):
    summary_rows.append({
        "sweep": sweep,
        "rmse_disc": _rmse(grp["d_disc_p00"]),
        "max_disc": np.max(np.abs(grp["d_disc_p00"])),
        "rmse_cont": _rmse(grp["d_cont_p00"]),
        "max_cont": np.max(np.abs(grp["d_cont_p00"])),
    })

summary = pd.DataFrame(summary_rows)
print(summary.to_string(index=False))
