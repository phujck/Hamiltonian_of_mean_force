"""
Plotting script for HMF v4 validation.
- Reads data from simulations/results/data/hmf_v4_validation.csv.
- Generates comparison plots.

Run with: python simulations/src/hmf_v4_plot.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_results():
    data_path = os.path.join("simulations", "results", "data", "hmf_v4_validation.csv")
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please run simulations/src/hmf_v4_generate.py first.")
        return

    df = pd.read_csv(data_path)
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Distance
    axes[0].semilogy(df["lambda"], df["dist_v4"], label="Trace Distance")
    axes[0].set_xlabel("Lambda")
    axes[0].set_ylabel("Distance")
    axes[0].set_title("Fidelity of HMF v4")
    axes[0].legend()
    axes[0].grid(True, which="both", ls="-", alpha=0.2)
    
    # Fields (Rotated)
    axes[1].plot(df["lambda"], df["hx_v4"], 'k-', label="Analytic h_x")
    axes[1].plot(df["lambda"], df["hx_num_rot"], 'r--', label="Numeric h_x (Rot)")
    axes[1].plot(df["lambda"], df["hz_v4"], 'b-', label="Analytic h_z")
    axes[1].plot(df["lambda"], df["hz_num"], 'g--', label="Numeric h_z")
    axes[1].set_xlabel("Lambda")
    axes[1].set_ylabel("Field Strength")
    axes[1].set_title("Effective Fields (Rotated Frame)")
    axes[1].legend()
    axes[1].grid(True)

    # Check y-component removal
    axes[2].plot(df["lambda"], df["hy_num_rot"], label="Numeric h_y (Rot)")
    axes[2].plot(df["lambda"], df["hy_num_lab"], alpha=0.3, label="Numeric h_y (Lab)")
    axes[2].set_xlabel("Lambda")
    axes[2].set_title("Resulting h_y (Should be ~0)")
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    output_fig = os.path.join("simulations", "results", "figures", "hmf_v4_validation.png")
    os.makedirs(os.path.dirname(output_fig), exist_ok=True)
    plt.savefig(output_fig, dpi=300)
    print(f"Validation plot saved to {output_fig}")

if __name__ == "__main__":
    plot_results()
