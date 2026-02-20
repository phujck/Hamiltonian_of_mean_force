"""
Verify specific HMF extraction logic on known states.
"""
import numpy as np
from scipy.linalg import expm, logm
from hmf_model_comparison_standalone import get_hmf_fields, SIGMA_X, SIGMA_Z

def check_state(hx_in, hz_in, beta):
    # Construct exact state
    H_target = hx_in * SIGMA_X + hz_in * SIGMA_Z
    rho = expm(-beta * H_target)
    rho /= np.trace(rho)
    
    # Extract
    hx_out, hz_out = get_hmf_fields(rho, beta)
    
    print(f"In:  hx={hx_in:.4f}, hz={hz_in:.4f}")
    print(f"Out: hx={hx_out:.4f}, hz={hz_out:.4f}")
    print(f"Diff: {abs(hx_in-hx_out):.4e}, {abs(hz_in-hz_out):.4e}")
    print("-" * 30)

print("--- Testing HMF Extraction Identity ---")
check_state(1.0, 0.0, 1.0)
check_state(0.0, 1.0, 1.0)
check_state(0.5, 0.5, 2.0)
check_state(0.0, 53.0, 2.0) # The large field case
