# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 00:07:44 2026

@author: dimit
"""

import numpy as np

def test_Q_power(P, n_procedures=10000, tol=1e-10):
    """
    Test whether Q^N is negligible for the given transition matrix P.
    
    Reports:
    - Eigenvalues of Q (all must be < 1 in magnitude)
    - Spectral radius of Q
    - Frobenius norm of Q^N
    - Comparison of exact vs approximate absorption probability
    """
    Q = P[:4, :4]
    R = P[:4, 4]
    
    # Eigenvalue check
    eigvals = np.linalg.eigvals(Q)
    spectral_radius = np.max(np.abs(eigvals))
    print(f"Eigenvalues of Q: {np.round(eigvals, 6)}")
    print(f"Spectral radius:  {spectral_radius:.8f}  (must be < 1)")
    
    # Theoretical bound: ||Q^N|| <= spectral_radius^N
    bound = spectral_radius ** n_procedures
    print(f"Theoretical bound ||Q^N|| <= {bound:.2e}")
    
    # Fundamental matrix
    N_mat = np.linalg.inv(np.eye(4) - Q)
    
    # Exact: use matrix_power (feasible since Q is 4x4)
    Q_n = np.linalg.matrix_power(Q, n_procedures)
    norm_Qn = np.linalg.norm(Q_n, 'fro')
    print(f"||Q^N|| Frobenius: {norm_Qn:.2e}  (negligible if < {tol:.0e})")
    
    # Exact absorption probability
    p_exact = ((np.eye(4) - Q_n) @ N_mat @ R)[0]
    
    # Approximate (Q^N = 0)
    p_approx = (N_mat @ R)[0]
    
    print(f"\nAbsorption probability from S0:")
    print(f"  Exact   (finite N={n_procedures}): {p_exact:.6e}")
    print(f"  Approx  (Q^N = 0):                {p_approx:.6e}")
    print(f"  Relative error:                   {abs(p_exact-p_approx)/p_approx:.2e}")
    print(f"\nLog-risk exact:  {-np.log10(max(p_exact,  1e-15)):.4f}")
    print(f"Log-risk approx: {-np.log10(max(p_approx, 1e-15)):.4f}")
    
    approximation_safe = norm_Qn < tol
    print(f"\n{'✅ Q^N ≈ 0: approximation is safe' if approximation_safe else '❌ Q^N not negligible: use exact formula'}")
    
    return p_exact, p_approx, norm_Qn

# --- Run test with a representative RG-3 model ---
from testbio41 import BiosafetyModel

scenarios = [
    ("Worst maintenance",  dict(training=50, maint_freq=0.5, maint_adh=0.4, inspection=75)),
    ("Best maintenance",   dict(training=50, maint_freq=2.0, maint_adh=0.9, inspection=75)),
    ("True MCMC model",    dict(training=65, maint_freq=1.2, maint_adh=0.85, inspection=75)),
]

for name, params in scenarios:
    print(f"\n{'='*60}")
    print(f"Scenario: {name}")
    print('='*60)
    model = BiosafetyModel(**params)
    test_Q_power(model.P, n_procedures=10000)
