import logging
from hamiltonian_testing import build_wz_hamiltonian
from datetime import datetime
from scipy.sparse.linalg import eigsh
import numpy as np

H_pauli, n_qubits = build_wz_hamiltonian(
    cutoff=16,
    N=3,
    a=1.0,
    potential="linear",
    boundary_condition="dirichlet",
    log_path=f"packages/wesszumino/wesszumino/test/hamiltonian_build.log",
    log_level=logging.DEBUG,      # DEBUG includes Pauli lists
    pauli_max_terms=200,          # cap terms per printed block
    log_running_total=False,      
)

H = H_pauli.to_matrix(sparse=True)
eigenvalues, eigenvectors = eigsh(H, k=8, which="SA", return_eigenvectors=True)
min_index = int(np.argmin(eigenvalues))
eigenvalues = np.sort(eigenvalues)

print(eigenvalues)
