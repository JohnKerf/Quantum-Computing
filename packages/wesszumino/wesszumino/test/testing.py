import logging
from hamiltonian_testing import build_wz_hamiltonian
from datetime import datetime


H, n_qubits = build_wz_hamiltonian(
    cutoff=2,
    N=3,
    a=1.0,
    potential="linear",
    boundary_condition="dirichlet",
    log_path=f"packages/wesszumino/wesszumino/test/hamiltonian_build.log",
    log_level=logging.DEBUG,      # DEBUG includes Pauli lists
    pauli_max_terms=200,          # cap terms per printed block
    log_running_total=False,      
)
