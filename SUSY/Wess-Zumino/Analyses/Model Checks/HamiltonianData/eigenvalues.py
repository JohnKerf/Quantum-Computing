import pennylane as qml
import os, json
import numpy as np
from datetime import datetime

from wesszumino import build_wz_hamiltonian

import git
repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_tree_dir


import pennylane as qml
import os, json
import numpy as np
from datetime import datetime
from scipy.sparse.linalg import eigsh

from wesszumino import build_wz_hamiltonian
from susy_qm import calculate_wz_hamiltonian


N = 5
a = 1.0
c = -0.2

#potential = "linear"
potential = 'quadratic'
#boundary_condition = 'dirichlet'
boundary_condition = 'periodic'

cutoffs = [2,4,8,16]

for cutoff in cutoffs:

    print(f"Running for cutoff {cutoff}")
    print("Creating pauli Hamiltonian")

    #num_qubits = int(1+np.log2(cutoff)) * N

    t1 = datetime.now() 
    #H = calculate_wz_hamiltonian(cutoff, N, a, potential, boundary_condition, c)
    H_pauli, num_qubits = build_wz_hamiltonian(
        cutoff,
        N,
        a,
        c=c,
        m=1.0,
        potential=potential,
        boundary_condition=boundary_condition,
        remove_zero_terms=True
    )
    Hp = datetime.now() - t1

    print("Converting to dense")
    t1 = datetime.now() 
    #H = qml.matrix(H_pauli, wire_order=list(range(num_qubits)))
    H = H_pauli.sparse_matrix(
        wire_order=list(range(num_qubits)),
        format="csr"
    )
    #H = qml.matrix(H_pauli, wire_order=list(range(num_qubits)), sparse=True)
    #H_pauli = qml.pauli_decompose(H, wire_order=list(range(num_qubits))).simplify()
    Hc = datetime.now() - t1

    print("Finding eigenvalues")
    t1 = datetime.now()
    #eigenvalues = np.sort(np.linalg.eig(H)[0])[:8]
    #eigenvalues = np.sort(np.linalg.eigvalsh(H))[:8]
    eigenvalues = eigsh(H, k=8, which="SA", return_eigenvectors=False)
    eigenvalues = np.sort(eigenvalues)
    Ht = datetime.now() - t1

    print("Saving data")
    data  = {"load_date": str(datetime.now()),
            "potential": potential,
            "cutoff": cutoff,
            "N": N,
            "a": a,
            "c": None if potential == "linear" else c,
            "boundary_condition": boundary_condition,
            "num_qubits": num_qubits,
            "H_size": H.shape,
            "H_pauli_time": str(Hp),
            "H_creation_time": str(Hc),
            "H_eigenvalue_time": str(Ht),
            "eigenvalues": [x.real.tolist() for x in eigenvalues],
            "num_paulis": len(H_pauli),
            "pauli_coeffs": H_pauli.coeffs,
            "pauli_terms": [str(op) for op in H_pauli.ops]
            }

    if potential == 'quadratic':
        folder = 'C' + str(abs(c)) + '/' + 'N'+ str(N)
    else:
        folder = 'N'+ str(N)


    dir_path = os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\Wess-Zumino\Analyses\Model Checks\HamiltonianData", 
                        boundary_condition, potential, 
                        folder
                        )
    os.makedirs(dir_path, exist_ok=True)

    with open(os.path.join(dir_path,"data.json"), "w") as json_file:
        json.dump(data, json_file, indent=4)
