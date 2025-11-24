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

from wesszumino import build_wz_hamiltonian, pauli_str_to_op



N = 5
a = 1.0
c = -0.2

#potential = "linear"
potential = 'quadratic'
boundary_condition = 'dirichlet'
#boundary_condition = 'periodic'

cutoffs = [2,4]

if potential == 'quadratic':
    folder = 'C' + str(abs(c)) + '/' + 'N'+ str(N)
else:
    folder = 'N'+ str(N)

for cutoff in cutoffs:

    print(f"Running for cutoff {cutoff}")

    H_path = os.path.join(repo_path, r"SUSY\Wess-Zumino\Analyses\Model Checks\HamiltonianData", boundary_condition, potential, folder, f"{potential}_{cutoff}.json")
    with open(H_path, 'r') as file:
        H_data = json.load(file)

    pauli_coeffs = H_data['pauli_coeffs']
    pauli_strings = H_data['pauli_terms']
    pauli_terms = [pauli_str_to_op(t) for t in pauli_strings]

    num_qubits = H_data['num_qubits']

    H_pauli = qml.Hamiltonian(pauli_coeffs, pauli_terms)

    H = H_pauli.sparse_matrix(
        wire_order=list(range(num_qubits)),
        format="csr"
    )

    eigenvalues, eigenvectors = eigsh(H, k=8, which="SA", return_eigenvectors=True)
    min_index = np.argmin(eigenvalues)
    min_eigenvector = np.asarray(eigenvectors[:, min_index])

    probs = np.abs(min_eigenvector)**2
    idx0 = int(np.argmax(probs))
    n = int(np.log2(len(min_eigenvector)))

    bitstring0 = format(idx0, f"0{n}b")
    basis_state0 = [int(b) for b in bitstring0]

    print("best index:", idx0)
    print("best bitstring:", bitstring0)
    print("basis_state list:", basis_state0)
    print("prob:", probs[idx0])

    print("Saving data")
    data  = {"load_date": str(datetime.now()),
            "potential": potential,
            "cutoff": cutoff,
            "N": N,
            "a": a,
            "c": None if potential == "linear" else c,
            "boundary_condition": boundary_condition,
            "num_qubits": num_qubits,
            "eigenvalues": [x.real.tolist() for x in eigenvalues],
            "best_basis_state": basis_state0,
            "prob": probs[idx0]
            }


    if potential == 'quadratic':
        folder = 'C' + str(abs(c)) + '/' + 'N'+ str(N)
    else:
        folder = 'N'+ str(N)


    dir_path = os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\Wess-Zumino\Analyses\Model Checks\HamiltonianData\Eigenvectors", 
                        boundary_condition, potential, 
                        folder
                        )
    os.makedirs(dir_path, exist_ok=True)

    np.save(os.path.join(dir_path,f"{potential}_{cutoff}.npy"), min_eigenvector) 
    with open(os.path.join(dir_path,f"{potential}_{cutoff}.json"), "w") as json_file:
        json.dump(data, json_file, indent=4)

