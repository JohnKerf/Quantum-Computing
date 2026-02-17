import os, json
import numpy as np
from datetime import datetime
from scipy.sparse.linalg import eigsh

import wesszumino as wz


def dense_storage_bytes(num_qubits: int, dtype=np.complex128) -> int:
    dim = 1 << num_qubits  # 2**num_qubits
    return dim * dim * np.dtype(dtype).itemsize

N = 2
a = 1.0
c = -0.8

potential = "linear"
#potential = 'quadratic'
boundary_condition = 'dirichlet'
#boundary_condition = 'periodic'

cutoffs = [32,64]

for cutoff in cutoffs:

    print(f"Running for cutoff {cutoff}")
    print("Creating pauli Hamiltonian")

    t1 = datetime.now() 
    H_pauli, num_qubits = wz.build_wz_hamiltonian_position(cutoff=cutoff, N=N, a=a, c=c, potential=potential, boundary_condition=boundary_condition)
    Hp = datetime.now() - t1

    print("Converting to sparse matrix")
    t1 = datetime.now() 
    H = H_pauli.to_matrix(sparse=True)
    Hc = datetime.now() - t1

    dense_mem = dense_storage_bytes(num_qubits, dtype=np.complex128)
    sparse_mem = int(H.data.nbytes + H.indices.nbytes + H.indptr.nbytes)

    print("Finding eigenvalues")
    t1 = datetime.now()
    eigenvalues, eigenvectors = eigsh(H, k=8, which="SA", return_eigenvectors=True)
    min_index = int(np.argmin(eigenvalues))
    eigenvalues = np.sort(eigenvalues)
    Ht = datetime.now() - t1

    print("Finding min eigenvector basis")
    min_eigenvector = np.asarray(eigenvectors[:, min_index])

    probs = np.abs(min_eigenvector)**2
    idx0 = int(np.argmax(probs))
    n = int(np.log2(len(min_eigenvector)))
    bitstring0 = format(idx0, f"0{n}b")
    basis_state0 = [int(b) for b in bitstring0]

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
            "H_dense_mem_estimate_bytes": dense_mem,
            "H_sparse_mem_estimate_bytes": sparse_mem,
            "H_pauli_time": str(Hp),
            "H_creation_time": str(Hc),
            "H_eigenvalue_time": str(Ht),
            "eigenvalues": [x.real.tolist() for x in eigenvalues],
            "best_basis_state": basis_state0,
            "basis_prob": probs[idx0],
            "num_paulis": len(H_pauli),
            "pauli_coeffs": np.real(H_pauli.coeffs).astype(float).tolist(),
            "pauli_labels": H_pauli.paulis.to_labels()
            }

    if potential == 'quadratic':
        folder = 'C' + str(abs(c)) + '/' + 'N'+ str(N)
    else:
        folder = 'N'+ str(N)

    dir_path = os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\Wess-Zumino\Analyses\Model Checks\HamiltonianDataPosition", 
                        boundary_condition, potential, 
                        folder
                        )
    os.makedirs(dir_path, exist_ok=True)

    with open(os.path.join(dir_path,f"{potential}_{cutoff}.json"), "w") as json_file:
        json.dump(data, json_file, indent=4)

    print("Done")
