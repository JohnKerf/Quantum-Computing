import os, json
import numpy as np
from datetime import datetime
from scipy.sparse.linalg import eigsh

import susy_qm as sqm


def dense_storage_bytes(num_qubits: int, dtype=np.complex128) -> int:
    dim = 1 << num_qubits  # 2**num_qubits
    return dim * dim * np.dtype(dtype).itemsize

potential = "QHO"

cutoffs = [128,256]

for cutoff in cutoffs:

    print(f"Running for cutoff {cutoff}")
    print("Creating pauli Hamiltonian")

    t1 = datetime.now() 
    H_pauli, num_qubits, H_dense = sqm.build_sqm_hamiltonian_position_basis_pauli(
        cutoff=cutoff,
        potential=potential,
        m=1.0, g=1.0, u=1.0,
        x_max=8.0,
        atol_from_operator=1e-12,
    )
    Hp = datetime.now() - t1

    print("Converting to sparse matrix")
    t1 = datetime.now() 
    H = H_pauli.to_matrix(sparse=True)
    Hc = datetime.now() - t1

    dense_mem = dense_storage_bytes(num_qubits, dtype=np.complex128)
    sparse_mem = int(H.data.nbytes + H.indices.nbytes + H.indptr.nbytes)

    print("Finding eigenvalues")
    t1 = datetime.now()
    if H.shape[0] < 2000:
        H_dense = H.todense()
        eigenvalues, eigenvectors = np.linalg.eigh(H_dense)
        min_index = int(np.argmin(eigenvalues))
        eigenvalues = np.sort(eigenvalues)[:8]
        used_dense = True
    else:
        eigenvalues, eigenvectors = eigsh(H, k=8, which="SA", return_eigenvectors=True)
        min_index = int(np.argmin(eigenvalues))
        eigenvalues = np.sort(eigenvalues)
        used_dense = False
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
            "num_qubits": num_qubits,
            "H_size": H.shape,
            "H_dense_mem_estimate_bytes": dense_mem,
            "H_sparse_mem_estimate_bytes": sparse_mem,
            "H_pauli_time": str(Hp),
            "H_creation_time": str(Hc),
            "H_eigenvalue_time": str(Ht),
            "used_dense": used_dense,
            "eigenvalues": [x.real.tolist() for x in eigenvalues],
            "best_basis_state": basis_state0,
            "basis_prob": float(probs[idx0]),
            "num_paulis": len(H_pauli),
            "pauli_coeffs": np.real(H_pauli.coeffs).astype(float).tolist(),
            "pauli_labels": H_pauli.paulis.to_labels()
            }

    dir_path = os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\SUSY QM\Analyses\Model Checks\HamiltonianData", potential)
    os.makedirs(dir_path, exist_ok=True)

    with open(os.path.join(dir_path,f"{potential}_{cutoff}.json"), "w") as json_file:
        json.dump(data, json_file, indent=4)

    print("Done")
