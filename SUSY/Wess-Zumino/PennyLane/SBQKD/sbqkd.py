import pennylane as qml
from pennylane import numpy as pnp
import os, json
import numpy as np
from collections import Counter
import pandas as pd
from datetime import datetime

from scipy.sparse.linalg import eigsh

import wesszumino as wz

import git
repo_path = git.Repo('.', search_parent_directories=True).working_tree_dir


if __name__ == "__main__":

    N = 3
    a = 1.0
    c = -0.2

    potential = "linear"
    #potential = 'quadratic'
    boundary_condition = 'dirichlet'
    #boundary_condition = 'periodic'

    cutoff = 8
    shots=4096

    #for shots in [500, 1000, 2000, 4000, 10000]:

    if potential == 'quadratic':
        folder = 'C' + str(abs(c)) + '/' + 'N'+ str(N)
    else:
        folder = 'N'+ str(N)

    H_path = os.path.join(repo_path, r"SUSY\Wess-Zumino\Analyses\Model Checks\HamiltonianData", boundary_condition, potential, folder, f"{potential}_{cutoff}.json")
    with open(H_path, 'r') as file:
        H_data = json.load(file)

    pauli_coeffs = H_data['pauli_coeffs']
    pauli_strings = H_data['pauli_terms']
    pauli_terms = [wz.pauli_str_to_op(t) for t in pauli_strings]

    num_qubits = H_data['num_qubits']

    dense_H_size = H_data['H_size']

    eigenvalues = H_data['eigenvalues']
    min_eigenvalue = np.min(eigenvalues)

    print(f"min_eigenvalue: {min_eigenvalue}")

    H_pauli = qml.Hamiltonian(pauli_coeffs, pauli_terms)
    pauli_terms = wz.pauli_terms_from_operator(H_pauli, wire_order=list(range(num_qubits)))

    nb = int(np.log2(cutoff))
    n = 1 + nb
    fw = [i * n for i in range(N)]

    pairs = [(fw[i], fw[i+1]) for i in range(len(fw)-1)]

    #Dirichlet-Linear
    #basis = [0]*n + [1] + [0]*nb #N2
    basis = [0]*n + [1] + [0]*nb + [0]*n #N3
    #basis = [0]*n + [1] + [0]*nb + [0]*n + [1] + [0]*nb #N4
    #basis = [0]*n + [1] + [0]*nb + [0]*n + [1] + [0]*nb + [0]*n #N5


    dev = qml.device("lightning.qubit", wires=num_qubits, shots=shots)
    @qml.qnode(dev)
    def circuit(t, n_steps):

        qml.BasisState(basis, wires=list(range(num_qubits)))

        qml.SingleExcitation(-0.44214317580171686, wires=[0,4])
        qml.SingleExcitation( 0.43171778953114925, wires=[4,8])
        qml.RY(-0.07876791939033408, wires=[2])
        qml.RY(-0.14272180107841284, wires=[6])
        qml.RY(-0.07877077330649465, wires =[10])

        #for pair in pairs:
            #qml.FermionicSingleExcitation(np.pi/2, wires=pair)
            #qml.CNOT(wires=pair)
            #qml.CRY(np.pi/2, wires=pair)

        qml.ApproxTimeEvolution(H_pauli, t, n_steps)

        return qml.counts(wires=list(range(num_qubits)))
    

    print(qml.draw(circuit)(1.0, 1))



    k=1
    n_steps=1
    dt=1.0
    max_k = 15
    tol = 1e-10

    converged=False
    samples = Counter()
    prev_energy = np.inf

    all_data = []
    all_energies = []

    while not converged and k <= max_k:

        print(f"Running for {k} Krylov dimension")

        t = dt*k

        t1 = datetime.now()
        counts = circuit(t=t, n_steps=n_steps)
        Ct = datetime.now() - t1
        samples.update(counts)


        sorted_states = sorted(samples.items(), key=lambda x: x[1], reverse=True)
        top_states = [s for s, c in sorted_states]
        idx = [int(s, 2) for s in top_states]

        H_reduced = wz.reduced_sparse_matrix_from_pauli_terms(pauli_terms, top_states)
    
        t1 = datetime.now()
        es = eigsh(H_reduced, k=1, which="SA", return_eigenvectors=False)
        HRt = datetime.now() - t1

        mi = np.argmin(es)
        me = es[mi].real

        diff_prev = np.abs(prev_energy-me)

        row = { "D": k,
                "t":t,
                "circuit_time": str(Ct),
                "num_samples": len(samples),
                "H_reduced_size": H_reduced.shape,
                "reduction": (1 - (H_reduced.shape[0] / dense_H_size[0]))*100,
                "H_reduced_e": me,
                "eigenvalue_time": str(HRt),
                "diff": np.abs(min_eigenvalue-me),
                "change_from_prev": None if diff_prev == np.inf else diff_prev
                }

        
        all_data.append(row)
        all_energies.append(me)

        converged = True if diff_prev < tol else False

        if converged == False and k == max_k: 
            print("max_k reached")
            break
        elif converged == False:
            prev_energy = me
            k+=1
        else:
            print(f"Converged")

    final_data = {
        "load_date": str(datetime.now()),
        "potential": potential,
        "cutoff": cutoff,
        "N": N,
        "a": a,
        "c": None if potential == "linear" else c,
        "boundary_condition": boundary_condition,
        "num_qubits": num_qubits,
        "num_paulis": len(pauli_strings),
        "dense_H_size": dense_H_size,
        "min_eigenvalue": min_eigenvalue,
        "basis": basis,
        "tol":tol,
        "dt":dt,
        "n_steps":n_steps,
        "max_k":max_k,
        "final_k": k,
        "converged": converged,
        "all_energies": all_energies,
        "all_run_data": all_data
    }


    folder_path = os.path.join(repo_path, r"SUSY\Wess-Zumino\PennyLane\SBQKD\basis", boundary_condition, potential, folder)
    os.makedirs(folder_path, exist_ok=True)

    with open(os.path.join(folder_path, f"{potential}_{cutoff}.json"), "w") as json_file:
        json.dump(final_data, json_file, indent=4)
        
        
