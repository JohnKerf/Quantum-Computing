import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import differential_evolution
from scipy.stats.qmc import Halton
import os
import json
import numpy as np
from datetime import datetime, timedelta
import time
from qiskit.quantum_info import SparsePauliOp
from susy_qm import calculate_Hamiltonian, create_vqd_plots

shots = 1024
potential = 'AHO'  # Change as needed
cut_offs_list = [16]

starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder = str(starttime)

base_path = os.path.join(r"C:\Users\Johnk\Documents\Quantum Computing Code\Quantum-Computing\SUSY\SUSY QM\PennyLane\SSVQE\Files", potential, folder)
os.makedirs(base_path, exist_ok=True)

print(f"Running SSVQE for {potential} potential")

for cut_off in cut_offs_list:
    print(f"Running for cutoff: {cut_off}")

    H = calculate_Hamiltonian(cut_off, potential)
    eigenvalues = np.sort(np.linalg.eig(H)[0])[:3]
    min_eigenvalue = min(eigenvalues.real)

    hamiltonian = SparsePauliOp.from_operator(H)
    num_qubits = hamiltonian.num_qubits

    num_layers = 1
    params_shape = qml.StronglyEntanglingLayers.shape(n_layers=num_layers, n_wires=num_qubits)

    def ansatz(params, wires):
        params = pnp.tensor(params.reshape(params_shape), requires_grad=True)
        qml.StronglyEntanglingLayers(weights=params, wires=wires, imprimitive=qml.CZ)

    dev = qml.device('default.qubit', wires=num_qubits, shots=shots)

    @qml.qnode(dev)
  
    def expected_value(params, phi):
        qml.StatePrep(phi, wires=range(num_qubits))
        ansatz(params, wires=range(num_qubits))
        return qml.expval(qml.Hermitian(H, wires=range(num_qubits)))


    def loss_f_ssvqe(params):
        cost = 0
        for phi in input_states:
            cost += expected_value(params, phi)
        return cost

    input_states = []
    for j in range(len(eigenvalues)):
        state = np.zeros(2**num_qubits)
        state[j] = 1.0
        input_states.append(state)

    bounds = [(0, 2 * np.pi) for _ in range(np.prod(params_shape))]
    max_iter = 500
    popsize = 20
    tol = 1e-2
    abs_tol = 1e-3

    seed = (os.getpid() * int(time.time())) % 123456789
    halton_sampler = Halton(d=np.prod(params_shape), seed=seed)
    halton_samples = 2 * np.pi * halton_sampler.random(n=popsize)

    print("Starting SSVQE optimization")
    res = differential_evolution(loss_f_ssvqe, bounds, maxiter=max_iter, tol=tol, atol=abs_tol, popsize=popsize, init=halton_samples, seed=seed)

    optimized_params = res.x
    optimized_energies = [expected_value(optimized_params, phi) for phi in input_states]

    print("Optimization complete")

    run = {
        "starttime": starttime,
        "potential": potential,
        "cutoff": cut_off,
        "exact_eigenvalues": [x.real.tolist() for x in eigenvalues],
        "ansatz": "StronglyEntanglingLayers-1layer",
        "optimized_energies": [float(e) for e in optimized_energies],
        "params": optimized_params.tolist(),
        "success": res.success,
        "iterations": res.nit,
        #"total_run_time": str(res.execution_time),
    }

    path = os.path.join(base_path, "{}_{}.json".format(potential, cut_off))
    with open(path, 'w') as json_file:
        json.dump(run, json_file, indent=4)

    print(f"Results saved to {path}")
