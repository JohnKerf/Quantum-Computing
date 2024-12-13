# PennyLane imports
import pennylane as qml
from pennylane import numpy as pnp

from scipy.optimize import differential_evolution
from scipy.stats.qmc import Halton

# General imports
import os
import json
import numpy as np
from datetime import datetime, timedelta

from qiskit.quantum_info import SparsePauliOp

# custom module
from susy_qm import calculate_Hamiltonian


potential = 'QHO'
#potential = 'AHO'
#potential = 'DW'

#cut_offs_list = [2,4,8,16]#,32]
cut_offs_list = [2]
shots = 1024


starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder = str(starttime)
#Create directory for files
base_path = r"C:\Users\Johnk\OneDrive\Desktop\PhD 2024\Quantum Computing Code\Quantum-Computing\SUSY\SUSY QM\PennyLane\VQE\Differential Evolution\Files\{}\\{}\\".format(potential, folder)
os.makedirs(base_path)

print(f"Running for {potential} potential")

for cut_off in cut_offs_list:

    print(f"Running for cutoff: {cut_off}")

    #calculate Hamiltonian and expected eigenvalues
    H = calculate_Hamiltonian(cut_off, potential)
    eigenvalues = np.sort(np.linalg.eig(H)[0])
    min_eigenvalue = min(eigenvalues.real)

    #create qiskit Hamiltonian Pauli string
    hamiltonian = SparsePauliOp.from_operator(H)
    num_qubits = hamiltonian.num_qubits


    # Device
    shots = shots
    #dev = qml.device('default.qubit', wires=num_qubits, shots=1024)
    dev = qml.device('lightning.qubit', wires=num_qubits, shots=shots)


    #Initial params shape
    num_layers = 1
    params_shape = qml.StronglyEntanglingLayers.shape(n_layers=num_layers, n_wires=num_qubits)

    device_time = timedelta()

    # Define the cost function
    @qml.qnode(dev)
    def cost_function(params):

        global device_time
        start = datetime.now()

        params = pnp.tensor(params.reshape(params_shape), requires_grad=True)
        qml.StronglyEntanglingLayers(weights=params, wires=range(num_qubits), imprimitive=qml.CZ)

        end = datetime.now()
        device_time += (end - start)

        return qml.expval(qml.Hermitian(H, wires=range(num_qubits)))
    
    
    # VQE
    vqe_start = datetime.now()

    #variables
    num_vqe_runs = 1
    max_iter = 10000
    strategy = "randtobest1bin"
    tol = 1e-3
    abs_tol = 1e-3
    popsize = 20

    # Generate Halton sequence
    num_dimensions = np.prod(params_shape)
    num_samples = popsize
    halton_sampler = Halton(d=num_dimensions)
    halton_samples = halton_sampler.random(n=num_samples)
    scaled_samples = 2 * np.pi * halton_samples

    #data arrays
    energies = []
    x_values = []
    success = []
    run_times = []
    num_iters = []
    num_evaluations = []

    #Optimizer
    bounds = [(0, 2 * np.pi) for _ in range(np.prod(params_shape))]

    for i in range(num_vqe_runs):

        run_start = datetime.now()

        if i % 10 == 0:
            print(f"Run: {i}")

        # Differential Evolution optimization
        res = differential_evolution(cost_function,
                                    bounds,
                                    maxiter=max_iter,
                                    tol=tol,
                                    atol=abs_tol,
                                    strategy=strategy,
                                    popsize=popsize,
                                    init=scaled_samples,
                                    )
        
        if res.success == False:
            print("Not converged")

        energies.append(res.fun)
        x_values.append(res.x)
        success.append(res.success)
        num_iters.append(res.nit)
        num_evaluations.append(res.nfev)

        run_end = datetime.now()
        run_time = run_end - run_start
        run_times.append(str(run_time))

    vqe_end = datetime.now()
    vqe_time = vqe_end - vqe_start

    # Save run
    run = {
        "starttime": starttime,
        "potential": potential,
        "cutoff": cut_off,
        "exact_eigenvalues": [x.real.tolist() for x in eigenvalues],
        "ansatz": "StronglyEntanglingLayers-1layer",
        "num_VQE": num_vqe_runs,
        "shots": shots,
        "Optimizer": {
            "name": "differential_evolution",
            "bounds": "[(0, 2 * np.pi) for _ in range(np.prod(params_shape))]",
            "maxiter": max_iter,
            "tolerance": tol,
            "abs_tolerance": abs_tol,
            "strategy": strategy,
            "popsize": popsize,
            'init': 'scaled_samples',
        },
        "results": energies,
        "params": [x.tolist() for x in x_values],
        "num_iters": num_iters,
        "num_evaluations": num_evaluations,
        "success": np.array(success, dtype=bool).tolist(),
        "run_times": run_times,
        "total_device_time": str(device_time),
        "total_run_time": str(vqe_time),
    }

    # Save the variable to a JSON file
    path = base_path + "{}_{}.json".format(potential, cut_off)
    with open(path, 'w') as json_file:
        json.dump(run, json_file, indent=4)


