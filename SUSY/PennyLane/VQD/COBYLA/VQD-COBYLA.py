# PennyLane imports
import pennylane as qml
from pennylane import numpy as pnp

from scipy.optimize import minimize

# General imports
import os
import json
import numpy as np
from datetime import datetime

from qiskit.quantum_info import SparsePauliOp

# custom module
from susy_qm import calculate_Hamiltonian, create_vqd_plots


potential = 'QHO'
#potential = 'AHO'
#potential = 'DW'

cut_offs_list = [2,4,8,16]#,32]

starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder = str(starttime)

#Create directory for files
base_path = r"C:\Users\johnkerf\Desktop\Quantum-Computing\Quantum-Computing\SUSY\PennyLane\VQD\COBYLA\Files\{}\\{}\\".format(potential, folder)
#base_path = r"C:\Users\Johnk\OneDrive\Desktop\PhD 2024\Quantum Computing Code\Quantum-Computing\SUSY\PennyLane\VQD\COBYLA\Files\{}\\{}\\".format(potential, folder)
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


    #Initial params shape
    num_layers = 1
    params_shape = qml.StronglyEntanglingLayers.shape(n_layers=num_layers, n_wires=num_qubits)

    def ansatz(params, wires):
        params = pnp.tensor(params.reshape(params_shape), requires_grad=True)
        qml.StronglyEntanglingLayers(weights=params, wires=wires, imprimitive=qml.CZ)

    # Device
    shots = None
    dev = qml.device('lightning.qubit', wires=2*num_qubits+1, shots=shots)

    #Swap test to calculate overlap
    @qml.qnode(dev)
    def swap_test(params1, params2):

        params1 = pnp.tensor(params1.reshape(params_shape), requires_grad=True)
        params2 = pnp.tensor(params2.reshape(params_shape), requires_grad=True)

        ancilla = 2*num_qubits
        qml.Hadamard(wires=ancilla)

        ansatz(params1, wires=range(num_qubits))
        ansatz(params2, wires=range(num_qubits, 2*num_qubits))

        qml.Barrier()
        for i in range(num_qubits):
            qml.CSWAP(wires=[ancilla, i, num_qubits + i])

        qml.Hadamard(wires=ancilla)

        return qml.probs(wires=ancilla)
    

    # Device
    shots = None
    dev2 = qml.device('lightning.qubit', wires=num_qubits, shots=shots)

    @qml.qnode(dev2)
    def expected_value(params):
        wires = range(num_qubits)
        ansatz(params, wires)
        return qml.expval(qml.Hermitian(H, wires=range(num_qubits)))


    def loss_f(params):
        energy = expected_value(params)

        penalty = 0

        if len(prev_param_list) != 0:
            for prev_param in prev_param_list:
                overlap =  1 - (2*swap_test(prev_param, params)[1])
                penalty += (beta*overlap)

        return energy + (penalty)
    

    def callback(xk):
        global iteration_counter
        current_cost = loss_f(xk)

        iteration_counter += 1
        counts.append(iteration_counter) 
        values.append(current_cost)
    
    
    #VQD
    vqd_start = datetime.now()

    #variables
    num_vqd_runs = 4
    if num_vqd_runs > len(eigenvalues):
        print("num_vqd_runs is greater than number of eigenvalues")
        raise
    
    max_iterations = 10000
    beta = 2.0
    tolerance = 1e-4

    #data arrays
    energies = []
    prev_param_list = []
    num_iters = []
    run_times = []
    success = []
    run_times = []

    all_counts = []
    all_values = []
    all_overlaps = []

    #Initial params
    scale = 0.5
    params_shape = qml.StronglyEntanglingLayers.shape(n_layers=1, n_wires=num_qubits)
    params = scale*np.pi * pnp.random.random(size=np.prod(params_shape))

    for i in range(num_vqd_runs):

        run_start = datetime.now()

        iteration_counter = 0
        counts = []
        values = []    

        if i % 1 == 0:
            print(f"Run: {i}")


        res = minimize(loss_f,
                    x0=params,
                    method='COBYLA',
                    options={'maxiter': max_iterations, 'tol':tolerance},
                    callback=callback)
        
        all_counts.append(counts)
        all_values.append(values)        


        energies.append(res.fun)
        prev_param_list.append(res.x) 
        num_iters.append(res.nfev)
        success.append(res.success)

        run_end = datetime.now()
        run_time = run_end - run_start
        run_times.append(run_time)

        # Calculate overlap with all previous states
        overlaps = []
        if i > 0:
            for j in range(i):
                overlap =  1 - (2*swap_test(prev_param_list[j], params)[1])
                overlaps.append(overlap)
            all_overlaps.append(overlaps)

    vqd_end = datetime.now()
    vqd_time = vqd_end - vqd_start


    #Save run
    run = {
        'potential': potential,
        'cutoff': cut_off,
        'exact_eigenvalues': [round(x.real,10).tolist() for x in eigenvalues],
        'ansatz': 'StronglyEntanglingLayers-1layer',
        'num_VQD': num_vqd_runs,
        'Optimizer': {'name': 'COBYLA',
                    'maxiter':max_iterations,
                    'tolerance': tolerance
                    },
        'converged_energies': energies,
        'converged_params': [x.tolist() for x in prev_param_list],
        'energies': all_values,
        'num_iters': all_counts,
        'success': np.array(success, dtype=bool).tolist(),
        'overlaps': all_overlaps,
        'run_times': [str(x) for x in run_times],
        'total_run_time': str(vqd_time)
    }

    # Save the variable to a JSON file
    path = base_path + "{}_{}.json".format(potential, cut_off)
    with open(path, 'w') as json_file:
        json.dump(run, json_file, indent=4)

create_vqd_plots(potential=potential, base_path=base_path, cut_off_list=cut_offs_list)

