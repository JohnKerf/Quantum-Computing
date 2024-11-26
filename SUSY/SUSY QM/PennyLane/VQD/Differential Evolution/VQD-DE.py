# PennyLane imports
import pennylane as qml
from pennylane import numpy as pnp

from scipy.optimize import differential_evolution

# General imports
import os
import json
import numpy as np
from datetime import datetime

from qiskit.quantum_info import SparsePauliOp

# custom module
from susy_qm import calculate_Hamiltonian, create_vqd_plots

shots = 1024
potential = 'QHO'
#potential = 'AHO'
#potential = 'DW'

cut_offs_list = [16]

starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder = str(starttime)

#Create directory for files
#base_path = r"C:\Users\johnkerf\Desktop\Quantum-Computing\Quantum-Computing\SUSY\PennyLane\VQD\COBYLA\Files\{}\\{}\\".format(potential, folder)
base_path = r"C:\Users\Johnk\OneDrive\Desktop\PhD 2024\Quantum Computing Code\Quantum-Computing\SUSY\SUSY QM\PennyLane\VQD\Differential Evolution\Files\{}\\{}\\".format(potential, folder)
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
    shots = shots
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
    shots = shots
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
    

    def callback(xk, convergence):
        global iteration_counter
        current_cost = loss_f(xk)

        iteration_counter += 1
        counts.append(iteration_counter) 
        values.append(current_cost)
    
    
    #VQD
    vqd_start = datetime.now()

    #Optimizer
    bounds = [(0, 2 * np.pi) for _ in range(np.prod(params_shape))]

    #variables
    num_vqd_runs = 10
    max_iterations = 500 #calcs per iteration = popsize*num_params
    beta = 2.0
    tolerance = 1e-3
    strategy = 'best1bin'
    popsize = 15

    #data arrays
    all_energies = []
    all_params = []
    all_success = []
    run_times = []
    all_num_iters = []
    all_evaluations = []

    num_energy_levels = 3
    if num_energy_levels > len(eigenvalues):
        print("num_vqd_runs is greater than number of eigenvalues")
        raise
    
    for i in range(num_vqd_runs):

        if i % 1 == 0:
            print(f"VQD run: {i}")

        energies = []
        prev_param_list = []
        num_iters = []
        success = []
        num_evaluations = []

        run_start = datetime.now()

        for j in range(num_energy_levels):

            iteration_counter = 0
            counts = []
            values = []    

            if j % 1 == 0:
                print(f"Energy level: {j}")


            # Differential Evolution optimization
            res = differential_evolution(loss_f, 
                                            bounds, 
                                            maxiter=max_iterations, 
                                            atol=tolerance,
                                            strategy=strategy, 
                                            popsize=popsize,
                                            callback=callback)
                

            energies.append(res.fun)
            prev_param_list.append(res.x) 
            num_iters.append(res.nfev)
            success.append(res.success)
            num_evaluations.append(res.nfev)

        all_energies.append(energies)
        all_params.append(prev_param_list)
        all_success.append(success)
        all_num_iters.append(num_iters)
        all_evaluations.append(num_evaluations)

        run_end = datetime.now()
        run_time = run_end - run_start
        run_times.append(run_time)

    vqd_end = datetime.now()
    vqd_time = vqd_end - vqd_start


    #Save run
    print("Saving data")
    
    run = {
        'potential': potential,
        'cutoff': cut_off,
        'exact_eigenvalues': [round(x.real,10).tolist() for x in eigenvalues],
        'ansatz': 'StronglyEntanglingLayers-1layer',
        'shots': shots,
        'num_VQD': num_vqd_runs,
        'num_energy_levels': num_energy_levels,
        'Optimizer': {'name': 'differential_evolution',
                            'bounds':'[(0, 2 * np.pi) for _ in range(np.prod(params_shape))]',
                            'maxiter':max_iterations,
                            'tolerance': tolerance,
                            'strategy': strategy,
                            'popsize': popsize
                            },
        'converged_energies': all_energies,
        'converged_params': [x.tolist() for x in prev_param_list],
        'num_iters': all_num_iters,
        'evaluations': all_evaluations,
        'success': np.array(all_success, dtype=bool).tolist(),
        'run_times': [str(x) for x in run_times],
        'total_run_time': str(vqd_time)
    }

    # Save the variable to a JSON file
    path = base_path + "{}_{}.json".format(potential, cut_off)
    with open(path, 'w') as json_file:
        json.dump(run, json_file, indent=4)

    # Plot
    create_vqd_plots(data=run, path=base_path)

