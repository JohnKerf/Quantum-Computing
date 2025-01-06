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
import time

from qiskit.quantum_info import SparsePauliOp

# custom module
from susy_qm import calculate_Hamiltonian, create_vqd_plots

shots = None
potential = 'QHO'
#potential = 'AHO'
#potential = 'DW'

cut_offs_list = [16]

starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder = str(starttime)

#Create directory for files
#base_path = r"C:\Users\johnkerf\Desktop\Quantum-Computing\Quantum-Computing\SUSY\PennyLane\VQD\COBYLA\Files\{}\\{}\\".format(potential, folder)
base_path = r"C:\Users\Johnk\OneDrive\Desktop\PhD 2024\Quantum Computing Code\Quantum-Computing\SUSY\SUSY QM\PennyLane\VQD\Differential Evolution\TestFiles\{}\\{}\\".format(potential, folder)
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

    swap_time = timedelta()
    #Swap test to calculate overlap
    @qml.qnode(dev)
    def swap_test(params1, params2):

        global swap_time
        start = datetime.now()
        #params1 = pnp.tensor(params1.reshape(params_shape), requires_grad=True)
        #params2 = pnp.tensor(params2.reshape(params_shape), requires_grad=True)

        ancilla = 2*num_qubits
        qml.Hadamard(wires=ancilla)

        ansatz(params1, wires=range(num_qubits))
        ansatz(params2, wires=range(num_qubits, 2*num_qubits))

        qml.Barrier()
        for i in range(num_qubits):
            qml.CSWAP(wires=[ancilla, i, num_qubits + i])

        qml.Hadamard(wires=ancilla)

        prob = qml.probs(wires=ancilla)

        end = datetime.now()
        swap_time += (end - start)

        return prob#qml.probs(wires=ancilla)
    
    multi_swap_time = timedelta()
    def multi_swap_test(params1_list, params2_list, P):
    
        global multi_swap_time
        
        results = []
        for _ in range(P):

            start = datetime.now()

            prob = swap_test(params1_list, params2_list)

            end = datetime.now()
            multi_swap_time += (end - start)

            results.append(prob[0])
        
        avg_prob = sum(results) / P
        overlap = 2 * avg_prob - 1

        return overlap
    
    eval_time = timedelta()
    @qml.qnode(dev)
    def expected_value(params):

        global eval_time
        start = datetime.now()

        wires = range(num_qubits)
        ansatz(params, wires)

        exval = qml.expval(qml.Hermitian(H, wires=range(num_qubits)))

        end = datetime.now()
        eval_time += (end - start)

        return exval#qml.expval(qml.Hermitian(H, wires=range(num_qubits)))


    device_time = timedelta()
    def loss_f(params, num_swap_tests):

        global device_time
        start = datetime.now()

        energy = expected_value(params)

        penalty = 0
        if len(prev_param_list) > 0:
            for prev_param in prev_param_list:
                overlap = multi_swap_test(prev_param, params, num_swap_tests)
                penalty += (beta*overlap)

        end = datetime.now()
        device_time += (end - start)

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
    num_vqd_runs = 1
    num_energy_levels = 1
    num_swap_tests = 1

    if num_energy_levels > len(eigenvalues):
        print("num_vqd_runs is greater than number of eigenvalues")
        raise

    max_iter = 500 #calcs per iteration = popsize*num_params
    beta = 2.0
    strategy = 'randtobest1bin'
    tol = 1e-2
    abs_tol = 1e-3
    popsize = 20

    #data arrays
    all_energies = []
    all_params = []
    all_success = []
    run_times = []
    all_num_iters = []
    all_evaluations = []
    seeds = []


    for i in range(num_vqd_runs):

        #seed
        seed = (os.getpid() * int(time.time())) % 123456789
        run_start = datetime.now()
        seeds.append(seed)

        # Generate Halton sequence
        num_dimensions = np.prod(params_shape)
        num_samples = popsize
        halton_sampler = Halton(d=num_dimensions, seed=seed)
        halton_samples = halton_sampler.random(n=num_samples)
        scaled_samples = 2 * np.pi * halton_samples

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
            res = differential_evolution(
                lambda params: loss_f(params, num_swap_tests),
                bounds,
                maxiter=max_iter,
                tol=tol,
                atol=abs_tol,
                strategy=strategy,
                popsize=popsize,
                init=scaled_samples,
                seed=seed
            )
                

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
        run_times.append(str(run_time))

    vqd_end = datetime.now()
    vqd_time = vqd_end - vqd_start


    #Save run
    print("Saving data")

    run = {
        "starttime": starttime,
        "potential": potential,
        "cutoff": cut_off,
        "exact_eigenvalues": [x.real.tolist() for x in eigenvalues],
        "ansatz": "StronglyEntanglingLayers-1layer",
        "num_VQD": num_vqd_runs,
        "num_energy_levels": num_energy_levels,
        "num_swap_tests": num_swap_tests,
        "beta": beta,
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
        "results": all_energies,
        "params": [[x.tolist() for x in param_list] for param_list in all_params],
        "num_iters": all_num_iters,
        "num_evaluations": all_evaluations,
        "success": [np.array(x, dtype=bool).tolist() for x in all_success],
        "run_times": run_times,
        "seeds": seeds,
        "total_run_time": str(vqd_time),
        "device_time": str(device_time),
        "swap_time": str(swap_time),
        "multi_swap_time": str(multi_swap_time),
        "eval_time": str(eval_time)
    }

    # Save the variable to a JSON file
    path = base_path + "{}_{}.json".format(potential, cut_off)
    with open(path, 'w') as json_file:
        json.dump(run, json_file, indent=4)

    # Plot
    #create_vqd_plots(data=run, path=base_path)

