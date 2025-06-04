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

from multiprocessing import Pool

from susy_qm import calculate_Hamiltonian


def cost_function(params, prev_param_list, H, params_shape, num_qubits, shots, beta, num_swap_tests):
   
    dev = qml.device("default.qubit", wires=2*num_qubits+1, shots=shots)

    def ansatz(params, wires):
        params_idx=0
        for i in wires:
            qml.RY(params[params_idx], wires=[i])
            params_idx +=1
     

    #Swap test to calculate overlap
    @qml.qnode(dev)
    def swap_test(params1, params2):

        start = datetime.now()

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
        global swap_time
        swap_time = (end - start)

        return prob
    
    
    def multi_swap_test(params1_list, params2_list, P):
    
        multi_swap_time = timedelta()

        results = []
        for i in range(num_swap_tests):
            prob = swap_test(params1_list, params2_list)
            results.append(prob[0])  # Probability of ancilla qubit in |0>
            multi_swap_time += swap_time
        
        avg_prob = sum(results) / P
        overlap = 2 * avg_prob - 1

        return overlap, multi_swap_time

    dev = qml.device("default.qubit", wires=2*num_qubits+1, shots=shots)
    @qml.qnode(dev)
    def expected_value(params):

        start = datetime.now()

        wires = range(num_qubits)
        ansatz(params, wires)
        exval = qml.expval(qml.Hermitian(H, wires=range(num_qubits)))

        end = datetime.now()
        global expval_time
        expval_time = (end - start)

        return exval

    def loss_f(params):

        total_swap_time = timedelta()

        energy = expected_value(params)

        penalty = 0

        if len(prev_param_list) != 0:
            for prev_param in prev_param_list:
                overlap, tst = multi_swap_test(prev_param, params, num_swap_tests)
                penalty += (beta*overlap)
                total_swap_time += tst

        device_time = total_swap_time + expval_time

        return energy + (penalty), device_time

    return loss_f(params)


def run_vqd(i, bounds, max_iter, tol, abs_tol, strategy, popsize, H, params_shape, num_qubits, shots, num_energy_levels, beta, num_swap_tests):
    
    # We need to generate a random seed for each process otherwise each parallelised run will have the same result
    seed = (os.getpid() * int(time.time())) % 123456789
    run_start = datetime.now()

    # Generate Halton sequence
    num_dimensions = np.prod(params_shape)
    num_samples = popsize
    halton_sampler = Halton(d=num_dimensions, seed=seed)
    halton_samples = halton_sampler.random(n=num_samples)
    scaled_samples = 2 * np.pi * halton_samples

    all_energies = []
    prev_param_list = []
    all_success = []
    all_num_iters = []
    all_evaluations = []
    all_dev_times = []

    device_time = timedelta()

    def wrapped_cost_function(params):
        result, dt = cost_function(params, prev_param_list, H, params_shape, num_qubits, shots, beta, num_swap_tests)
        nonlocal device_time
        device_time += dt
        all_dev_times.append(dt)
        return result

    for _ in range(num_energy_levels):
        
        # Differential Evolution optimization
        res = differential_evolution(
            wrapped_cost_function,
            bounds,
            maxiter=max_iter,
            tol=tol,
            atol=abs_tol,
            strategy=strategy,
            popsize=popsize,
            init=scaled_samples,
            seed=seed
        )


        all_energies.append(res.fun)
        prev_param_list.append(res.x)
        all_success.append(res.success)
        all_num_iters.append(res.nit)
        all_evaluations.append(res.nfev)

    run_end = datetime.now()
    run_time = run_end - run_start
    

    return {
        "seed": seed,
        "energies": all_energies,
        "params": prev_param_list,
        "success": all_success,
        "num_iters": all_num_iters,
        "evaluations": all_evaluations,
        "run_time": run_time,
        "device_time": device_time
    }


if __name__ == "__main__":
    
    potential_list = ["QHO"]#, "AHO", "DW"]
    cut_offs_list = [16]
    shots = None

    for potential in potential_list:

        starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        #base_path = os.path.join("/users/johnkerf/SUSY/VQD/QM/Files", potential)
        base_path = os.path.join(r"C:\Users\Johnk\OneDrive\Desktop\PhD 2024\Quantum Computing Code\Quantum-Computing\SUSY\SUSY QM\PennyLane\VQD\Differential Evolution\ParaTestFiles", potential)
        os.makedirs(base_path, exist_ok=True)

        print(f"Running for {potential} potential")

        for cut_off in cut_offs_list:

            print(f"Running for cutoff: {cut_off}")

            # Calculate Hamiltonian and expected eigenvalues
            H = calculate_Hamiltonian(cut_off, potential)
            eigenvalues = np.sort(np.linalg.eig(H)[0])
            min_eigenvalue = min(eigenvalues.real)

            # Create qiskit Hamiltonian Pauli string
            hamiltonian = SparsePauliOp.from_operator(H)
            num_qubits = hamiltonian.num_qubits

            # Initial params shape
            num_layers = 1
            params_shape = qml.StronglyEntanglingLayers.shape(n_layers=num_layers, n_wires=num_qubits)

            # Optimizer
            bounds = [(0, 2 * np.pi) for _ in range(np.prod(params_shape))]

            num_vqd_runs = 1
            num_energy_levels = 3
            beta = 2.0
            num_swap_tests = 1

            max_iter = 500
            strategy = "randtobest1bin"
            tol = 1e-2
            abs_tol = 1e-3
            popsize = 20

            # Start multiprocessing for VQE runs
            with Pool(processes=1) as pool:
                vqd_results = pool.starmap(
                    run_vqd,
                    [
                        (i, bounds, max_iter, tol, abs_tol, strategy, popsize, H, params_shape, num_qubits, shots, num_energy_levels, beta, num_swap_tests)
                        for i in range(num_vqd_runs)
                    ],
                )

            # Collect results
            seeds = [res["seed"] for res in vqd_results]
            all_energies = [result["energies"] for result in vqd_results]
            all_params = [result["params"] for result in vqd_results]
            all_success = [result["success"] for result in vqd_results]
            all_num_iters = [result["num_iters"] for result in vqd_results]
            all_evaluations = [result["evaluations"] for result in vqd_results]
            run_times = [str(res["run_time"]) for res in vqd_results]
            total_run_time = sum([res["run_time"] for res in vqd_results], timedelta())
            total_device_time = sum([res['device_time'] for res in vqd_results], timedelta())

            vqd_end = datetime.now()
            vqd_time = vqd_end - datetime.strptime(starttime, "%Y-%m-%d_%H-%M-%S")

            # Save run
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
                "parallel_run_time": str(vqd_time),
                "total_VQD_time": str(total_run_time),
                "total_device_time": str(total_device_time)
            }

            # Save the variable to a JSON file
            path = os.path.join(base_path, "{}_{}.json".format(potential, cut_off))
            with open(path, "w") as json_file:
                json.dump(run, json_file, indent=4)

            print("Done")
