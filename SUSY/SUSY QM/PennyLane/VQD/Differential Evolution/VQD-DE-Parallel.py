import pennylane as qml
from pennylane import numpy as pnp

from scipy.optimize import differential_evolution
from scipy.stats.qmc import Halton

import os
import json
import numpy as np
from datetime import datetime
import time

from qiskit.quantum_info import SparsePauliOp

from multiprocessing import Pool

from susy_qm import calculate_Hamiltonian


def cost_function(params, prev_param_list, H, params_shape, num_qubits, shots, beta):
   
    dev = qml.device("default.qubit", wires=2*num_qubits+1, shots=shots)

    def ansatz(params, wires):
        params = pnp.tensor(params.reshape(params_shape), requires_grad=True)
        qml.StronglyEntanglingLayers(weights=params, wires=wires, imprimitive=qml.CZ)

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

    @qml.qnode(dev)
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

    return loss_f(params)


def run_vqd(i, bounds, max_iter, tol, abs_tol, strategy, popsize, H, params_shape, num_qubits, shots, num_energy_levels, beta):
    
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

    for i in range(num_energy_levels):
        
        

        # Differential Evolution optimization
        res = differential_evolution(
            lambda params: cost_function(params, prev_param_list, H, params_shape, num_qubits, shots, beta),
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
        "run_times": str(run_time),
    }


if __name__ == "__main__":
    
    potential_list = ["QHO"]#, "AHO", "DW"]
    cut_offs_list = [16]
    shots = 1024

    for potential in potential_list:

        starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        #base_path = os.path.join("/users/johnkerf/SUSY/VQD/QM/Files", potential)
        base_path = os.path.join(r"C:\Users\Johnk\OneDrive\Desktop\PhD 2024\Quantum Computing Code\Quantum-Computing\SUSY\SUSY QM\PennyLane\VQD\Differential Evolution\Files", potential)
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
            num_energy_levels = 2
            beta = 2.0

            max_iter = 200
            strategy = "randtobest1bin"
            tol = 1e-3
            abs_tol = 1e-3
            popsize = 20

            # Start multiprocessing for VQE runs
            with Pool(processes=1) as pool:
                vqd_results = pool.starmap(
                    run_vqd,
                    [
                        (i, bounds, max_iter, tol, abs_tol, strategy, popsize, H, params_shape, num_qubits, shots, num_energy_levels, beta)
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
            run_times = [result["run_times"] for result in vqd_results]

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
            }

            # Save the variable to a JSON file
            path = os.path.join(base_path, "{}_{}.json".format(potential, cut_off))
            with open(path, "w") as json_file:
                json.dump(run, json_file, indent=4)

            print("Done")
