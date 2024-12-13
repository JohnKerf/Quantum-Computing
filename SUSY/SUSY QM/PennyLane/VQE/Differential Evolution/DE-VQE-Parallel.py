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


def cost_function(params, H, params_shape, num_qubits, shots):
   
    dev = qml.device("default.qubit", wires=num_qubits, shots=shots)

    @qml.qnode(dev)
    def circuit(params):
        params = pnp.tensor(params.reshape(params_shape), requires_grad=True)
        qml.StronglyEntanglingLayers(weights=params, wires=range(num_qubits), imprimitive=qml.CZ)
        return qml.expval(qml.Hermitian(H, wires=range(num_qubits)))

    return circuit(params)


def run_vqe(i, bounds, max_iter, tol, abs_tol, strategy, popsize, H, params_shape, num_qubits, shots):

    # We need to generate a random seed for each process otherwise each parallelised run will have the same result
    seed = (os.getpid() * int(time.time())) % 123456789
    run_start = datetime.now()

    # Generate Halton sequence
    num_dimensions = np.prod(params_shape)
    num_samples = popsize
    halton_sampler = Halton(d=num_dimensions, seed=seed)
    halton_samples = halton_sampler.random(n=num_samples)
    scaled_samples = 2 * np.pi * halton_samples

    # Differential Evolution optimization
    res = differential_evolution(
        lambda params: cost_function(params, H, params_shape, num_qubits, shots),
        bounds,
        maxiter=max_iter,
        tol=tol,
        atol=abs_tol,
        strategy=strategy,
        popsize=popsize,
        init=scaled_samples,
        seed=seed
    )

    run_end = datetime.now()
    run_time = run_end - run_start

    return {
        "seed": seed,
        "energy": res.fun,
        "params": res.x.tolist(),
        "success": res.success,
        "num_iters": res.nit,
        "num_evaluations": res.nfev,
        "run_time": str(run_time),
    }


if __name__ == "__main__":
    
    potential_list = ["QHO", "AHO", "DW"]
    cut_offs_list = [2, 4, 8, 16]
    shots = 1024

    for potential in potential_list:

        starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_path = os.path.join("/users/johnkerf/SUSY/VQE/QM/Files", potential)
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

            num_vqe_runs = 100
            max_iter = 10000
            strategy = "randtobest1bin"
            tol = 1e-3
            abs_tol = 1e-3
            popsize = 20

            # Start multiprocessing for VQE runs
            with Pool(processes=40) as pool:
                vqe_results = pool.starmap(
                    run_vqe,
                    [
                        (i, bounds, max_iter, tol, abs_tol, strategy, popsize, H, params_shape, num_qubits, shots)
                        for i in range(num_vqe_runs)
                    ],
                )

            # Collect results
            seeds = [res["seed"] for res in vqe_results]
            energies = [res["energy"] for res in vqe_results]
            x_values = [res["params"] for res in vqe_results]
            success = [res["success"] for res in vqe_results]
            num_iters = [res["num_iters"] for res in vqe_results]
            num_evaluations = [res["num_evaluations"] for res in vqe_results]
            run_times = [res["run_time"] for res in vqe_results]

            vqe_end = datetime.now()
            vqe_time = vqe_end - datetime.strptime(starttime, "%Y-%m-%d_%H-%M-%S")

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
                "params": x_values,
                "num_iters": num_iters,
                "num_evaluations": num_evaluations,
                "success": np.array(success, dtype=bool).tolist(),
                "run_times": run_times,
                "seeds": seeds,
                "total_run_time": str(vqe_time),
            }

            # Save the variable to a JSON file
            path = os.path.join(base_path, "{}_{}.json".format(potential, cut_off))
            with open(path, "w") as json_file:
                json.dump(run, json_file, indent=4)
