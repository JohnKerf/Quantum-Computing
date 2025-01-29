import numpy as np
from datetime import datetime, timedelta
import time

from susy_qm import calculate_wz_hamiltonian

import pennylane as qml
from pennylane import numpy as pnp

from qiskit.quantum_info import SparsePauliOp

from scipy.optimize import differential_evolution
from scipy.stats.qmc import Halton

import json
import os

from multiprocessing import Pool


def cost_function(params, sel_params_shape, ran_params_shape, seed, num_qubits, min_eigenvector, shots):
   
    dev = qml.device("default.qubit", wires=num_qubits, shots=shots)

    @qml.qnode(dev)
    def ansatz(sel_params, rand_params):
        qml.StronglyEntanglingLayers(sel_params, wires=range(num_qubits), imprimitive=qml.CZ)
        qml.RandomLayers(rand_params, wires=range(num_qubits), imprimitive=qml.CZ, seed=seed)
        return qml.state()
    
    sel_params = params[:np.prod(sel_params_shape)]
    ran_params = params[np.prod(sel_params_shape):]
    sel_params = pnp.tensor(sel_params.reshape(sel_params_shape), requires_grad=True)
    ran_params = pnp.tensor(ran_params.reshape(ran_params_shape), requires_grad=True)

    ansatz_state = ansatz(sel_params, ran_params)
    ansatz_state = np.array(ansatz_state)

    overlap = np.vdot(min_eigenvector, ansatz_state)
    overlap_squared = np.abs(overlap)**2

    return 1 - overlap_squared


def run_overlap_check(i, sel_params_shape, ran_params_shape, min_eigenvector, maxiter, num_qubits, shots, bounds, tol, abs_tol, strategy, popsize):

    # We need to generate a random seed for each process otherwise each parallelised run will have the same result
    seed = (os.getpid() * int(time.time())) % 123456789
    run_start = datetime.now()

    # Generate Halton sequence
    num_dimensions = np.prod(sel_params_shape) + np.prod(ran_params_shape)
    num_samples = popsize
    halton_sampler = Halton(d=num_dimensions, seed=seed)
    halton_samples = halton_sampler.random(n=num_samples)
    scaled_samples = 2 * np.pi * halton_samples

    # Differential Evolution optimization
    res = differential_evolution(
        cost_function,
        args=((sel_params_shape, ran_params_shape, seed, num_qubits, min_eigenvector, shots)),
        bounds=bounds,
        maxiter=maxiter,
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
        "cost": res.fun,
        "params": res.x.tolist(),
        "success": res.success,
        "num_iters": res.nfev,
        "run_time": run_time,
    }

if __name__ == "__main__":

    starttime = datetime.now()

    # Parameters
    N = 3
    cutoff = 2
    a = 1.0
    potential = "linear"
    boundary_condition = 'dirichlet'
    c = -0.2

    # Hamiltonian data
    H = calculate_wz_hamiltonian(cutoff, N, a, potential, boundary_condition, c)

    eigenvalues, eigenvectors = np.linalg.eig(H)

    min_index = np.argmin(eigenvalues)
    min_eigenvalue = eigenvalues[min_index]
    min_eigenvector = np.asarray(eigenvectors[:, min_index])

    hamiltonian = SparsePauliOp.from_operator(H)
    num_qubits = hamiltonian.num_qubits

    # ansatz param shapes
    sel_num_layers = 2
    sel_params_shape = qml.StronglyEntanglingLayers.shape(n_layers=sel_num_layers, n_wires=num_qubits)

    ran_num_layers = 3
    ran_params_shape = qml.RandomLayers.shape(n_layers=ran_num_layers, n_rotations=num_qubits)

    num_params = np.prod(sel_params_shape) + np.prod(ran_params_shape)

    #variables
    shots = None
    num_tests = 4
    maxiter = 10
    strategy = "randtobest1bin"
    tol = 1e-3
    abs_tol = 1e-3
    popsize = 20

    bounds = [(0, 2 * np.pi) for _ in range(num_params)]

    # Start multiprocessing
    with Pool(processes=4) as pool:
        overlap_results = pool.starmap(
            run_overlap_check,
            [
                (i, sel_params_shape, ran_params_shape, min_eigenvector, maxiter, num_qubits, shots, bounds, tol, abs_tol, strategy, popsize)
                for i in range(num_tests)
            ],
        )

    # Collect results
    seeds = [res["seed"] for res in overlap_results]
    cost = [res["cost"] for res in overlap_results]
    params = [res["params"] for res in overlap_results]
    success = [res["success"] for res in overlap_results]
    num_iters = [res["num_iters"] for res in overlap_results]
    run_times = [str(res["run_time"]) for res in overlap_results]
    total_run_time = sum([res["run_time"] for res in overlap_results], timedelta())

    end = datetime.now()
    paralleltime = end - starttime

    minindex = np.argmin(cost)
    min_cost = cost[minindex]
    min_params = params[minindex]
    min_seed = seeds[minindex]

    # Save to a JSON file
    data = {
        "starttime": str(starttime),
        "endtime": str(end),
        "potential": potential,
        "boundary_condition": boundary_condition,
        "cutoff": cutoff,
        "N": N,
        "a": a,
        "c": None if potential == "linear" else c,
        "shots": shots,
        "sel_num_layers": sel_num_layers,
        "ran_num_layers": ran_num_layers,
        "num_tests": num_tests,
        "Optimizer": {
                "name": "differential_evolution",
                "bounds": "[(0, 2 * np.pi) for _ in range(np.prod(params_shape))]",
                "maxiter": maxiter,
                "tolerance": tol,
                "abs_tolerance": abs_tol,
                "strategy": strategy,
                "popsize": popsize,
                'init': 'scaled_samples',
            },
        "num_qubits": num_qubits,
        "num_params": int(num_params),
        "min_eigenvector": [x.real.tolist() for x in min_eigenvector],
        "results": cost,
        #"params": params,
        "num_iters": num_iters,
        "success": np.array(success, dtype=bool).tolist(),
        "seeds": seeds,
        "min_cost": min_cost,
        "min_params": min_params,
        "min_seed": min_seed,
        #"run_times": run_times,
        "total_runtime": str(total_run_time),
        "parallel_time": str(paralleltime)
    }

    folder = starttime.strftime("%Y-%m-%d_%H-%M-%S")
    base_path = os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\Wess-Zumino\Ansatz Testing\Files", folder)
    os.makedirs(base_path, exist_ok=True)
    file_path = os.path.join(base_path, 'data.json')

    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    print("Done")

