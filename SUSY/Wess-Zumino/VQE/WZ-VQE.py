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
from susy_qm import calculate_wz_hamiltonian

# prallel processing
from multiprocessing import Pool


def cost_function(params, H, num_qubits, shots):
   
    dev = qml.device("default.qubit", wires=num_qubits, shots=shots)
    start = datetime.now()

    @qml.qnode(dev)
    def circuit(params):

        basis_state = [0,0,1,0,0,1]
        qml.BasisState(basis_state, wires=range(num_qubits))

        qml.RY(params[0], wires=[3])
        qml.RY(params[1], wires=[5])
        qml.CRY(params[2], wires=[3,5])
        qml.FermionicSingleExcitation(params[3], wires=[0,1])
        qml.FermionicSingleExcitation(params[4], wires=[1,2])

        return qml.expval(qml.Hermitian(H, wires=range(num_qubits)))
    
    end = datetime.now()
    device_time = (end - start)
    
    return circuit(params), device_time


def run_vqe(i, bounds, max_iter, tol, abs_tol, strategy, popsize, H, num_params, num_qubits, shots):

    # We need to generate a random seed for each process otherwise each parallelised run will have the same result
    seed = (os.getpid() * int(time.time())) % 123456789
    run_start = datetime.now()

    # Generate Halton sequence
    num_dimensions = num_params
    num_samples = popsize
    halton_sampler = Halton(d=num_dimensions, seed=seed)
    halton_samples = halton_sampler.random(n=num_samples)
    scaled_samples = 2 * np.pi * halton_samples

    device_time = timedelta()

    def wrapped_cost_function(params):
        result, dt = cost_function(params, H, num_qubits, shots)
        nonlocal device_time
        device_time += dt
        return result

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

    run_end = datetime.now()
    run_time = run_end - run_start

    return {
        "seed": seed,
        "energy": res.fun,
        "params": res.x.tolist(),
        "success": res.success,
        "num_iters": res.nit,
        "num_evaluations": res.nfev,
        "run_time": run_time,
        "device_time": device_time
    }


if __name__ == "__main__":
    
    # Parameters
    N = 3
    a = 1.0
    c = -0.2

    potential = "linear"
    #potential = 'quadratic'
    boundary_condition = 'dirichlet'
    #boundary_condition = 'periodic'
    cutoffs = [2]
    shots = 1024

    
    starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if potential == 'quadratic':
        folder = 'C' + str(abs(c)) + '/' + 'N'+ str(N) + '/' + str(starttime)
    else:
        folder = 'N'+ str(N) + '/' + str(starttime)

    base_path = os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\Wess-Zumino\VQE\Files", boundary_condition, potential, folder)
    os.makedirs(base_path, exist_ok=True)

    print(str(starttime))
    print(f"Running for {N} sites")
    print(f"Running for {boundary_condition} boundary conditions")
    print(f"Running for {potential} potential")

    for cutoff in cutoffs:
        print(f"Running for cutoff: {cutoff}")

        # Calculate Hamiltonian and expected eigenvalues
        H = calculate_wz_hamiltonian(cutoff, N, a, potential, boundary_condition, c)
        eigenvalues = np.sort(np.linalg.eig(H)[0])[:4]
        min_eigenvalue = min(eigenvalues.real)

        # Create qiskit Hamiltonian Pauli string
        hamiltonian = SparsePauliOp.from_operator(H)
        num_qubits = hamiltonian.num_qubits
        print("Num qubits: ", num_qubits)


        # Optimizer
        num_params = 5
        bounds = [(0, 2 * np.pi) for _ in range(num_params)]
        num_vqe_runs = 4
        max_iterations = 10000
        strategy = "randtobest1bin"
        popsize = 20
        tol = 1e-3
        abs_tol = 1e-3

        # Start multiprocessing for VQE runs
        with Pool(processes=4) as pool:
            vqe_results = pool.starmap(
                run_vqe,
                [
                    (i, bounds, max_iterations, tol, abs_tol, strategy, popsize, H, num_params, num_qubits, shots)
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
        run_times = [str(res["run_time"]) for res in vqe_results]
        total_run_time = sum([res["run_time"] for res in vqe_results], timedelta())
        total_device_time = sum([res['device_time'] for res in vqe_results], timedelta())

        vqe_end = datetime.now()
        vqe_time = vqe_end - datetime.strptime(starttime, "%Y-%m-%d_%H-%M-%S")

        # Save run
        run = {
            "starttime": starttime,
            "potential": potential,
            "boundary_condition": boundary_condition,
            "cutoff": cutoff,
            "N": N,
            "a": a,
            "c": None if potential == "linear" else c,
            "exact_eigenvalues": [x.real.tolist() for x in eigenvalues],
            "ansatz": 'circuit.txt',
            "num_VQE": num_vqe_runs,
            "shots": shots,
            "Optimizer": {
                "name": "differential_evolution",
                "bounds": "[(0, 2 * np.pi)]",
                "maxiter": max_iterations,
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
            "parallel_run_time": str(vqe_time),
            "total_VQE_time": str(total_run_time),
            "total_device_time": str(total_device_time)
        }

        # Save the variable to a JSON file
        path = os.path.join(base_path, "{}_{}.json".format(potential, cutoff))
        with open(path, "w") as json_file:
            json.dump(run, json_file, indent=4)
