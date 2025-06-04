import pennylane as qml
from pennylane import numpy as pnp

from scipy.optimize import differential_evolution
from scipy.stats.qmc import Halton

import os
import json
import numpy as np
from datetime import datetime, timedelta
import time

from multiprocessing import Pool

from susy_qm import calculate_Hamiltonian


def cost_function(params, H, HP, num_qubits, shots):
   
    dev = qml.device("default.qubit", wires=num_qubits, shots=shots)
    start = datetime.now()
  
    @qml.qnode(dev)
    def circuit(params):

        for i, term in enumerate(HP):
            qml.ApproxTimeEvolution(term, params[i], 3)
            
        return qml.expval(qml.Hermitian(H, wires=range(num_qubits)))
     
    
    end = datetime.now()
    device_time = (end - start)

    return circuit(params), device_time


def run_vqe(i, bounds, max_iter, tol, abs_tol, strategy, popsize, H, HP, num_qubits, shots, num_params):

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
        result, dt = cost_function(params, H, HP, num_qubits, shots)
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
    
    potential = "DW"
    shots = 1024
    cutoff_list = [16]#, 4, 8, 16, 32, 64, 128, 256]

    for cutoff in cutoff_list:

        print(f"Running for {potential} potential and cutoff {cutoff}")

        starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_path = os.path.join(r"C:\Users\johnkerf\Desktop\Quantum-Computing\Quantum-Computing\SUSY\SUSY QM\PennyLane\VQE\test", potential, str(starttime))
        os.makedirs(base_path, exist_ok=True)


        # Calculate Hamiltonian and expected eigenvalues
        H = calculate_Hamiltonian(cutoff, potential)
        eigenvalues = np.sort(np.linalg.eig(H)[0])[:4]

        HP = qml.pauli_decompose(H)
        print(HP)
        
        num_qubits = int(1 + np.log2(cutoff))

        # Optimizer
        num_params = len(HP)
        bounds = [(0, 2 * np.pi) for _ in range(num_params)]

        num_vqe_runs = 1
        max_iter = 2000
        strategy = "randtobest1bin"
        tol = 1e-3
        abs_tol = 1e-3
        popsize = 20

        vqe_starttime = datetime.now()

        # Start multiprocessing for VQE runs
        with Pool(processes=1) as pool:
            vqe_results = pool.starmap(
                run_vqe,
                [
                    (i, bounds, max_iter, tol, abs_tol, strategy, popsize, H, HP, num_qubits, shots, num_params)
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
        device_times = [str(res["device_time"]) for res in vqe_results]
        total_device_time = sum([res['device_time'] for res in vqe_results], timedelta())

        vqe_end = datetime.now()
        vqe_time = vqe_end - vqe_starttime

        # Save run
        run = {
            "starttime": starttime,
            "endtime": vqe_end.strftime("%Y-%m-%d_%H-%M-%S"),
            "potential": potential,
            "cutoff": cutoff,
            "exact_eigenvalues": [x.real.tolist() for x in eigenvalues],
            "ansatz": "circuit.txt",
            "num_VQE": num_vqe_runs,
            "shots": shots,
            "Optimizer": {
                "name": "differential_evolution",
                "bounds": "[(0, 2 * np.pi)",
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
            "device_times": device_times,
            "parallel_run_time": str(vqe_time),
            "total_VQE_time": str(total_run_time),
            "total_device_time": str(total_device_time),
            "seeds": seeds,
        }

        # Save the variable to a JSON file
        path = os.path.join(base_path, "{}_{}.json".format(potential, cutoff))
        with open(path, "w") as json_file:
            json.dump(run, json_file, indent=4)
