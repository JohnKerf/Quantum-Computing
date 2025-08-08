import pennylane as qml
from pennylane import numpy as pnp
from pennylane.pauli import group_observables

from scipy.optimize import differential_evolution
from scipy.stats.qmc import Halton

import os
import json
import numpy as np
from datetime import datetime, timedelta
import time

from multiprocessing import Pool

from susy_qm import calculate_Hamiltonian

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def cost_function(params, H_decomp, num_qubits, shots, device, seed):
   
    dev = qml.device(device, wires=num_qubits, shots=shots, seed=seed)
    start = datetime.now()
  
    paulis = H_decomp.ops
    coeffs = H_decomp.coeffs

    groups = group_observables(paulis)
    
    @qml.qnode(dev)
    def circuit(params, groups):
        
        basis = [1] + [0]*(num_qubits-1)
        qml.BasisState(basis, wires=range(num_qubits))

        qml.RY(params[0], wires=[1])
        qml.RY(params[1], wires=[2])
        qml.CRY(params[2], wires=[2,1])
        
        return [qml.expval(op) for op in groups]

    energy = 0
    for group in groups:
        results = circuit(params, group)
        for op, res in zip(group, results):
            idx = paulis.index(op)
            energy += coeffs[idx] * res
     
    end = datetime.now()
    device_time = (end - start)

    return energy, device_time

    

def run_vqe(i, bounds, max_iter, tol, abs_tol, strategy, popsize, H_decomp, num_qubits, shots, num_params, device):

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
        result, dt = cost_function(params, H_decomp, num_qubits, shots, device, seed)
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
    
    potential = "AHO"
    device = 'default.qubit'
    shots = 10000
    cutoff = 8

    print(f"Running for {potential} potential and cutoff {cutoff}")

    starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_path = os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\SUSY QM\PennyLane\qml.Sample\Qiskit-Vs-PL_pauli\PL\FilesADAPT-Overlap", potential)
    os.makedirs(base_path, exist_ok=True)


    # Calculate Hamiltonian and expected eigenvalues
    H = calculate_Hamiltonian(cutoff, potential)
    
    eigenvalues = np.sort(np.linalg.eig(H)[0])[:4]
    num_qubits = int(1 + np.log2(cutoff))
    
    H_decomp = qml.pauli_decompose(H, wire_order=range(num_qubits))

    # Optimizer
    num_params = 3
    bounds = [(-2*np.pi, 2 * np.pi) for _ in range(num_params)]

    num_vqe_runs = 8
    max_iter = 10000
    strategy = "randtobest1bin"
    tol = 1e-8
    abs_tol = 0
    popsize = 20

    vqe_starttime = datetime.now()

    # Start multiprocessing for VQE runs
    with Pool(processes=num_vqe_runs) as pool:
        vqe_results = pool.starmap(
            run_vqe,
            [
                (i, bounds, max_iter, tol, abs_tol, strategy, popsize, H_decomp, num_qubits, shots, num_params, device)
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
        "device": device,
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

    print("Done")
