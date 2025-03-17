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

from scipy.sparse import lil_matrix
import scipy as sp


def cost_function(params, H, num_qubits, shots):
   
    dev = qml.device("default.qubit", wires=num_qubits, shots=shots)
    start = datetime.now()
  

    @qml.qnode(dev)
    def circuit(params):
        param_index=0
        for i in range(num_qubits):
            qml.RY(params[param_index], wires=i)
            param_index += 1

        ## Apply entanglement
        #for j in reversed(range(1, num_qubits)):
        #    qml.CNOT(wires=[j, j-1])

        # Apply RY rotations
        #for k in range(num_qubits):
        #    qml.RY(params[param_index], wires=k)
        #    param_index += 1
        
        return qml.expval(qml.Hermitian(H, wires=range(num_qubits)))
 
     
    
    end = datetime.now()
    device_time = (end - start)

    return circuit(params), device_time


def run_vqe(i, bounds, max_iter, tol, abs_tol, strategy, popsize, H, num_qubits, shots):

    # We need to generate a random seed for each process otherwise each parallelised run will have the same result
    seed = (os.getpid() * int(time.time())) % 123456789
    run_start = datetime.now()

    # Generate Halton sequence
    num_dimensions = num_qubits
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

def sparse_ghz_state(n_qubits):

    # Size of the state
    state_size = 2**n_qubits
   
    # Create a sparse state_vector with only non-zero elements
    # Create an empty sparse row vector
    state_vector = lil_matrix((1, state_size))  
   
    # Set the amplitudes for non-zero element
    state_vector[0, 0] = 1/np.sqrt(2)  # |000...0>
    state_vector[0, state_size-1] = 1/np.sqrt(2)  # |111...1>
   
    return state_vector


if __name__ == "__main__":
       

    starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_path = os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\SUSY QM\Entanglement\files")
    os.makedirs(base_path, exist_ok=True)

    
#======================================================================================================================

    num_qubits = 2
    I_n_qubit =sp.sparse.eye(2**num_qubits)

    Hbell = I_n_qubit.toarray() -2 * np.outer(sparse_ghz_state(num_qubits).toarray(), np.conj(sparse_ghz_state(num_qubits)).toarray())
    eigenvalues = np.sort(np.linalg.eig(Hbell)[0])
    min_eigenvalue = min(eigenvalues.real)
    hamiltonian = SparsePauliOp.from_operator(Hbell, atol = 1e-8)
    num_qubits = hamiltonian.num_qubits

    # Optimizer
    bounds = [(0, 2 * np.pi) for _ in range(num_qubits)]

    num_vqe_runs = 8
    max_iter = 10000
    strategy = "randtobest1bin"
    tol = 1e-3
    abs_tol = 1e-3
    popsize = 20
    shots = 1024

    vqe_starttime = datetime.now()

    # Start multiprocessing for VQE runs
    with Pool(processes=8) as pool:
        vqe_results = pool.starmap(
            run_vqe,
            [
                (i, bounds, max_iter, tol, abs_tol, strategy, popsize, Hbell, num_qubits, shots)
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
    vqe_time = vqe_end - vqe_starttime

    # Save run
    run = {
        "starttime": starttime,
        "endtime": vqe_end.strftime("%Y-%m-%d_%H-%M-%S"),
        #"potential": potential,
        #"cutoff": cut_off,
        "exact_eigenvalues": [x.real.tolist() for x in eigenvalues],
        "ansatz": "circuit.txt",
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
        "parallel_run_time": str(vqe_time),
        "total_VQE_time": str(total_run_time),
        "total_device_time": str(total_device_time)
    }

    # Save the variable to a JSON file
    path = os.path.join(base_path, "data2.json")
    with open(path, "w") as json_file:
        json.dump(run, json_file, indent=4)
