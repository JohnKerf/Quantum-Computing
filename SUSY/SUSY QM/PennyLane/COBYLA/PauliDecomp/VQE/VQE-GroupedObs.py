import pennylane as qml
from pennylane import numpy as pnp
from pennylane.pauli import group_observables

from scipy.optimize import minimize

import os
import json
import numpy as np
from datetime import datetime, timedelta
import time

from multiprocessing import Pool

from susy_qm import calculate_Hamiltonian

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def cost_function(params, H_decomp, num_qubits, shots, device, seed):
   
    dev = qml.device(device, wires=num_qubits, shots=shots, seed=seed)
    start = datetime.now()
  
    paulis = H_decomp.ops
    coeffs = H_decomp.coeffs

    groups = group_observables(paulis)
    
    @qml.qnode(dev)
    def circuit(params, groups):

        '''
        #################### DW ##################
        # 2
        #qml.RY(params[0], wires=[1])

        # 4
        #basis = [1] + [0]*(num_qubits-1)
        #qml.BasisState(basis, wires=range(num_qubits))
        #qml.RY(params[0], wires=[1])
        #qml.RY(params[1], wires=[2])

        # 8+
        qml.RY(params[0], wires=[num_qubits-3])
        qml.RY(params[1], wires=[num_qubits-1])
        qml.CRY(params[2], wires=[num_qubits-1, num_qubits-2])
        qml.RY(params[3], wires=[num_qubits-2])
        qml.RY(params[4], wires=[num_qubits-1])
        '''

        '''
        #################### AHO ##################
        # 2
        #basis = [1] + [0]*(num_qubits-1)
        #qml.BasisState(basis, wires=range(num_qubits))
        #qml.RY(params[0], wires=[0])

        # 4
        #basis = [1] + [0]*(num_qubits-1)
        #qml.BasisState(basis, wires=range(num_qubits))
        #qml.RY(params[0], wires=[1])

        # 8+
        basis = [1] + [0]*(num_qubits-1)
        qml.BasisState(basis, wires=range(num_qubits))
        qml.RY(params[0], wires=[num_qubits-3])
        qml.RY(params[1], wires=[num_qubits-2])
        '''

        '''
        #################### QHO ##################
        # 2+
        basis = [1] + [0]*(num_qubits-1)
        qml.BasisState(basis, wires=range(num_qubits))
        qml.RY(params[0], wires=[0])
        '''

        #basis = [1] + [0]*(num_qubits-1)
        #qml.BasisState(basis, wires=range(num_qubits))
        #qml.RY(params[0], wires=[num_qubits-3])
        #qml.RY(params[1], wires=[num_qubits-2])
        #qml.CRY(params[2], wires=[num_qubits-2, num_qubits-3])

        basis = [1] + [0]*(num_qubits-1)
        qml.BasisState(basis, wires=range(num_qubits))

        qml.RY(params[0], wires=[num_qubits-2])
        qml.RY(params[1], wires=[num_qubits-3])
        qml.RY(params[2], wires=[num_qubits-4])
        qml.CRY(params[3], wires=[num_qubits-2, num_qubits-3])
        qml.CRY(params[4], wires=[num_qubits-2, num_qubits-4])
        qml.CRY(params[5], wires=[num_qubits-3, num_qubits-4])
        qml.CRY(params[6], wires=[num_qubits-4, num_qubits-3])
        qml.RY(params[7], wires=[num_qubits-4]) 

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

    

def run_vqe(i, max_iter, tol, H_decomp, num_qubits, shots, num_params, device):

    # We need to generate a random seed for each process otherwise each parallelised run will have the same result
    seed = (os.getpid() * int(time.time())) % 123456789
    run_start = datetime.now()

  
    device_time = timedelta()

    def wrapped_cost_function(params):
        result, dt = cost_function(params, H_decomp, num_qubits, shots, device, seed)
        nonlocal device_time
        device_time += dt
        return result

    np.random.seed(seed)
    x0 = np.random.random(size=num_params)*2*np.pi

    res = minimize(
            wrapped_cost_function,
            x0,
            method= "COBYLA",
            options= {'maxiter':max_iter, 'tol':tol}
        )

    run_end = datetime.now()
    run_time = run_end - run_start

    return {
        "seed": seed,
        "energy": res.fun,
        "params": res.x.tolist(),
        "success": res.success,
        "num_iters": res.nfev,
        "run_time": run_time,
        "device_time": device_time
    }


if __name__ == "__main__":
    
    potential = "AHO"
    device = 'default.qubit'
    #device = 'qiskit.aer'
    #shots = 100000
    #shots = 10000
    shots = None
    #cutoff = 2

    cutoffs = [16]#,4,8,16]

    for cutoff in cutoffs:

        print(f"Running for {potential} potential and cutoff {cutoff}")

        starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_path = os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\SUSY QM\PennyLane\COBYLA\PauliDecomp\Files2", str(shots), potential)
        os.makedirs(base_path, exist_ok=True)

        # Calculate Hamiltonian and expected eigenvalues
        H = calculate_Hamiltonian(cutoff, potential)
        
        eigenvalues = np.sort(np.linalg.eig(H)[0])[:4]
        num_qubits = int(1 + np.log2(cutoff))
        
        H_decomp = qml.pauli_decompose(H, wire_order=range(num_qubits))

        # Optimizer
        num_params = 3

        num_vqe_runs = 100
        max_iter = 10000
        tol = 1e-8


        vqe_starttime = datetime.now()

        # Start multiprocessing for VQE runs
        with Pool(processes=10) as pool:
            vqe_results = pool.starmap(
                run_vqe,
                [
                    (i, max_iter, tol, H_decomp, num_qubits, shots, num_params, device)
                    for i in range(num_vqe_runs)
                ],
            )

        # Collect results
        seeds = [res["seed"] for res in vqe_results]
        energies = [res["energy"] for res in vqe_results]
        x_values = [res["params"] for res in vqe_results]
        success = [res["success"] for res in vqe_results]
        num_iters = [res["num_iters"] for res in vqe_results]
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
                "name": "COBYLA",
                "maxiter": max_iter,
                "tolerance": tol
            },
            "results": energies,
            "params": x_values,
            "num_iters": num_iters,
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
