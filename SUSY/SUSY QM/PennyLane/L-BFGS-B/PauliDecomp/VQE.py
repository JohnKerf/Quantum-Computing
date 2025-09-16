import pennylane as qml
from pennylane import numpy as pnp
from pennylane.pauli import group_observables

from scipy.optimize import minimize

import os, json, time, logging
import numpy as np
from datetime import datetime, timedelta

from multiprocessing import Pool

from susy_qm import calculate_Hamiltonian

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def setup_logger(logfile_path, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(logfile_path)
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def cost_function(paulis, coeffs, num_qubits, dev):
       
    def energy(params):

        @qml.qnode(dev)
        def circuit(params):

            '''
            #################### DW ##################
            # 2
            qml.RY(params[0], wires=[1])

            # 4
            #basis = [1] + [0]*(num_qubits-1)
            #qml.BasisState(basis, wires=range(num_qubits))
            #qml.RY(params[0], wires=[1])
            #qml.RY(params[1], wires=[2])

            # 8+
            #qml.RY(params[0], wires=[num_qubits-3])
            #qml.RY(params[1], wires=[num_qubits-1])
            #qml.CRY(params[2], wires=[num_qubits-1, num_qubits-2])
            #qml.RY(params[3], wires=[num_qubits-2])
            #qml.RY(params[4], wires=[num_qubits-1])
            '''

            '''
            #################### AHO ##################
            # 2
            basis = [1] + [0]*(num_qubits-1)
            qml.BasisState(basis, wires=range(num_qubits))
            qml.RY(params[0], wires=[0])

            # 4
            #basis = [1] + [0]*(num_qubits-1)
            #qml.BasisState(basis, wires=range(num_qubits))
            #qml.RY(params[0], wires=[1])

            # 8+
            #basis = [1] + [0]*(num_qubits-1)
            #qml.BasisState(basis, wires=range(num_qubits))
            #qml.RY(params[0], wires=[num_qubits-3])
            #qml.RY(params[1], wires=[num_qubits-2])
            '''

            '''
            #################### QHO ##################
            # 2+
            basis = [1] + [0]*(num_qubits-1)
            qml.BasisState(basis, wires=range(num_qubits))
            qml.RY(params[0], wires=[0])
            '''

            #'''
            #################### RA ##################
            n = num_qubits-1
            wires = range(num_qubits)
            for i, w in enumerate(wires):
                qml.RY(params[i], wires=w)

            for i in range(1, num_qubits):
                qml.CNOT(wires=[wires[i-1], wires[i]])

            qml.CNOT(wires=[wires[-1], wires[0]])

            for i, w in enumerate(wires):
                qml.RY(params[n + i], wires=w)
            #'''

            return [qml.expval(op) for op in paulis]

        expvals = circuit(params)                 
        energy = pnp.dot(coeffs, expvals)
        
        return energy
    
    grad_fn = qml.grad(energy)
    #params = qml.numpy.array(params, requires_grad=True)

    def f_and_g(x):
        x_pl = qml.numpy.array(x, requires_grad=True)
        val = energy(x_pl)
        grad = grad_fn(x_pl)
       
        return float(val), np.asarray(grad, dtype=float)
    
    return f_and_g

    

def run_vqe(i, max_iter, tol, paulis, coeffs, num_qubits, shots, num_params, device, eps, lam, p):

    # We need to generate a random seed for each process otherwise each parallelised run will have the same result
    seed = (os.getpid() * int(time.time())) % 123456789
    dev = qml.device(device, wires=num_qubits, shots=shots, seed=seed)
    np.random.seed(seed)

    x0 = np.random.random(size=num_params)*2*np.pi
    bounds = [(0, 2 * np.pi) for _ in range(num_params)]

    valgrad = cost_function(paulis, coeffs, num_qubits, dev)

    last_energy = None
    iteration_count = 0
    def wrapped_cost_function(params):
        nonlocal last_energy, iteration_count
        result = cost_function(params, paulis, coeffs, num_qubits, dev)

        last_energy = result
        iteration_count +=1

        neg = max(0.0, -(result + eps))

        return result + lam * (neg ** p)
    
    def f_only(x):
        energy, grad = valgrad(x)
        neg = max(0.0, -(energy + eps))

        return energy + lam * (neg ** p)
    
    def jac_only(x):
        val, grad = valgrad(x)
        return grad

    
    run_start = datetime.now()

    res = minimize(
        f_only,
        x0,
        bounds=bounds,
        method="L-BFGS-B",
        jac=jac_only,
        options={"maxiter": max_iter, "ftol": tol},
    )

    run_end = datetime.now()
    run_time = run_end - run_start

    return {
        "seed": seed,
        "energy": res.fun,
        "params": res.x.tolist(),
        "success": res.success,
        "num_iters": res.nfev,
        "run_time": run_time
    }


if __name__ == "__main__":
    
    potential = "DW"
    device = 'default.qubit'
    #device = 'qiskit.aer'
    
    #shots_list = [None, 10000, 100000]
    shots_list = [10000]
    cutoffs = [2,4,8,16,32,64]

    lam = 15
    p = 2

    for shots in shots_list:
        for cutoff in cutoffs:

            if potential == "AHO":
                i = np.log2(cutoff)
                factor = 2**(((i-1)*i)/2)
                eps = 0.5 / factor
            else:
                eps = 0

            print(f"Running for {potential} potential and cutoff {cutoff}")

            starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            base_path = os.path.join("/users/johnkerf/fastscratch/Data/PennyLane/L-BFGS-B/PauliDecomp/VQE/FilesNP-RA", str(shots), potential)
            os.makedirs(base_path, exist_ok=True)

            # Calculate Hamiltonian and expected eigenvalues
            H = calculate_Hamiltonian(cutoff, potential)
            
            eigenvalues = np.sort(np.linalg.eig(H)[0])[:4]
            num_qubits = int(1 + np.log2(cutoff))
            
            
            H_decomp = qml.pauli_decompose(H, wire_order=range(num_qubits))
            paulis = H_decomp.ops
            coeffs = H_decomp.coeffs

            # Optimizer
            num_params = 2*num_qubits
            num_vqe_runs = 100
            max_iter = 10000
            tol = 1e-8


            vqe_starttime = datetime.now()

            # Start multiprocessing for VQE runs
            with Pool(processes=100) as pool:
                vqe_results = pool.starmap(
                    run_vqe,
                    [
                        (i, max_iter, tol, paulis, coeffs, num_qubits, shots, num_params, device, eps, lam, p)
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
                    "name": "L-BFGS-B",
                    "maxiter": max_iter,
                    "tolerance": tol
                },
                "cost function":{
                    "type": "small negatives",
                    "p":p,
                    "lam":lam,
                    "eps":eps
                },
                "results": energies,
                "params": x_values,
                "num_iters": num_iters,
                "success": np.array(success, dtype=bool).tolist(),
                "run_times": run_times,
                "parallel_run_time": str(vqe_time),
                "total_VQE_time": str(total_run_time),
                "seeds": seeds,
            }

            # Save the variable to a JSON file
            path = os.path.join(base_path, "{}_{}.json".format(potential, cutoff))
            with open(path, "w") as json_file:
                json.dump(run, json_file, indent=4)

            print("Done")
