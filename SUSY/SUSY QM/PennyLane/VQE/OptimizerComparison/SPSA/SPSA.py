import pennylane as qml

import os
import json
import numpy as np
from datetime import datetime, timedelta
import time

from qiskit.quantum_info import SparsePauliOp

from multiprocessing import Pool

from susy_qm import calculate_Hamiltonian


def cost_function(params, H, num_qubits, shots):
   
    dev = qml.device("default.qubit", wires=num_qubits, shots=shots)
    start = datetime.now()
  

    @qml.qnode(dev)
    def circuit(params):
        param_index=0
        for i in range(num_qubits):
            qml.RY(params[param_index], wires=i)
            param_index += 1
            
        return qml.expval(qml.Hermitian(H, wires=range(num_qubits)))  

    
    end = datetime.now()
    device_time = (end - start)

    return circuit(params), device_time

def run_optimizer(opt, cost_function, init_param, num_steps, interval, execs_per_step, tol):
    # Copy the initial parameters to make sure they are never overwritten
    param = init_param.copy()

    # Obtain the device used in the cost function
    #dev = cost_function.device

    # Initialize the memory for cost values during the optimization
    cost_history = []
    params_history = []
    # Monitor the initial cost value
    cost_history.append(cost_function(param))
    exec_history = [0]
    eval_history = [0]

    converged = False

    #print(
    #    f"\nRunning the {opt.__class__.__name__} optimizer for {num_steps} iterations."
    #)
    for step in range(num_steps):
        # Print out the status of the optimization
        #if step % interval == 0:
        #    print(
        #        f"Step {step:3d}: Circuit executions: {exec_history[step]:4d}, "
        #        f"Cost = {cost_history[step]}"
        #    )

        # Perform an update step
        param = opt.step(cost_function, param)
        if step > 0:
            if abs(cost_function(param) - cost_function(params_history[-1])) < tol:
                converged = True
                break 
            else:
                params_history.append(param)
                cost_history.append(cost_function(param))
                exec_history.append((step + 1))
                eval_history.append((step + 1) * execs_per_step)

        else:
            params_history.append(param)
            cost_history.append(cost_function(param))
            exec_history.append((step + 1))
            eval_history.append((step + 1) * execs_per_step)

    #print(
    #    f"Step {num_steps:3d}: Circuit executions: {exec_history[-1]:4d}, "
    #    f"Cost = {cost_history[-1]}"
    #)
    return cost_history, exec_history, eval_history, params_history, converged


def run_vqe(i, max_iter, tol, H, num_qubits, shots):

    # We need to generate a random seed for each process otherwise each parallelised run will have the same result
    seed = (os.getpid() * int(time.time())) % 123456789
    run_start = datetime.now()

    # Optimizer
    np.random.seed(seed)
    x0 = np.random.random(size=num_qubits)*2*np.pi

    device_time = timedelta()

    def wrapped_cost_function(params):
        result, dt = cost_function(params, H, num_qubits, shots)
        nonlocal device_time
        device_time += dt
        return result
    
    num_steps_spsa = max_iter
    opt = qml.SPSAOptimizer(maxiter=num_steps_spsa)#, c=0.15, a=0.2)
    # We spend 2 circuit evaluations per step:
    execs_per_step = 20
    cost_history_spsa, exec_history_spsa, eval_history, params_history, converged = run_optimizer(
        opt, wrapped_cost_function, x0, num_steps_spsa, 20, execs_per_step, tol
)



    run_end = datetime.now()
    run_time = run_end - run_start

    return {
        "seed": seed,
        "energy": cost_history_spsa[-1],
        "params": params_history[-1].tolist(),
        "success": converged,
        "num_iters":exec_history_spsa[-1],
        "num_evaluations": eval_history[-1],
        "run_time": run_time,
        "device_time": device_time
    }


if __name__ == "__main__":
    
    potential_list = ["DW"]
    cut_offs_list = [16]
    shots = 64

    for potential in potential_list:

        starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_path = os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\SUSY QM\PennyLane\VQE\OptimizerComparison\SPSA", potential, str(shots))
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

            num_vqe_runs = 4
            max_iter = 10000
            tol = 1e-12
    
            vqe_starttime = datetime.now()

            # Start multiprocessing for VQE runs
            with Pool(processes=4) as pool:
                vqe_results = pool.starmap(
                    run_vqe,
                    [
                        (i, max_iter, tol, H, num_qubits, shots)
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
                "potential": potential,
                "cutoff": cut_off,
                "exact_eigenvalues": [x.real.tolist() for x in eigenvalues],
                "ansatz": "circuit.txt",
                "num_VQE": num_vqe_runs,
                "shots": shots,
                "Optimizer": {
                    "name": "COBYLA",
                    "x0": "np.random.uniform(0, 2 * np.pi, size=num_qubits)",
                    "maxiter": max_iter,
                    "tolerance": tol
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
            path = os.path.join(base_path, "{}_{}.json".format(potential, cut_off))
            with open(path, "w") as json_file:
                json.dump(run, json_file, indent=4)
