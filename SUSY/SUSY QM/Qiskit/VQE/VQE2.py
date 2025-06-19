# vqe_qiskit_runner.py
from qiskit.circuit import QuantumCircuit, Parameter
#from qiskit_ibm_runtime import Estimator
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit_aer import AerSimulator
from scipy.optimize import differential_evolution
from scipy.stats.qmc import Halton
from qiskit.primitives import Estimator, BackendEstimator

import numpy as np
import os
import json
import time
import logging
from datetime import datetime, timedelta
from multiprocessing import Pool

from susy_qm import calculate_Hamiltonian

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



def setup_logger(logfile_path, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(logfile_path)
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

'''
def build_ansatz(params, num_qubits):

    theta = Parameter("θ")
    qc = QuantumCircuit(num_qubits)
    q = list(reversed(range(num_qubits)))

    qc.ry(theta, q[0])

    qc.ry(theta, q[num_qubits-2])
    qc.ry(theta, q[num_qubits-3])
    qc.ry(theta, q[num_qubits-4])

    #qc.ry(params[0], q[num_qubits - 3])
    #qc.ry(params[1], q[num_qubits - 1])
    #qc.cry(params[2], q[num_qubits - 1], q[num_qubits - 2])
    #qc.ry(params[3], q[num_qubits - 2])
    #qc.ry(params[4], q[num_qubits - 1])

    return qc
'''
'''
def build_ansatz(num_qubits):

    params = [Parameter(f"θ{i}") for i in range(4)]

    qc = QuantumCircuit(num_qubits)
    q = list(reversed(range(num_qubits)))

    #qc.x(q[0])
    qc.ry(params[0], q[0])
    qc.ry(params[1], q[1])
    qc.ry(params[2], q[2])
    qc.ry(params[3], q[3])

    return qc, params

def cost_function(params, H_matrix, num_qubits, shots, estimator, observable):

    start = datetime.now()

    circuit, param_objs  = build_ansatz(num_qubits)
    param_dict = dict(zip(param_objs, params))
    bound_circuit = circuit.assign_parameters(param_dict, inplace=False)
    job = estimator.run([(circuit, observable)])
    energy = job.result()[0].data.evs

    end = datetime.now()

    return energy, end - start
'''

def cost_function(params, H_matrix, num_qubits, shots):
    
    start = datetime.now()

    # Build observable
    observable = SparsePauliOp.from_operator(H_matrix)

    # Build ansatz and parameter map
    param_objs = [Parameter(f"θ{i}") for i in range(4)]
    qc = QuantumCircuit(num_qubits)
    for i in range(4):
        qc.ry(param_objs[i], i)

    param_dict = dict(zip(param_objs, params))
    bound_circuit = qc.assign_parameters(param_dict, inplace=False)

    backend = AerSimulator()
    estimator = BackendEstimator(backend=backend, options={"shots": shots})

    result = estimator.run(circuits=[bound_circuit], observables=[observable]).result()
    energy = result.values[0]

    # Use a new Estimator per call (safe in multiprocessing)
    #estimator = Estimator(backend=AerSimulator(), options={"default_shots": shots})
    #estimator = BackendEstimator(backend=AerSimulator(), options={"shots": shots})
    #result = estimator.run([(bound_circuit, observable)]).result()
    #energy = result.values[0]
    #job = estimator.run([(bound_circuit, observable)])
    #energy = job.result()[0].data.evs

    end = datetime.now()
    return energy, end - start


def run_vqe(i, bounds, max_iter, tol, abs_tol, strategy, popsize, H, num_qubits, shots, num_params, log_dir):
    
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"vqe_run_{i}.log")
    logger = setup_logger(log_path, f"logger_{i}")

    seed = (os.getpid() * int(time.time())) % 123456789
    run_start = datetime.now()

    halton_sampler = Halton(d=num_params, seed=seed)
    scaled_samples = 2 * np.pi * halton_sampler.random(n=popsize)
    device_time = timedelta()

    def wrapped_cost_function(params):
        result, dt = cost_function(params, H, num_qubits, shots)
        nonlocal device_time
        device_time += dt
        return result

    iteration_count = 0

    def iteration_callback(xk, convergence=None):
        nonlocal iteration_count
        iteration_count += 1
        if iteration_count % 50 == 0:
            logger.info(f"VQE run {i} is on iteration {iteration_count}")

    logger.info(f"Starting VQE run {i} (seed={seed})")

    res = differential_evolution(
        wrapped_cost_function,
        bounds,
        maxiter=max_iter,
        tol=tol,
        atol=abs_tol,
        strategy=strategy,
        popsize=popsize,
        init=scaled_samples,
        seed=seed,
        workers=1,
        callback=iteration_callback
    )

    run_end = datetime.now()

    logger.info(f"Finished VQE run {i} - Energy: {res.fun:.6f}, Success: {res.success}, Iterations: {res.nit}")

    return {
        "seed": seed,
        "energy": res.fun,
        "params": res.x.tolist(),
        "success": res.success,
        "num_iters": res.nit,
        "num_evaluations": res.nfev,
        "run_time": str(run_end - run_start),
        "device_time": str(device_time)
    }

if __name__ == "__main__":
    potential = "AHO"
    shots = 1024
    cutoff_list = [8]

    for cutoff in cutoff_list:

        print(f"Running for {potential} potential and cutoff {cutoff}")
        starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_path = os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\SUSY QM\Qiskit\VQE\test2", potential, str(starttime))
        log_path = os.path.join(base_path, 'logs')
        os.makedirs(base_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)

        H = calculate_Hamiltonian(cutoff, potential)
        eigenvalues = np.sort(np.linalg.eig(H)[0])[:4]
        num_qubits = int(1 + np.log2(cutoff))
        num_params = 4
        bounds = [(0, 2 * np.pi) for _ in range(num_params)]

        num_vqe_runs = 1
        max_iter = 10000
        strategy = "randtobest1bin"
        tol = 1e-3
        abs_tol = 1e-3
        popsize = 20

        #observable = SparsePauliOp.from_operator(H)

        #backend = AerSimulator()
        #estimator = Estimator(backend, options={"default_shots": shots})
        #print(estimator.options)

        vqe_start = datetime.now()
        '''
        with Pool(processes=4) as pool:
            vqe_results = pool.starmap(
                run_vqe,
                [
                    (i, bounds, max_iter, tol, abs_tol, strategy, popsize, H, num_qubits, shots, num_params, log_path,)
                    for i in range(num_vqe_runs)
                ],
            )
        '''
        vqe_results = []
        for i in range(num_vqe_runs):
            result = run_vqe(i, bounds, max_iter, tol, abs_tol, strategy, popsize, H, num_qubits, shots, num_params, log_path)
            vqe_results.append(result)

        vqe_end = datetime.now()

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
                "bounds": str(bounds),
                "maxiter": max_iter,
                "tolerance": tol,
                "abs_tolerance": abs_tol,
                "strategy": strategy,
                "popsize": popsize,
                'init': 'halton_scaled',
            },
            "results": [r["energy"] for r in vqe_results],
            "params": [r["params"] for r in vqe_results],
            "num_iters": [r["num_iters"] for r in vqe_results],
            "num_evaluations": [r["num_evaluations"] for r in vqe_results],
            "success": [r["success"] for r in vqe_results],
            "run_times": [r["run_time"] for r in vqe_results],
            "device_times": [r["device_time"] for r in vqe_results],
            "parallel_run_time": str(vqe_end - vqe_start),
        }

        with open(os.path.join(base_path, f"{potential}_{cutoff}.json"), "w") as f:
            json.dump(run, f, indent=4)

        print("Done")
