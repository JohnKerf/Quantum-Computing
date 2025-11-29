# vqe_qiskit_runner.py
import os, json, time, logging
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats.qmc import Halton

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import EstimatorOptions

from susy_qm import calculate_Hamiltonian

from multiprocessing import Pool

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



def cost_function(params, estimator, observable, param_objs, qc):

    start = datetime.now()

    param_dict = dict(zip(param_objs, params))
    bound = qc.assign_parameters(param_dict, inplace=False)

    result = estimator.run([(bound, observable)]).result()
    energy = result[0].data.evs.sum()

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

    options = EstimatorOptions()
    options.default_shots = shots
    backend = AerSimulator()
    estimator = Estimator(backend, options=options)

    observable = SparsePauliOp.from_operator(H)
    param_objs = [Parameter(f"θ{i}") for i in range(2*num_qubits)]

    #qc = QuantumCircuit(num_qubits)
    #for i in range(num_qubits):
    #    qc.ry(param_objs[i], i)

    qc = QuantumCircuit(num_qubits)
    param_index=0
    for i in range(num_qubits):
        qc.ry(param_objs[param_index], i)
        param_index += 1
    
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)

    for i in range(num_qubits):
        qc.ry(param_objs[param_index], i)
        param_index += 1


    def wrapped_cost(params):
        result, dt = cost_function(params, estimator, observable, param_objs, qc)
        nonlocal device_time
        device_time += dt
        return result


    iteration_count = 0
    def callback(xk, convergence=None):
        nonlocal iteration_count
        iteration_count += 1
        if iteration_count % 50 == 0:
            energy, _ = cost_function(xk, estimator, observable, param_objs, qc)
            logger.info(f"Iteration {iteration_count}: Energy = {energy:.8f}")

    logger.info(f"Starting VQE run {i} (seed={seed})")

    result = differential_evolution(
        wrapped_cost,
        bounds,
        maxiter=max_iter,
        tol=tol,
        atol=abs_tol,
        strategy=strategy,
        popsize=popsize,
        init=scaled_samples,
        seed=seed,
        workers=1,
        callback=callback
    )

    run_end = datetime.now()
    logger.info(f"Completed VQE run {i}: Energy = {result.fun:.6f}")

    return {
        "seed": seed,
        "energy": result.fun,
        "params": result.x.tolist(),
        "success": result.success,
        "num_iters": result.nit,
        "num_evaluations": result.nfev,
        "run_time": run_end - run_start,
        "device_time": device_time
    }

if __name__ == "__main__":
    potential = "DW"
    shots = 10000
    cutoff = 16

    print(f"Running VQE for {potential} with cutoff {cutoff}")
    starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_path = os.path.join("/users/johnkerf/Quantum Computing/SUSY-QM/Qiskit-Vs-PL_pauli/Qiskit/Files", potential)
    log_path = os.path.join(base_path, "logs")
    os.makedirs(log_path, exist_ok=True)

    H = calculate_Hamiltonian(cutoff, potential)
    eigenvalues = np.sort(np.linalg.eigvals(H))[:4]
    num_qubits = int(1 + np.log2(cutoff))
    num_params = 2*num_qubits
    bounds = [(0, 2 * np.pi) for _ in range(num_params)]

    max_iter = 10000
    strategy = "randtobest1bin"
    tol = 1e-3
    abs_tol = 1e-3
    popsize = 20
    num_vqe_runs = 100

    vqe_starttime = datetime.now()

    with Pool(processes=100) as pool:
            vqe_results = pool.starmap(
                run_vqe,
                [
                    (i, bounds, max_iter, tol, abs_tol, strategy, popsize, H, num_qubits, shots, num_params, log_path)
                    for i in range(num_vqe_runs)
                ],
            )

    vqe_end = datetime.now()
    vqe_time = vqe_end - vqe_starttime

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

    output = {
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

    with open(os.path.join(base_path, f"{potential}_{cutoff}.json"), "w") as f:
        json.dump(output, f, indent=4)

    print("Done.")
