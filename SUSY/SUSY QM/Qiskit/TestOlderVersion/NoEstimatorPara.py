# vqe_qiskit_runner.py
import os, json, time, logging
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats.qmc import Halton

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from scipy.linalg import eigh, qr

from multiprocessing import Pool

from susy_qm import calculate_Hamiltonian2

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

def build_ansatz(params_labels):
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(params_labels[i], i)
    return qc

def cost_function(params, params_labels, projector_circuits, sim, shots):
    start = datetime.now()
    
    bindings = dict(zip(params_labels, params))
    bound_circuits = [qc.assign_parameters(bindings, inplace=False) for _, qc in projector_circuits]
    transpiled = transpile(bound_circuits, backend=sim)
    result = sim.run(transpiled, shots=shots).result()

    energy = 0.0
    for i, (eigval, _) in enumerate(projector_circuits):
        counts = result.results[i].data.counts
        counts = {k[::-1]: v for k, v in counts.items()}
        prob_0 = counts.get('0x0', 0) / shots
        energy += eigval * prob_0

    end = datetime.now()

    return energy, end - start



def run_vqe(i, bounds, max_iter, tol, abs_tol, strategy, popsize, num_params, log_dir, params_labels, projector_circuits, shots):

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"vqe_run_{i}.log")
    logger = setup_logger(log_path, f"logger_{i}")

    run_start = datetime.now()

    seed = (os.getpid() * int(time.time())) % 123456789
    halton_sampler = Halton(d=num_params, seed=seed)
    scaled_samples = 2 * np.pi * halton_sampler.random(n=popsize)

    sim = AerSimulator(method="automatic", seed_simulator=seed)

    device_time = timedelta()

    def wrapped_cost(params):
        result, dt = cost_function(params, params_labels, projector_circuits, sim, shots)
        nonlocal device_time
        device_time += dt
        return result

    iteration_count = 0
    def callback(xk, convergence=None):
        nonlocal iteration_count
        iteration_count += 1
        if iteration_count % 50 == 0:
            energy, _ = cost_function(xk, params_labels, projector_circuits, sim, shots)
            logger.info(f"Iteration {iteration_count}: Energy = {energy:.8f}")

    logger.info(f"Starting VQE run {i} (seed={seed})")

    result = differential_evolution(
        wrapped_cost,
        bounds=bounds,
        maxiter=max_iter,
        tol=tol,
        atol=abs_tol,
        strategy=strategy,
        popsize=popsize,
        init=scaled_samples,
        seed=seed,
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

    potential = "QHO"
    shots = 1024
    cutoff = 2

    print(f"Running VQE for {potential} with cutoff {cutoff}")
    starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_path = os.path.join(r"C:\Users\johnkerf\Desktop\Quantum-Computing\Quantum-Computing\SUSY\SUSY QM\Qiskit\TestOlderVersion\VQEFiles2", potential, str(starttime))
    log_path = os.path.join(base_path, "logs")
    os.makedirs(log_path, exist_ok=True)

    H = calculate_Hamiltonian2(cutoff, potential)
    eigvals, eigvecs = eigh(H)
    n = int(np.log2(H.shape[0]))
    shots = 1024

    num_params = n
    params_labels = [Parameter(f'theta_{i}') for i in range(num_params)]

    ansatz_template = build_ansatz(params_labels)

    # Precompute projectors (U† circuits) once
    projector_circuits = []
    for i in range(len(eigvals)):
        eigval = eigvals[i]
        eigvec = eigvecs[:, i]
        cols = [eigvec] + [np.eye(2**n)[:, j] for j in range(1, 2**n)]
        U = qr(np.column_stack(cols))[0]
        U_dag = U.conj().T

        qc = ansatz_template.copy()
        qc.unitary(U_dag, range(n))
        qc.measure_all()
        projector_circuits.append((eigval, qc))

    bounds = [(0, 2 * np.pi) for _ in range(num_params)]

    max_iter = 1000
    strategy = "randtobest1bin"
    tol = 1e-3
    abs_tol = 1e-3
    popsize = 20
    num_vqe_runs = 2

    vqe_starttime = datetime.now()

    with Pool(processes=2) as pool:
        vqe_results = pool.starmap(
            run_vqe,
            [
                (i, bounds, max_iter, tol, abs_tol, strategy, popsize, num_params, log_path, params_labels, projector_circuits, shots)
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
    
    run = {
            "starttime": starttime,
            "endtime": vqe_end.strftime("%Y-%m-%d_%H-%M-%S"),
            "potential": potential,
            "cutoff": cutoff,
            "exact_eigenvalues": [x.real.tolist() for x in eigvals],
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
        json.dump(run, f, indent=4)

    print("Done.")
