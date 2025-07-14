# vqe_qiskit_runner.py
import os, json, time, logging
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats.qmc import Halton

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import Operator

from susy_qm import calculate_Hamiltonian2

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Global persistent estimator objects ---
estimator = None
observable = None
param_objs = None
circuit_template = None

def setup_logger(logfile_path, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(logfile_path)
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def cost_function(params, H):

    start = datetime.now()

    observable = SparsePauliOp.from_operator(Operator(H))
    param_objs = [Parameter(f"θ{i}") for i in range(4)]

    qc = QuantumCircuit(num_qubits)
    qc.ry(param_objs[0], 0)     
    qc.ry(param_objs[1], 1)     
    qc.ry(param_objs[2], 2)     
    qc.ry(param_objs[3], 3)

    aer_sim = AerSimulator(method='automatic')
    pm = generate_preset_pass_manager(backend=aer_sim, optimization_level=3)
    isa_qc = pm.run(qc)

    estimator = AerEstimator(run_options={"seed": 42, "shots":shots})
    result = estimator.run([isa_qc], [observable], [params]).result()
    
    energy = result.values[0]

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

    def wrapped_cost(params, H):
        result, dt = cost_function(params, H)
        nonlocal device_time
        device_time += dt
        return result

    iteration_count = 0
    def callback(xk, convergence=None):
        nonlocal iteration_count
        iteration_count += 1
        if iteration_count % 50 == 0:
            energy, _ = cost_function(xk)
            logger.info(f"Iteration {iteration_count}: Energy = {energy:.8f}")

    logger.info(f"Starting VQE run {i} (seed={seed})")

    result = differential_evolution(
        wrapped_cost,
        args=(H,),
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
        "run_time": str(run_end - run_start),
        "device_time": str(device_time)
    }

if __name__ == "__main__":

    potential = "AHO"
    shots = 1024
    cutoff = 8

    print(f"Running VQE for {potential} with cutoff {cutoff}")
    starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_path = os.path.join(r"C:\Users\johnkerf\Desktop\Quantum-Computing\Quantum-Computing\SUSY\SUSY QM\Qiskit\TestOlderVersion\VQEFiles", potential, str(starttime))
    log_path = os.path.join(base_path, "logs")
    os.makedirs(log_path, exist_ok=True)

    H = calculate_Hamiltonian2(cutoff, potential)
    eigenvalues = np.sort(np.linalg.eigvals(H))[:4]
    num_qubits = int(1 + np.log2(cutoff))

    num_params = 4
    bounds = [(0, 2 * np.pi) for _ in range(num_params)]

    max_iter = 1000
    strategy = "randtobest1bin"
    tol = 1e-3
    abs_tol = 1e-3
    popsize = 20
    num_vqe_runs = 1

    results = []
    for i in range(num_vqe_runs):
        res = run_vqe(i, bounds, max_iter, tol, abs_tol, strategy, popsize, H, num_qubits, shots, num_params, log_path)
        results.append(res)

    output = {
        "potential": potential,
        "cutoff": cutoff,
        "exact_eigenvalues": [x.real for x in eigenvalues],
        "shots": shots,
        "results": results,
    }

    with open(os.path.join(base_path, f"{potential}_{cutoff}.json"), "w") as f:
        json.dump(output, f, indent=4)

    print("Done.")
