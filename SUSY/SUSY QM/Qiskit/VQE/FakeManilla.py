import os, json, time, logging
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats.qmc import Halton

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import SparsePauliOp

from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit_ibm_runtime import RuntimeOptions
from qiskit import transpile

from susy_qm import calculate_Hamiltonian

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from qiskit.circuit.library import RYGate

def apply_cry(theta, control, target, circuit):
    """Apply a CRY gate using standard decomposition."""
    circuit.ry(theta / 2, target)
    circuit.cx(control, target)
    circuit.ry(-theta / 2, target)
    circuit.cx(control, target)

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

def initialize_vqe(H_matrix, num_qubits, shots):
    global estimator, observable, param_objs, circuit_template

    observable = SparsePauliOp.from_operator(H_matrix)
    param_objs = [Parameter(f"θ{i}") for i in range(4)]

    print(num_qubits)
    qc = QuantumCircuit(num_qubits)
    qc.ry(param_objs[0], 0)     # θ₁ on q0
    qc.ry(param_objs[1], 2)     # θ₂ on q2
    apply_cry(param_objs[2], control=2, target=1, circuit=qc)
    qc.ry(param_objs[3], 1)     # θ₄ on q1

    print("[DEBUG] Initial circuit:")
    print(qc)

    fake_backend = FakeManilaV2()
    options = {
        "seed_simulator": 42,      # simulator option promoted
        "resilience_level": 0
    }
    estimator = Estimator(mode=fake_backend, options=options)

    transpiled_qc = transpile(qc, backend=fake_backend, optimization_level=1)
    circuit_template = transpiled_qc

def cost_function(params):
    global estimator, observable, param_objs, circuit_template


    start = datetime.now()

    param_dict = dict(zip(param_objs, params))
    bound = circuit_template.assign_parameters(param_dict, inplace=False)

    print("[CHECK] bound.num_qubits =", bound.num_qubits)
    print("[CHECK] bound.qregs:", bound.qregs)
    print("[CHECK] circuit:")
    print(bound)

    #runtime_options = RuntimeOptions(shots=shots)
    job = estimator.run([(bound, observable)])
    result = job.result()
    energy = result[0]  # EstimatorV2 returns list of results

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

    def wrapped_cost(params):
        result, dt = cost_function(params)
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
        "run_time": str(run_end - run_start),
        "device_time": str(device_time)
    }

if __name__ == "__main__":
    potential = "DW"
    shots = 1024
    cutoff_list = [8]

    for cutoff in cutoff_list:
        print(f"Running VQE for {potential} with cutoff {cutoff}")
        starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_path = os.path.join(
            r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\SUSY QM\Qiskit\VQE\test3",
            potential, str(starttime)
        )
        log_path = os.path.join(base_path, "logs")
        os.makedirs(log_path, exist_ok=True)

        H = calculate_Hamiltonian(cutoff, potential)
        print(H.shape[0])
        eigenvalues = np.sort(np.linalg.eigvals(H))[:4]
        num_qubits = int(1 + np.log2(cutoff))
        print(num_qubits)
        num_params = 4
        bounds = [(0, 2 * np.pi) for _ in range(num_params)]

        max_iter = 1000
        strategy = "randtobest1bin"
        tol = 1e-3
        abs_tol = 1e-3
        popsize = 20
        num_vqe_runs = 1

        initialize_vqe(H, num_qubits, shots)

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
