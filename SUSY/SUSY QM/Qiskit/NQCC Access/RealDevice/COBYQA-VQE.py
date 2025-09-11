# vqe_qiskit_runner.py
import os, json, time, logging
from datetime import datetime, timedelta
import numpy as np

from scipy.optimize import minimize

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import EstimatorOptions, EstimatorV2 as Estimator
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from susy_qm import calculate_Hamiltonian

from multiprocessing import Pool

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import git
repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_tree_dir

#service = QiskitRuntimeService(name="NQCC-Q3")
NQCC_IBM_QUANTUM_API_KEY = ""
ibm_instance_crn = ""
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=NQCC_IBM_QUANTUM_API_KEY, instance=ibm_instance_crn)
backend_name = 'ibm_kingston'
real_backend = service.backend(backend_name)

def setup_logger(logfile_path, name, enabled=True):
    if not enabled:
        
        logger = logging.getLogger(f"{name}_disabled")
        logger.handlers = []               
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        logger.setLevel(logging.CRITICAL)
        return logger

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(logfile_path)
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger



def  cost_function(params, ansatz, hamiltonian, estimator):

    start = datetime.now()

    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]

    end = datetime.now()

    return energy, end - start


def run_vqe(i, max_iter, initial_tr_radius, final_tr_radius, H, num_qubits, shots, num_params, log_dir, log_enabled):

    if log_enabled: os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"vqe_run_{i}.log")
    logger = setup_logger(log_path, f"logger_{i}", enabled=log_enabled)

    seed = (os.getpid() * int(time.time())) % 123456789

    run_start = datetime.now()
    device_time = timedelta()

    observable = SparsePauliOp.from_operator(H)
    param_objs = [Parameter(f"θ{i}") for i in range(num_params)]

    qc = QuantumCircuit(num_qubits)
    n = num_qubits-1
    qc.x(n)
    qc.ry(param_objs[0], n)


    target = real_backend.target
    pm = generate_preset_pass_manager(target=target, optimization_level=3)
    ansatz_isa = pm.run(qc)

    hamiltonian_isa = observable.apply_layout(layout=ansatz_isa.layout)


    def wrapped_cost(params):
        result, dt = cost_function(params, ansatz_isa, hamiltonian_isa, estimator)
        nonlocal device_time
        device_time += dt
        return result
    
    iteration_count = 0
    def callback(xk, convergence=None):
        nonlocal iteration_count
        iteration_count += 1
        if log_enabled and iteration_count % 1 == 0:
            energy, _ = cost_function(xk, ansatz_isa, hamiltonian_isa, estimator)
            logger.info(f"Iteration {iteration_count}: Energy = {energy:.8f}")


    logger.info(f"Starting VQE run {i} (seed={seed})")

    np.random.seed(seed)
    x0 = np.random.random(size=num_params)*2*np.pi
    bounds = [(0, 2 * np.pi) for _ in range(num_params)]

    with Session(backend=real_backend) as session:
        estimator = Estimator(mode=session)
        estimator.options.default_shots = shots
        #estimator.options.resilience_level = 2

        logger.info(estimator.options)

        res = minimize(
            wrapped_cost,
            x0,
            bounds=bounds,
            method= "COBYQA",
            options= {
                'maxiter':max_iter, 
                'maxfev':max_iter, 
                'initial_tr_radius':initial_tr_radius, 
                'final_tr_radius':final_tr_radius, 
                'scale':True, 
                'disp':False},
            callback=callback
        )

    run_end = datetime.now()
    if log_enabled: logger.info(f"Completed VQE run {i}: Energy = {res.fun:.6f}")
    if log_enabled: logger.info({
        "seed": seed,
        "energy": res.fun,
        "params": res.x.tolist(),
        "success": res.success,
        "num_iters": int(res.nfev),
        "run_time": run_end - run_start,
        "device_time": device_time
    })

    return {
        "seed": seed,
        "energy": res.fun,
        "params": res.x.tolist(),
        "success": res.success,
        "num_iters": int(res.nfev),
        "run_time": run_end - run_start,
        "device_time": device_time
    }

if __name__ == "__main__":

    log_enabled = True

    potential = "QHO"
    shotslist = [1024]
    cutoffs = [2]

    for shots in shotslist:
        for cutoff in cutoffs:

            print(f"Running for {potential} potential and cutoff {cutoff} and shots {shots}")

            starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            base_path = os.path.join(repo_path,r"SUSY\SUSY QM\Qiskit\NQCC Access\RealDevice\{}\Files".format(backend_name), str(shots), potential)
            os.makedirs(base_path, exist_ok=True)

            log_path = os.path.join(base_path, f"logs_{str(cutoff)}")
            
            H = calculate_Hamiltonian(cutoff, potential)
            eigenvalues = np.sort(np.linalg.eigvals(H))[:4]
            num_qubits = int(1 + np.log2(cutoff))

            num_params = 1
            num_vqe_runs = 1
            max_iter = 20
            initial_tr_radius = 0.8
            final_tr_radius = 1e-3

            vqe_starttime = datetime.now()

            i=1
            vqe_results =  run_vqe(i, max_iter, initial_tr_radius, final_tr_radius, H, num_qubits, shots, num_params, log_path, log_enabled)

            # Collect results
            seeds = vqe_results["seed"]
            energies = vqe_results["energy"]
            x_values = vqe_results["params"]
            success = vqe_results["success"]
            num_iters = vqe_results["num_iters"]
            run_time = str(vqe_results["run_time"])
            device_time = str(vqe_results["device_time"])

            vqe_end = datetime.now()
            vqe_time = vqe_end - vqe_starttime

            # Save run
            run = {
                "backend": backend_name,
                "starttime": starttime,
                "endtime": vqe_end.strftime("%Y-%m-%d_%H-%M-%S"),
                "potential": potential,
                "cutoff": cutoff,
                "exact_eigenvalues": [x.real.tolist() for x in eigenvalues],
                "ansatz": "circuit.txt",
                "num_VQE": num_vqe_runs,
                "shots": shots,
                "Optimizer": {
                    "name": "COBYLA",
                    "maxiter": max_iter,
                    'maxfev': max_iter,
                    "initial_tr_radius": initial_tr_radius,
                    "final_tr_radius": final_tr_radius
                },
                "results": energies,
                "params": x_values,
                "num_iters": num_iters,
                "success": np.array(success, dtype=bool).tolist(),
                "run_time": run_time,
                "device_time": device_time,
                "VQE_run_time": str(vqe_time),
                "seeds": seeds,
            }

            # Save the variable to a JSON file
            path = os.path.join(base_path, "{}_{}.json".format(potential, cutoff))
            with open(path, "w") as json_file:
                json.dump(run, json_file, indent=4)

            print("Done")
