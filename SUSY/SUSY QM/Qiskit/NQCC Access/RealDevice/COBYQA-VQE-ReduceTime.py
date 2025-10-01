# vqe_qiskit_runner.py
import os, json, time, logging
from datetime import datetime
import numpy as np

from scipy.optimize import minimize

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from susy_qm import calculate_Hamiltonian


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import git
repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_tree_dir

#service = QiskitRuntimeService(name="NQCC-Q3")
path = r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\apikey.json"
with open(path, encoding="utf-8") as f:
    api_key = json.load(f).get("apikey")

NQCC_IBM_QUANTUM_API_KEY = api_key
ibm_instance_crn = "crn:v1:bluemix:public:quantum-computing:us-east:a/d4f95db0515b47b7ba61dba8a424f873:55736fd5-c0a0-4f44-8180-ce6e81d6c9d0::"
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=NQCC_IBM_QUANTUM_API_KEY, instance=ibm_instance_crn)
backend_name = 'ibm_kingston'
real_backend = service.backend(backend_name)

#noise_model = NoiseModel.from_backend(real_backend)
#aer_backend = AerSimulator(noise_model=noise_model)
    
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




def run_vqe(i, max_iter, initial_tr_radius, final_tr_radius, H, num_qubits, shots, num_params, log_dir, log_enabled, eps, lam, p):

    if log_enabled: os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"vqe_run_{i}.log")
    logger = setup_logger(log_path, f"logger_{i}", enabled=log_enabled)

    seed = (os.getpid() * int(time.time())) % 123456789

    run_start = datetime.now()

    observable = SparsePauliOp.from_operator(H)
    param_objs = [Parameter(f"θ{i}") for i in range(num_params)]

    qc = QuantumCircuit(num_qubits)
    n = num_qubits-1
    qc.x(n)
    qc.ry(param_objs[0], n)


    target = real_backend.target
    #target = aer_backend.target
    pm = generate_preset_pass_manager(target=target, optimization_level=3)
    ansatz_isa = pm.run(qc)

    hamiltonian_isa = observable.apply_layout(layout=ansatz_isa.layout)

    last_energy = None
    iteration_count = 0
    def cost_function(params):
        nonlocal last_energy, iteration_count
        pub = (ansatz_isa, [hamiltonian_isa], [params])
        result = estimator.run(pubs=[pub]).result()
        energy = result[0].data.evs[0]

        last_energy = energy
        iteration_count +=1

        neg = max(0.0, -(energy + eps))
        
        return energy + lam * (neg ** p)
    
    
    def callback(xk, convergence=None):
        if log_enabled:
            logger.info(f"Iteration {iteration_count}: Energy = {last_energy:.8f}")


    logger.info(f"Starting VQE run {i} (seed={seed})")

    np.random.seed(seed)
    x0 = np.random.random(size=num_params)*2*np.pi
    bounds = [(0, 2 * np.pi) for _ in range(num_params)]

    with Session(backend=real_backend) as session:
    #with Session(backend=aer_backend) as session:
     
        estimator = Estimator(mode=session)
        estimator.options.default_shots = shots
        estimator.options.resilience_level = 0
        logger.info(estimator.options)

        res = minimize(
            cost_function,
            x0,
            bounds=bounds,
            method= "COBYQA",
            options= {
                'maxiter':max_iter, 
                'maxfev':2*max_iter, 
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
        "message": res.message
        #"device_time": device_time
    })

    return {
        "seed": seed,
        "energy": res.fun,
        "params": res.x.tolist(),
        "success": res.success,
        "num_iters": int(res.nfev),
        "run_time": run_end - run_start
        #"device_time": device_time
    }

if __name__ == "__main__":

    log_enabled = True

    potential = "QHO"
    cutoff = 2

    shots = 1024
    num_params = 1

    num_vqe_runs = 1
    max_iter = 50
    initial_tr_radius = 0.3
    final_tr_radius = 1e-8

    lam = 15
    p = 2

    if potential == "AHO":
        i = np.log2(cutoff)
        factor = 2**(((i-1)*i)/2)
        eps = 0.5 / factor
    else:
        eps = 0

    print(f"Running for {potential} potential and cutoff {cutoff} and shots {shots}")

    starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_path = os.path.join(repo_path,r"SUSY\SUSY QM\Qiskit\NQCC Access\RealDevice\{}\Files".format(backend_name), str(starttime), str(shots), potential)
    os.makedirs(base_path, exist_ok=True)

    log_path = os.path.join(base_path, f"logs_{str(cutoff)}")
    
    H = calculate_Hamiltonian(cutoff, potential)
    eigenvalues = np.sort(np.linalg.eigvals(H))[:4]
    num_qubits = int(1 + np.log2(cutoff))

    vqe_starttime = datetime.now()

    i=1
    vqe_results =  run_vqe(i, max_iter, initial_tr_radius, final_tr_radius, H, num_qubits, shots, num_params, log_path, log_enabled, eps, lam, p)

    # Collect results
    seeds = vqe_results["seed"]
    energies = vqe_results["energy"]
    x_values = vqe_results["params"]
    success = vqe_results["success"]
    num_iters = vqe_results["num_iters"]
    run_time = str(vqe_results["run_time"])
    #device_time = str(vqe_results["device_time"])

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
            "name": "COBYQA",
            "maxiter": max_iter,
            'maxfev': 2*max_iter,
            "initial_tr_radius": initial_tr_radius,
            "final_tr_radius": final_tr_radius
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
        "run_time": run_time,
        #"device_time": device_time,
        "VQE_run_time": str(vqe_time),
        "seeds": seeds,
    }

    # Save the variable to a JSON file
    path = os.path.join(base_path, "{}_{}.json".format(potential, cutoff))
    with open(path, "w") as json_file:
        json.dump(run, json_file, indent=4)

    print("Done")
