# vqe_qiskit_runner.py
import os, json, time, logging
from datetime import datetime, timedelta
import numpy as np

from scipy.optimize import minimize

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorOptions, EstimatorV2 as Estimator
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from susy_qm import calculate_Hamiltonian

from multiprocessing import Pool

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import git
repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_tree_dir


def setup_logger(logfile_path, name):
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




def run_vqe(i, max_iter, tol, H, num_qubits, shots, num_params, log_dir):

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"vqe_run_{i}.log")
    logger = setup_logger(log_path, f"logger_{i}")

    seed = (os.getpid() * int(time.time())) % 123456789

    run_start = datetime.now()
    device_time = timedelta()

    backend = AerSimulator()

    observable = SparsePauliOp.from_operator(H)

    param_objs = [Parameter(f"θ{i}") for i in range(num_params)]

    '''
    ############ QHO ##################
    qc = QuantumCircuit(num_qubits)
    n = num_qubits-1
    qc.x(n)
    qc.ry(param_objs[0], n)
    '''

    '''
    ############ AHO ##################
    qc = QuantumCircuit(num_qubits)

    # 2
    #qc.x(1)
    #qc.ry(param_objs[0], 1)

    # 4
    #qc.x(2)
    #qc.ry(param_objs[0], 1)

    # 8+
    n = num_qubits-1
    qc.x(n)
    qc.ry(param_objs[0], 1)
    qc.ry(param_objs[1], 2)
    '''

    #'''
    ############ DW ##################
    qc = QuantumCircuit(num_qubits)

    qc.ry(param_objs[0], 0)

    #qc.x(2)
    #qc.ry(param_objs[0], 0)
    #qc.ry(param_objs[1], 1)

    #qc.ry(param_objs[0], 0)   
    #qc.ry(param_objs[1], 2)
    #qc.cry(param_objs[2], 0, 1)
    #qc.ry(param_objs[3], 1)
    #qc.ry(param_objs[4], 0)
    #'''

    target = backend.target
    pm = generate_preset_pass_manager(target=target, optimization_level=3)
    ansatz_isa = pm.run(qc)

    hamiltonian_isa = observable.apply_layout(layout=ansatz_isa.layout)

    def cost_only(theta):
        return cost_function([theta], ansatz_isa, hamiltonian_isa, estimator)[0]

    def grad_theta(theta):
        s = np.pi/2
        return 0.5*(cost_only(theta + s) - cost_only(theta - s))

    def wrapped_cost_scalar(theta):
        return wrapped_cost([theta[0]])

    def wrapped_grad(theta):
        return np.array([grad_theta(theta[0])])


    def wrapped_cost(params):
        result, dt = cost_function(params, ansatz_isa, hamiltonian_isa, estimator)
        nonlocal device_time
        device_time += dt
        return result


    logger.info(f"Starting VQE run {i} (seed={seed})")

    np.random.seed(seed)
    x0 = np.random.random(size=num_params)*2*np.pi
    bounds = [(0, 2 * np.pi) for _ in range(num_params)]

    with Session(backend=backend) as session:
        estimator = Estimator(mode=session)
        estimator.options.default_shots = shots

        res = minimize(
            wrapped_cost_scalar,
            x0=x0,
            jac=wrapped_grad,            # <-- provide gradient!
            method="L-BFGS-B",
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': tol}
        )

    run_end = datetime.now()
    logger.info(f"Completed VQE run {i}: Energy = {res.fun:.6f}")

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

    potential = "DW"
    shotslist = [None, 10000, 100000]
    cutoffs = [2]

    for shots in shotslist:
        for cutoff in cutoffs:

            print(f"Running for {potential} potential and cutoff {cutoff} and shots {shots}")

            starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            base_path = os.path.join(repo_path,r"SUSY\SUSY QM\Qiskit\L-BFGS-B\Files", str(shots), potential)
            log_path = os.path.join(base_path, f"logs_{str(cutoff)}")
            os.makedirs(log_path, exist_ok=True)

            H = calculate_Hamiltonian(cutoff, potential)
            eigenvalues = np.sort(np.linalg.eigvals(H))[:4]
            num_qubits = int(1 + np.log2(cutoff))

            num_params = 1
            num_vqe_runs = 100
            max_iter = 10000
            tol = 1e-8

            vqe_starttime = datetime.now()

            #Start multiprocessing for VQE runs
            with Pool(processes=10) as pool:
                vqe_results = pool.starmap(
                    run_vqe,
                    [
                        (i, max_iter, tol, H, num_qubits, shots, num_params, log_path)
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
                "shots": shots,
                "Optimizer": {
                    "name": "L--BFGS-B",
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
