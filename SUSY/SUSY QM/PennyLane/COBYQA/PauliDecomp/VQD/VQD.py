import pennylane as qml

from scipy.optimize import minimize

import os, json, time, logging
import numpy as np
from datetime import datetime, timedelta


from multiprocessing import Pool

from susy_qm import calculate_Hamiltonian, ansatze

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


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


def cost_function(params, prev_param_list, num_qubits, beta, dev, swap_dev, eps, lam, p, paulis, coeffs):
 
   
    def ansatz(params, prev=False): 

        basis = [0]*num_qubits
        wires = np.arange(num_qubits)

        if prev==True:
            qml.BasisState(basis, wires=range(num_qubits, 2*num_qubits))
            wires = wires + num_qubits
        else:
            qml.BasisState(basis, wires=range(num_qubits))
  
        n = num_qubits-1
        for i, w in enumerate(wires):
            qml.RY(params[i], wires=w)

        for i in range(1, num_qubits):
            qml.CNOT(wires=[wires[i-1], wires[i]])

        #qml.CNOT(wires=[wires[-1], wires[0]])

        for i, w in enumerate(wires):
            qml.RY(params[n + i], wires=w)


    #Swap test to calculate overlap
    @qml.qnode(swap_dev)
    def swap_test(params1, params2):

        ansatz(params1)
        ansatz(params2, prev=True)

        qml.Barrier()
        for i in range(num_qubits):
            qml.CNOT(wires=[i, i+num_qubits])    
            qml.Hadamard(wires=i)      

        prob = qml.probs(wires=range(2*num_qubits))

        return prob
    
    
    @qml.qnode(dev)
    def expected_value(params):
        ansatz(params)
        return [qml.expval(op) for op in paulis]
    
    
    def overlap(params, prev_params):

        probs = swap_test(params, prev_params)

        overlap = 0
        for idx, p in enumerate(probs):

            bitstring = format(idx, '0{}b'.format(2*num_qubits))

            counter_11 = 0
            for i in range(num_qubits):
                a = int(bitstring[i])
                b = int(bitstring[i+num_qubits])
                if (a == 1 and b == 1):
                    counter_11 +=1

            overlap += p*(-1)**counter_11

        return overlap
        

    def loss_f(params):

        expvals = expected_value(params)
        energy = float(np.dot(coeffs, expvals))    

        neg = max(0.0, -(energy + eps))
        penilised_e = energy + lam * (neg ** p)

        penalty = 0

        if len(prev_param_list) != 0:
            for prev_param in prev_param_list:
                ol = overlap(params,prev_param)
                penalty += beta*ol


        return penilised_e + penalty

    return loss_f(params)


def run_vqd(i, paulis, coeffs, run_info, log_enabled, log_dir):

    if log_enabled: os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"vqe_run_{i}.log")
    logger = setup_logger(log_path, f"logger_{i}", enabled=log_enabled)
    
    # We need to generate a random seed for each process otherwise each parallelised run will have the same result
    seed = (os.getpid() * int(time.time())) % 123456789

    num_qubits = run_info["num_qubits"]
    num_params = run_info["num_params"]

    dev = qml.device(run_info["device"], wires=num_qubits, shots=run_info["shots"], seed=seed)
    swap_dev = qml.device(run_info["device"], wires=2*num_qubits, shots=run_info["swap_shots"], seed=seed)

    np.random.seed(seed)
    x0 = np.random.random(size=num_params)*2*np.pi
    if run_info["use_bounds"]:
        bounds = [(0, 2 * np.pi) for _ in range(num_params)]
    else:
        bounds = [(None, None) for _ in range(num_params)]

    run_info["seed"] = seed

    if log_enabled: logger.info(json.dumps(run_info, indent=4, default=str))
    if log_enabled: logger.info(f"Starting VQE run {i} (seed={seed})")

    all_energies = []
    prev_param_list = []
    all_success = []
    all_num_iters = []
    all_evaluations = []

    run_start = datetime.now()
    
    Tdev = qml.Tracker(dev)
    Tswap = qml.Tracker(swap_dev)

    for _ in range(run_info["num_energy_levels"]):

        Tdev.reset() 
        Tswap.reset()
        
        with Tdev, Tswap:
            res = minimize(
                cost_function,
                x0,
                args=(prev_param_list, num_qubits, run_info["beta"], dev, swap_dev, run_info["eps"], run_info["lam"], run_info["p"], paulis, coeffs),
                bounds=bounds,
                method= "COBYQA",
                options= run_info["optimizer_options"]
            )

        totals_d = getattr(Tdev, "totals", {})
        num_evals_d = int(totals_d.get("executions", 0))
        totals_s = getattr(Tswap, "totals", {})
        num_evals_s = int(totals_s.get("executions", 0))

        num_evals = num_evals_d + num_evals_s

        all_energies.append(res.fun)
        prev_param_list.append(res.x)
        all_success.append(res.success)
        all_num_iters.append(res.nit)
        all_evaluations.append(num_evals)

    run_end = datetime.now()
    run_time = run_end - run_start

    results_data = {
        "seed": seed,
        "energies": all_energies,
        "params": prev_param_list,
        "success": all_success,
        "num_iters": all_num_iters,
        "evaluations": all_evaluations,
        "run_time": run_time
    }
    
    if log_enabled: logger.info(json.dumps(results_data, indent=4, default=str))

    return results_data


if __name__ == "__main__":

    log_enabled = False

    device = "default.qubit"
    
    potential = "QHO"
    cutoffs = [2,4,8,16,32]

    ansatz_name = "real_amplitudes"

    shots=10000
    swap_shots=10000

    lam = 15
    p = 2
    
    num_vqd_runs = 100
    num_energy_levels = 3    
    beta = 5.0

    max_iter = 10000
    initial_tr_radius = 0.2
    final_tr_radius = 1e-8


    starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_path = os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\SUSY QM\PennyLane\COBYQA\PauliDecomp\VQD\Files\Test", str(shots), potential)
    os.makedirs(base_path, exist_ok=True)

    print(f"Running for {potential} potential")

    use_bounds = False if shots == None else True

    for cutoff in cutoffs:

        log_path = os.path.join(base_path, f"logs_{str(cutoff)}")

        if potential == "AHO":
            i = np.log2(cutoff)
            factor = 2**(((i-1)*i)/2)
            eps = 0.5 / factor
        else:
            eps = 0

        print(f"Running for cutoff: {cutoff}")

        # Calculate Hamiltonian and expected eigenvalues
        H = calculate_Hamiltonian(cutoff, potential)
        eigenvalues = np.sort(np.linalg.eig(H)[0])[:3]
        min_eigenvalue = min(eigenvalues.real)

        num_qubits = int(np.log2(cutoff)+1)    

        H_decomp = qml.pauli_decompose(H, wire_order=range(num_qubits))
        paulis = H_decomp.ops
        coeffs = H_decomp.coeffs

        num_params = 2*num_qubits

        run_info = {"device":device,
                    "Potential":potential,
                    "cutoff": cutoff,
                    "ansatz_name": ansatz_name,
                    "num_qubits": num_qubits,
                    "num_paulis": len(paulis),
                    "num_params": num_params,
                    "shots": shots,
                    "swap_shots": swap_shots,
                    "beta": beta,
                    "lam": lam,
                    "p":p,
                    "eps":eps,
                    "num_vqd_runs": num_vqd_runs,
                    "num_energy_levels": num_energy_levels,
                    "optimizer_options": {
                        'maxiter':max_iter, 
                        'maxfev':max_iter, 
                        'initial_tr_radius':initial_tr_radius, 
                        'final_tr_radius':final_tr_radius, 
                        'scale':True, 
                        'disp':False
                        },
                    "use_bounds": use_bounds,
                    "path": base_path
                    }

        print(json.dumps(run_info, indent=4, default=str))

        # Start multiprocessing for VQE runs
        with Pool(processes=10) as pool:
            vqd_results = pool.starmap(
                run_vqd,
                [
                    (i, paulis, coeffs, run_info, log_enabled, log_path)
                    for i in range(num_vqd_runs)
                ],
            )

        # Collect results
        seeds = [res["seed"] for res in vqd_results]
        all_energies = [result["energies"] for result in vqd_results]
        all_params = [result["params"] for result in vqd_results]
        all_success = [result["success"] for result in vqd_results]
        all_num_iters = [result["num_iters"] for result in vqd_results]
        all_evaluations = [result["evaluations"] for result in vqd_results]
        run_times = [str(res["run_time"]) for res in vqd_results]
        total_run_time = sum([res["run_time"] for res in vqd_results], timedelta())

        vqd_end = datetime.now()
        vqd_time = vqd_end - datetime.strptime(starttime, "%Y-%m-%d_%H-%M-%S")

        # Save run
        run = {
            "device": device,
            "starttime": starttime,
            "endtime": vqd_end.strftime("%Y-%m-%d_%H-%M-%S"),
            "potential": potential,
            "cutoff": cutoff,
            "ansatz": ansatz_name,
            "num_qubits": num_qubits,
            "num_paulis": len(paulis),
            "num_params": num_params,
            "exact_eigenvalues": [x.real.tolist() for x in eigenvalues],
            "num_VQD": num_vqd_runs,
            "num_energy_levels": num_energy_levels,
            "beta": beta,
            "shots": shots,
            "swap_shots": swap_shots,
            "Optimizer": {
                "name": "COBYQA",
                "maxiter": max_iter,
                'maxfev': max_iter,
                "initial_tr_radius": initial_tr_radius,
                "final_tr_radius": final_tr_radius
            },
            "cost function":{
                "type": "small negatives",
                "p":p,
                "lam":lam,
                "eps":eps
            },
            "results": all_energies,
            "params": [[x.tolist() for x in param_list] for param_list in all_params],
            "num_iters": all_num_iters,
            "num_evaluations": all_evaluations,
            "success": [np.array(x, dtype=bool).tolist() for x in all_success],
            "run_times": run_times,
            "parallel_run_time": str(vqd_time),
            "total_VQD_time": str(total_run_time),
            "seeds": seeds
        }

        # Save the variable to a JSON file
        path = os.path.join(base_path, "{}_{}.json".format(potential, cutoff))
        with open(path, "w") as json_file:
            json.dump(run, json_file, indent=4)

        print("Done")
