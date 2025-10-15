import pennylane as qml
from pennylane import numpy as pnp
from pennylane.pauli import group_observables

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


def cost_function(params, paulis, coeffs, num_qubits, dev, ansatz, max_gate):

    #groups = group_observables(paulis)
    #gates = ansatze.truncate_ansatz(ansatz, params, num_qubits, max_gates=max_gate)
       
    @qml.qnode(dev)
    def circuit(params):

        ansatz(params, num_qubits, include_fermion=False)
   
        #for gate in gates:
        #    qml.apply(gate)   

        #return [qml.expval(op) for op in group] ##grouped
        return [qml.expval(op) for op in paulis] ##ungrouped

    ### un-grouped paulis ###
    expvals = circuit(params)                 
    energy = float(np.dot(coeffs, expvals)) 

    ### grouped paulis ###
    #energy = 0
    #for group in groups:
    #    results = circuit(params, group)
    #    for op, res in zip(group, results):
    #        idx = paulis.index(op)
    #        energy += coeffs[idx] * res

    return energy

    
def run_vqe(i, max_iter, tol, initial_tr_radius, final_tr_radius, paulis, coeffs, num_qubits, shots, num_params, device, log_dir, log_enabled, eps, lam, p, ansatz, max_gate, use_bounds, run_info):

    if log_enabled: os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"vqe_run_{i}.log")
    logger = setup_logger(log_path, f"logger_{i}", enabled=log_enabled)

    # We need to generate a random seed for each process otherwise each parallelised run will have the same result
    seed = (os.getpid() * int(time.time())) % 123456789
    dev = qml.device(device, wires=num_qubits, shots=shots, seed=seed)
    run_start = datetime.now()

    last_energy = None
    iteration_count = 0
    def wrapped_cost_function(params):
        nonlocal last_energy, iteration_count
        result = cost_function(params, paulis, coeffs, num_qubits, dev, ansatz, max_gate)

        last_energy = result
        iteration_count +=1

        neg = max(0.0, -(result + eps))

        return result + lam * (neg ** p)


    def callback(xk, convergence=None):
        if log_enabled and iteration_count % 10 == 0:
            logger.info(f"Iteration {iteration_count}: Energy = {last_energy:.8f}")

    np.random.seed(seed)
    x0 = np.random.random(size=num_params)*2*np.pi

    if use_bounds:
        bounds = [(0, 2 * np.pi) for _ in range(num_params)]
    else:
        bounds = [(None, None) for _ in range(num_params)]

    run_info["seed"] = seed
    run_info["bounds"] = bounds

    if log_enabled: logger.info(json.dumps(run_info, indent=4, default=str))
    if log_enabled: logger.info(f"Starting VQE run {i} (seed={seed})")

    with qml.Tracker(dev) as tracker:
        res = minimize(
                wrapped_cost_function,
                x0,
                bounds=bounds,
                method= "COBYQA",
                options= {
                    'maxiter':max_iter, 
                    'maxfev':max_iter, 
                    #'tol':tol, 
                    'initial_tr_radius':initial_tr_radius, 
                    'final_tr_radius':final_tr_radius, 
                    'scale':True, 
                    'disp':False},
                callback=callback
            )

    run_end = datetime.now()
    run_time = run_end - run_start

    if log_enabled: logger.info(f"Completed VQE run {i}: Energy = {res.fun:.6f}")
    if log_enabled: logger.info(f"optimizer message: {res.message}")

    totals = getattr(tracker, "totals", {})
    num_evals = int(totals.get("executions", 0))

    results_data = {
        "seed": seed,
        "energy": res.fun,
        "params": res.x.tolist(),
        "success": res.success,
        "num_iters": res.nit,
        "num_evaluations": num_evals,
        "run_time": run_time
    }

    if log_enabled: logger.info(json.dumps(results_data, indent=4, default=str))

    return results_data

if __name__ == "__main__":
    
    log_enabled = True

    potential = "DW"
    device = 'default.qubit'
    #device = 'qiskit.aer'

    #shotslist = [10000, None, 100000]
    shotslist = [None] #for shots=None dont use bounds
    cutoffs = [16]

    lam = 15
    p = 2

    max_gate=0
    
    # Optimizer
    num_vqe_runs = 100
    max_iter = 10000
    tol = 1e-8
    initial_tr_radius = 1.0
    final_tr_radius = 1e-8

    for shots in shotslist:

        use_bounds = False if shots == None else True

        print(f"Using bounds: {use_bounds}")
        
        for cutoff in cutoffs:

            ansatze_type = 'exact' #exact, Reduced, CD3

            if potential == "QHO":
                ansatz_name = f"CQAVQE_QHO_{ansatze_type}"
            elif (potential != "QHO") and (cutoff <= 64):
                ansatz_name = f"CQAVQE_{potential}{cutoff}_{ansatze_type}"
            else:
                ansatz_name = f"CQAVQE_{potential}16_{ansatze_type}"

            #ansatz_name = f"CQAVQE_{potential}{cutoff}_CD3"
            #ansatz_name = "real_amplitudes"

            ansatz = ansatze.get(ansatz_name)
            

            if potential == "AHO":
                i = np.log2(cutoff)
                factor = 2**(((i-1)*i)/2)
                eps = 0.5 / factor
            else:
                eps = 0

            print(f"Running for {potential} potential and cutoff {cutoff} and shots {shots}")

            #for max_gate in range(1, num_params+1):
            #print(f"Running for gate {max_gate}")

            starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            base_path = os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\SUSY QM\PennyLane\COBYQA\PauliDecomp\VQE\ReducedHamExact", str(shots), potential)
            os.makedirs(base_path, exist_ok=True)

            log_path = os.path.join(base_path, f"logs_{str(cutoff)}")

            # Calculate Hamiltonian and expected eigenvalues
            H = calculate_Hamiltonian(cutoff, potential)

            if potential in ["QHO","AHO"] or ((potential == "DW") and (cutoff == 4)):
                H = H[cutoff:, cutoff:]
            else:
                H = H[:cutoff, :cutoff]
            
            eigenvalues = np.sort(np.linalg.eig(H)[0])[:4]
            num_qubits = int(np.log2(cutoff))

            if ansatz_name == 'real_amplitudes':
                num_params = 2*num_qubits
            else:
                num_params = ansatz.n_params
            
            H_decomp = qml.pauli_decompose(H, wire_order=range(num_qubits))
            paulis = H_decomp.ops
            coeffs = H_decomp.coeffs


            run_info = {"device":device,
                        "Potential":potential,
                        "cutoff": cutoff,
                        "num_qubits": num_qubits,
                        "num_paulis": len(paulis),
                        "num_params": num_params,
                        "min_eigenvalue": eigenvalues[0].real,
                        "shots": shots,
                        "lam": lam,
                        "p":p,
                        "eps":eps,
                        "max_gate":max_gate,
                        "num_vqe_runs": num_vqe_runs,
                        "max_iter": max_iter,
                        "tol": tol, 
                        "initial_tr_radius": initial_tr_radius,
                        "final_tr_radius": final_tr_radius,
                        "use_bounds": use_bounds,
                        "ansatz_name": ansatz_name,
                        "path": base_path
                        }

            print(json.dumps(run_info, indent=4, default=str))


            vqe_starttime = datetime.now()

            # Start multiprocessing for VQE runs
            with Pool(processes=10) as pool:
                vqe_results = pool.starmap(
                    run_vqe,
                    [
                        (i, max_iter, tol, initial_tr_radius, final_tr_radius, paulis, coeffs, num_qubits, shots, num_params, device, log_path, log_enabled, eps, lam, p, ansatz, max_gate, use_bounds, run_info)
                        for i in range(num_vqe_runs)
                    ],
                )

            # Collect results
            seeds = [res["seed"] for res in vqe_results]
            energies = [res["energy"] for res in vqe_results]
            x_values = [res["params"] for res in vqe_results]
            success = [res["success"] for res in vqe_results]
            num_iters = [res["num_iters"] for res in vqe_results]
            num_evals = [res["num_evaluations"] for res in vqe_results]
            run_times = [str(res["run_time"]) for res in vqe_results]
            total_run_time = sum([res["run_time"] for res in vqe_results], timedelta())

            vqe_end = datetime.now()
            vqe_time = vqe_end - vqe_starttime

            # Save run
            run = {
                "starttime": starttime,
                "endtime": vqe_end.strftime("%Y-%m-%d_%H-%M-%S"),
                "potential": potential,
                "cutoff": cutoff,
                "exact_eigenvalues": [x.real.tolist() for x in eigenvalues],
                "ansatz": ansatz_name,
                "num_VQE": num_vqe_runs,
                "device": device,
                "shots": shots,
                "Optimizer": {
                    "name": "COBYQA",
                    "maxiter": max_iter,
                    'maxfev': max_iter,
                    "tolerance": tol,
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
                "num_evaluations": num_evals,
                "success": np.array(success, dtype=bool).tolist(),
                "run_times": run_times,
                "parallel_run_time": str(vqe_time),
                "total_VQE_time": str(total_run_time),
                "seeds": seeds
            }

            # Save the variable to a JSON file
            path = os.path.join(base_path, "{}_{}.json".format(potential, cutoff))
            with open(path, "w") as json_file:
                json.dump(run, json_file, indent=4)

            print("Done")
