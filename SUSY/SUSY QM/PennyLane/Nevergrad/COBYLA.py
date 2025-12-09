import pennylane as qml
from pennylane import numpy as pnp

# SciPy no longer needed for optimization, but you can keep this import if used elsewhere
# from scipy.optimize import minimize

import os, json, time, logging
import numpy as np
from datetime import datetime, timedelta

from multiprocessing import Pool

from susy_qm import calculate_Hamiltonian, ansatze

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import git
repo_path = git.Repo('.', search_parent_directories=True).working_tree_dir

# NEW: Nevergrad import
import nevergrad as ng


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

    gates = ansatze.truncate_ansatz(ansatz, params, num_qubits, max_gate=max_gate)

    @qml.qnode(dev)
    def circuit():
        for gate in gates:
            qml.apply(gate)
        return [qml.expval(op) for op in paulis]

    expvals = circuit()
    energy = float(np.dot(coeffs, expvals))

    return energy


def run_vqe(i, paulis, coeffs, ansatz, run_info, log_enabled, log_dir):

    if log_enabled:
        os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"vqe_run_{i}.log")
    logger = setup_logger(log_path, f"logger_{i}", enabled=log_enabled)

    num_qubits = run_info["num_qubits"]
    num_params = run_info["num_params"]

    seed = (os.getpid() * int(time.time())) % 123456789
    dev = qml.device(
        run_info["device"],
        wires=num_qubits,
        shots=run_info["shots"],
        seed=seed,
    )
    run_start = datetime.now()

    last_energy = None
    iteration_count = 0

    def wrapped_cost_function(params):
        """Objective passed to Nevergrad COBYLA (params: np.ndarray)."""
        nonlocal last_energy, iteration_count

        result = cost_function(
            params, paulis, coeffs, num_qubits, dev, ansatz, run_info["max_gate"]
        )

        last_energy = result
        iteration_count += 1

        # penalty for negative energies (your small-negatives trick)
        neg = max(0.0, -(result + run_info["eps"]))
        obj = result + run_info["lam"] * (neg ** run_info["p"])

        if log_enabled and iteration_count % 10 == 0:
            logger.info(
                f"Iteration {iteration_count}: "
                f"Energy = {last_energy:.8f}, Objective = {obj:.8f}"
            )

        return obj

    # Initial point
    np.random.seed(seed)
    x0 = np.random.random(size=num_params) * 2 * np.pi

    # Bounds for Nevergrad parametrization
    use_bounds = run_info["use_bounds"]
    if use_bounds:
        lower, upper = 0.0, 2 * np.pi
    else:
        lower, upper = None, None

    run_info["seed"] = seed

    if log_enabled:
        logger.info(json.dumps(run_info, indent=4, default=str))
        logger.info(f"Starting VQE run {i} (seed={seed})")

    # === Nevergrad COBYLA setup ===
    max_iter = run_info["optimizer_options"]["maxiter"]
    tol = run_info["optimizer_options"].get("tol", 0.0)       # improvement threshold
    patience = run_info["optimizer_options"].get("patience", 1000)

    if use_bounds:
        parametrization = ng.p.Array(init=x0).set_bounds(lower, upper)
    else:
        parametrization = ng.p.Array(init=x0)

    # SciPy COBYLA wrapped in Nevergrad
    cobyla_family = ng.families.NonObjectOptimizer(method="COBYLA", random_restart=False)
    optimizer = cobyla_family(
        parametrization=parametrization,
        budget=max_iter,
        num_workers=1,   # must be sequential for NonObjectOptimizer
    )

    best_obj = None
    best_params = None
    best_energy = None
    no_improve_rounds = 0

    with qml.Tracker(dev) as tracker:
        for k in range(max_iter):
            # ask for a candidate
            x = optimizer.ask()
            theta = np.asarray(x.value, dtype=float)

            # evaluate objective
            obj = wrapped_cost_function(theta)

            # tell optimizer
            optimizer.tell(x, obj)

            # update best-so-far
            if best_obj is None or obj < best_obj:
                prev_best = best_obj
                best_obj = obj
                best_params = theta.copy()
                best_energy = last_energy

                if prev_best is not None and tol > 0:
                    improvement = prev_best - best_obj
                    if improvement < tol:
                        no_improve_rounds += 1
                    else:
                        no_improve_rounds = 0
                else:
                    no_improve_rounds = 0
            else:
                if tol > 0:
                    no_improve_rounds += 1

            # early stopping if no significant improvement for 'patience' iterations
            if tol > 0 and no_improve_rounds >= patience:
                if log_enabled:
                    logger.info(
                        f"Early stopping at iteration {iteration_count}: "
                        f"no improvement > {tol} over {patience} iterations."
                    )
                break

    # fallback (should not happen if max_iter > 0)
    if best_obj is None:
        best_params = x0
        best_obj = wrapped_cost_function(best_params)
        best_energy = last_energy

    final_obj = float(best_obj)
    final_energy = float(best_energy)

    run_end = datetime.now()
    run_time = run_end - run_start

    if log_enabled:
        logger.info(
            f"Completed VQE run {i}: "
            f"Objective = {final_obj:.6f}, Energy = {final_energy:.6f}"
        )

    totals = getattr(tracker, "totals", {})
    num_evals = int(totals.get("executions", 0))
    success = False if k==(max_iter-1) else True

    results_data = {
        "seed": seed,
        # penalized objective (what COBYLA minimized)
        "energy": final_obj,
        # physical energy without penalty
        "physical_energy": final_energy,
        "params": best_params.tolist(),
        "success": success,   
        "num_iters": k+1,
        "num_evaluations": num_evals,
        "run_time": run_time,
    }

    if log_enabled:
        logger.info(json.dumps(results_data, indent=4, default=str))

    return results_data



if __name__ == "__main__":

    log_enabled = False
    num_processes = 10

    potential = "DW"
    cutoff = 8

    device = 'default.qubit'

    shots = None
    use_bounds = False if shots is None else True
    

    lam = 15
    p = 2

    # Optimizer
    num_vqe_runs = 10
    max_iter = 10
    tol = 1e-10  

    optimizer_options = {
        'maxiter': max_iter,
        'tol': tol,
    }

    ansatze_type = 'real_amplitudes'  # exact, reduced, CD3, real_amplitudes
    num_layers = 1
    max_gate = None  # gate truncation

    if ansatze_type in ['exact', 'reduced']:
        if potential == "QHO":
            ansatz_name = f"CQAVQE_QHO_{ansatze_type}"
        elif (potential != "QHO") and (cutoff <= 64):
            ansatz_name = f"CQAVQE_{potential}{cutoff}_{ansatze_type}"
        else:
            ansatz_name = f"CQAVQE_{potential}16_{ansatze_type}"
    else:
        ansatz_name = ansatze_type

    ansatz = ansatze.get(ansatz_name)

    if potential == "AHO":
        i = np.log2(cutoff)
        factor = 2 ** (((i - 1) * i) / 2)
        eps = 0.5 / factor
    else:
        eps = 0

    starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_path = os.path.join(repo_path, r"SUSY\SUSY QM\PennyLane\Nevergrad\Files2", str(shots), potential)
    os.makedirs(base_path, exist_ok=True)

    log_path = os.path.join(base_path, f"logs_{str(cutoff)}")

    # Calculate Hamiltonian and expected eigenvalues
    H = calculate_Hamiltonian(cutoff, potential)

    eigenvalues = np.sort(np.linalg.eig(H)[0])[:4]
    num_qubits = int(1 + np.log2(cutoff))

    if ansatze_type in ['exact', 'reduced']:
        num_params = ansatz.n_params
    else:
        num_params = ansatz.n_params(num_qubits, num_layers)

    if max_gate is not None:
        num_params = max_gate if num_params > max_gate else num_params

    H_decomp = qml.pauli_decompose(H, wire_order=range(num_qubits))
    paulis = H_decomp.ops
    coeffs = H_decomp.coeffs

    run_info = {
        "device": device,
        "Potential": potential,
        "cutoff": cutoff,
        "ansatz_name": ansatz_name,
        "max_gate": max_gate,
        "num_qubits": num_qubits,
        "num_paulis": len(paulis),
        "num_params": num_params,
        "shots": shots,
        "lam": lam,
        "p": p,
        "eps": eps,
        "num_vqe_runs": num_vqe_runs,
        "optimizer_options": optimizer_options,
        "use_bounds": use_bounds,
        "path": base_path,
    }

    print(json.dumps(run_info, indent=4, default=str))

    vqe_starttime = datetime.now()

    # Start multiprocessing for VQE runs
    with Pool(processes=num_processes) as pool:
        vqe_results = pool.starmap(
            run_vqe,
            [
                (i, paulis, coeffs, ansatz, run_info, log_enabled, log_path)
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
        "device": device,
        "starttime": starttime,
        "endtime": vqe_end.strftime("%Y-%m-%d_%H-%M-%S"),
        "potential": potential,
        "cutoff": cutoff,
        "num_qubits": num_qubits,
        "num_paulis": len(paulis),
        "num_params": num_params,
        "exact_eigenvalues": [x.real.tolist() for x in eigenvalues],
        "ansatz": ansatz_name,
        "max_gate": max_gate,
        "num_VQE": num_vqe_runs,
        "shots": shots,
        "Optimizer": {
            "name": "NGOpt",
            "optimizer_options": optimizer_options,
        },
        "cost function": {
            "type": "small negatives",
            "p": p,
            "lam": lam,
            "eps": eps,
        },
        "results": energies,
        "params": x_values,
        "num_evaluations": num_evals,
        "num_iters": num_iters,
        "success": np.array(success, dtype=bool).tolist(),
        "run_times": run_times,
        "parallel_run_time": str(vqe_time),
        "total_VQE_time": str(total_run_time),
        "seeds": seeds,
    }

    path = os.path.join(base_path, "{}_{}.json".format(potential, cutoff))
    with open(path, "w") as json_file:
        json.dump(run, json_file, indent=4)

    print("Done")
