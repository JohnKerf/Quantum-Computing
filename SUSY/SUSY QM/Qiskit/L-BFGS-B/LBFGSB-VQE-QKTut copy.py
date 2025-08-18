# vqe_qiskit_runner.py
import os, json, time, logging
from datetime import datetime, timedelta
import numpy as np
from multiprocessing import Pool

from scipy.optimize import minimize

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import Session
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from susy_qm import calculate_Hamiltonian

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


# ---------- Ansatz (DW) with CRY decomposed ----------
def build_dw_ansatz_with_decomposed_cry(num_qubits=3):
    """
    Logical ansatz you described:
        ry(θ0, q0)
        ry(θ1, q2)
        cry(θ2, control=q0, target=q1)   <-- decomposed
        ry(θ3, q1)
        ry(θ4, q0)

    Decomposition:
        cry(θ2) ≡ ry(+θ2/2, q1); cx(0,1); ry(-θ2/2, q1); cx(0,1)

    Returns:
        qc              : QuantumCircuit
        logical_params  : [θ0, θ1, θ2, θ3, θ4]
        physical_params : dict with keys 'phi_a','phi_b' for +θ2/2 and -θ2/2
    """
    θ0 = Parameter("θ0")
    θ1 = Parameter("θ1")
    θ2 = Parameter("θ2")  # logical CRY angle
    θ3 = Parameter("θ3")
    θ4 = Parameter("θ4")

    φa = Parameter("φa")  # +θ2/2
    φb = Parameter("φb")  # -θ2/2

    qc = QuantumCircuit(num_qubits)
    qc.ry(θ0, 0)
    qc.ry(θ1, 2)

    # Decomposed CRY(θ2) with control=0, target=1
    qc.ry(φa, 1)
    qc.cx(0, 1)
    qc.ry(φb, 1)
    qc.cx(0, 1)

    qc.ry(θ3, 1)
    qc.ry(θ4, 0)

    logical_params = [θ0, θ1, θ2, θ3, θ4]
    physical_params = {"phi_a": φa, "phi_b": φb}
    return qc, logical_params, physical_params


def bind_from_logical(x, logical_params, physical_params):
    """
    Map logical vector x = [θ0, θ1, θ2, θ3, θ4] to a dict of circuit Parameters.
    """
    θ0, θ1, θ2, θ3, θ4 = [float(v) for v in x]
    return {
        logical_params[0]: θ0,
        logical_params[1]: θ1,
        logical_params[3]: θ3,
        logical_params[4]: θ4,
        physical_params["phi_a"]: +0.5 * θ2,
        physical_params["phi_b"]: -0.5 * θ2,
    }

def vector_from_logical(x, logical_params, physical_params, param_order):
    bind = bind_from_logical(x, logical_params, physical_params)
    return np.array([bind[p] for p in param_order], dtype=float)


# ---------- Estimator cost & exact parameter-shift gradient ----------
def make_cost_and_grad(ansatz_isa, hamiltonian_isa, estimator,
                       logical_params, physical_params, param_order,
                       device_time_accumulator):

    def cost_with_vector(vec):
        start = datetime.now()
        pub = (ansatz_isa, [hamiltonian_isa], [vec])   # <-- vector, not dict
        result = estimator.run(pubs=[pub]).result()
        energy = float(result[0].data.evs[0])
        end = datetime.now()
        device_time_accumulator["dt"] += (end - start)
        return energy

    def E(x):
        vec = vector_from_logical(x, logical_params, physical_params, param_order)
        return cost_with_vector(vec)

    def cost_vec(x):
        return E(np.asarray(x, dtype=float))

    def grad_vec(x):
        x = np.asarray(x, dtype=float)
        grad = np.zeros_like(x)
        s = np.pi/2

        # θ0
        xp = x.copy(); xp[0] += s
        xm = x.copy(); xm[0] -= s
        grad[0] = 0.5*(E(xp) - E(xm))

        # θ1
        xp = x.copy(); xp[1] += s
        xm = x.copy(); xm[1] -= s
        grad[1] = 0.5*(E(xp) - E(xm))

        # θ3
        xp = x.copy(); xp[3] += s
        xm = x.copy(); xm[3] -= s
        grad[3] = 0.5*(E(xp) - E(xm))

        # θ4
        xp = x.copy(); xp[4] += s
        xm = x.copy(); xm[4] -= s
        grad[4] = 0.5*(E(xp) - E(xm))

        # θ2 via φa/φb shifts
        base_vec = vector_from_logical(x, logical_params, physical_params, param_order)

        # We need to find indices of φa and φb in param_order to shift them directly.
        φa = physical_params["phi_a"]; φb = physical_params["phi_b"]
        idx_a = param_order.index(φa)
        idx_b = param_order.index(φb)

        vec_ap = base_vec.copy(); vec_ap[idx_a] += s
        vec_am = base_vec.copy(); vec_am[idx_a] -= s
        term_a = 0.5*(cost_with_vector(vec_ap) - cost_with_vector(vec_am))

        vec_bp = base_vec.copy(); vec_bp[idx_b] += s
        vec_bm = base_vec.copy(); vec_bm[idx_b] -= s
        term_b = 0.5*(cost_with_vector(vec_bp) - cost_with_vector(vec_bm))

        grad[2] = 0.5*(term_a - term_b)
        return grad

    return cost_vec, grad_vec


def run_vqe(i, max_iter, tol, H, num_qubits, shots, log_dir):

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"vqe_run_{i}.log")
    logger = setup_logger(log_path, f"logger_{i}")

    seed = (os.getpid() * int(time.time())) % 123456789

    run_start = datetime.now()
    device_time = {"dt": timedelta()}  # mutable for closure accumulation

    # Backend selection: exact if shots is None
    if shots is None:
        backend = AerSimulator(method="statevector")
    else:
        backend = AerSimulator()

    # Observable from dense H
    observable = SparsePauliOp.from_operator(Operator(H))

    # ----- Build ansatz -----
    # DW ansatz uses three qubits (0,1,2) per your snippet
    qc, logical_params, physical_params = build_dw_ansatz_with_decomposed_cry(num_qubits)

    # Transpile to ISA and map observable
    target = backend.target
    pm = generate_preset_pass_manager(target=target, optimization_level=3)
    ansatz_isa = pm.run(qc)
    param_order = list(ansatz_isa.parameters)   # ordered list of Parameters
    hamiltonian_isa = observable.apply_layout(layout=ansatz_isa.layout)


    logger.info(f"Starting VQE run {i} (seed={seed})")

    # Initial point and bounds (angles in [0, 2π])
    np.random.seed(seed)
    num_params = 5
    x0 = np.random.random(size=num_params) * 2 * np.pi
    bounds = [(0.0, 2*np.pi)] * num_params

    with Session(backend=backend) as session:
        estimator = Estimator(mode=session)
        # exact expectations if shots is None
        estimator.options.default_shots = shots

        # Build cost and exact gradient
        cost_vec, grad_vec = make_cost_and_grad(
            ansatz_isa, hamiltonian_isa, estimator,
            logical_params, physical_params, param_order,
            device_time
        )

        # Optimize with L-BFGS-B + analytic jacobian
        res = minimize(
            cost_vec,
            x0=x0,
            jac=grad_vec,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter, "ftol": tol}
        )

    run_end = datetime.now()
    logger.info(f"Completed VQE run {i}: Energy = {res.fun:.12f}")

    return {
        "seed": seed,
        "energy": float(res.fun),
        "params": np.asarray(res.x, dtype=float).tolist(),
        "success": bool(res.success),
        "num_iters": int(res.nfev),
        "run_time": run_end - run_start,
        "device_time": device_time["dt"],
    }


if __name__ == "__main__":

    potential = "DW"
    shotslist = [None]#, 10000, 100000]
    cutoffs = [8]

    for shots in shotslist:
        for cutoff in cutoffs:

            print(f"Running for {potential} potential and cutoff {cutoff} and shots {shots}")

            starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            base_path = os.path.join(repo_path, r"SUSY\SUSY QM\Qiskit\L-BFGS-B\Files2", str(shots), potential)
            log_path = os.path.join(base_path, f"logs_{str(cutoff)}")
            os.makedirs(log_path, exist_ok=True)

            H = calculate_Hamiltonian(cutoff, potential)
            eigenvalues = np.sort(np.linalg.eigvals(H))[:4]
            num_qubits = int(1 + np.log2(cutoff))  # your prior rule; DW ansatz uses qubits 0,1,2

            num_vqe_runs = 100
            max_iter = 10000
            tol = 1e-8

            vqe_starttime = datetime.now()

            # Start multiprocessing for VQE runs
            with Pool(processes=10) as pool:
                vqe_results = pool.starmap(
                    run_vqe,
                    [
                        (i, max_iter, tol, H, num_qubits, shots, log_path)
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
                "ansatz": "DW with decomposed CRY (RY(+θ2/2); CX; RY(-θ2/2); CX)",
                "num_VQE": num_vqe_runs,
                "shots": shots,
                "Optimizer": {
                    "name": "L-BFGS-B",
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
