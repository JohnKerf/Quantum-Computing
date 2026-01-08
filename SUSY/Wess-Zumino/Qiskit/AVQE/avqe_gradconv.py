from qiskit.primitives import StatevectorEstimator
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.primitives import StatevectorEstimator
from qiskit.circuit.library import XXPlusYYGate
from scipy.optimize import minimize
from qiskit.quantum_info import PauliList, SparsePauliOp

import os, json, time
import numpy as np
from datetime import datetime, timedelta

from multiprocessing import Pool

from collections import Counter, defaultdict
import heapq
from dataclasses import dataclass
from typing import Tuple, Optional

import git
repo_path = git.Repo('.', search_parent_directories=True).working_tree_dir

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


@dataclass(frozen=True)
class OpSpec:
    name: str
    qubits: Tuple[int, ...]
    beta: Optional[float] = None   # only for XXPlusYY


def apply_opspec(qc, theta, op: OpSpec):
    if op.name == "RY":
        (q,) = op.qubits
        qc.ry(theta, q)
    elif op.name == "RZ":
        (q,) = op.qubits
        qc.rz(theta, q)
    elif op.name == "CRY":
        c, t = op.qubits
        qc.cry(theta, c, t)
    elif op.name == "XXPlusYY":
        a, b = op.qubits
        beta = -np.pi/2 if op.beta is None else op.beta
        qc.append(XXPlusYYGate(theta, beta), [a, b])
    else:
        raise ValueError(f"Unknown op {op.name}")


def make_pool(N, cutoff, phi_beta=-np.pi/2):
    n_site = int(1 + np.log2(cutoff))
    n_b = n_site - 1
    pool = []

    for site in range(N):
        base = site * n_site

        # fermion is LAST qubit in the site block
        f = base + n_b

        # bosons are the first n_b qubits
        b = [base + j for j in range(n_b)]

        pool.append(OpSpec("RY", (f,)))

        for w in b:
            pool.append(OpSpec("RY", (w,)))
            pool.append(OpSpec("CRY", (f, w)))
            pool.append(OpSpec("CRY", (w, f)))

    for site in range(N - 1):
        base0 = site * n_site
        base1 = (site + 1) * n_site

        b0 = [base0 + j for j in range(n_b)]
        b1 = [base1 + j for j in range(n_b)]
        for u in b0:
            for v in b1:
                pool.append(OpSpec("CRY", (u, v)))

        f0 = base0 + n_b
        f1 = base1 + n_b
        pool.append(OpSpec("XXPlusYY", (f0, f1), beta=phi_beta))

    return pool




def build_ansatz(num_qubits, basis_state_bits, ops):
    thetas = ParameterVector("theta", len(ops))  # ASCII is safer
    qc = QuantumCircuit(num_qubits)

    for q, bit in enumerate(basis_state_bits):
        if bit:
            qc.x(q)

    for k, spec in enumerate(ops):
        apply_opspec(qc, thetas[k], spec)

    return qc, thetas


def finite_diff_grad(estimator, circuit, observable, x, eps=1e-4):
   
    x = np.asarray(x, dtype=float)
    params = list(circuit.parameters)
    n = len(params)

    if x.shape != (n,):
        raise ValueError(f"x must have shape ({n},), got {x.shape}.")

    # 2n parameter sets in a single batch
    X = np.repeat(x[None, :], 2 * n, axis=0)
    for k in range(n):
        X[2 * k, k]     += eps
        X[2 * k + 1, k] -= eps

    pub = (circuit, [observable], X.tolist())
    res = estimator.run([pub]).result()

    evs = np.asarray(res[0].data.evs, dtype=float).squeeze()
    if evs.ndim != 1:
        raise ValueError(f"Unexpected evs shape after squeeze: {evs.shape}")

    grad = (evs[0::2] - evs[1::2]) / (2.0 * eps)
    return grad


def finite_diff_grad_component(estimator, circuit, observable, x, k, eps=1e-4):
    """Finite-difference d/dx_k <H> at point x using only 2 evaluations."""
    x = np.asarray(x, dtype=float)
    n = len(list(circuit.parameters))
    if x.shape != (n,):
        raise ValueError(f"x must have shape ({n},), got {x.shape}.")

    X = np.repeat(x[None, :], 2, axis=0)
    X[0, k] += eps
    X[1, k] -= eps

    pub = (circuit, [observable], X.tolist())
    res = estimator.run([pub]).result()

    evs = np.asarray(res[0].data.evs, dtype=float).squeeze()
    if evs.ndim != 1:
        raise ValueError(f"Unexpected evs shape after squeeze: {evs.shape}")


    return float((evs[0] - evs[1]) / (2.0 * eps))


def grad_for_candidate_op_at_pos(
    estimator,
    H_pauli,                 # SparsePauliOp
    num_qubits: int,
    basis_state: list[int],
    selected_ops,            # list[OpSpec]
    selected_params,         # array-like shape (len(selected_ops),)
    candidate_op,            # OpSpec to test
    pos: int,                # insertion position in [0, len(selected_ops)]
    *,
    param0: float = 0.0,
    fd_eps: float = 1e-4,
):
    """Gradient for inserting candidate_op at position pos, wrt the inserted parameter."""
    L = len(selected_ops)
    if not (0 <= pos <= L):
        raise ValueError(f"pos must be in [0, {L}], got {pos}")

    ops = list(selected_ops)
    ops.insert(pos, candidate_op)
    qc, _ = build_ansatz(num_qubits, basis_state, ops)

    sel = np.asarray(selected_params, dtype=float)
    x = np.insert(sel, pos, float(param0)) if sel.size else np.array([float(param0)], dtype=float)

    return finite_diff_grad_component(estimator, qc, H_pauli, x, k=pos, eps=fd_eps)


def grad_for_candidate_op(
    estimator,
    H_pauli,                 # SparsePauliOp
    num_qubits: int,
    basis_state: list[int],
    selected_ops,             # list of OpSpec already in ansatz
    selected_params,          # np.array shape (len(selected_ops),)
    candidate_op,             # OpSpec to test (appended)
    *,
    param0: float = 0.0,
    fd_eps: float = 1e-4,
):
    # Build circuit for (selected + candidate)
    ops = list(selected_ops) + [candidate_op]
    qc, _ = build_ansatz(num_qubits, basis_state, ops)

    # Parameter vector in the same order as qc.parameters / build_ansatz order
    x = np.zeros(len(ops), dtype=float)
    if len(selected_ops) > 0:
        x[:-1] = np.asarray(selected_params, dtype=float)
    x[-1] = float(param0)

    # Full gradient; we only need component for the NEW (last) parameter
    g = finite_diff_grad(estimator, qc, H_pauli, x, eps=fd_eps)
    return float(g[-1])



def cost_function(params, ansatz_isa, hamiltonian_isa, estimator):

    pub = (ansatz_isa, [hamiltonian_isa], params)
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]

    return energy



def op_key(op): return (op.name, op.qubits)

def pool_add(pool, pool_keys, op):
    k = op_key(op)
    if k not in pool_keys:
        pool.append(op); pool_keys.add(k)

def pool_remove(pool, pool_keys, op):
    k = op_key(op)
    if k in pool_keys:
        pool_keys.remove(k)
        for j, existing in enumerate(pool):
            if op_key(existing) == k:
                pool.pop(j); break

def is_1q(op): return len(op.qubits) == 1
def is_2q(op): return len(op.qubits) == 2


def run_adapt_vqe(run_idx, H_pauli, run_info):

    pool = make_pool(run_info['N'], run_info['cutoff'], phi_beta=-np.pi/2) 
    pool_keys = {op_key(op) for op in pool}

    incident_2q = defaultdict(list)
    for op in pool:
        if is_2q(op):
            w0, w1 = op.qubits
            incident_2q[w0].append(op)
            incident_2q[w1].append(op)

    # track which wires have had *selected* 1q gates
    wires_with_1q = set()

    estimator = StatevectorEstimator()

    num_qubits = run_info["num_qubits"]
    basis_state = run_info["basis_state"]

    seed = (os.getpid() * int(time.time())) % (123456789 * (run_idx + 1))

    run_start = datetime.now()

    positions = []
    best_gradients = []
    op_list = []
    op_params = []
    energies = []
    min_e = float('nan')
    success = False

    eps_screen = 1e-1      # your random screening range
    fd_eps = 1e-4 


    # -----------------------------
    # Gradient-based convergence / logging
    # -----------------------------
    grad_conv_thresh = run_info.get("grad_conv_thresh", 1e-6)
    grad_conv_patience = run_info.get("grad_conv_patience", 3)  # consecutive iterations below threshold
    min_iters_before_grad_conv = run_info.get("min_iters_before_grad_conv", 5)

    # Switch to exhaustive (all operators x all insertion positions) after this iteration index
    scan_after = run_info.get("scan_all_positions_after", 2)

    grad_history = []
    _below_count = 0

    for i in range(run_info["num_steps"]):
        print(f"Run {run_idx} step {i}")
        max_ops_list = []

        # -----------------------------
        # Compute gradients + pick next operator (with optional all-position scan)
        # -----------------------------
        L = len(op_list)

        best_abs_grad_iter = -1.0
        insert_pos = L  # default: append
        most_common_gate = None

        # MODE A: your existing screening (append-only)
        if i < scan_after:
            print(f"Run {i}: computing gradient (append-only screening)")
            for param0 in np.random.uniform(-eps_screen, eps_screen, size=run_info["num_grad_checks"]):

                grad_list = []
                max_abs_this_param0 = -1.0

                for cand in pool:
                    g = grad_for_candidate_op(
                        estimator=estimator,
                        H_pauli=H_pauli,
                        num_qubits=num_qubits,
                        basis_state=basis_state,
                        selected_ops=op_list,
                        selected_params=op_params,
                        candidate_op=cand,
                        param0=float(param0),
                        fd_eps=fd_eps,
                    )
                    ag = abs(g)
                    grad_list.append((cand, ag))

                    if ag > max_abs_this_param0:
                        max_abs_this_param0 = ag
                    if ag > best_abs_grad_iter:
                        best_abs_grad_iter = ag

                max_op, _ = max(grad_list, key=lambda t: t[1])
                max_ops_list.append(max_op)

            most_common_gate, count = Counter(max_ops_list).most_common(1)[0]
            insert_pos = L  # append

        # MODE B: exhaustive scan over all operators x all insertion positions
        else:
            print(f"Run {i}: computing gradient (ALL ops x ALL positions)")
            #param0 = 0.0
            for param0 in np.random.uniform(-eps_screen, eps_screen, size=run_info["num_grad_checks"]):
                for pos in range(L + 1):
                    for cand in pool:
                        g = grad_for_candidate_op_at_pos(
                            estimator=estimator,
                            H_pauli=H_pauli,
                            num_qubits=num_qubits,
                            basis_state=basis_state,
                            selected_ops=op_list,
                            selected_params=op_params,
                            candidate_op=cand,
                            pos=pos,
                            param0=param0,
                            fd_eps=fd_eps,
                        )
                        ag = abs(g)
                        if ag > best_abs_grad_iter:
                            best_abs_grad_iter = ag
                            most_common_gate = cand
                            insert_pos = pos


                # in exhaustive mode, most_common_gate is set by max |grad|
                if most_common_gate is None:
                    print("Pool empty")
                    # should only happen if pool is empty
                    best_abs_grad_iter = 0.0

        # -----------------------------
        # Gradient-based convergence check (stop BEFORE adding a new operator)
        # -----------------------------
        if best_abs_grad_iter < grad_conv_thresh and i >= min_iters_before_grad_conv:
            _below_count += 1
        else:
            _below_count = 0

        if _below_count >= grad_conv_patience:
            print(f"Gradient converged: max|grad|={best_abs_grad_iter:.3e} < {grad_conv_thresh:.3e}")
            success = True
            final_params = np.asarray(op_params, dtype=float)  # current optimized params
            break

        # -----------------------------
        # Apply the chosen operator (append or insert), then optimize parameters
        # -----------------------------
        if most_common_gate is None:
            # No available operator to add
            print("Pool is empty; stopping.")
            success = True
            final_params = np.asarray(op_params, dtype=float)
            break

        op_list.insert(insert_pos, most_common_gate)
        positions.append(insert_pos)
        best_gradients.append(best_abs_grad_iter)

        print(f"Run {i}: finished gradients - selected {most_common_gate.name} on {most_common_gate.qubits} at pos={insert_pos}")

        # 1) remove the selected op from the active pool (so it can’t be immediately re-picked)
        pool_remove(pool, pool_keys, most_common_gate)

        if is_2q(most_common_gate):
            # add local 1q gates on both qubits of this 2q gate
            w0, w1 = most_common_gate.qubits

            pool_add(pool, pool_keys, OpSpec("RY", (w0,)))
            pool_add(pool, pool_keys, OpSpec("RZ", (w0,)))

            pool_add(pool, pool_keys, OpSpec("RY", (w1,)))
            pool_add(pool, pool_keys, OpSpec("RZ", (w1,)))

        elif is_1q(most_common_gate):
            # if we’ve now applied 1q gates on BOTH ends of a 2q op, re-enable that 2q op
            (w,) = most_common_gate.qubits
            wires_with_1q.add(w)

            # only check 2q ops touching this wire
            for op2 in incident_2q[w]:
                a, b = op2.qubits
                other = b if a == w else a
                if other in wires_with_1q:
                    pool_add(pool, pool_keys, op2)

        # --- optimize parameters as before ---
        np.random.seed(seed)
        x0 = np.insert(np.asarray(op_params, dtype=float), insert_pos, 0.0)

        ansatz_qc, theta = build_ansatz(num_qubits, basis_state, op_list) 
        
        print(f"Run {i}: running VQE")

        res = minimize(
            cost_function,
            x0,
            args=(ansatz_qc, H_pauli, estimator),
            method="COBYQA",
            options=run_info["optimizer_options"]
        )


        min_e = res.fun
        op_params = res.x
        energies.append(min_e)

        print(res.success)

    run_end = datetime.now()
    run_time = run_end - run_start

    if success == False:
        final_params = op_params

    final_ops = []
    for op, param, grad, pos in zip(op_list, final_params, best_gradients, positions):
        final_ops.append({
            "name": op.name,
            "param": float(param),
            "qubits": list(op.qubits),
            "grad": grad,
            "pos": pos
        })


    return {
        "seed": seed,
        "energies": energies,
        "min_energy": (None if (min_e is None or (isinstance(min_e, float) and (min_e != min_e))) else float(min_e)),
        "op_list": final_ops,
        "grad_history": grad_history,
        "success": success,
        "num_iters": i+1,
        "run_time": run_time
    }


if __name__ == "__main__":

    num_processes=1

    cutoff = 2
    shots = None
    # Parameters
    N = 3
    a = 1.0
    c = -0.8

    potential = "linear"
    #potential = 'quadratic'
    boundary_condition = 'dirichlet'
    #boundary_condition = 'periodic'

    # Optimizer
    num_steps = 10
    num_grad_checks = 10
    num_vqe_runs = 1
    max_iter = 10000
    initial_tr_radius = 0.1
    final_tr_radius = 1e-8
    scale=True

    optimizer_options = {
                    'maxiter':max_iter, 
                    'maxfev':max_iter, 
                    'initial_tr_radius':initial_tr_radius, 
                    'final_tr_radius':final_tr_radius, 
                    'scale':scale
                    }

    print(f"Running for {potential} potential, cutoff {cutoff}")


    starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if potential == 'quadratic':
        folder = 'C' + str(abs(c)) + '/' + 'N'+ str(N)
    else:
        folder = 'N'+ str(N)

    base_path = os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\Wess-Zumino\Qiskit\AVQE\Files", boundary_condition, potential, folder)
    os.makedirs(base_path, exist_ok=True)

    print("Loading Hamiltonian")
    H_path = os.path.join(repo_path, r"SUSY\Wess-Zumino\Analyses\Model Checks\HamiltonianData", boundary_condition, potential, folder, f"{potential}_{cutoff}.json")
    with open(H_path, 'r') as file:
        H_data = json.load(file)

    pauli_coeffs = np.asarray(H_data["pauli_coeffs"], dtype=np.complex128)
    pauli_labels = H_data["pauli_labels"]

    H_pauli = SparsePauliOp(PauliList(pauli_labels), pauli_coeffs).simplify(atol=1e-12)

    num_qubits = H_data['num_qubits']

    eigenvalues = H_data['eigenvalues']
    min_eigenvalue = np.min(eigenvalues)

    print(f"Min eignvalue: {min_eigenvalue.real}")
  

    nb = int(np.log2(cutoff))
    n = 1 + nb
    
    basis_state = H_data['best_basis_state'][::-1]


    run_info = {"Potential":potential,
                "cutoff": cutoff,
                "N": N,
                "a": a,
                "c": None if potential == "linear" else c,
                "num_qubits": num_qubits,
                "min_eigenvalue":min_eigenvalue.real,
                "num_steps":num_steps,
                "num_grad_checks":num_grad_checks,
                "num_vqe_runs": num_vqe_runs,
                "optimizer_options": optimizer_options,
                "basis_state":basis_state,
                "path": base_path,
                }
    

    vqe_starttime = datetime.now()

    print("Starting ADAPT-VQE")
    # Start multiprocessing for VQE runs
    with Pool(processes=num_processes) as pool:
        vqe_results = pool.starmap(
            run_adapt_vqe,
            [
                (i, H_pauli, run_info)
                for i in range(num_vqe_runs)
            ],
        )

    print("Finished ADAPT-VQE")
    # Collect results
    seeds = [res["seed"] for res in vqe_results]
    all_energies = [res["energies"] for res in vqe_results]
    min_energies = [res["min_energy"] for res in vqe_results]
    op_lists = [res["op_list"] for res in vqe_results]
    success = [res["success"] for res in vqe_results]
    num_iters = [res["num_iters"] for res in vqe_results]
    run_times = [str(res["run_time"]) for res in vqe_results]
    total_run_time = sum([res["run_time"] for res in vqe_results], timedelta())
    grad_history = [res["grad_history"] for res in vqe_results]

    vqe_end = datetime.now()
    vqe_time = vqe_end - vqe_starttime

    # Save run
    run = {
        "starttime": starttime,
        "endtime": vqe_end.strftime("%Y-%m-%d_%H-%M-%S"),
        "potential": potential,
        "cutoff": cutoff,
        "N": N,
        "a": a,
        "c": None if potential == "linear" else c,
        "exact_eigenvalues": eigenvalues,
        "shots": shots,
        "Optimizer": {
                "name": "COBYQA",
                "optimizer_options":optimizer_options
            },
        "num_VQE": num_vqe_runs,
        "num_steps":num_steps,
        "num_grad_checks":num_grad_checks,
        "basis_state": basis_state,
        #"operator_pool": [str(op) for op in operator_pool],
        "all_energies": all_energies,
        "min_energies": min_energies,
        "op_list": op_lists,
        #"grad_history": grad_history,
        "num_iters": num_iters,
        "success": np.array(success, dtype=bool).tolist(),
        "run_times": run_times,
        "seeds": seeds,
        "parallel_run_time": str(vqe_time),
        "total_VQE_time": str(total_run_time)
    }

    # Save the variable to a JSON file
    path = os.path.join(base_path, "{}_{}.json".format(potential, cutoff))
    with open(path, "w") as json_file:
        json.dump(run, json_file, indent=4)

    print("Done")