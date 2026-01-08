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
    thetas = ParameterVector("theta", len(ops)) 
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

    evs = np.asarray(res[0].data.evs, dtype=float)
    if evs.ndim == 2:
        evs = evs[0]

    grad = (evs[0::2] - evs[1::2]) / (2.0 * eps)
    return grad


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

    op_list = []
    op_params = []
    energies = []
    success = False

    eps_screen = 1e-1      # your random screening range
    fd_eps = 1e-4 

    for i in range(run_info["num_steps"]):
        print(f"Run {run_idx} step {i}")
        max_ops_list = []

        print(f"Run {i}: computing gradient")
        for param0 in np.random.uniform(-eps_screen, eps_screen, size=run_info["num_grad_checks"]):

            grad_list = []
            for cand in pool:  # pool is list[OpSpec]
                g = grad_for_candidate_op(
                    estimator=estimator,
                    H_pauli=H_pauli,
                    num_qubits=num_qubits,
                    basis_state=basis_state,
                    selected_ops=op_list,          # list[OpSpec]
                    selected_params=op_params,     # np.array
                    candidate_op=cand,
                    param0=param0,
                    fd_eps=fd_eps,
                )
                grad_list.append((cand, abs(g)))

            max_op, max_grad = max(grad_list, key=lambda t: t[1])
            max_ops_list.append(max_op)

        most_common_gate, count = Counter(max_ops_list).most_common(1)[0]
        op_list.append(most_common_gate)

        print(f"Run {i}: finished gradients - selected {most_common_gate.name} on {most_common_gate.qubits}")

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
        x0 = np.append(op_params, 0.0)     

        ansatz_qc, theta = build_ansatz(num_qubits, basis_state, op_list) 
        
        print(f"Run {i}: running VQE")

        res = minimize(
            cost_function,
            x0,
            args=(ansatz_qc, H_pauli, estimator),
            method="COBYQA",
            options=run_info["optimizer_options"]
        )

        if i != 0:
            pre_min_e = min_e

        min_e = res.fun
        pre_op_params = op_params.copy()
        op_params = res.x
        energies.append(min_e)

        print(res.success)

        if i >=10:
            if abs(pre_min_e - min_e) < 1e-8:
                print("Energy converged")
                energies.pop()
                op_list.pop()
                final_params = pre_op_params
                success = True
                break
            if abs(run_info["min_eigenvalue"] - min_e) < 1e-8:
                print("Energy close to min eigenvalue")
                success = True
                final_params = op_params
                break
        
    run_end = datetime.now()
    run_time = run_end - run_start

    if success == False:
        final_params = op_params

    final_ops = []
    for op, param in zip(op_list, final_params):
        final_ops.append({
            "name": op.name,
            "param": float(param),
            "qubits": list(op.qubits),
        })


    return {
        "seed": seed,
        "energies": energies,
        "min_energy": min_e,
        "op_list": final_ops,
        "success": success,
        "num_iters": i+1,
        "run_time": run_time
    }


if __name__ == "__main__":

    num_processes=10

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
    num_steps = 20
    num_grad_checks = 5
    num_vqe_runs = 10
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

    base_path = os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\Wess-Zumino\Qiskit\AVQE\Files2", boundary_condition, potential, folder)
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
