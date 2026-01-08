import pennylane as qml
from pennylane import numpy as pnp

from scipy.optimize import minimize

from scipy.optimize import differential_evolution
from scipy.stats.qmc import Halton

import os
import json
import numpy as np
from datetime import datetime, timedelta
import time

from multiprocessing import Pool

from collections import Counter, defaultdict

from wesszumino import build_wz_hamiltonian, pauli_str_to_op

import git
repo_path = git.Repo('.', search_parent_directories=True).working_tree_dir

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def compute_grad(param, H, num_qubits, operator_ham, op_list, op_params, basis_state, grad_dev):

    @qml.qnode(grad_dev, interface="autograd")
    def expval_circuit(p):

        qml.BasisState(basis_state, wires=range(num_qubits))

        for theta, op in zip(op_params, op_list):
            type(op)(theta, wires=op.wires)

        type(operator_ham)(p, wires=operator_ham.wires)

        return qml.expval(H)    

    p = pnp.tensor(param, requires_grad=True)
    grad = qml.grad(expval_circuit)(p)

    return grad


def cost_function(params, H, num_qubits, op_list, basis_state, dev):
     
    @qml.qnode(dev)
    def circuit(params):

        qml.BasisState(basis_state, wires=range(num_qubits))

        param_index = 0
        for op in op_list:
            o = type(op)
            o(params[param_index], wires=op.wires)
            param_index +=1

        return qml.expval(H)

    energy = circuit(params)               

    return energy


def op_key(op):
    # ignore parameter values; use just (gate-type, wires)
    return (type(op), tuple(int(w) for w in op.wires.tolist()))

def pool_add(pool, pool_keys, op):
    k = op_key(op)
    if k not in pool_keys:
        pool.append(op)
        pool_keys.add(k)

def pool_remove(pool, pool_keys, op):
    k = op_key(op)
    if k in pool_keys:
        pool_keys.remove(k)
        # remove the first matching key (robust to parameter differences)
        for j, existing in enumerate(pool):
            if op_key(existing) == k:
                pool.pop(j)
                break

def is_1q(op): return len(op.wires) == 1
def is_2q(op): return len(op.wires) == 2




def run_adapt_vqe(run_idx, H, run_info):

    seed = (os.getpid() * int(time.time())) % 123456789 * (run_idx + 1)

    num_qubits = run_info["num_qubits"]
    shots = run_info["shots"]
    basis_state = run_info["basis_state"]

    eps = np.pi#run_info.get("eps", 1e-2)
    grad_tol = run_info.get("grad_tol", 1e-6)
    debug_grads = run_info.get("debug_grads", False)

    op_list = []
    op_params = []
    energies = []
    success = False

    dev = qml.device(run_info["device"], wires=num_qubits, shots=shots, seed=seed)

    # --- pool setup (master + active + fast membership) ---
    master_pool = run_info["operator_pool"].copy()
    pool = master_pool.copy()
    pool_keys = {op_key(op) for op in pool}

    # 2-qubit ops incident on each wire (for quick re-add)
    incident_2q = defaultdict(list)
    for op in master_pool:
        if is_2q(op):
            w0, w1 = (int(w) for w in op.wires.tolist())
            incident_2q[w0].append(op)
            incident_2q[w1].append(op)

    # track which wires have had *selected* 1q gates
    wires_with_1q = set()

    run_start = datetime.now()

    for i in range(run_info["num_steps"]):
        print(f"Run {run_idx} step {i}")

        # ---------------------------
        # 1) SELECT NEXT OP BY MAX GRAD OVER PROBES
        # ---------------------------
        rng = np.random.default_rng(seed)
        probes = rng.uniform(-eps, eps, size=run_info["num_grad_checks"])

        best_op = None
        best_score = -1.0

        if debug_grads:
            print(f"Run {run_idx} step {i}: computing gradients (eps={eps}, probes={probes})")

        for op in pool:
            # score(op) = max_p |dE/dp| at small probe angles p
            score = 0.0
            for p in probes:
                g = compute_grad(p, H, num_qubits, op, op_list, op_params, basis_state, dev)
                score = max(score, abs(float(g)))

            if debug_grads:
                print(op, "score:", score)

            if score > best_score:
                best_score = score
                best_op = op

        # gradient-based stopping criterion (ADAPT-style)
        print(f"Run {run_idx} step {i}: best grad score = {best_score:.3e}")
        if best_score < grad_tol:
            print(f"Stopping ADAPT: best_score ({best_score:.3e}) < grad_tol ({grad_tol:.3e})")
            success = True
            break

        # append a *template* op (parameter value irrelevant here)
        chosen = type(best_op)(0.0, wires=best_op.wires)
        op_list.append(chosen)
        print(f"Run {run_idx} step {i}: selected {chosen}")

        # ---------------------------
        # 2) UPDATE POOL (your logic)
        # ---------------------------
        pool_remove(pool, pool_keys, chosen)

        if is_2q(chosen):
            w0, w1 = (int(w) for w in chosen.wires.tolist())
            pool_add(pool, pool_keys, qml.RY(0.0, wires=w0))
            pool_add(pool, pool_keys, qml.RZ(0.0, wires=w0))
            pool_add(pool, pool_keys, qml.RY(0.0, wires=w1))
            pool_add(pool, pool_keys, qml.RZ(0.0, wires=w1))

        elif is_1q(chosen):
            w = int(chosen.wires.tolist()[0])
            wires_with_1q.add(w)
            for op2 in incident_2q[w]:
                a, b = (int(x) for x in op2.wires.tolist())
                other = b if a == w else a
                if other in wires_with_1q:
                    pool_add(pool, pool_keys, op2)

        # ---------------------------
        # 3) OPTIMIZE ALL PARAMS (VQE step)
        # ---------------------------
        num_dimensions = len(op_list)
        bounds = [(0.0, 2.0 * np.pi) for _ in range(num_dimensions)]

        # initial guess: previous params + 0 for the new parameter
        x0 = np.concatenate([np.asarray(op_params, dtype=float), np.array([0.0])])

        print(f"Run {run_idx} step {i}: running VQE")

        res = minimize(
            cost_function,
            x0,
            args=(H, num_qubits, op_list, basis_state, dev),
            method="COBYQA",
            options=run_info["optimizer_options"]
        )

        op_params = np.asarray(res.x, dtype=float)
        min_e = float(res.fun)
        energies.append(min_e)

        print(f"Run {run_idx} step {i}: VQE done. E = {min_e:.12f}, success={res.success}")
       
        if abs(run_info["min_eigenvalue"] - min_e) < run_info.get("energy_tol", 1e-8):
            print("Energy close to min eigenvalue")
            success = True
            break
        
    run_end = datetime.now()
    run_time = run_end - run_start
   

    final_ops = []
    for op, param in zip(op_list,op_params):
        dict = {"name": op.name,
                "param": param,
                "wires": op.wires.tolist()}
        final_ops.append(dict)

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

    device = 'lightning.qubit'
    num_processes=10
    
    cutoff = 2
    shots = None

    # Parameters
    N = 4
    a = 1.0
    c = -0.8

    potential = "linear"
    #potential = 'quadratic'
    boundary_condition = 'dirichlet'
    #boundary_condition = 'periodic'


    # Optimizer
    num_steps = 10
    num_grad_checks = 5
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

    base_path = os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\Wess-Zumino\PennyLane\AVQE\Files3", boundary_condition, potential, folder)
    os.makedirs(base_path, exist_ok=True)

    print("Loading Hamiltonian")
    H_path = os.path.join(repo_path, r"SUSY\Wess-Zumino\Analyses\Model Checks\HamiltonianData", boundary_condition, potential, folder, f"{potential}_{cutoff}.json")
    with open(H_path, 'r') as file:
        H_data = json.load(file)

    pauli_coeffs = H_data['pauli_coeffs']
    pauli_strings = H_data['pauli_terms']
    pauli_terms = [pauli_str_to_op(t) for t in pauli_strings]

    num_qubits = H_data['num_qubits']

    eigenvalues = H_data['eigenvalues']
    min_eigenvalue = np.min(eigenvalues)

    print(f"Min eignvalue: {min_eigenvalue.real}")

    print("Making dense H")
    pauli_H = qml.Hamiltonian(pauli_coeffs, pauli_terms)
    #dense_H = qml.matrix(pauli_H, wire_order=list(range(num_qubits)))
    dense_H = None
    print("Finished making dense H")


    #Create operator pool
    operator_pool = []
    phi = 0.0

    n_site = int(1 + np.log2(cutoff))
    n_b = n_site - 1

    for site in range(N):
        f = site * n_site
        b = [f + 1 + j for j in range(n_b)]

        # Fermion single-qubit gates:
        # RZ: phase (keeps number); RY: mixes |0>,|1> (changes number sector)
        #operator_pool.append(qml.RX(phi, wires=f))
        operator_pool.append(qml.RZ(phi, wires=f))
        operator_pool.append(qml.RY(phi, wires=f))  # <-- adds/removes fermion

        # Boson local gates (change boson number in cutoff-2 truncation)
        for w in b:
            operator_pool.append(qml.RY(phi, wires=w))  # add/remove boson
            operator_pool.append(qml.RZ(phi, wires=w))

            # Local fermion-boson entanglers
            operator_pool.append(qml.CRY(phi, wires=[f, w]))
            operator_pool.append(qml.CRY(phi, wires=[w, f]))

    # Inter-site couplings
    for site in range(N-1):
        # Boson-boson
        b0 = [(site*n_site)     + 1 + j for j in range(n_b)]
        b1 = [((site+1)*n_site) + 1 + j for j in range(n_b)]

        for u in b0:
            for v in b1:
                operator_pool.append(qml.CRY(phi, wires=[u, v]))

        # Fermion hopping / mixing between sites (still useful even if RY present)
        f0 = site * n_site
        f1 = (site+1) * n_site
        #operator_pool.append(qml.FermionicSingleExcitation(phi, wires=[f0, f1]))
        #operator_pool.append(qml.CRY(phi, wires=[f0, f1]))
        operator_pool.append(qml.SingleExcitation(phi, wires=[f0, f1]))
        


    nb = int(np.log2(cutoff))
    n = 1 + nb
    fw = [i * n for i in range(N)]

    # Choose basis state
    #Dirichlet-Linear
    #basis_state = [0]*n + [1] + [0]*nb #N2
    #basis_state = [0]*n + [1] + [0]*nb + [0]*n #N3
    basis_state = [0]*n + [1] + [0]*nb + [0]*n + [1] + [0]*nb #N4
    #basis_state = [0]*n + [1] + [0]*nb + [0]*n + [1] + [0]*nb + [0]*n #N5  

    #basis_state = H_data['best_basis_state']
    #basis_state = [0, 0, 0, 1, 0, 0, 0, 0, 0]
     

    run_info = {"device":device,
                "Potential":potential,
                "cutoff": cutoff,
                "N": N,
                "a": a,
                "c": None if potential == "linear" else c,
                "num_qubits": num_qubits,
                "min_eigenvalue":min_eigenvalue.real,
                "shots": shots,
                "num_steps":num_steps,
                "num_grad_checks":num_grad_checks,
                "phi":phi,
                "num_vqe_runs": num_vqe_runs,
                "optimizer_options": optimizer_options,
                "basis_state":basis_state,
                "operator_pool":operator_pool,
                "path": base_path,
                "pauli_coeffs": pauli_coeffs,
                "pauli_terms": pauli_terms
                }
    

    vqe_starttime = datetime.now()

    print("Starting ADAPT-VQE")
    # Start multiprocessing for VQE runs
    with Pool(processes=num_processes) as pool:
        vqe_results = pool.starmap(
            run_adapt_vqe,
            [
                (i, pauli_H, run_info)
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
        "phi": phi,
        "basis_state": basis_state,
        "operator_pool": [str(op) for op in operator_pool],
        "all_energies": all_energies,
        "min_energies": min_energies,
        #"initial_op_list":combined_intital,
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
