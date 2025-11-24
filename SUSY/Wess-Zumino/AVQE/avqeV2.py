import pennylane as qml
from pennylane import numpy as pnp

from scipy.optimize import minimize

import os
import json
import numpy as np
from datetime import datetime, timedelta
import time

from multiprocessing import Pool

from collections import Counter

from wesszumino import build_wz_hamiltonian, pauli_str_to_op

import git
repo_path = git.Repo('.', search_parent_directories=True).working_tree_dir

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)




def compute_grad(param, H, paulis, coeffs, num_qubits, operator_ham, op_list, op_params, basis_state, dev):

    coeffs_p = pnp.array(coeffs, dtype=float)
    basis_state_p = pnp.array(basis_state, dtype=int)

    @qml.qnode(dev, interface="autograd")
    def expval_circuit(p):

        qml.BasisState(basis_state_p, wires=range(num_qubits))

        for theta, op in zip(op_params, op_list):
            type(op)(theta, wires=op.wires)

        type(operator_ham)(p, wires=operator_ham.wires)

        return [qml.expval(op) for op in paulis]

    def cost_fn(p):
        expvals = expval_circuit(p)          
        return pnp.dot(coeffs_p, expvals)    

    p = pnp.tensor(param, requires_grad=True)
    grad = qml.grad(cost_fn)(p)

    return grad



def cost_function(params, H, paulis, coeffs, num_qubits, shots, op_list, basis_state, dev):
   
    start = datetime.now()
  
    @qml.qnode(dev)
    def circuit(params):

        qml.BasisState(basis_state, wires=range(num_qubits))

        param_index = 0
        for op in op_list:
            o = type(op)
            o(params[param_index], wires=op.wires)
            param_index +=1

        #return qml.expval(qml.Hermitian(H, wires=range(num_qubits)))
        return [qml.expval(op) for op in paulis]
    
    expvals = circuit(params)                 
    energy = float(np.dot(coeffs, expvals)) 

    end = datetime.now()
    device_time = (end - start)

    return energy, device_time


def run_adapt_vqe(i, H, run_info):

    num_qubits = run_info["num_qubits"] 
    shots = run_info["shots"]  
    basis_state = run_info["basis_state"]
    phi = run_info["phi"]
    paulis = run_info['pauli_terms']
    coeffs = run_info['pauli_coeffs']


    # We need to generate a random seed for each process otherwise each parallelised run will have the same result
    seed = (os.getpid() * int(time.time())) % 123456789
    dev = qml.device(run_info["device"], wires=num_qubits, shots=run_info["shots"], seed=seed)
    run_start = datetime.now()

    device_time = timedelta()

    def wrapped_cost_function(params):
        result, dt = cost_function(params, H, paulis, coeffs, num_qubits, shots, op_list, basis_state, dev)
        nonlocal device_time
        device_time += dt
        return result
    

    # Main ADAPT-VQE script
    op_list = []#run_info["initial_op_list"]
    op_params = []#run_info["initial_params"]
    energies = []

    pool = run_info["operator_pool"].copy()
    success = False

    for i in range(run_info["num_steps"]):

        max_ops_list = []
        
        if i != 0:
            
            pool.remove(most_common_gate)

            if (type(most_common_gate) == qml.FermionicSingleExcitation) or (type(most_common_gate) == qml.CRY):
                cq = most_common_gate.wires[0]
                tq = most_common_gate.wires[1]

                if (qml.RY(phi, wires=cq) not in pool):
                    pool.append(qml.RY(phi, wires=cq))

                if (qml.RZ(phi, wires=cq) not in pool):
                    pool.append(qml.RZ(phi, wires=cq))

                if (qml.RY(phi, wires=tq) not in pool):
                    pool.append(qml.RY(phi, wires=tq))

                if (qml.RZ(phi, wires=tq) not in pool):
                    pool.append(qml.RZ(phi, wires=tq))

        
        for param in np.random.uniform(phi, phi, size=run_info["num_grad_checks"]):
            grad_list = []
            for op in pool:
                grad = compute_grad(param, H, paulis, coeffs, num_qubits, op, op_list, op_params, basis_state, dev)
                o=type(op)

                if (o == qml.CNOT) or (o == qml.CZ):
                    grad_op = o(wires=op.wires)
                else:
                    grad_op = o(param, wires=op.wires)

                grad_list.append((grad_op,abs(grad)))

            max_op, max_grad = max(grad_list, key=lambda x: x[1])
            max_ops_list.append(max_op)

        counter = Counter(max_ops_list)
        most_common_gate, count = counter.most_common(1)[0]
        op_list.append(most_common_gate)


        np.random.seed(seed)
        x0 = np.concatenate((op_params, np.array([np.random.random()*2*np.pi])))
        
        res = minimize(
            wrapped_cost_function,
            x0,
            method= "COBYQA",
            options= run_info["optimizer_options"]
        )
        
        if i!=0: pre_min_e = min_e
        min_e = res.fun
        pre_op_params = op_params.copy()
        op_params = res.x

        energies.append(min_e)

        if i!=0:
            if abs(pre_min_e - min_e) < 1e-8:
                print("Energy converged")
                energies.pop()
                op_list.pop()
                final_params = pre_op_params
                success = True
                break
            if abs(run_info["min_eigenvalue"]-min_e) < 1e-12:
                print("Energy close to min eigenvalue")
                success = True
                final_params = op_params
                break
        
    run_end = datetime.now()
    run_time = run_end - run_start

    if success == False:
        final_params = op_params

    final_ops = []
    for op, param in zip(op_list,final_params):
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
        "run_time": run_time,
        "device_time": device_time
    }


if __name__ == "__main__":

    device = 'default.qubit'
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
    num_grad_checks = 1
    num_vqe_runs = 10
    max_iter = 10000
    initial_tr_radius = 1.0
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

    base_path = os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\Wess-Zumino\AVQE\Files", boundary_condition, potential, folder)
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
        operator_pool.append(qml.FermionicSingleExcitation(phi, wires=[f0, f1]))
        


    nb = int(np.log2(cutoff))
    n = 1 + nb
    fw = [i * n for i in range(N)]

    pairs = [(fw[i], fw[i+1]) for i in range(len(fw)-1)]

    initial_op_list = []
    initial_params = [np.pi/2]*N
    for pair in pairs:
        initial_op_list.append(qml.FermionicSingleExcitation(phi,wires=pair))

    # Choose basis state
    #basis_state = [0]*num_qubits
    #basis_state = [0]*(num_qubits-1) + [1]
    #basis_state = [1] + [0]*(num_qubits-1) 
    #basis_state = [0]*n + [1]+[0]*nb + [1]+[0]*nb + [1]+[0]*nb + [0]*n
    basis_state = [0]*n + [1]+[0]*nb + [1]+[0]*nb + [0]*n
    #basis_state = [1,0,0,0,0,0]
    #basis_state = [0,0,1,0,1,0,0,0]      
     

    combined_intital = []
    for op, param in zip(initial_op_list,initial_params):
        dict = {"name": op.name,
                "param": param,
                "wires": op.wires.tolist(),
                "energy":0.0}
        combined_intital.append(dict)

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
                "initial_op_list":initial_op_list, 
                "initial_params":initial_params,
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
                (i, dense_H, run_info)
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
    total_device_time = sum([res['device_time'] for res in vqe_results], timedelta())

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
        "total_VQE_time": str(total_run_time),
        "total_device_time": str(total_device_time)
    }

    # Save the variable to a JSON file
    path = os.path.join(base_path, "{}_{}.json".format(potential, cutoff))
    with open(path, "w") as json_file:
        json.dump(run, json_file, indent=4)

    print("Done")
