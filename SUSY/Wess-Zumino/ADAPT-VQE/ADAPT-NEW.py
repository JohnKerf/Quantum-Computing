import pennylane as qml
from pennylane import numpy as pnp

from scipy.optimize import differential_evolution
from scipy.stats.qmc import Halton

import os
import json
import numpy as np
from datetime import datetime, timedelta
import time

from qiskit.quantum_info import SparsePauliOp

from multiprocessing import Pool

from collections import Counter

from susy_qm import calculate_wz_hamiltonian


def compute_grad(param, H, num_qubits, operator_ham, op_list, op_params, basis_state):

    dev2 = qml.device("default.qubit", wires=num_qubits, shots=None)
    @qml.qnode(dev2)

    def grad_circuit(param, operator_ham, op_list, op_params):

        qml.BasisState(basis_state, wires=range(num_qubits))

        param_index = 0
        for op in op_list:
            o = type(op)
            o(op_params[param_index], wires=op.wires)
            param_index +=1

        oph = type(operator_ham)
        oph(param, wires=operator_ham.wires)

        return qml.expval(qml.Hermitian(H, wires=range(num_qubits)))
    
    params = pnp.tensor(param, requires_grad=True)
    grad_fn = qml.grad(grad_circuit)
    grad = grad_fn(params, operator_ham, op_list, op_params)
    
    return grad



def cost_function(params, H, num_qubits, shots, op_list, basis_state):
   
    dev = qml.device("default.qubit", wires=num_qubits, shots=shots)
    start = datetime.now()
  
    @qml.qnode(dev)
    def circuit(params):

        qml.BasisState(basis_state, wires=range(num_qubits))

        param_index = 0
        for op in op_list:
            o = type(op)
            o(params[param_index], wires=op.wires)
            param_index +=1

        return qml.expval(qml.Hermitian(H, wires=range(num_qubits)))

     
    end = datetime.now()
    device_time = (end - start)

    return circuit(params), device_time


def run_adapt_vqe(i, max_iter, tol, abs_tol, strategy, popsize, H, num_qubits, shots, num_steps, phi, num_grad_checks, operator_pool, basis_state, min_eigenvalue, initial_op_list, initial_params):

    # We need to generate a random seed for each process otherwise each parallelised run will have the same result
    seed = (os.getpid() * int(time.time())) % 123456789
    run_start = datetime.now()
    device_time = timedelta()

    def wrapped_cost_function(params):
        result, dt = cost_function(params, H, num_qubits, shots, op_list, basis_state)
        nonlocal device_time
        device_time += dt
        return result
    

    # Main ADAPT-VQE script
    op_list = initial_op_list.copy()
    op_params = initial_params.copy()

    print(op_list)

    energies = []

    pool = operator_pool.copy()
    success = False

    single_ops = [qml.RY,qml.RZ,qml.RX]

    for i in range(num_steps):

        max_ops_list = []
        
        if i != 0:
            
            pool.remove(most_common_gate)

            if (type(most_common_gate) == qml.FermionicSingleExcitation) or (type(most_common_gate) == qml.CRY):
                cq = most_common_gate.wires[0]
                tq = most_common_gate.wires[1]

                for sop in single_ops:
                    if (sop(phi, wires=cq) not in pool):
                        pool.append(sop(phi, wires=cq))

                    if (sop(phi, wires=tq) not in pool):
                        pool.append(sop(phi, wires=tq))
        
        for param in np.random.uniform(phi, phi, size=num_grad_checks):
            grad_list = []
            for op in pool:
                grad = compute_grad(param, H, num_qubits, op, op_list, op_params, basis_state)
                o=type(op)
                grad_op = o(param, wires=op.wires)

                grad_list.append((grad_op,abs(grad)))

            max_op, max_grad = max(grad_list, key=lambda x: x[1])
            max_ops_list.append(max_op)

        counter = Counter(max_ops_list)
        most_common_gate, count = counter.most_common(1)[0]
        op_list.append(most_common_gate)

        # Generate Halton sequence
        num_dimensions = len(op_list)
        num_samples = popsize
        halton_sampler = Halton(d=num_dimensions, seed=seed)
        halton_samples = halton_sampler.random(n=num_samples)
        scaled_samples = 2 * np.pi * halton_samples

        bounds = [(0, 2 * np.pi) for _ in range(num_dimensions)]
        x0 = np.concatenate((op_params, np.array([0.0])))
        
        res = differential_evolution(wrapped_cost_function,
                                        bounds,
                                        x0=x0,
                                        maxiter=max_iter,
                                        tol=tol,
                                        atol=abs_tol,
                                        strategy=strategy,
                                        popsize=popsize,
                                        init=scaled_samples,
                                        seed=seed
                                        )
        
        if i!=0: pre_min_e = min_e
        min_e = res.fun
        pre_op_params = op_params.copy()
        op_params = res.x

        energies.append(min_e)

        if i!=0:
            if abs(pre_min_e - min_e) < 1e-8:
                #print("gradient converged")
                energies.pop()
                op_list.pop()
                final_params = pre_op_params
                success = True
                break
            if abs(min_eigenvalue-min_e) < 1e-6:
                success = True
                final_params = op_params
                break

    run_end = datetime.now()
    run_time = run_end - run_start

    if success == False:
        final_params = op_params

    final_ops = []
    print(op_list)
    for op, param in zip(op_list,final_params):
        dict = {"name": op.name,
                "param": param,
                "wires": op.wires.tolist()
                }
        final_ops.append(dict)

    print(final_ops)

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
    
   
    # Parameters
    N = 3
    a = 1.0
    c = -0.8

    potential = "linear"
    #potential = 'quadratic'
    boundary_condition = 'dirichlet'
    #boundary_condition = 'periodic'
    cutoff = 2
    shots = 1024
    
    starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if potential == 'quadratic':
        folder = 'C' + str(abs(c)) + '/' + 'N'+ str(N) + '/' + str(starttime)
    else:
        folder = 'N'+ str(N) + '/' + str(starttime)

    base_path = os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\Wess-Zumino\ADAPT-VQE\Test", boundary_condition, potential, folder)
    os.makedirs(base_path, exist_ok=True)

    print(str(starttime))
    print(f"Running for {N} sites")
    print(f"Running for {boundary_condition} boundary conditions")
    print(f"Running for {potential} potential, cutoff {cutoff}")

    # Calculate Hamiltonian and expected eigenvalues
    ham_start_time = datetime.now()
    H = calculate_wz_hamiltonian(cutoff, N, a, potential, boundary_condition, c)
    ham_calc_time = datetime.now() - ham_start_time
    print(f"Time taken to create hamiltonian: {str(ham_calc_time)}")

    eigenvalues = np.sort(np.linalg.eig(H)[0])[:4]
    min_eigenvalue = min(eigenvalues)

    #create qiskit Hamiltonian Pauli string
    hamiltonian = SparsePauliOp.from_operator(H)
    num_qubits = hamiltonian.num_qubits

    #Create operator pool
    operator_pool = []
    phi = 0.0
    for i in range(num_qubits):
        operator_pool.append(qml.RY(phi,wires=[i]))
        #operator_pool.append(qml.RZ(phi,wires=[i]))
        #operator_pool.append(qml.RX(phi,wires=[i]))

    c_pool = []

    for control in range(N,num_qubits):
            for target in range(N,num_qubits):
                if control != target:
                    c_pool.append(qml.CRY(phi=phi, wires=[control, target]))
                    #c_pool.append(qml.FermionicSingleExcitation(phi, wires=[control, target]))

    for i in range(N-1):
        c_pool.append(qml.FermionicSingleExcitation(phi, wires=[i, i+1]))
        #c_pool.append(qml.FermionicSingleExcitation(phi, wires=[1, 2]))

    operator_pool = operator_pool + c_pool  


    #initial_op_list = []
    #initial_params = []
    #initial_op_list = [qml.RY(phi,wires=[1]), qml.FermionicSingleExcitation(phi,wires=[0,1])]
    #initial_params = [np.pi, np.pi/4] 

    #initial_op_list = [qml.RY(phi,wires=[3]), qml.RY(phi,wires=[5]), qml.CRY(phi,wires=[3,5]), qml.FermionicSingleExcitation(phi,wires=[0,1]), qml.FermionicSingleExcitation(phi,wires=[1,2])]
    #initial_params = [np.pi/2, np.pi, np.pi, 5.832757824583785, 2.6755479551819445] 
    initial_op_list = [qml.RY(phi,wires=[3]), qml.RY(phi,wires=[5]), qml.CRY(phi,wires=[3,5]), qml.FermionicSingleExcitation(phi,wires=[0,1]), qml.FermionicSingleExcitation(phi,wires=[1,2])]
    initial_params = [np.pi/2, np.pi, np.pi, 5.832757824583785, 2.6755479551819445] 

    combined_intital = []
    for op, param in zip(initial_op_list,initial_params):
        dict = {"name": op.name,
                "param": param,
                "wires": op.wires.tolist(),
                "energy":0.0}
        combined_intital.append(dict)

    # Choose basis state
    #basis_state = [0]*(num_qubits)
    #basis_state = [1,0,0,1,0,0]
    basis_state = [0,0,1,0,0,1]
    #basis_state = [1] + [0]*(num_qubits-1)

    # Optimizer
    num_steps = 3
    num_grad_checks = 1
    num_vqe_runs = 1
    max_iter = 500
    strategy = "randtobest1bin"
    tol = 1e-3
    abs_tol = 1e-2
    popsize = 20

    vqe_starttime = datetime.now()

    print("Starting ADAPT-VQE")
    # Start multiprocessing for VQE runs
    with Pool(processes=1) as pool:
        vqe_results = pool.starmap(
            run_adapt_vqe,
            [
                (i, max_iter, tol, abs_tol, strategy, popsize, H, num_qubits, shots, num_steps, phi, num_grad_checks, operator_pool, basis_state, min_eigenvalue, initial_op_list, initial_params)
                for i in range(num_vqe_runs)
            ],
        )

    print("Finished ADAPT-VQE")
    # Collect results
    seeds = [res["seed"] for res in vqe_results]
    all_energies = [res["energies"] for res in vqe_results]
    min_energies = [res["min_energy"] for res in vqe_results]
    #op_params = [str(res["op_params"]) for res in vqe_results]
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
        "exact_eigenvalues": [x.real.tolist() for x in np.sort(eigenvalues)],
        "ansatz": "circuit.txt",
        "shots": shots,
        "Optimizer": {
            "name": "differential_evolution",
            "bounds": "[(0, 2 * np.pi)]",
            "maxiter": max_iter,
            "tolerance": tol,
            "abs_tolerance": abs_tol,
            "strategy": strategy,
            "popsize": popsize,
            'init': 'scaled_samples',
        },
        "num_VQE": num_vqe_runs,
        "num_steps":num_steps,
        "num_grad_checks":num_grad_checks,
        "phi": phi,
        "basis_state": basis_state,
        "operator_pool": [str(op) for op in operator_pool],
        "all_energies": all_energies,
        "min_energies": min_energies,
        "initial_op_list": combined_intital,
        "op_list": op_lists,
        "num_iters": num_iters,
        "success": np.array(success, dtype=bool).tolist(),
        "run_times": run_times,
        "seeds": seeds,
        "parallel_run_time": str(vqe_time),
        "total_VQE_time": str(total_run_time),
        "total_device_time": str(total_device_time),
        "ham_calc_time": str(ham_calc_time)
    }

    # Save the variable to a JSON file
    path = os.path.join(base_path, "{}_{}.json".format(potential, cutoff))
    with open(path, "w") as json_file:
        json.dump(run, json_file, indent=4)
