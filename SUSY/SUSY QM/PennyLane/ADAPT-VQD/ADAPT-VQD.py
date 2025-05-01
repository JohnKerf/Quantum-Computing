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

from susy_qm import calculate_Hamiltonian

###########################################################################################################
def create_circuit(params, num_qubits, op_list, current_basis, pre_basis=[], use_trial=False, trial_op=None, swap=False):

    param_index = 0

    if swap:
        qml.BasisState(pre_basis, wires=range(num_qubits, 2*num_qubits))
        for op in op_list:
            o = type(op)
            if o == qml.CRY:
                w0 = op.wires[0] + num_qubits
                w1 = op.wires[1] + num_qubits
                o(params[param_index], wires=[w0,w1])
                param_index += 1
            else:
                wire = op.wires[0] + num_qubits
                o(params[param_index], wires=wire)
                param_index += 1
    else:
        qml.BasisState(current_basis, wires=range(num_qubits))
        for op in op_list:
            o = type(op)
            o(params[param_index], wires=op.wires)
            param_index +=1

        if use_trial:
            to = type(trial_op)
            to(0.0, wires=trial_op.wires)



def multi_swap_test(num_qubits, pre_basis, current_basis, prev_op_list, prev_params, op_list, op_params, use_trial, trial_op=None, num_swap_tests=1):

    swap_dev = qml.device("default.qubit", wires=2*num_qubits, shots=None)
    @qml.qnode(swap_dev)
    def swap_test(num_qubits, pre_basis, current_basis, prev_op_list, prev_params, op_list, op_params, use_trial, trial_op):

        create_circuit(prev_params, num_qubits, prev_op_list, current_basis, pre_basis, swap=True)
        create_circuit(op_params, num_qubits, op_list, current_basis, pre_basis, use_trial, trial_op=trial_op)

        qml.Barrier()
        for i in range(num_qubits):
            qml.CNOT(wires=[i, i+num_qubits])    
            qml.Hadamard(wires=i)      

        prob = qml.probs(wires=range(2*num_qubits))

        return prob

    def overlap(pre_basis, current_basis, prev_op_list, prev_params, op_list, op_params, use_trial, trial_op):

        probs = swap_test(num_qubits, pre_basis, current_basis, prev_op_list, prev_params, op_list, op_params, use_trial, trial_op)

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



    results = []
    for _ in range(num_swap_tests):

        ol = overlap(pre_basis, current_basis, prev_op_list, prev_params, op_list, op_params, use_trial, trial_op)
        results.append(ol)
    
    avg_ol = sum(results) / num_swap_tests

    return avg_ol


def compute_grad(trial_param, H, num_qubits, trial_op, op_list, op_params, basis_state):

    dev2 = qml.device("default.qubit", wires=num_qubits, shots=None)
    @qml.qnode(dev2)

    def grad_circuit(trial_param, trial_op, op_list, op_params):

        qml.BasisState(basis_state, wires=range(num_qubits))

        param_index = 0
        for op in op_list:
            o = type(op)
            o(op_params[param_index], wires=op.wires)
            param_index +=1

        oph = type(trial_op)
        oph(trial_param, wires=trial_op.wires)

        return qml.expval(qml.Hermitian(H, wires=range(num_qubits)))
    
    params = pnp.tensor(trial_param, requires_grad=True)
    grad_fn = qml.grad(grad_circuit)
    grad = grad_fn(params, trial_op, op_list, op_params)
    
    return grad


def grad_plus_overlap(H, num_qubits, e_level, basis_list, trial_op, trial_param, op_list, op_params, prev_op_list, prev_params, beta):

    current_basis = basis_list[e_level]

    grad = compute_grad(trial_param, H, num_qubits, trial_op, op_list, op_params, current_basis)
  
    penalty = 0
    pre_level = 0
    if len(prev_op_list) != 0:
            for prev_op, prev_param in zip(prev_op_list, prev_params):
                pre_basis = basis_list[pre_level]
                ol = multi_swap_test(num_qubits, pre_basis, current_basis, prev_op, prev_param, op_list, op_params, use_trial=True, trial_op=trial_op)
                penalty += (beta*ol)
                pre_level+=1

    
    return abs(grad), penalty


def cost_function(params, H, num_qubits, shots, op_list, prev_op_list, prev_params, basis_list, e_level, beta):

    current_basis = basis_list[e_level]

    dev = qml.device("default.qubit", wires=num_qubits, shots=shots)
    start = datetime.now()

    @qml.qnode(dev)
    def energy_expval(params, num_qubits, op_list, basis_state):

        create_circuit(params, num_qubits, op_list, basis_state)

        return qml.expval(qml.Hermitian(H, wires=range(num_qubits)))

    energy = energy_expval(params, num_qubits, op_list, current_basis)

    penalty = 0
    pre_level = 0
    if len(prev_op_list) != 0:
        for prev_op, prev_param in zip(prev_op_list, prev_params):
                    pre_basis = basis_list[pre_level]
                    ol = multi_swap_test(num_qubits, pre_basis, current_basis, prev_op, prev_param, op_list, params, use_trial=False)
                    penalty += (beta*ol)
                    pre_level+=1

    end = datetime.now()
    device_time = (end - start)

    return (energy + (penalty)), device_time




def run_adapt_vqd(i, max_iter, tol, abs_tol, strategy, popsize, H, num_qubits, shots, num_energy_levels, num_adapt_steps, phi, num_grad_checks, operator_pool, basis_list, eigenvalues, beta):

    # We need to generate a random seed for each process otherwise each parallelised run will have the same result
    seed = (os.getpid() * int(time.time())) % 123456789
    run_start = datetime.now()

    device_time = timedelta()

    def wrapped_cost_function(params):
        result, dt = cost_function(params, H, num_qubits, shots, op_list, prev_op_list, prev_params, basis_list, e_level, beta)
        nonlocal device_time
        device_time += dt
        return result
    
    # Main ADAPT-VQE script
    prev_op_list = []
    prev_params = []
    all_energies = []
    success_list = []
    final_ops_list = []
    single_ops = [qml.RY]


    pool = operator_pool.copy()
    success = False

    for e_level in range(num_energy_levels):

        op_list = []
        op_params = []
        energies = []
        pool = operator_pool.copy()
        success = False

        current_eigenval = eigenvalues[e_level]
        #print(f"Looking for energy level {e_level} with eigenvalue {current_eigenval}")

        for i in range(num_adapt_steps):

            #print(f"Running for adapt step: {i}")

            max_ops_list = []
            
            if i != 0:
                
                pool.remove(most_common_gate)

                if type(most_common_gate) == qml.CRY:
                    cq = most_common_gate.wires[0]
                    tq = most_common_gate.wires[1]

                    for sop in single_ops:
                        if (sop(phi, wires=cq) not in pool):
                            pool.append(sop(phi, wires=cq))

                        if (sop(phi, wires=tq) not in pool):
                            pool.append(sop(phi, wires=tq))
            
            for trial_param in np.random.uniform(phi, phi, size=num_grad_checks):
                grad_list = []
                grads = []
                penalties = []
                for trial_op in pool:
                    grad, penalty = grad_plus_overlap(H, num_qubits, e_level, basis_list, trial_op, trial_param, op_list, op_params, prev_op_list, prev_params, beta)
                    grads.append(grad)
                    penalties.append(penalty)
                    o=type(trial_op)
                    grad_op = o(trial_param, wires=trial_op.wires)

                    grad_list.append(grad_op)

                penalties = np.array(penalties)
                grad_norm = grads if max(grads) == 0 else grads / max(grads)
                penalty_norm = penalties if max(penalties) == 0 else penalties / max(penalties)

                #grad_norm = np.where(max(grads) != 0, grads / max(grads), 0)
                #penalty_norm = np.where(max(penalties) != 0, np.array(penalties) / max(penalties), 0)

                gp = grad_norm - penalty_norm
                max_gp = np.argmax(gp)

                max_op = grad_list[max_gp]
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
            
            #print('Starting VQE')

            res = differential_evolution(wrapped_cost_function,
                                            bounds=bounds,
                                            #args=(H, num_qubits, shots, op_list, prev_op_list, prev_params, basis_list, e_level, beta),
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
                if abs(current_eigenval-min_e) < 1e-6:
                    #print("Converged to min e")
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
                    "wires": op.wires.tolist()
                    }
            final_ops.append(dict)

        prev_op_list.append(op_list)
        prev_params.append(final_params)
        all_energies.append(energies)

        final_ops_list.append(final_ops)
        success_list.append(success)

    return {
        "seed": seed,
        "energies": all_energies,
        #"min_energy": min_e,
        #"op_params": op_params,
        "op_list": final_ops_list,
        "success": success_list,
        #"num_iters": i+1,
        "run_time": run_time,
        "device_time": device_time
    }


if __name__ == "__main__":
    
    potential = "AHO"
    cutoff = 16
    shots = 1024

    print(f"Running for {potential} potential, cutoff {cutoff}")

    starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_path = os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\SUSY QM\PennyLane\ADAPT-VQD\TestFiles", potential)
    os.makedirs(base_path, exist_ok=True)


    # Calculate Hamiltonian and expected eigenvalues
    H = calculate_Hamiltonian(cutoff, potential)

    eigenvalues = np.sort(np.linalg.eig(H)[0])[:4]
    min_eigenvalue = np.min(eigenvalues)

    #create qiskit Hamiltonian Pauli string
    hamiltonian = SparsePauliOp.from_operator(H)
    num_qubits = hamiltonian.num_qubits

    #Create operator pool
    operator_pool = []
    phi = 0.0
    for i in range(num_qubits):
        operator_pool.append(qml.RY(phi,wires=[i]))

    c_pool = []

    for control in range(num_qubits):
            for target in range(num_qubits):
                if control != target:
                    c_pool.append(qml.CRY(phi=phi, wires=[control, target]))

    operator_pool = operator_pool + c_pool    
    

    # Optimizer
    num_energy_levels = 3
    num_adapt_steps = 3
    num_grad_checks = 10
    beta = 1.0

    num_vqd_runs = 16
    max_iter = 200
    strategy = "randtobest1bin"
    tol = 1e-3
    abs_tol = 1e-2
    popsize = 20

    #QHO
    #basis_list = [[1] + [0]*(num_qubits-1),
    #            [0]*(num_qubits),
    #            [1] + [0]*(num_qubits-1)
    #            ]

    #AHO
    basis_list = [[1] + [0]*(num_qubits-1),
                [1] + [0]*(num_qubits-1),
                [0]*(num_qubits)
                ]

    vqe_starttime = datetime.now()

    print("Starting ADAPT-VQD")
    # Start multiprocessing for VQE runs
    with Pool(processes=4) as pool:
        vqe_results = pool.starmap(
            run_adapt_vqd,
            [
                (i, max_iter, tol, abs_tol, strategy, popsize, H, num_qubits, shots, num_energy_levels, num_adapt_steps, phi, num_grad_checks, operator_pool, basis_list, eigenvalues, beta)
                for i in range(num_vqd_runs)
            ],
        )

    print("Finished ADAPT-VQD")
    # Collect results
    seeds = [res["seed"] for res in vqe_results]
    all_energies = [res["energies"] for res in vqe_results]
    #min_energies = [res["min_energy"] for res in vqe_results]
    #op_params = [str(res["op_params"]) for res in vqe_results]
    op_lists = [res["op_list"] for res in vqe_results]
    success = [res["success"] for res in vqe_results]
    #num_iters = [res["num_iters"] for res in vqe_results]
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
        "exact_eigenvalues": [x.real.tolist() for x in eigenvalues],
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
        "num_VQD": num_vqd_runs,
        "num_energy_levels": num_energy_levels,
        "num_steps":num_adapt_steps,
        "num_grad_checks":num_grad_checks,
        "phi": phi,
        "beta": beta,
        "basis_state": basis_list,
        "operator_pool": [str(op) for op in operator_pool],
        "all_energies": all_energies,
        #"min_energies": min_energies,
        #"op_params": op_params,
        "op_list": op_lists,
        #"num_iters": num_iters,
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
