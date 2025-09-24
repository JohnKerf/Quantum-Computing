import pennylane as qml
from pennylane.pauli import group_observables

from scipy.optimize import minimize

import os
import json
import numpy as np
from datetime import datetime, timedelta
import time


from multiprocessing import Pool

from susy_qm import calculate_Hamiltonian

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def cost_function(params, prev_param_list, H_decomp, num_qubits, shots, beta, num_swap_tests):

    paulis = H_decomp.ops
    coeffs = H_decomp.coeffs
    groups = group_observables(paulis)    
   
    swap_dev = qml.device("default.qubit", wires=2*num_qubits, shots=None)

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

        qml.CNOT(wires=[wires[-1], wires[0]])

        for i, w in enumerate(wires):
            qml.RY(params[n + i], wires=w)

    #Swap test to calculate overlap
    @qml.qnode(swap_dev)
    def swap_test(params1, params2):

        start = datetime.now()

        ansatz(params1)
        ansatz(params2, prev=True)

        qml.Barrier()
        for i in range(num_qubits):
            qml.CNOT(wires=[i, i+num_qubits])    
            qml.Hadamard(wires=i)      

        prob = qml.probs(wires=range(2*num_qubits))

        end = datetime.now()
        global swap_time
        swap_time = (end - start)

        return prob
    
    dev = qml.device("default.qubit", wires=num_qubits, shots=shots)
    @qml.qnode(dev)
    def expected_value(params, groups):

        start = datetime.now()

        ansatz(params)
        #exval = qml.expval(qml.Hermitian(H, wires=range(num_qubits)))

        end = datetime.now()
        global expval_time
        expval_time = (end - start)

        return [qml.expval(op) for op in groups]
    
    
    def overlap(params, prev_params):

        #overlap_time = timedelta()
        probs = swap_test(params, prev_params)
        #overlap_time += swap_time

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

        return overlap#, overlap_time
    
    
    def multi_swap_test(params, prev_params):

        multi_swap_time = timedelta()
        
        results = []
        for _ in range(num_swap_tests):

            ol = overlap(params, prev_params)
            multi_swap_time += swap_time
            results.append(ol)
        
        avg_ol = sum(results) / num_swap_tests

        return avg_ol, multi_swap_time
    

    def loss_f(params):

        total_time = timedelta()

        energy = 0
        for group in groups:
            results = expected_value(params, group)
            for op, res in zip(group, results):
                idx = paulis.index(op)
                energy += coeffs[idx] * res

        penalty = 0

        if len(prev_param_list) != 0:
            for prev_param in prev_param_list:
                ol, ol_time = multi_swap_test(params,prev_param)
                penalty += (beta*ol)
                total_time += ol_time

        device_time = total_time + expval_time

        return energy + (penalty), device_time

    return loss_f(params)


def run_vqd(i, max_iter, tol, initial_tr_radius, final_tr_radius, H_decomp, num_qubits, num_params, shots, num_energy_levels, beta, num_swap_tests):
    
    # We need to generate a random seed for each process otherwise each parallelised run will have the same result
    seed = (os.getpid() * int(time.time())) % 123456789
    run_start = datetime.now()

    np.random.seed(seed)
    x0 = np.random.random(size=num_params)*2*np.pi
    bounds = [(0, 2 * np.pi) for _ in range(num_params)]

    all_energies = []
    prev_param_list = []
    all_success = []
    all_num_iters = []
    all_evaluations = []
    all_dev_times = []

    device_time = timedelta()

    def wrapped_cost_function(params):
        result, dt = cost_function(params, prev_param_list, H_decomp, num_qubits, shots, beta, num_swap_tests)
        nonlocal device_time
        device_time += dt
        all_dev_times.append(dt)
        return result

    for _ in range(num_energy_levels):
        
        # Differential Evolution optimization
        res = minimize(
            wrapped_cost_function,
            x0,
            #bounds=bounds,
            method= "COBYQA",
            options= {
                'maxiter':max_iter, 
                'maxfev':max_iter, 
                #'tol':tol, 
                'initial_tr_radius':initial_tr_radius, 
                'final_tr_radius':final_tr_radius, 
                'scale':True, 
                'disp':False}
        )


        all_energies.append(res.fun)
        prev_param_list.append(res.x)
        all_success.append(res.success)
        all_num_iters.append(res.nit)
        all_evaluations.append(res.nfev)

    run_end = datetime.now()
    run_time = run_end - run_start
    

    return {
        "seed": seed,
        "energies": all_energies,
        "params": prev_param_list,
        "success": all_success,
        "num_iters": all_num_iters,
        "evaluations": all_evaluations,
        "run_time": run_time,
        "device_time": device_time
    }


if __name__ == "__main__":
    
    potential_list = ["QHO"]#, "AHO", "DW"]
    cut_offs_list = [8,16]#,8,16]
    shots_list = [10000]#, 10000, 100000]
    
    num_vqd_runs = 100
    num_energy_levels = 3    
    num_swap_tests = 1
    beta = 2.0

    max_iter = 10000
    tol = 1e-8
    initial_tr_radius = 0.1
    final_tr_radius = 1e-11

    #rads = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    #rads = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    #betas = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0]

    #for beta in betas:
    #for beta in betas:
    #print(f"Running for {str(final_tr_radius)} final_tr_radius")

    for shots in shots_list:

        print(f"Running for {str(shots)} shots")

        for potential in potential_list:

            starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            base_path = os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\SUSY QM\PennyLane\COBYQA\PauliDecomp\VQD\Files\NegPen2", str(shots), potential)
            os.makedirs(base_path, exist_ok=True)

            print(f"Running for {potential} potential")

            for cut_off in cut_offs_list:

                print(f"Running for cutoff: {cut_off}")

                # Calculate Hamiltonian and expected eigenvalues
                H = calculate_Hamiltonian(cut_off, potential)
                eigenvalues = np.sort(np.linalg.eig(H)[0])[:3]
                min_eigenvalue = min(eigenvalues.real)

                num_qubits = int(np.log2(cut_off)+1)    

                H_decomp = qml.pauli_decompose(H, wire_order=range(num_qubits))

                # Optimizer
                num_params = 2*num_qubits

                # Start multiprocessing for VQE runs
                with Pool(processes=10) as pool:
                    vqd_results = pool.starmap(
                        run_vqd,
                        [
                            (i, max_iter, tol, initial_tr_radius, final_tr_radius, H_decomp, num_qubits, num_params, shots, num_energy_levels, beta, num_swap_tests)
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
                total_device_time = sum([res['device_time'] for res in vqd_results], timedelta())

                vqd_end = datetime.now()
                vqd_time = vqd_end - datetime.strptime(starttime, "%Y-%m-%d_%H-%M-%S")

                # Save run
                run = {
                    "starttime": starttime,
                    "potential": potential,
                    "cutoff": cut_off,
                    "exact_eigenvalues": [x.real.tolist() for x in eigenvalues],
                    "ansatz": "StronglyEntanglingLayers-1layer",
                    "num_VQD": num_vqd_runs,
                    "num_energy_levels": num_energy_levels,
                    "num_swap_tests": num_swap_tests,
                    "beta": beta,
                    "shots": shots,
                    "Optimizer": {
                        "name": "COBYQA",
                        "maxiter": max_iter,
                        'maxfev': max_iter,
                        "tolerance": tol,
                        "initial_tr_radius": initial_tr_radius,
                        "final_tr_radius": final_tr_radius
                    },
                    "results": all_energies,
                    "params": [[x.tolist() for x in param_list] for param_list in all_params],
                    "num_iters": all_num_iters,
                    "num_evaluations": all_evaluations,
                    "success": [np.array(x, dtype=bool).tolist() for x in all_success],
                    "run_times": run_times,
                    "seeds": seeds,
                    "parallel_run_time": str(vqd_time),
                    "total_VQD_time": str(total_run_time),
                    "total_device_time": str(total_device_time)
                }

                # Save the variable to a JSON file
                path = os.path.join(base_path, "{}_{}.json".format(potential, cut_off))
                with open(path, "w") as json_file:
                    json.dump(run, json_file, indent=4)

                print("Done")
