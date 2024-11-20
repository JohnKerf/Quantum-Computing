# PennyLane imports
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.optimize import AdamOptimizer

# General imports
import os
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from qiskit.quantum_info import SparsePauliOp

#custom module
from susy_qm import calculate_Hamiltonian, create_vqe_plots


#potential = 'QHO'
potential = 'AHO'
#potential = 'DW'

cut_offs_list = [2,4,8,16]#,32]
tol_list = [1e-1, 1e-2, 1e-3, 1e-4]
#tol_list = [1e-6]

for tolerance in tol_list:

    starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = str(starttime)
    #Create directory for files
    os.makedirs(r"C:\Users\Johnk\OneDrive\Desktop\PhD 2024\Quantum Computing Code\Quantum-Computing\SUSY\PennyLane\SUSY VQE\Shot Noise\Adam\Files\{}\\{}".format(potential, folder))

    print(f"Running for {potential} potential")

    print(f"Running for tolerance: {str(tolerance)}")

    for cut_off in cut_offs_list:

        print(f"Running for cutoff: {cut_off}")

        #calculate Hamiltonian and expected eigenvalues
        H = calculate_Hamiltonian(cut_off, potential)
        eigenvalues = np.sort(np.linalg.eig(H)[0])
        min_eigenvalue = min(eigenvalues.real)

        #create qiskit Hamiltonian Pauli string
        hamiltonian = SparsePauliOp.from_operator(H)
        num_qubits = hamiltonian.num_qubits

        # Device
        shots = 1024
        #dev = qml.device('default.qubit', wires=num_qubits, shots=100)
        dev = qml.device('lightning.qubit', wires=num_qubits, shots=shots)

        # Define the cost function
        @qml.qnode(dev)
        def cost_function(params):
            qml.StronglyEntanglingLayers(weights=params, wires=range(num_qubits), imprimitive=qml.CZ)
            return qml.expval(qml.Hermitian(H, wires=range(num_qubits)))
        
        
        #Optimizer
        stepsize = 0.5
        optimizer = AdamOptimizer(stepsize=stepsize)

        # VQE
        vqe_start = datetime.now()

        #variables
        num_vqe_runs = 100
        max_iterations = 10000
        #tolerance = 1e-8
        moving_avg_length = 5
        #gradient_tol = 1e-4

        #data arrays
        energies = []
        param_values = []
        success = []
        run_times = []
        num_iters = []

        for j in range(num_vqe_runs):

            if j % 10==0:
                print(f"VQE run: {j}")

            run_start = datetime.now()
            converged = False
            prev_energy = None

            moving_average_check = False
            gradient_norm_check = False

            #Initial params
            scale = 0.25
            params_shape = qml.StronglyEntanglingLayers.shape(n_layers=1, n_wires=num_qubits)
            params = scale*np.pi * pnp.random.random(size=params_shape)

            iter_energies = []

            for i in range(max_iterations):

                params, energy = optimizer.step_and_cost(cost_function, params)
                iter_energies.append(energy)

                # Moving average convergence check
                if len(iter_energies) > moving_avg_length:
                    energy_moving_avg = np.mean(np.abs(np.diff(iter_energies[-moving_avg_length:])))
                    if energy_moving_avg < tolerance:
                        #print(f"Converged at iteration {i} with moving average change = {energy_moving_avg}")
                        moving_average_check = True
                        converged = True
                        break

                # Gradient norm convergence check
                #grads = optimizer.compute_grad(cost_function, (params,), {})
                #grad_norm = np.linalg.norm(grads[0])
                #if grad_norm < gradient_tol:
                #    #print(f"Converged at iteration {i} with gradient norm = {grad_norm}")
                #    gradient_norm_check = True

                #if moving_average_check & gradient_norm_check:
                #    #print("moving_average_check and gradient_norm_check converged")
                #    converged = True
                #    break


                ## Check for convergence
                #if prev_energy is not None and np.abs(energy - prev_energy) < tolerance:
                #    converged = True
                #    break

                prev_energy = energy
            
            if converged == False:
                print("Not converged")

            energies.append(energy)
            param_values.append(params)
            success.append(converged)
            num_iters.append(i+1)

            run_end = datetime.now()
            run_time = run_end - run_start
            run_times.append(run_time)

        vqe_end = datetime.now()
        vqe_time = vqe_end - vqe_start

        #Save run
        run = {
            'potential': potential,
            'cutoff': cut_off,
            'exact_eigenvalues': [round(x.real,10).tolist() for x in eigenvalues],
            'ansatz': 'StronglyEntanglingLayers-1layer',
            'shots': shots,
            'ansatz_scale': scale,
            'num_VQE': num_vqe_runs,
            'Optimizer': {'name': 'AdamOptimizer',
                        'stepsize':stepsize,
                        'maxiter':max_iterations,
                        'tolerance': tolerance,
                        'moving_avg_length': moving_avg_length
                        },
            'results': energies,
            'params': [x.tolist() for x in param_values],
            'num_iters': num_iters,
            'success': np.array(success, dtype=bool).tolist(),
            'run_times': [str(x) for x in run_times],
            'total_run_time': str(vqe_time)
        }

        # Save the variable to a JSON file
        path = r"C:\Users\Johnk\OneDrive\Desktop\PhD 2024\Quantum Computing Code\Quantum-Computing\SUSY\PennyLane\SUSY VQE\Shot Noise\Adam\Files\{}\\{}\{}_{}.json".format(potential, folder, potential, cut_off)
        with open(path, 'w') as json_file:
            json.dump(run, json_file, indent=4)


    # create plots
    base_path = r"C:\Users\Johnk\OneDrive\Desktop\PhD 2024\Quantum Computing Code\Quantum-Computing\SUSY\PennyLane\SUSY VQE\Shot Noise\Adam\Files\{}\\{}\\"
    create_vqe_plots(potential=potential, base_path=base_path, folder=folder, cut_off_list=cut_offs_list)
   
