# PennyLane imports
import pennylane as qml
from pennylane import numpy as pnp

from scipy.optimize import differential_evolution

# General imports
import os
import json
import numpy as np
from datetime import datetime

from qiskit.quantum_info import SparsePauliOp

# custom module
from susy_qm import calculate_Hamiltonian, create_plots


potential = 'QHO'
#potential = 'AHO'
#potential = 'DW'

#cut_offs_list = [2,4,8,16]#,32]
cut_offs_list = [2]
#tol_list = [1e-3, 1e-4, 1e-5, 1e-6]
tol_list = [1e-3]

for tolerance in tol_list:

    starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = str(starttime)
    #Create directory for files
    os.makedirs(r"C:\Users\Johnk\OneDrive\Desktop\PhD 2024\Quantum Computing Code\Quantum-Computing\SUSY\PennyLane\VQD\Shot Noise\Files\{}\\{}".format(potential, folder))

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
        #shots = None
        dev = qml.device('lightning.qubit', wires=num_qubits, shots=shots)


        #Initial params shape
        num_layers = 1
        params_shape = qml.StronglyEntanglingLayers.shape(n_layers=num_layers, n_wires=num_qubits)


        # Define the vqd cost function
        @qml.qnode(dev)
        def vqd_cost_function(params, prev_states):
            params = pnp.tensor(params.reshape(params_shape), requires_grad=True)
            qml.StronglyEntanglingLayers(weights=params, wires=range(num_qubits), imprimitive=qml.CZ)
            energy = qml.expval(qml.Hermitian(H, wires=range(num_qubits)))

            # Add penalty for overlap with previous states
            penalty = 0
            for state in prev_states:
                overlap = abs(np.vdot(params.flatten(), state)) ** 2
                penalty += overlap
            
            return energy + penalty
        
        

        # VQD
        vqd_start = datetime.now()

        #variables
        num_vqd_runs = 100
        max_iterations = 10000
        #tolerance = 1e-3
        strategy = 'best1bin'
        popsize = 15

        #data arrays
        energies = []
        prev_states = []

        x_values = []
        success = []
        run_times = []
        num_iters = []
        num_evaluations = []

        for i in range(num_vqd_runs):

            run_start = datetime.now()

            if i % 10 == 0:
                print(f"Run: {i}")

            #Optimizer
            bounds = [(0, 2 * np.pi) for _ in range(np.prod(params_shape))]

            # Differential Evolution optimization
            res = differential_evolution(lambda x: vqd_cost_function(x, prev_states), 
                                            bounds, 
                                            maxiter=max_iterations, 
                                            atol=tolerance,
                                            strategy=strategy, 
                                            popsize=popsize)
            
            if res.success == False:
                print("Not converged")

            energies.append(res.fun)
            x_values.append(res.x)
            success.append(res.success)
            num_iters.append(res.nit)
            num_evaluations.append(res.nfev)

            run_end = datetime.now()
            run_time = run_end - run_start
            run_times.append(run_time)

        vqd_end = datetime.now()
        vqd_time = vqd_end - vqd_start

        #Save run
        run = {
            'potential': potential,
            'cutoff': cut_off,
            'exact_eigenvalues': [round(x.real,10).tolist() for x in eigenvalues],
            'ansatz': 'StronglyEntanglingLayers-1layer',
            'num_VQE': num_vqd_runs,
            'Optimizer': {'name': 'differential_evolution',
                        'bounds':'[(0, 2 * np.pi) for _ in range(np.prod(params_shape))]',
                        'maxiter':max_iterations,
                        'tolerance': tolerance,
                        'strategy': strategy,
                        'popsize': popsize
                        },
            'results': energies,
            'params': [x.tolist() for x in x_values],
            'num_iters': num_iters,
            'num_evaluations': num_evaluations,
            'success': np.array(success, dtype=bool).tolist(),
            'run_times': [str(x) for x in run_times],
            'total_run_time': str(vqd_time)
        }

        # Save the variable to a JSON file
        path = r"C:\Users\Johnk\OneDrive\Desktop\PhD 2024\Quantum Computing Code\Quantum-Computing\SUSY\PennyLane\VQD\Shot Noise\Files\{}\\{}\{}_{}.json".format(potential, folder, potential, cut_off)
        with open(path, 'w') as json_file:
            json.dump(run, json_file, indent=4)


    base_path = r"C:\Users\Johnk\OneDrive\Desktop\PhD 2024\Quantum Computing Code\Quantum-Computing\SUSY\PennyLane\VQD\Shot Noise\Files\{}\\{}\\"
    create_plots(potential=potential, base_path=base_path, folder=folder, cut_off_list=cut_offs_list)

