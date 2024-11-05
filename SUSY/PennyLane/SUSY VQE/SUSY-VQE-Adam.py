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
from susy_qm import calculate_Hamiltonian


#potential = 'QHO'
potential = 'AHO'
#potential = 'DW'

cut_offs_list = [2,4,8]#,16,32]
tol_list = [1e-2, 1e-4, 1e-6, 1e-8]
tol_list = [1e-6]

for tolerance in tol_list:

    starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #Create directory for files
    os.makedirs(r"C:\Users\Johnk\OneDrive\Desktop\PhD 2024\Quantum Computing Code\Quantum-Computing\PennyLane\SUSY VQE\Adam\Files\{}\\{}".format(potential, str(starttime)))

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
        #dev = qml.device('default.qubit', wires=num_qubits, shots=100)
        dev = qml.device('lightning.qubit', wires=num_qubits, shots=100)

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
        path = r"C:\Users\Johnk\OneDrive\Desktop\PhD 2024\Quantum Computing Code\Quantum-Computing\PennyLane\SUSY VQE\Adam\Files\{}\\{}\{}_{}.json".format(potential, str(starttime), potential, cut_off)
        with open(path, 'w') as json_file:
            json.dump(run, json_file, indent=4)


    # Load all data and create graphs
    print("Creating plots")
    data_dict = {}
    base_path = r"C:\Users\Johnk\OneDrive\Desktop\PhD 2024\Quantum Computing Code\Quantum-Computing\PennyLane\SUSY VQE\Adam\Files\{}\\{}\{}_{}.json"

    for n in cut_offs_list:
        file_path = base_path.format(potential, str(starttime), potential, n)
        with open(file_path, 'r') as json_file:
            data_dict[f'c{n}'] = json.load(json_file)


    #Create and save plots
    num_cutoffs = len(cut_offs_list)
    nrows = int(np.ceil(num_cutoffs/2))
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(30, 5*nrows))
    axes = axes.flatten()

    for idx, (cutoff, cutoff_data) in enumerate(data_dict.items()):
        
        results = cutoff_data['results']
        x_values = range(len(results))

        # Calculating statistics
        mean_value = np.mean(results)
        median_value = np.median(results)
        min_value = np.min(results)

        # Creating the plot
        ax = axes[idx]
        ax.plot(x_values, results, marker='o', label='Energy Results')

        # Plot mean, median, and min lines
        ax.axhline(y=mean_value, color='r', linestyle='--', label=f'Mean = {mean_value:.6f}')
        ax.axhline(y=median_value, color='g', linestyle='-', label=f'Median = {median_value:.6f}')
        ax.axhline(y=min_value, color='b', linestyle='-.', label=f'Min = {min_value:.6f}')

        ax.set_ylim(min_value - 0.01, max(results) + 0.01)
        ax.set_xlabel('Run')
        ax.set_ylabel('Ground State Energy')
        ax.set_title(f"{potential}: Cutoff = {cutoff_data['cutoff']}")
        ax.legend()
        ax.grid(True)

    # Hide any remaining unused axes
    for idx in range(num_cutoffs, len(axes)):
        fig.delaxes(axes[idx])

    print("Saving plots")
    plt.tight_layout()
    plt.savefig(r"C:\Users\Johnk\OneDrive\Desktop\PhD 2024\Quantum Computing Code\Quantum-Computing\PennyLane\SUSY VQE\Adam\Files\{}\\{}\results.png".format(potential, str(starttime)))

    print("Done")

