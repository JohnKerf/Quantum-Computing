# PennyLane imports
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.optimize import AdamOptimizer

from scipy.optimize import differential_evolution

# General imports
import os
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from qiskit.quantum_info import SparsePauliOp


def create_matrix(cut_off, type):
    # Initialize a zero matrix of the specified size
    matrix = np.zeros((cut_off, cut_off), dtype=np.complex128)
    
    # Fill the off-diagonal values with square roots of integers
    for i in range(cut_off):
        if i > 0:  # Fill left off-diagonal
            if type == 'q':
                matrix[i][i - 1] = (1/np.sqrt(2)) * np.sqrt(i)  # sqrt(i) for left off-diagonal
            else:
                matrix[i][i - 1] = (1j/np.sqrt(2)) * np.sqrt(i)

        if i < cut_off - 1:  # Fill right off-diagonal
            if type == 'q':
                matrix[i][i + 1] = (1/np.sqrt(2)) * np.sqrt(i + 1)  # sqrt(i + 1) for right off-diagonal
            else:
                matrix[i][i + 1] = (-1j/np.sqrt(2)) * np.sqrt(i + 1)

    return matrix


# Function to calculate the Hamiltonian
def calculate_Hamiltonian(cut_off, potential):
    # Generate the position (q) and momentum (p) matrices
    q = create_matrix(cut_off, 'q')  # q matrix
    p = create_matrix(cut_off, 'p')  # p matrix

    # Calculate q^2 and q^3 for potential terms
    q2 = np.matmul(q, q)
    q3 = np.matmul(q2, q)

    #fermionic identity
    I_f = np.eye(2)

    #bosonic identity
    I_b = np.eye(cut_off)

    # Superpotential derivatives
    if potential == 'QHO':
        W_prime = q  # W'(q) = q
        W_double_prime = I_b #W''(q) = 1

    elif potential == 'AHO':
        W_prime = q + q3  # W'(q) = q + q^3
        W_double_prime = I_b + 3 * q2  # W''(q) = 1 + 3q^2

    elif potential == 'DW':
        W_prime = q + q2 + I_b  # W'(q) = q + q^2 + 1
        W_double_prime = I_b + 2 * q  # W''(q) = 1 + 2q

    else:
        print("Not a valid potential")
        raise

    # Kinetic term: p^2
    p2 = np.matmul(p, p)

    # Commutator term [b^†, b] = -Z
    Z = np.array([[1, 0], [0, -1]])  # Pauli Z matrix for fermion number
    commutator_term = np.kron(Z, W_double_prime)

    # Construct the block-diagonal kinetic term (bosonic and fermionic parts)
    # Bosonic part is the same for both, hence we use kron with the identity matrix
    kinetic_term = np.kron(I_f, p2)

    # Potential term (W' contribution)
    potential_term = np.kron(I_f, np.matmul(W_prime, W_prime))

    # Construct the full Hamiltonian
    H_SQM = 0.5 * (kinetic_term + potential_term + commutator_term)
    H_SQM[np.abs(H_SQM) < 10e-12] = 0
    
    return H_SQM


#potential = 'QHO'
potential = 'AHO'
#potential = 'DW'

cut_offs_list = [2,4,8]#,16,32]
cut_offs_list = [16]
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
        shots = 1024
        dev = qml.device('default.qubit', wires=num_qubits, shots=1024)
        #dev = qml.device('lightning.qubit', wires=num_qubits, shots=shots)


        #Initial params shape
        num_layers = 1
        params_shape = qml.StronglyEntanglingLayers.shape(n_layers=num_layers, n_wires=num_qubits)


        # Define the cost function
        @qml.qnode(dev)
        def cost_function(params):
            params = pnp.tensor(params.reshape(params_shape), requires_grad=True)
            qml.StronglyEntanglingLayers(weights=params, wires=range(num_qubits), imprimitive=qml.CZ)
            return qml.expval(qml.Hermitian(H, wires=range(num_qubits)))
        
        

        # VQE
        vqe_start = datetime.now()

        #variables
        num_vqe_runs = 100
        max_iterations = 10000
        tolerance = 1e-6
        strategy = 'best1bin'
        popsize = 15
        params_scale = 0.25

        #data arrays
        energies = []
        x_values = []
        success = []
        run_times = []
        num_iters = []

        for i in range(num_vqe_runs):

            #params = params_scale*np.pi * pnp.random.random(size=params_shape)

            run_start = datetime.now()

            if i % 10 == 0:
                print(f"Run: {i}")

            #Optimizer
            bounds = [(0, 2 * np.pi) for _ in range(np.prod(params_shape))]

            # Differential Evolution optimization
            res = differential_evolution(cost_function, 
                                            bounds, 
                                            maxiter=max_iterations, 
                                            tol=tolerance,
                                            strategy=strategy, 
                                            popsize=popsize)
            
            if res.success == False:
                print("Not converged")

            energies.append(res.fun)
            x_values.append(res.x)
            success.append(res.success)
            num_iters.append(res.nfev)

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
            'ansatz_scale': params_scale,
            'num_VQE': num_vqe_runs,
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

