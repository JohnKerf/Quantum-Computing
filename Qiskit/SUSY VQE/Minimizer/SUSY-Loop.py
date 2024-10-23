# General imports
import numpy as np
import json
import os
from datetime import datetime

# Pre-defined ansatz circuit and operator class for Hamiltonian
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp

# SciPy minimizer routine
from scipy.optimize import minimize

# Plotting functions
import matplotlib.pyplot as plt

# runtime imports
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

aer_sim = AerSimulator(method='statevector')
pm = generate_preset_pass_manager(backend=aer_sim, optimization_level=3)

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
    q2 = np.dot(q, q)
    q3 = np.dot(q2, q)

    #fermionic identity
    I_f = np.eye(2)

    #bosonic identity
    I_b = np.eye(cut_off)

    # Superpotential derivatives
    if potential == 'QHO':
        W_prime = q  # W'(q) = q
        W_double_prime = I_b

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
    p2 = np.dot(p, p)

    # Commutator term [b^†, b] = -Z
    Z = np.array([[1, 0], [0, -1]])  # Pauli Z matrix for fermion number
    commutator_term = np.kron(Z, W_double_prime)

    # Construct the block-diagonal kinetic term (bosonic and fermionic parts)
    # Bosonic part is the same for both, hence we use kron with the identity matrix
    kinetic_term = np.kron(I_f, p2)

    # Potential term (W' contribution)
    potential_term = np.kron(I_f, np.dot(W_prime, W_prime))

    # Construct the full Hamiltonian
    H_SQM = 0.5 * (kinetic_term + potential_term + commutator_term)
    H_SQM[np.abs(H_SQM) < 10e-10] = 0
    
    return H_SQM


#potential = 'QHO'
#potential = 'AHO'
potential = 'DW'

cut_offs_list = [2,4,8,16]#,32]

starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#Create directory for files
os.makedirs(r"C:\Users\Johnk\OneDrive\Desktop\PhD 2024\Quantum Computing Code\Quantum-Computing\Qiskit\SUSY VQE\Minimizer\Files\{}\\{}".format(potential, str(starttime)))

print(f"Running for {potential} potential")

for cut_off in cut_offs_list:

    print(f"Running for cutoff: {cut_off}")

    #calculate Hamiltonian and expected eigenvalues
    H = calculate_Hamiltonian(cut_off, potential)
    eigenvalues = np.sort(np.linalg.eig(H)[0])
    min_eigenvalue = min(eigenvalues.real)

    #create qiskit Hamiltonian Pauli string
    hamiltonian = SparsePauliOp.from_operator(H)
    ansatz = RealAmplitudes(num_qubits=hamiltonian.num_qubits, reps=1)
    ansatz_isa = pm.run(ansatz)
    hamiltonian_isa = hamiltonian.apply_layout(layout=ansatz_isa.layout)


    # Create cost function
    cost_history_dict = {
    "prev_vector": None,
    "iters": 0,
    "cost_history": [],
    }


    def cost_func(params, ansatz, hamiltonian, estimator):
        """Return estimate of energy from estimator

        Parameters:
            params (ndarray): Array of ansatz parameters
            ansatz (QuantumCircuit): Parameterized ansatz circuit
            hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
            estimator (EstimatorV2): Estimator primitive instance
            cost_history_dict: Dictionary for storing intermediate results

        Returns:
            float: Energy estimate
        """
        pub = (ansatz, [hamiltonian], [params])
        #result = estimator.run(pubs=[pub], precision=0.001).result()
        result = estimator.run(pubs=[pub]).result()
        energy = result[0].data.evs[0]

        cost_history_dict["iters"] += 1
        cost_history_dict["prev_vector"] = params
        cost_history_dict["cost_history"].append(energy)

        return energy
    
    #define initial guess
    num_params = ansatz.num_parameters
    x0 = 2 * np.pi * np.random.random(num_params)

    # VQE
    num_vqe_runs = 100
    max_iterations = 10000
    energies = []
    x_values = []

    backend=aer_sim
    estimator = Estimator(mode=backend)

    for i in range(num_vqe_runs):

        if i % 10 == 0:
            print(f"Run: {i}")

        res = minimize(
            cost_func,
            x0,
            method= "COBYLA",
            args= (ansatz_isa, hamiltonian_isa, estimator),
            options= {'maxiter':max_iterations}
        )
        energies.append(res.fun)
        x_values.append(res.x)

        closest_e = min(enumerate(energies), key=lambda x: abs(x[1] - min_eigenvalue))[0]
        closest_x = x_values[closest_e]
        x0 = closest_x

    #Save run
    run = {
        'potential': potential,
        'cutoff': cut_off,
        'exact_eigenvalues': [round(x.real,10).tolist() for x in eigenvalues],
        'ansatz': 'RealAmplitudes',
        'num_VQE': num_vqe_runs,
        'backend': 'aer_simulator',
        'min_function': {'name': 'minimizer',
                        'method': "COBYLA",
                        'maxiter':max_iterations
                        },
        'results': energies,
        'x_values': [x.tolist() for x in x_values]
    }

    # Save the variable to a JSON file
    path = r"C:\Users\Johnk\OneDrive\Desktop\PhD 2024\Quantum Computing Code\Quantum-Computing\Qiskit\SUSY VQE\Minimizer\Files\{}\\{}\{}_{}.json".format(potential, str(starttime), potential, cut_off)
    with open(path, 'w') as json_file:
        json.dump(run, json_file, indent=4)


# Load all data and create graphs
print("Creating plots")
data_dict = {}
base_path = r"C:\Users\Johnk\OneDrive\Desktop\PhD 2024\Quantum Computing Code\Quantum-Computing\Qiskit\SUSY VQE\Minimizer\Files\{}\\{}\{}_{}.json"

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
plt.savefig(r"C:\Users\Johnk\OneDrive\Desktop\PhD 2024\Quantum Computing Code\Quantum-Computing\Qiskit\SUSY VQE\Minimizer\Files\{}\\{}\results.png".format(potential, str(starttime)))

print("Done")

