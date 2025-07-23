import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from susy_qm import calculate_Hamiltonian
import matplotlib.pyplot as plt
import seaborn as sns

from multiprocessing import Pool

from datetime import datetime

num_energy_samples = 10000

cutoff = 16
potential = 'DW'

# Calculate Hamiltonian and expected eigenvalues
H = calculate_Hamiltonian(cutoff, potential)
eigenvalues, eigenvectors = np.linalg.eig(H)
min_3_ev = eigenvalues.argsort()[:4]
min_eigenvector = np.asarray(eigenvectors[:, min_3_ev[0]])

num_qubits = int(1 + np.log2(cutoff))

H_decomp = qml.pauli_decompose(H, wire_order=range(num_qubits))
coeffs = np.array(H_decomp.coeffs)

shots_list = [2, 4, 8, 32, 128, 512, 1024, 2048, 4096, 10000]


if __name__ == "__main__":

    start = datetime.now()
    all_energies = []

    for shots in shots_list:

        print(f"Running for shots: {shots}")

        dev = qml.device("default.qubit", wires=num_qubits, shots=shots, seed=42)
        @qml.qnode(dev)
        def circuit(params):
            param_index = 0
            for i in range(num_qubits):
                qml.RY(params[param_index], wires=i)
                param_index += 1

            for j in reversed(range(1, num_qubits)):
                qml.CNOT(wires=[j, j - 1])

            for k in range(num_qubits):
                qml.RY(params[param_index], wires=k)
                param_index += 1

            return qml.expval(qml.Hermitian(H, wires=range(num_qubits)))


        def eval_sample_energy(i):

            rng = np.random.default_rng(seed=i)
            params = rng.random(2 * num_qubits) * 2 * np.pi
            energy = circuit(params)        

            return energy
        
        with Pool(processes=100) as pool:
            energies = pool.map(eval_sample_energy, range(num_energy_samples))  

        #energies = [e for chunk in energy_chunks for e in chunk]
        print(f"Collected {len(energies)} energy samples (should be {num_energy_samples})")

        all_energies.append(energies)

        plt.figure()
        sns.histplot(energies, kde=True)
        plt.axvline(x=0.89159936, color='red', linestyle='--', linewidth=1.0)
        plt.title(f"Histogram of Sampled Energies (shots={shots})")
        plt.savefig(
            f"/users/johnkerf/Quantum Computing/SUSY-QM/EnergySampling/ExpvalSampling/Histograms/Histo_{shots}.png"
        )

    # Plot KDEs
    plt.figure()
    for i, e in enumerate(all_energies):
        sns.kdeplot(e, label=f"{shots_list[i]}")

    plt.legend()
    plt.xlabel("Energy")
    plt.ylabel("Density")
    plt.title("KDEs of Multiple Energy Distributions (Sampled)")
    plt.savefig(
        "/users/johnkerf/Quantum Computing/SUSY-QM/EnergySampling/ExpvalSampling/Histograms/probs.png"
    )

    end = datetime.now()
    time_taken = end-start

    print("Done")
    print(f"Time taken = {str(time_taken)}")
