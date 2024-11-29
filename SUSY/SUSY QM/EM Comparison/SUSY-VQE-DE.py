# PennyLane imports
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import differential_evolution
import os
import json
import numpy as np
from datetime import datetime
from qiskit.quantum_info import SparsePauliOp
from multiprocessing import Pool
from susy_qm import calculate_Hamiltonian, create_vqe_plots


def cost_function(params, H, params_shape, num_qubits, shots):
    """
    Cost function to evaluate the energy expectation value.
    Each worker creates its own device instance.
    """
    dev = qml.device("lightning.qubit", wires=num_qubits, shots=shots)

    @qml.qnode(dev)
    def circuit(params):
        params = pnp.tensor(params.reshape(params_shape), requires_grad=True)
        qml.StronglyEntanglingLayers(weights=params, wires=range(num_qubits), imprimitive=qml.CZ)
        return qml.expval(qml.Hermitian(H, wires=range(num_qubits)))

    return circuit(params)


def run_vqe(i, bounds, tolerance, strategy, popsize, H, params_shape, num_qubits, shots):
    """
    Run a single VQE optimization.
    Each worker creates its own device.
    """
    run_start = datetime.now()

    print(f"Run: {i}")

    # Differential Evolution optimization
    res = differential_evolution(
        lambda params: cost_function(params, H, params_shape, num_qubits, shots),
        bounds,
        maxiter=500,
        atol=tolerance,
        strategy=strategy,
        popsize=popsize,
    )

    run_end = datetime.now()
    run_time = run_end - run_start

    return {
        "energy": res.fun,
        "params": res.x.tolist(),
        "success": res.success,
        "num_iters": res.nit,
        "num_evaluations": res.nfev,
        "run_time": str(run_time),
    }


if __name__ == "__main__":
    # Configuration
    potential = "QHO"
    cut_offs_list = [2]
    tol_list = [1e-1]
    shots = 1024

    for tolerance in tol_list:
        starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = str(tolerance)
        base_path = r"C:\Users\Johnk\OneDrive\Desktop\PhD 2024\Quantum Computing Code\Quantum-Computing\SUSY\SUSY QM\EM Comparison\Files\{}\\{}\\".format(potential, folder)
        os.makedirs(base_path, exist_ok=True)

        print(f"Running for {potential} potential")
        print(f"Running for tolerance: {str(tolerance)}")

        for cut_off in cut_offs_list:
            print(f"Running for cutoff: {cut_off}")

            # Calculate Hamiltonian and expected eigenvalues
            H = calculate_Hamiltonian(cut_off, potential)
            eigenvalues = np.sort(np.linalg.eig(H)[0])
            min_eigenvalue = min(eigenvalues.real)

            # Create qiskit Hamiltonian Pauli string
            hamiltonian = SparsePauliOp.from_operator(H)
            num_qubits = hamiltonian.num_qubits

            # Initial params shape
            num_layers = 1
            params_shape = qml.StronglyEntanglingLayers.shape(n_layers=num_layers, n_wires=num_qubits)

            # Optimizer
            bounds = [(0, 2 * np.pi) for _ in range(np.prod(params_shape))]
            num_vqe_runs = 100
            strategy = "best1bin"
            popsize = 15

            # Start multiprocessing for VQE runs
            with Pool(processes=4) as pool:
                vqe_results = pool.starmap(
                    run_vqe,
                    [
                        (i, bounds, tolerance, strategy, popsize, H, params_shape, num_qubits, shots)
                        for i in range(num_vqe_runs)
                    ],
                )

            # Collect results
            energies = [res["energy"] for res in vqe_results]
            x_values = [res["params"] for res in vqe_results]
            success = [res["success"] for res in vqe_results]
            num_iters = [res["num_iters"] for res in vqe_results]
            num_evaluations = [res["num_evaluations"] for res in vqe_results]
            run_times = [res["run_time"] for res in vqe_results]

            vqe_end = datetime.now()
            vqe_time = vqe_end - datetime.strptime(starttime, "%Y-%m-%d_%H-%M-%S")

            # Save run
            run = {
                "potential": potential,
                "cutoff": cut_off,
                "exact_eigenvalues": [round(x.real, 10).tolist() for x in eigenvalues],
                "ansatz": "StronglyEntanglingLayers-1layer",
                "num_VQE": num_vqe_runs,
                "shots": shots,
                "Optimizer": {
                    "name": "differential_evolution",
                    "bounds": "[(0, 2 * np.pi) for _ in range(np.prod(params_shape))]",
                    "maxiter": 500,
                    "tolerance": tolerance,
                    "strategy": strategy,
                    "popsize": popsize,
                },
                "results": energies,
                "params": x_values,
                "num_iters": num_iters,
                "num_evaluations": num_evaluations,
                "success": np.array(success, dtype=bool).tolist(),
                "run_times": run_times,
                "total_run_time": str(vqe_time),
            }

            # Save the variable to a JSON file
            path = base_path + "{}_{}.json".format(potential, cut_off)
            with open(path, "w") as json_file:
                json.dump(run, json_file, indent=4)
