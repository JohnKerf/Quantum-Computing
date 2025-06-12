import os
import json
import numpy as np
from datetime import datetime, timedelta
import time
from multiprocessing import Pool

from scipy.optimize import differential_evolution
from scipy.stats.qmc import Halton

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_aer import AerSimulator
from qiskit.circuit.library import RealAmplitudes

from susy_qm import calculate_Hamiltonian


def create_parameterized_ansatz(params, num_qubits):
    """DW ansatz with parameterized gates."""
    qc = QuantumCircuit(num_qubits)
    for i in range(len(params)):
        qc.ry(params[i], i)
    return qc


def cost_function(params, transpiled_qc, theta, observable, estimator):
    start = datetime.now()
    bound_circuit = transpiled_qc.assign_parameters({theta[i]: params[i] for i in range(len(params))})
    energy = estimator.run([(bound_circuit, observable)]).result()[0].data.evs
    end = datetime.now()
    return np.real(energy), (end - start)


def run_vqe(i, bounds, max_iter, tol, abs_tol, strategy, popsize,
            H_matrix, num_qubits, shots, num_params, transpiled_qc,
            theta, observable, estimator):
    seed = (os.getpid() * int(time.time())) % 123456789
    run_start = datetime.now()

    halton_sampler = Halton(d=num_params, seed=seed)
    halton_samples = halton_sampler.random(n=popsize)
    scaled_samples = 2 * np.pi * halton_samples

    device_time = timedelta()

    def wrapped_cost_function(params):
        result, dt = cost_function(params, transpiled_qc, theta, observable, estimator)
        nonlocal device_time
        device_time += dt
        return result
    
    iteration_counter = {"iter": 0}
    def iteration_callback(xk, convergence=None):
        if iteration_counter['iter'] % 50 == 0:
            print(f"[Run {i}] Iteration {iteration_counter['iter']}")
        iteration_counter["iter"] += 1

    res = differential_evolution(
        wrapped_cost_function,
        bounds,
        maxiter=max_iter,
        tol=tol,
        atol=abs_tol,
        strategy=strategy,
        popsize=popsize,
        init=scaled_samples,
        seed=seed,
        callback=iteration_callback
    )

    run_end = datetime.now()
    run_time = run_end - run_start

    return {
        "seed": seed,
        "energy": res.fun,
        "params": res.x.tolist(),
        "success": res.success,
        "num_iters": res.nit,
        "num_evaluations": res.nfev,
        "run_time": run_time,
        "device_time": device_time
    }


if __name__ == "__main__":

    potential = "AHO"
    shots = 1024
    cutoff_list = [4]

    for cutoff in cutoff_list:
        print(f"Running for {potential} potential and cutoff {cutoff}")
        starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        base_path = os.path.join(
            r"C:\Users\johnkerf\Desktop\Quantum-Computing\Quantum-Computing\SUSY\SUSY QM\Qiskit\VQE\test",
            potential)
        os.makedirs(base_path, exist_ok=True)

        H_matrix = calculate_Hamiltonian(cutoff, potential)
        eigenvalues = np.sort(np.linalg.eig(H_matrix)[0])[:4]

        num_qubits = int(1 + np.log2(cutoff))
        num_params = num_qubits
        bounds = [(0, 2 * np.pi) for _ in range(num_params)]

        simulator = AerSimulator()
        estimator = Estimator(mode=simulator)
        estimator.options.default_shots = 1024
        
        # Prepare parameterized ansatz
        #theta = [Parameter(f'theta_{i}') for i in range(num_params)]
        #qc_template = create_parameterized_ansatz(theta, num_qubits)
        #transpiled_qc = transpile(qc_template, simulator)

        # Use Qiskit's RealAmplitudes ansatz
        reps = 1  # You can increase reps for deeper circuits
        ansatz = RealAmplitudes(num_qubits, reps=reps, entanglement='reverse_linear')
        theta = ansatz.parameters  # This is a ParameterVector
        transpiled_qc = transpile(ansatz, simulator)
        num_params = len(theta)  # Update num_params accordingly
        bounds = [(0, 2 * np.pi) for _ in range(num_params)]

        # Prepare observable
        observable = SparsePauliOp.from_operator(Operator(H_matrix))

        num_vqe_runs = 1
        max_iter = 2000
        strategy = "randtobest1bin"
        tol = 1e-3
        abs_tol = 1e-3
        popsize = 20

        vqe_starttime = datetime.now()

        with Pool(processes=1) as pool:
            vqe_results = pool.starmap(
                run_vqe,
                [
                    (i, bounds, max_iter, tol, abs_tol, strategy, popsize, H_matrix,
                     num_qubits, shots, num_params, transpiled_qc, theta, observable, estimator)
                    for i in range(num_vqe_runs)
                ],
            )

        seeds = [res["seed"] for res in vqe_results]
        energies = [res["energy"] for res in vqe_results]
        x_values = [res["params"] for res in vqe_results]
        success = [res["success"] for res in vqe_results]
        num_iters = [res["num_iters"] for res in vqe_results]
        num_evaluations = [res["num_evaluations"] for res in vqe_results]
        run_times = [str(res["run_time"]) for res in vqe_results]
        total_run_time = sum([res["run_time"] for res in vqe_results], timedelta())
        device_times = [str(res["device_time"]) for res in vqe_results]
        total_device_time = sum([res['device_time'] for res in vqe_results], timedelta())

        vqe_end = datetime.now()
        vqe_time = vqe_end - vqe_starttime

        run = {
            "starttime": starttime,
            "endtime": vqe_end.strftime("%Y-%m-%d_%H-%M-%S"),
            "potential": potential,
            "cutoff": cutoff,
            "exact_eigenvalues": [x.real.tolist() for x in eigenvalues],
            "ansatz": "Qiskit-DW-16 Estimator",
            "num_VQE": num_vqe_runs,
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
            "results": energies,
            "params": x_values,
            "num_iters": num_iters,
            "num_evaluations": num_evaluations,
            "success": np.array(success, dtype=bool).tolist(),
            "run_times": run_times,
            "device_times": device_times,
            "parallel_run_time": str(vqe_time),
            "total_VQE_time": str(total_run_time),
            "total_device_time": str(total_device_time),
            "seeds": seeds,
        }

        path = os.path.join(base_path, f"{potential}_{cutoff}.json")
        with open(path, "w") as json_file:
            json.dump(run, json_file, indent=4)

        print("Done.")
