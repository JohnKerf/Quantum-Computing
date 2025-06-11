import os
import json
import numpy as np
from datetime import datetime, timedelta
import time
from multiprocessing import Pool

from scipy.optimize import differential_evolution
from scipy.stats.qmc import Halton

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter

from susy_qm import calculate_Hamiltonian


#service = QiskitRuntimeService()
#backend = service.backend('ibm_brisbane')
#noise_model = NoiseModel.from_backend(backend)

simulator = AerSimulator()
#    noise_model=noise_model,
#    basis_gates=noise_model.basis_gates,
#    coupling_map=backend.configuration().coupling_map,
#    shots=1024
#)

estimator = Estimator(mode=simulator)
estimator.options.default_shots = 1024


def create_ansatz(params, num_qubits):
    """DW ansatz for cutoff 16 using Qiskit with correct qubit mapping."""
    def flip(q): return num_qubits - 1 - q

    qc = QuantumCircuit(num_qubits)

    #qc.x(4)
    qc.ry(params[0], 0)
    qc.ry(params[1], 1)
    qc.ry(params[2], 2)
    #qc.ry(params[3], 3)
    #qc.ry(params[4], 4)
    #qc.ry(params[0], num_qubits-1)
    #qc.ry(params[0], flip(num_qubits - 3))  # PennyLane: q2
    #qc.ry(params[1], flip(num_qubits - 1))  # PennyLane: q4
    #qc.cry(params[2], flip(num_qubits - 1), flip(num_qubits - 2))  # q4 → q3
    #qc.ry(params[3], flip(num_qubits - 2))  # PennyLane: q3
    #qc.ry(params[4], flip(num_qubits - 1))  # PennyLane: q3

    return qc


theta = [Parameter(f'theta_{i}') for i in range(num_params)]
qc_template = create_parameterized_ansatz(theta, num_qubits)
transpiled_qc = transpile(qc_template, simulator)
observable = SparsePauliOp.from_operator(Operator(H_matrix))

def cost_function(params, H_matrix, num_qubits, shots):
    start = datetime.now()

    qc = create_ansatz(params, num_qubits)

    observable = SparsePauliOp.from_operator(Operator(H_matrix))

    pm = generate_preset_pass_manager(optimization_level=1)
    isa_circuit = pm.run(qc)
    isa_observable = observable.apply_layout(isa_circuit.layout)
    energy = estimator.run([(isa_circuit, isa_observable)]).result()[0].data.evs

    end = datetime.now()
    return np.real(energy), (end - start)


def run_vqe(i, bounds, max_iter, tol, abs_tol, strategy, popsize, H_matrix, num_qubits, shots, num_params):
    seed = (os.getpid() * int(time.time())) % 123456789
    run_start = datetime.now()

    halton_sampler = Halton(d=num_params, seed=seed)
    halton_samples = halton_sampler.random(n=popsize)
    scaled_samples = 2 * np.pi * halton_samples

    device_time = timedelta()

    def wrapped_cost_function(params):
        result, dt = cost_function(params, H_matrix, num_qubits, shots)
        nonlocal device_time
        device_time += dt
        return result

    res = differential_evolution(
        wrapped_cost_function,
        bounds,
        maxiter=max_iter,
        tol=tol,
        atol=abs_tol,
        strategy=strategy,
        popsize=popsize,
        init=scaled_samples,
        seed=seed
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
    shots = 1024 # Set to None for analytic / noiseless mode
    cutoff_list = [4]  # Extend this list if needed

    for cutoff in cutoff_list:
        print(f"Running for {potential} potential and cutoff {cutoff}")
        starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        base_path = os.path.join(
            r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\SUSY QM\Qiskit\VQE\test",
            potential)
        os.makedirs(base_path, exist_ok=True)

        H_matrix = calculate_Hamiltonian(cutoff, potential)
        eigenvalues = np.sort(np.linalg.eig(H_matrix)[0])[:4]

        num_qubits = int(1 + np.log2(cutoff))
        num_params = 3
        bounds = [(0, 2 * np.pi) for _ in range(num_params)]

        num_vqe_runs = 2
        max_iter = 2000
        strategy = "randtobest1bin"
        tol = 1e-3
        abs_tol = 1e-3
        popsize = 20

        vqe_starttime = datetime.now()

        with Pool(processes=2) as pool:
            vqe_results = pool.starmap(
                run_vqe,
                [
                    (i, bounds, max_iter, tol, abs_tol, strategy, popsize, H_matrix, num_qubits, shots, num_params)
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
            "shots": shots if shots else "statevector",
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
