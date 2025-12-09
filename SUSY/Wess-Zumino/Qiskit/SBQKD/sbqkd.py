import os, json, logging, time, dataclasses
import numpy as np
from collections import Counter
from datetime import datetime

from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import SamplerV2 as Sampler, QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from scipy.sparse.linalg import eigsh

import wesszumino as wz

import git
repo_path = git.Repo('.', search_parent_directories=True).working_tree_dir

path = os.path.join(repo_path, r"open-apikey.json")
with open(path, encoding="utf-8") as f:
    api_key = json.load(f).get("apikey")

IBM_QUANTUM_API_KEY = api_key
ibm_instance_crn = "crn:v1:bluemix:public:quantum-computing:us-east:a/3ff62345f67c45e48e47a7f57d2f39f5:83214c75-88ab-4e55-8a87-6502ecc7cc9b::" #US
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=IBM_QUANTUM_API_KEY, instance=ibm_instance_crn)


def setup_logger(logfile_path, name, enabled=True):
    if not enabled:
        
        logger = logging.getLogger(f"{name}_disabled")
        logger.handlers = []               
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        logger.setLevel(logging.CRITICAL)
        return logger

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(logfile_path)
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def get_backend(backend_name, use_noise_model, noise_model_options, resilience_level, seed, shots, tags):

    noise_model = None

    if backend_name == "Aer":
        if use_noise_model:
            #real_backend = service.backend("ibm_kingston")
            real_backend = service.backend("ibm_torino")
            noise_model = NoiseModel.from_backend(
                real_backend,
                gate_error=noise_model_options["gate_error"],
                readout_error=noise_model_options["readout_error"],   
                thermal_relaxation=noise_model_options["thermal_relaxation"],
            )
            backend = AerSimulator(noise_model=noise_model)

            if log_enabled: logger.info(noise_model.noise_instructions)


        else:
            backend = AerSimulator(method="statevector")
    else:
        backend = service.backend(backend_name)


    if backend_name == "Aer":

        sampler = AerSampler(
            options={
                "backend_options": {
                    "method": "automatic",
                    "noise_model": noise_model if use_noise_model else None,
                    "seed_simulator": seed
                },
                "run_options": {
                    "shots": shots
                }
            }
        )
    else:
        sampler = Sampler(mode=backend)
        sampler.options.environment.job_tags = tags
        sampler.options.default_shots = shots
        sampler.options.resilience_level = resilience_level

    return backend, sampler


def create_circuit(backend, optimization_level, num_qubits, basis, H_pauli, t, num_trotter_steps):
    
    qr = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qr)

    for q, bit in enumerate(basis):
        if bit == 1:
            qc.x(q)        
    
    evol_gate = PauliEvolutionGate(H_pauli,time=t,synthesis=LieTrotter(reps=num_trotter_steps))
    qc.append(evol_gate, qr)
    qc.measure_all()


    target = backend.target
    pm = generate_preset_pass_manager(target=target, optimization_level=optimization_level)
    circuit_isa = pm.run(qc)

    return circuit_isa


def get_counts(sampler, qc, shots):
    
    pubs = [(qc)]
    job = sampler.run(pubs, shots=shots)

    job_id = job.job_id()

    counts = job.result()[0].data.meas.get_counts()

    try:
        job_metrics = job.metrics()
    except AttributeError:
        #logger.info("No usage/metrics available for this estimator type.")
        job_metrics = None

    return counts, job_id, job_metrics



if __name__ == "__main__":

    starttime = datetime.now()

    log_enabled = True
    seed = (os.getpid() * int(time.time())) % 123456789

    N = 3
    a = 1.0
    c = -0.2

    potential = "linear"
    #potential = 'quadratic'
    boundary_condition = 'dirichlet'
    #boundary_condition = 'periodic'

    cutoff = 8

    #backend_name = 'ibm_kingston'
    #backend_name = 'ibm_torino'
    backend_name = "Aer"

    # Noise model options
    use_noise_model = 0
    gate_error=False
    readout_error=False  
    thermal_relaxation=False

    noise_model_options = {
        "gate_error":gate_error,
        "readout_error":readout_error,   
        "thermal_relaxation":thermal_relaxation
        }

    shots = 2000
    optimization_level = 3
    resilience_level = 0

    #for shots in [200,500,1000,2000,4000,10000]:

    k=1
    n_steps=1
    dt=1.0
    max_k = 10
    tol = 1e-10

    tags=["Open-access", "SBQKD", f"shots:{shots}", f"{boundary_condition}", f"{potential}", f"N={N}", f"cutoff={cutoff}"]


    if potential == 'quadratic':
        folder = 'C' + str(abs(c)) + '/' + 'N'+ str(N)
    else:
        folder = 'N'+ str(N)

    base_path = os.path.join(repo_path, r"SUSY\Wess-Zumino\Qiskit\SBQKD\Files", backend_name, boundary_condition, potential, folder, starttime.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(base_path, exist_ok=True)
    log_path = os.path.join(base_path, f"logs_{str(cutoff)}")

    if log_enabled: 
        os.makedirs(log_path, exist_ok=True)
        log_path = os.path.join(log_path, f"vqe_run.log")
        logger = setup_logger(log_path, f"logger", enabled=log_enabled)

    if log_enabled: logger.info(f"Running VQE for {potential} potential and cutoff {cutoff}")

    H_path = os.path.join(repo_path, r"SUSY\Wess-Zumino\PennyLane\Analyses\Model Checks\HamiltonianData4", boundary_condition, potential, folder, f"{potential}_{cutoff}.json")
    with open(H_path, 'r') as file:
        H_data = json.load(file)

    pauli_coeffs = H_data['pauli_coeffs']
    pauli_strings = H_data['pauli_terms']
    pauli_terms = [wz.pauli_str_to_op(t) for t in pauli_strings]

    num_qubits = H_data['num_qubits']

    dense_H_size = H_data['H_size']

    eigenvalues = H_data['eigenvalues']
    min_eigenvalue = np.min(eigenvalues)

    if log_enabled: logger.info(f"min_eigenvalue: {min_eigenvalue}")

    H_pauli = wz.pl_to_qk_hamiltonian(pauli_terms, pauli_coeffs, num_qubits)
    pauli_terms = []
    for label, coeff in H_pauli.to_list():
        pauli_terms.append((coeff, label)) 

    nb = int(np.log2(cutoff))
    n = 1 + nb

    #Dirichlet-Linear
    #basis = [0]*n + [1] + [0]*nb #N2
    basis = [0]*n + [1] + [0]*nb + [0]*n #N3
    #basis = [0]*n + [1] + [0]*nb + [0]*n + [1] + [0]*nb #N4
    #basis = [0]*n + [1] + [0]*nb + [0]*n + [1] + [0]*nb + [0]*n #N5

    converged=False
    samples = Counter()
    prev_energy = np.inf

    all_data = []
    all_energies = []
    job_info = {}

    backend, sampler = get_backend(backend_name, use_noise_model, noise_model_options, resilience_level, seed, shots, tags)

    sampler_options = dataclasses.asdict(sampler.options)
    if log_enabled: logger.info(json.dumps(sampler_options, indent=4, default=str))

    while not converged and k <= max_k:

        if log_enabled: logger.info(f"Running for Krylov dimension {k}")
        print(f"Running for Krylov dimension {k}")

        t = dt*k

        qc = create_circuit(backend, optimization_level, num_qubits, basis, H_pauli, t, n_steps)

        t1 = datetime.now()
        counts, job_id, job_metrics  = get_counts(sampler, qc, shots)
        Ct = datetime.now() - t1

        job_info[job_id] = job_metrics
        samples.update(counts)


        sorted_states = sorted(samples.items(), key=lambda x: x[1], reverse=True)
        top_states = [s for s, c in sorted_states]
        idx = [int(s, 2) for s in top_states]

        H_reduced = wz.reduced_sparse_matrix_from_pauli_terms(pauli_terms, top_states)
    
        t1 = datetime.now()
        es = eigsh(H_reduced, k=1, which="SA", return_eigenvectors=False)
        HRt = datetime.now() - t1

        mi = np.argmin(es)
        me = es[mi].real

        diff_prev = np.abs(prev_energy-me)

        row = { "D": k,
                "t":t,
                "circuit_time": str(Ct),
                "num_samples": len(samples),
                "H_reduced_size": H_reduced.shape,
                "reduction": (1 - (H_reduced.shape[0] / dense_H_size[0]))*100,
                "H_reduced_e": me,
                "eigenvalue_time": str(HRt),
                "diff": np.abs(min_eigenvalue-me),
                "change_from_prev": None if diff_prev == np.inf else diff_prev
                }
        
        if log_enabled: logger.info(json.dumps(row, indent=4, default=str))
        
        all_data.append(row)
        all_energies.append(me)

        converged = True if diff_prev < tol else False

        if converged == False and k == max_k: 
            if log_enabled: logger.info("max_k reached")
            print("max_k reached")
            break
        elif converged == False:
            prev_energy = me
            k+=1
        else:
            if log_enabled: logger.info(f"Converged")
            print("Converged")

    endtime = datetime.now()

    final_data = {
        "starttime": starttime.strftime("%Y-%m-%d_%H-%M-%S"),
        "endtime": endtime.strftime("%Y-%m-%d_%H-%M-%S"),
        "time_taken": str(endtime-starttime),
        "backend": backend_name,
        "use_noise_model": use_noise_model,
        "noise_model_options": noise_model_options if use_noise_model else None,
        "optimization_level": optimization_level,
        "resilience_level": resilience_level,
        "shots": shots,
        "tags": tags,
        "potential": potential,
        "boundary_condition": boundary_condition,
        "cutoff": cutoff,
        "N": N,
        "a": a,
        "c": None if potential == "linear" else c,
        "num_qubits": num_qubits,
        "num_paulis": len(pauli_strings),
        "dense_H_size": dense_H_size,
        "eigenvalues": eigenvalues,
        "basis": basis,
        "tol":tol,
        "dt":dt,
        "n_trotter_steps":n_steps,
        "max_k":max_k,
        "final_k": k,
        "converged": converged,
        "all_energies": all_energies,
        "all_run_data": all_data,
        "job_info": job_info,
        "sampler_options": sampler_options
    }


    with open(os.path.join(base_path, f"{potential}_{cutoff}.json"), "w") as json_file:
        json.dump(final_data, json_file, indent=4, default=str)

    if log_enabled: logger.info("Done")
    print("Done")
    
        
