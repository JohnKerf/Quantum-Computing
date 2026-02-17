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
from qiskit.quantum_info import PauliList, SparsePauliOp
from scipy.sparse.linalg import eigsh

import wesszumino as wz

path = os.path.join( r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\open-apikey.json")
#path = r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\apikey.json"
with open(path, encoding="utf-8") as f:
    api_key = json.load(f).get("apikey")

IBM_QUANTUM_API_KEY = api_key
ibm_instance_crn = "crn:v1:bluemix:public:quantum-computing:us-east:a/3ff62345f67c45e48e47a7f57d2f39f5:83214c75-88ab-4e55-8a87-6502ecc7cc9b::" #Open
#ibm_instance_crn = "crn:v1:bluemix:public:quantum-computing:us-east:a/d4f95db0515b47b7ba61dba8a424f873:ed0704ac-ad7d-4366-9bcc-4217fb64abd1::" #NQCC

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
        if resilience_level == 1:
            sampler.options.twirling.enable_measure = True
            sampler.options.twirling.enable_gates = False
        elif resilience_level == 2:
            sampler.options.twirling.enable_measure = True
            sampler.options.twirling.enable_gates = True
            sampler.options.dynamical_decoupling.enable = True
        else:
            sampler.options.twirling.enable_measure = False
            sampler.options.twirling.enable_gates = False

    return backend, sampler

def create_circuit(backend, avqe_circuit, basis_state, optimization_level, num_qubits, H_pauli, t, num_trotter_steps):
    
    qr = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qr)

    #qc.append(avqe_circuit, qr)
    for q, bit in enumerate(basis_state):
        if bit == 1:
            qc.x(q)

    evol_gate = PauliEvolutionGate(H_pauli,time=t,synthesis=LieTrotter(reps=num_trotter_steps))
    qc.append(evol_gate, qr)
    qc.measure_all()

    target = backend.target
    pm = generate_preset_pass_manager(target=target, optimization_level=optimization_level)
    try:
        circuit_isa = pm.run(qc)
    except:
        pm = generate_preset_pass_manager(target=target, optimization_level=optimization_level, translation_method="synthesis")
        circuit_isa = pm.run(qc)

    
    return circuit_isa, circuit_cost_metrics(circuit_isa)

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

def filter_counts_by_fermion_number(counts,fermion_qubits,num_fermions):

    kept = {}
    rejected = 0
    for key, c in counts.items():
        b = [int(ch) for ch in key[::-1]]
        w = sum(b[i] for i in fermion_qubits)

        if w == num_fermions:
            kept[key] = c
        else:
            rejected += c

    return kept, rejected

def _twoq_only_depth(qc):
    """Depth counting only 2-qubit operations (ignoring barriers/measure)."""
    qc2 = QuantumCircuit(qc.num_qubits, qc.num_clbits)
    q_index = {q: i for i, q in enumerate(qc.qubits)}
    c_index = {c: i for i, c in enumerate(qc.clbits)}

    for inst in qc.data:
        op = inst.operation
        if op.name in {"barrier", "measure"}:
            continue
        if op.num_qubits == 2:
            qargs = [qc2.qubits[q_index[q]] for q in inst.qubits]
            cargs = [qc2.clbits[c_index[c]] for c in inst.clbits] if inst.clbits else []
            qc2.append(op, qargs, cargs)

    return qc2.depth()

def circuit_cost_metrics(qc):
    ops = qc.count_ops()
    ops_str = {str(k): int(v) for k, v in ops.items()}

    n2q = sum(
        1
        for inst in qc.data
        if inst.operation.num_qubits == 2 and inst.operation.name not in {"barrier", "measure"}
    )

    return {
        "depth": qc.depth(),
        "size": qc.size(),
        "num_2q_ops": n2q,
        "depth_2q": _twoq_only_depth(qc),
        "count_ops": ops_str,
    }

def truncate_by_coeff_weight(pauli_coeffs, pauli_labels, keep_ratio=0.999, min_keep=0):

    c = np.asarray(pauli_coeffs)
    lab = np.asarray(pauli_labels)

    abs_c = np.abs(c)
    order = np.argsort(abs_c)[::-1]
    abs_sorted = abs_c[order]
    w = abs_sorted**2 #squared gives more aggressive truncation

    cum = np.cumsum(w)
    total = float(cum[-1])
    target = keep_ratio * total

    m = int(np.searchsorted(cum, target, side="left") + 1)
    m = max(m, int(min_keep))
    m = min(m, len(c))

    keep_idx = order[:m]
    truncated = float(total - cum[m-1])

    info = {
        "m": m,
        "n": len(pauli_coeffs),
        "keep_frac_terms": m / len(pauli_coeffs),
        "keep_ratio": keep_ratio,
        "truncated": truncated,
        "total_weight": total
    }
    return c[keep_idx], lab[keep_idx], keep_idx, info

def run_skqd(run_idx, H_pauli, H_info, Ansatz_info, pauli_terms, avqe_circuit, backend_info, run_info, base_path, log_enabled, logger):

    starttime = datetime.now()

    dt = run_info["dt"]
    max_k = run_info["max_k"]
    tol = run_info["tol"]
    conserve_fermion = run_info["conserve_fermion"]
    fermion_qubits = run_info["fermion_qubits"]
    num_fermions = run_info["num_fermions"]
    trotter_patience = run_info["trotter_patience"]
    energy_patience = run_info["energy_patience"]
    max_n_steps = run_info["max_n_steps"]
                    
    shots = backend_info["shots"]
    backend_name = backend_info["backend_name"]
    num_qubits = H_info["num_qubits"]
    dense_H_size = H_info["dense_H_size"]
    min_eigenvalue = np.min(H_info["eigenvalues"])

    seed = (os.getpid() * int(time.time())) % 123456789 - run_idx

    backend, sampler = get_backend(backend_info["backend_name"], backend_info["use_noise_model"], backend_info["noise_model_options"], backend_info["resilience_level"], seed, shots, backend_info["tags"])
    sampler_options = dataclasses.asdict(sampler.options)

    k=1
    n_steps = 1

    pre_samples = 0

    patience_count = 0  
    trotter_patience_count = 0         
    prev_energy = None           
    converged = False

    samples = Counter()

    all_data = []
    all_counts = []
    all_energies = []
    job_info = {}

    while not converged and k <= max_k:

        if log_enabled: logger.info(f"[Run {run_idx}] Running for Krylov dimension {k}")
        print(f"Running for Krylov dimension {k}")

        t = (dt*k) / n_steps
        
        if backend_name != "Aer": 
            print("Transpiling...")
            if log_enabled: logger.info("Transpiling...")
        qc, circuit_cost = create_circuit(backend, avqe_circuit, H_info["basis"], backend_info["optimization_level"], num_qubits, H_pauli, t, n_steps)
        if backend_name != "Aer": 
            print("Finished transpiling")
            if log_enabled: logger.info("Finished transpiling")


        t1 = datetime.now()
        if backend_name != "Aer": 
            print("Getting counts...")
            if log_enabled: logger.info("Getting counts...")
        counts, job_id, job_metrics = get_counts(sampler, qc, shots) #counts are returned in binary notation i.e. q0q1...qn and not standard qiskit noation
        if backend_name != "Aer": 
            print(f"Received counts")
            if log_enabled: logger.info("Received counts")
        Ct = datetime.now() - t1

        if backend_name != "Aer":
            job_info[job_id] = job_metrics

            jobs = job_info.values()
            QPU_usage = 0.0
            for job in jobs:
                usage = job.get("usage")
                QPU_usage += float(usage["seconds"])

            if log_enabled: logger.info(f"[Run {run_idx}] Job ID: {job_id} - QPU usage: {usage} - Total usage {QPU_usage}")

        # trim per Krylov step
        raw_counts = counts
        raw_unique = len(raw_counts)

        fermion_rejected = 0

        if conserve_fermion:
            counts, rej_w = filter_counts_by_fermion_number(counts, fermion_qubits=fermion_qubits, num_fermions=num_fermions)
            fermion_rejected += rej_w
        else:
            counts = raw_counts

        kept_unique = len(counts)
        kept_shots = sum(counts.values())

        shot_processing = {
            "raw_unique_basis_states": raw_unique, # Unique shots found per krylov step
            "fermion_basis_rejected": raw_unique - kept_unique, # Num basis states rejected due to fermion number conservation
            "kept_unique_basis_states": kept_unique, # Num unique - fermion_rejected
            "fermion_shots_rejected": fermion_rejected, # Num shots rejected due to fermion number conservation
            "kept_shots": kept_shots, # Total number of shots spanning the kept_unique basis states
        }

        if log_enabled: logger.info("[Run %d]\n%s",run_idx,json.dumps(shot_processing, indent=4, default=str))

        # Update global samples with the trimmed counts
        pre_samples = len(samples)
        samples.update(counts)
        
        sorted_states = sorted(samples.items(), key=lambda x: x[1], reverse=True)
        top_states = [s for s, c in sorted_states]
        all_counts.append(dict(sorted_states)) 

        if backend_name != "Aer": 
            print("Reducing matrix")
            if log_enabled: logger.info("Reducing matrix")
        H_reduced = wz.reduced_sparse_matrix_from_pauli_terms(pauli_terms, top_states)
        
        t1 = datetime.now()

        if backend_name != "Aer": 
            print("Finding eigenvalues")
            if log_enabled: logger.info("Finding eigenvalues")
        if H_reduced.shape[0] < 2000:
            H_dense = H_reduced.todense()
            me = np.min(np.linalg.eigvals(H_dense)).real
            used_dense = True
        else:
            me = eigsh(H_reduced, k=1, which="SA", return_eigenvectors=False)[0].real
            used_dense = False
        HRt = datetime.now() - t1

        if prev_energy is None:
            diff_prev = None
        else:
            diff_prev = float(np.abs(prev_energy - me))
            
        prev_energy = me

        row = { 
            "D": k,
            "t":t,
            "num_trotter_steps": n_steps,
            "circuit_time": str(Ct),
            "shot_processing": shot_processing,
            "new_basis_count": len(samples) - pre_samples,
            "total_basis_count": len(samples),
            "H_reduced_size": H_reduced.shape,
            "reduction": (1 - (H_reduced.shape[0] / dense_H_size[0]))*100,
            "H_reduced_e": me,
            "used_dense": used_dense,
            "eigenvalue_time": str(HRt),
            "diff": np.abs(min_eigenvalue-me),
            "change_from_prev": None if diff_prev == np.inf else diff_prev,
            "circuit_cost": circuit_cost
            }
        
        if log_enabled: logger.info("[Run %d]\n%s",run_idx,json.dumps(row, indent=4, default=str))
        
        all_data.append(row)
        all_energies.append(me)

        if diff_prev is None:
            k+=1
            continue

        if diff_prev < tol:
            patience_count += 1
            trotter_patience_count+=1
        else:
            patience_count = 0

        if patience_count >= energy_patience:
            converged = True
            break
        elif k == max_k:
            break
        else:
            k+=1

        if trotter_patience_count >= trotter_patience:
            n_steps +=1
            trotter_patience_count=0
            if n_steps > max_n_steps:
                break

    endtime = datetime.now()

    final_data = {
            "seed": seed,
            "starttime": starttime.strftime("%Y-%m-%d_%H-%M-%S"),
            "endtime": endtime.strftime("%Y-%m-%d_%H-%M-%S"),
            "time_taken": str(endtime-starttime),
            "H_info": H_info,
            "backend_info": backend_info,
            "Ansatz_info" : Ansatz_info,
            "run_info":run_info,
            "final_k": len(all_data),
            "converged": converged,
            "all_energies": all_energies,
            "all_run_data": all_data,
            "num_jobs": len(jobs) if backend_name != "Aer" else None,
            "QPU_usage": QPU_usage if backend_name != "Aer" else None,
            "job_info": job_info,
            "sampler_options": sampler_options
        }


    #with open(os.path.join(base_path, f"run_{run_idx}.json"), "w") as json_file:
    with open(os.path.join(base_path, f"{potential}_{cutoff}.json"), "w") as json_file:
        json.dump(final_data, json_file, indent=4, default=str)

    if log_enabled: logger.info(f"[Run {run_idx}] Done")
    print(f"Run {run_idx} done")


if __name__ == "__main__":

    starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_enabled = True
        
    N = 4
    a = 1.0
    c = -0.2
    potential = "linear"
    #potential="quadratic"
    boundary_condition = 'dirichlet'

    #backend_name = 'ibm_kingston'
    backend_name = 'ibm_torino'
    #backend_name = 'ibm_marrakesh'

    #backend_name = "Aer"

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

    shots = 30000
    optimization_level = 3
    resilience_level = 0 # 1 = readout , 2 = readout + gate

    # trimming
    conserve_fermion = True
    keep_ratio = 0.999 

    dt=1.0
    max_k = 30
    tol = 1e-8

    trotter_patience = 2
    energy_patience = 3        
    max_n_steps = 5

    for cutoff in [16]:

        print(f"Running for {potential} and cutoff {cutoff}")

        tags=["Open-plan", "SKQD", f"shots:{shots}", f"{boundary_condition}", f"{potential}", f"N={N}", f"cutoff={cutoff}"]


        if potential == 'quadratic':
            folder = 'C' + str(abs(c)) + '/' + 'N'+ str(N)
        else:
            folder = 'N'+ str(N)

        base_path = os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\Wess-Zumino\Qiskit\SKQD\NQCC\Open Plan", boundary_condition, potential, folder, str(starttime))
        #base_path = os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\Wess-Zumino\Qiskit\SKQD\test2", boundary_condition, potential, folder)#, str(starttime))
        os.makedirs(base_path, exist_ok=True)
        log_path = os.path.join(base_path, f"logs_{str(cutoff)}")

        if log_enabled: 
            os.makedirs(log_path, exist_ok=True)
            log_path = os.path.join(log_path, f"vqe_run.log")
            logger = setup_logger(log_path, f"logger", enabled=log_enabled)
        else:
            logger = None

        if log_enabled: logger.info(f"Running VQE for {potential} potential and cutoff {cutoff}")

        H_path = os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\Wess-Zumino\Analyses\Model Checks\HamiltonianData", boundary_condition, potential, folder, f"{potential}_{cutoff}.json")
        with open(H_path, 'r') as file:
            H_data = json.load(file)

        pauli_coeffs = H_data["pauli_coeffs"]
        pauli_labels = H_data["pauli_labels"]
        num_qubits = H_data['num_qubits']
        dense_H_size = H_data['H_size']
        eigenvalues = H_data['eigenvalues']
        min_eigenvalue = np.min(eigenvalues)

        if log_enabled: logger.info(f"min_eigenvalue: {min_eigenvalue}")

        qps = int(np.log2(cutoff)) + 1
        fermion_qubits = [(s + 1) * qps - 1 for s in range(N)] 

        basis = [0] * num_qubits
        basis[2*qps - 1 :: 2*qps] = [1] * (N // 2)
        basis_state = basis

        num_fermions = sum(basis_state[q] for q in fermion_qubits)

        kept_coeffs, kept_labels, keep_idx, trunc_info = truncate_by_coeff_weight(pauli_coeffs, pauli_labels, keep_ratio=keep_ratio)
        keep_idx = np.sort(keep_idx)
        kept_coeffs = np.asarray(pauli_coeffs)[keep_idx]
        kept_labels = np.asarray(pauli_labels)[keep_idx]
        if log_enabled:
            logger.info(f"Truncation: {trunc_info}")

        H_pauli = SparsePauliOp(PauliList(kept_labels.tolist()), kept_coeffs.tolist())
        pauli_terms = list(zip(pauli_coeffs, pauli_labels))
        #pauli_terms = list(zip(kept_coeffs.tolist(), kept_labels.tolist()))



        
        include_basis=True
        include_rys=False
        include_xxyys=False
        avqe_circuit = wz.build_avqe_pattern_ansatz(N=N, cutoff=cutoff, include_basis=include_basis, include_rys=include_rys, include_xxyys=include_xxyys)

        H_info = {"potential": potential,
                "boundary_condition": boundary_condition,
                "cutoff": cutoff,
                "N": N,
                "a": a,
                "c": None if potential == "linear" else c,
                "num_qubits": num_qubits,
                "num_paulis": len(pauli_labels),
                "keep_ratio": keep_ratio,
                "trunc_info": trunc_info,
                "dense_H_size": dense_H_size,
                "eigenvalues": eigenvalues,
                "basis": basis_state
            }
        
        Ansatz_info = {
                "include_basis": include_basis,
                "include_rys": include_rys,
                "include_xxyys": include_xxyys
                }


        backend_info = {"backend_name": backend_name, 
                        "use_noise_model": use_noise_model, 
                        "noise_model_options": noise_model_options, 
                        "resilience_level": resilience_level, 
                        "optimization_level": optimization_level,
                        "shots": shots, 
                        "tags": tags}
        
        run_info = {"dt": dt,
                    "max_k": max_k,
                    "tol": tol,
                    "conserve_fermion": conserve_fermion,
                    "fermion_qubits": fermion_qubits,
                    "num_fermions": num_fermions,
                    "trotter_patience": trotter_patience,
                    "energy_patience": energy_patience,
                    "max_n_steps": max_n_steps
                    }
        
    
        # with Pool(processes=num_processes) as pool:
        #         pool.starmap(
        #             run_skqd,
        #             [ (i, H_mix, H_info, Ansatz_info, pauli_terms, avqe_circuit, backend_info, run_info, base_path, log_enabled) for i in range(num_processes)],
        #         )


        run_skqd(0, H_pauli, H_info, Ansatz_info, pauli_terms, avqe_circuit, backend_info, run_info, base_path, log_enabled, logger)
    
        
