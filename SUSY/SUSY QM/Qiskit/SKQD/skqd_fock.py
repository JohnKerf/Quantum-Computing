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

import susy_qm as sqm

import git
repo_path = git.Repo('.', search_parent_directories=True).working_tree_dir


path = os.path.join( r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\open-apikey.json")
#path = r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\apikey.json"
with open(path, encoding="utf-8") as f:
    api_key = json.load(f).get("apikey")

IBM_QUANTUM_API_KEY = api_key
ibm_instance_crn = "crn:v1:bluemix:public:quantum-computing:us-east:a/3ff62345f67c45e48e47a7f57d2f39f5:83214c75-88ab-4e55-8a87-6502ecc7cc9b::" #Open
#ibm_instance_crn = "crn:v1:bluemix:public:quantum-computing:us-east:a/d4f95db0515b47b7ba61dba8a424f873:ed0704ac-ad7d-4366-9bcc-4217fb64abd1::" #NQCC

service = QiskitRuntimeService(channel="ibm_quantum_platform", token=IBM_QUANTUM_API_KEY, instance=ibm_instance_crn)
COMPILE_BACKEND_NAME = "ibm_torino"#"ibm_marrakesh"
compile_backend = service.backend(COMPILE_BACKEND_NAME)
compile_target = compile_backend.target

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

def create_circuit(target, basis_state, optimization_level, num_qubits, H_pauli, t, num_trotter_steps, transpiler_seed):
    
    qr = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qr)

    for q, bit in enumerate(reversed(basis_state)):
        if bit == 1:
            qc.x(q)    

    evol_gate = PauliEvolutionGate(H_pauli,time=t,synthesis=LieTrotter(reps=num_trotter_steps))
    qc.append(evol_gate, qr)
    qc.measure_all()

    pm = generate_preset_pass_manager(target=target, optimization_level=optimization_level, seed_transpiler=transpiler_seed)
    try:
        circuit_isa = pm.run(qc)
    except:
        pm = generate_preset_pass_manager(target=target, optimization_level=optimization_level, translation_method="synthesis", seed_transpiler=transpiler_seed)
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
        "total_weight": total,
        "keep_idx": keep_idx
    }
    return c[keep_idx], lab[keep_idx], info

def run_skqd(H_pauli, H_info, pauli_terms, basis_state, backend_info, run_info, base_path, log_enabled):

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

    seed = (os.getpid() * int(time.time())) % 123456789

    backend, sampler = get_backend(backend_info["backend_name"], backend_info["use_noise_model"], backend_info["noise_model_options"], backend_info["resilience_level"], seed, shots, backend_info["tags"])
    if backend_info["use_noise_model"] == 1:
        target = compile_target
    else:
        target = backend.target
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

        if log_enabled: logger.info(f"Running for Krylov dimension {k}")

        t = dt*k
        
        qc, circuit_cost = create_circuit(target, basis_state, backend_info["optimization_level"], num_qubits, H_pauli, t, n_steps, backend_info["transpiler_seed"])

        t1 = datetime.now()
        counts, job_id, job_metrics = get_counts(sampler, qc, shots) #counts are returned in binary notation i.e. q0q1...qn and not standard qiskit noation
        Ct = datetime.now() - t1

        if backend_name != "Aer":
            job_info[job_id] = job_metrics

            jobs = job_info.values()
            QPU_usage = 0.0
            for job in jobs:
                usage = job.get("usage")
                QPU_usage += float(usage["seconds"])

            if log_enabled: logger.info(f"Job ID: {job_id} - QPU usage: {QPU_usage}")

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

        if log_enabled: logger.info(json.dumps(shot_processing, indent=4, default=str))

        # Update global samples with the trimmed counts
        pre_samples = len(samples)
        samples.update(counts)
        
        sorted_states = sorted(samples.items(), key=lambda x: x[1], reverse=True)
        top_states = [s for s, c in sorted_states]
        all_counts.append(dict(sorted_states)) 

        H_reduced = sqm.reduced_sparse_matrix_from_pauli_terms(pauli_terms, top_states)
        
        t1 = datetime.now()

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
        
        if log_enabled: logger.info(json.dumps(row, indent=4, default=str))
        
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


    with open(os.path.join(base_path, f"{potential}_{cutoff}.json"), "w") as json_file:
        json.dump(final_data, json_file, indent=4, default=str)

    if log_enabled: logger.info("Done")
    print(f"done")



if __name__ == "__main__":

    log_enabled = False

        
    potential = "DW"
    #cutoff = 32

    #backend_name = 'ibm_kingston'
    #backend_name = 'ibm_torino'
    backend_name = "Aer"

    # Noise model options
    use_noise_model = 0
    gate_error=False
    readout_error=False  
    thermal_relaxation=False
    transpiler_seed = 42

    noise_model_options = {
        "gate_error":gate_error,
        "readout_error":readout_error,   
        "thermal_relaxation":thermal_relaxation
        }

    shots = 10000
    optimization_level = 3
    resilience_level = 2 # 1 = readout , 2 = readout + gate

    # trimming
    conserve_fermion = True
    keep_ratio = 0.95 

    dt=0.5
    max_k = 1
    tol = 1e-10

    trotter_patience = 2
    energy_patience = 3        
    max_n_steps = 3

    for potential in ['QHO', 'AHO', 'DW']:
        for cutoff in [2,4,8,16,32,64,128,256, 512, 1024]:

            print(f"Running for {potential} and cutoff {cutoff}")

            tags=["SKQD", f"shots:{shots}", f"{potential}", f"cutoff={cutoff}"]


            base_path = os.path.join(repo_path,r"SUSY\SUSY QM\Qiskit\SKQD\BasisTesting\Aer\Fock+Trunc", potential)
            os.makedirs(base_path, exist_ok=True)
            log_path = os.path.join(base_path, f"logs_{str(cutoff)}")

            if log_enabled: 
                os.makedirs(log_path, exist_ok=True)
                log_path = os.path.join(log_path, f"vqe_run.log")
                logger = setup_logger(log_path, f"logger", enabled=log_enabled)

            if log_enabled: logger.info(f"Running for {potential} potential and cutoff {cutoff}")

            H = sqm.calculate_Hamiltonian(cutoff, potential)

            H_pauli = SparsePauliOp.from_operator(H)
            #pauli_labels = H_pauli.paulis
            #pauli_coeffs = H_pauli.coeffs

            pauli_coeffs = np.real(H_pauli.coeffs).astype(float).tolist() 
            pauli_labels = H_pauli.paulis.to_labels()

            kept_coeffs, kept_labels, trunc_info = truncate_by_coeff_weight(pauli_coeffs, pauli_labels, keep_ratio=keep_ratio)
            keep_idx = np.sort(trunc_info["keep_idx"])
            kept_coeffs = np.asarray(pauli_coeffs)[keep_idx]
            kept_labels = np.asarray(pauli_labels)[keep_idx]

            if log_enabled:
                logger.info(f"Truncation: {trunc_info}")

            H_pauli = SparsePauliOp(PauliList(kept_labels.tolist()), kept_coeffs.tolist()).simplify(atol=1e-12)
            pauli_terms = list(zip(pauli_coeffs, pauli_labels))

            #pauli_terms = [(complex(c), p.to_label()) for c, p in zip(H_pauli.coeffs, H_pauli.paulis)]

            num_qubits = int(1 + np.log2(cutoff))
            dense_H_size = H.shape
            eigenvalues = np.sort(np.linalg.eigvals(H))[:4]
            min_eigenvalue = np.min(eigenvalues)

            if log_enabled: logger.info(f"min_eigenvalue: {min_eigenvalue}")

            if potential == 'DW':
                if cutoff == 4:
                    basis_state = [1] + [0]*(num_qubits-1)
                else:
                    basis_state = [0]*(num_qubits)
            else:
                basis_state = [1] + [0]*(num_qubits-1)

            fermion_qubits = [num_qubits-1]
            num_fermions = sum(basis_state[::-1][q] for q in fermion_qubits)

            
            H_info = {"potential": potential,
                    "cutoff": cutoff,
                    "num_qubits": num_qubits,
                    "num_paulis": len(pauli_labels),
                    "dense_H_size": dense_H_size,
                    "eigenvalues": [float(e) for e in eigenvalues.real],
                    "basis": basis_state,
                    "keep_ratio": keep_ratio,
                    "trunc_info": trunc_info
                }
            

            backend_info = {"backend_name": backend_name, 
                            "use_noise_model": use_noise_model, 
                            "noise_model_options": noise_model_options, 
                            "resilience_level": resilience_level, 
                            "optimization_level": optimization_level,
                            "shots": shots, 
                            "tags": tags,
                            "transpiler_seed": transpiler_seed}
            
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


            run_skqd(H_pauli, H_info, pauli_terms, basis_state, backend_info, run_info, base_path, log_enabled)
    
        
