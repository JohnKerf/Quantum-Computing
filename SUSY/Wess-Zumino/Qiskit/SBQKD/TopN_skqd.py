import os, json, logging, time, dataclasses
import numpy as np
from collections import Counter
from datetime import datetime

from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library import PauliEvolutionGate, XXPlusYYGate
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




def trim_counts(counts, p_keep):
    if not counts:
        return {}
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    total = sum(c for _, c in items)
    running = 0
    kept = {}
    for key, c in items:
        kept[key] = c
        running += c
        if running / total >= p_keep:
            break
    return kept



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


def create_circuit(backend, avqe_circuit, optimization_level, basis_state, num_qubits, H_pauli, t, num_trotter_steps):
    
    qr = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qr)

    # for q, bit in enumerate(basis_state):
    #     if bit == 1:
    #         qc.x(q)    

    qc.append(avqe_circuit, qr)

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


def filter_counts_by_fermion_number(
    counts: dict[str, int],
    *,
    fermion_qubits: list[int],
    num_fermions: int,
    tol: int = 0, 
) -> tuple[dict[str, int], int]:
    kept = {}
    rejected = 0
    for key, c in counts.items():
        b = [int(ch) for ch in key[::-1]]
        w = sum(b[i] for i in fermion_qubits)
        if abs(w - num_fermions) <= tol:
            kept[key] = c
        else:
            rejected += c
    return kept, rejected


def circuit_cost_metrics(qc):

    ops = qc.count_ops()
    ops_str = {str(k): int(v) for k, v in ops.items()}
    n2q = sum(1 for ci in qc.data if ci.operation.num_qubits == 2 and ci.operation.name not in {"barrier", "measure"})

    return {
        "depth": qc.depth(),
        "size": qc.size(),
        "num_2q_ops": n2q,
        "count_ops": ops_str,
    }




def truncate_by_mass(pauli_coeffs, pauli_labels, keep_mass=0.999, norm="l1", min_keep=64):
    c = np.asarray(pauli_coeffs, dtype=np.complex128)
    lab = np.asarray(pauli_labels)

    abs_c = np.abs(c)
    order = np.argsort(abs_c)[::-1]
    abs_sorted = abs_c[order]

    if norm == "l1":
        w = abs_sorted
    elif norm == "l2":
        w = abs_sorted**2
    else:
        raise ValueError("norm must be 'l1' or 'l2'")

    cum = np.cumsum(w)
    total = float(cum[-1])
    target = keep_mass * total

    m = int(np.searchsorted(cum, target, side="left") + 1)
    m = max(m, int(min_keep))
    m = min(m, len(c))

    keep_idx = order[:m]
    tail = float(total - cum[m-1])

    info = {
        "m": m,
        "n": len(c),
        "keep_frac_terms": m / len(c),
        "keep_mass": keep_mass,
        "tail_mass": tail,
        "total_mass": total,
        "norm": norm,
    }
    return c[keep_idx], lab[keep_idx], info



if __name__ == "__main__":
    starttime = datetime.now()

    log_enabled = False
    seed = (os.getpid() * int(time.time())) % 123456789

        
    N = 5
    a = 1.0
    c = -0.2
    potential = "linear"
    boundary_condition = 'dirichlet'
    cutoff = 8

    # trimming
    CONSERVE_FERMIONS = True
    TRIM_STATES = False
    P_KEEP = 0.995

    #backend_name = 'ibm_kingston'
    #backend_name = 'ibm_torino'
    backend_name = "Aer"

    # Noise model options
    use_noise_model = 0
    gate_error=True
    readout_error=True  
    thermal_relaxation=True

    noise_model_options = {
        "gate_error":gate_error,
        "readout_error":readout_error,   
        "thermal_relaxation":thermal_relaxation
        }

    shots = 10000
    optimization_level = 3
    resilience_level = 2 # 1 = readout , 2 = readout + gate

    #k=1
    #n_steps=2
    dt=3.0
    dt_slice_max=1.0
    max_k = 40
    tol = 1e-8

    keep_mass = 0.85#9999
    min_keep = 64

    TROTTER_PATIENCE = 1

    for cutoff in [4]:

        print(f"Running for {potential} and cutoff {cutoff}")

        #tags=["Open-access", "SBQKD", f"shots:{shots}", f"{boundary_condition}", f"{potential}", f"N={N}", f"cutoff={cutoff}"]
        tags=["NQCC-Q4", "SBQKD", f"shots:{shots}", f"{boundary_condition}", f"{potential}", f"N={N}", f"cutoff={cutoff}"]


        if potential == 'quadratic':
            folder = 'C' + str(abs(c)) + '/' + 'N'+ str(N)
        else:
            folder = 'N'+ str(N)

        #base_path = os.path.join(r"/users/johnkerf/fastscratch/Data/WessZumino/SBQKD/Files", backend_name, boundary_condition, potential, folder, starttime.strftime("%Y-%m-%d_%H-%M-%S"))
        base_path = os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\Wess-Zumino\Qiskit\SBQKD\TopN-NoNoise-Full", boundary_condition, potential, folder)#, starttime.strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(base_path, exist_ok=True)
        log_path = os.path.join(base_path, f"logs_{str(cutoff)}")

        if log_enabled: 
            os.makedirs(log_path, exist_ok=True)
            log_path = os.path.join(log_path, f"vqe_run.log")
            logger = setup_logger(log_path, f"logger", enabled=log_enabled)

        if log_enabled: logger.info(f"Running VQE for {potential} potential and cutoff {cutoff}")

        H_path = os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\Wess-Zumino\Analyses\Model Checks\HamiltonianData", boundary_condition, potential, folder, f"{potential}_{cutoff}.json")
        with open(H_path, 'r') as file:
            H_data = json.load(file)

        pauli_coeffs = H_data["pauli_coeffs"]
        pauli_labels = H_data["pauli_labels"]
        
        kept_coeffs, kept_labels, trunc_info = truncate_by_mass(pauli_coeffs, pauli_labels, keep_mass=keep_mass, norm="l1", min_keep=min_keep)

        H_pauli = SparsePauliOp(PauliList(kept_labels.tolist()), kept_coeffs.tolist())
        #pauli_terms = list(zip(kept_coeffs.tolist(), kept_labels.tolist()))

        if log_enabled:
            logger.info(f"Truncation: {trunc_info}")


        #H_pauli = SparsePauliOp(PauliList(pauli_labels), pauli_coeffs)
        pauli_terms = list(zip(pauli_coeffs, pauli_labels))

        num_qubits = H_data['num_qubits']
        dense_H_size = H_data['H_size']
        eigenvalues = H_data['eigenvalues']
        min_eigenvalue = np.min(eigenvalues)

        if log_enabled: logger.info(f"min_eigenvalue: {min_eigenvalue}")

        basis_state = H_data['best_basis_state'][::-1]
        qps = int(np.log2(cutoff)) + 1
        fermion_qubits = [(s + 1) * qps - 1 for s in range(N)]
        num_fermions = sum(basis_state[q] for q in fermion_qubits)

        converged=False
        samples = Counter()
        prev_energy = np.inf

        all_data = []
        all_counts = []
        all_energies = []
        job_info = {}

        backend, sampler = get_backend(backend_name, use_noise_model, noise_model_options, resilience_level, seed, shots, tags)

        sampler_options = dataclasses.asdict(sampler.options)
        if log_enabled: logger.info(json.dumps(sampler_options, indent=4, default=str))

        include_basis=True
        include_rys=True
        include_xxyys=True
        avqe_circuit = wz.build_avqe_pattern_ansatz(N=N, cutoff=cutoff, include_basis=include_basis, include_rys=include_rys, include_xxyys=include_xxyys)

        k=1
        n_steps = 1
        patience_count = 0
        while not converged and k <= max_k:

            if log_enabled: logger.info(f"Running for Krylov dimension {k}")
            print(f"Running for Krylov dimension {k}")

            t = dt*k
            #n_steps = max(1, int(np.ceil(abs(float(t)) / dt_slice_max)))
            
            qc, circuit_cost = create_circuit(backend, avqe_circuit, optimization_level, basis_state, num_qubits, H_pauli, t, n_steps)

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

                logger.info(f"Job ID: {job_id} - QPU usage: {QPU_usage}")

            # trim per Krylov step
            raw_counts = counts
            raw_unique = len(raw_counts)
            raw_shots = sum(raw_counts.values())

            post_rejected = 0
            trim_rejected = 0

            if CONSERVE_FERMIONS:
                counts, rej_w = filter_counts_by_fermion_number(
                    counts,
                    fermion_qubits=fermion_qubits,
                    num_fermions=num_fermions,
                    tol=0,
                )
                post_rejected += rej_w
                print(f"Rejected {rej_w} states from conserving fermion number")
            else:
                counts = raw_counts

            if TRIM_STATES:
                pre_trim_shots = sum(counts.values())
                counts = trim_counts(counts, P_KEEP)
                trim_rejected = pre_trim_shots - sum(counts.values())
                print(f"Trimmed {trim_rejected} states with prob < {P_KEEP}")

            kept_unique = len(counts)
            kept_shots = sum(counts.values())

            shot_processing = {
                "raw_unique": raw_unique,
                "raw_shots": raw_shots,
                "postselect_rejected_shots": post_rejected,
                "trim_rejected_shots": trim_rejected,
                "kept_unique": kept_unique,
                "kept_shots": kept_shots,
            }

            if log_enabled: logger.info(json.dumps(shot_processing, indent=4, default=str))

            # Update global samples with the trimmed counts
            samples.update(counts)
            

            sorted_states = sorted(samples.items(), key=lambda x: x[1], reverse=True)
            top_states = [s for s, c in sorted_states]
            all_counts.append(dict(sorted_states)) 

            H_reduced = wz.reduced_sparse_matrix_from_pauli_terms(pauli_terms, top_states)
            
            t1 = datetime.now()
            #me = eigsh(H_reduced, k=1, which="SA", return_eigenvectors=False)[0].real
            if H_reduced.shape[0] < 2000:
                H_dense = H_reduced.todense()
                me = np.min(np.linalg.eigvals(H_dense)).real
                used_dense = True
            else:
                me = eigsh(H_reduced, k=1, which="SA", return_eigenvectors=False)[0].real
                used_dense = False
            HRt = datetime.now() - t1

            diff_prev = np.abs(prev_energy-me)

            row = { 
                "D": k,
                "t":t,
                "num_trotter_steps": n_steps,
                "circuit_time": str(Ct),
                "num_samples": len(samples),
                "shot_processing": shot_processing,
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

            converged = True if diff_prev < tol else False

            if converged == False and k == max_k: 
                if log_enabled: logger.info("max_k reached")
                print("max_k reached")
                break
            elif converged == False:
                prev_energy = me
                k+=1
            else:
                if patience_count < TROTTER_PATIENCE:
                    n_steps +=1
                    print(f"patience count {patience_count}")
                    print(f"Energy stagnated - increasing trotter steps to {n_steps}")
                    patience_count+=1
                    converged = False
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
            "Ansatz" : {
                        "include_basis": include_basis,
                        "include_rys": include_rys,
                        "include_xxyys": include_xxyys
                        },
            "CONSERVE_FERMIONS": CONSERVE_FERMIONS,
            "TRIM_STATES": TRIM_STATES,
            "P_KEEP": P_KEEP,
            "shots": shots,
            "tags": tags,
            "potential": potential,
            "boundary_condition": boundary_condition,
            "cutoff": cutoff,
            "N": N,
            "a": a,
            "c": None if potential == "linear" else c,
            "num_qubits": num_qubits,
            "num_paulis": len(pauli_labels),
            "dense_H_size": dense_H_size,
            "eigenvalues": eigenvalues,
            "basis": basis_state,
            "tol":tol,
            "dt":dt,
            #"dt_slice_max": dt_slice_max,
            "keep_mass": keep_mass,
            "min_keep": min_keep,
            #"n_trotter_steps":n_steps,
            "max_k":max_k,
            "final_k": k,
            "converged": converged,
            "all_energies": all_energies,
            "all_run_data": all_data,
            "num_jobs": len(jobs) if backend_name != "Aer" else None,
            "QPU_usage": QPU_usage if backend_name != "Aer" else None,
            "job_info": job_info,
            "sampler_options": sampler_options,
            #"all_counts": all_counts
        }


        with open(os.path.join(base_path, f"{potential}_{cutoff}.json"), "w") as json_file:
            json.dump(final_data, json_file, indent=4, default=str)

        if log_enabled: logger.info("Done")
        print("Done")
    
        
