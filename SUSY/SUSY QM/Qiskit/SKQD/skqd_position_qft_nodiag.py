import os, json, logging, time, dataclasses
import numpy as np
from collections import Counter
from datetime import datetime

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import SamplerV2 as Sampler, QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import SparsePauliOp, Operator, PauliList
from scipy.sparse.linalg import eigsh
from qiskit.circuit.library import QFTGate, DiagonalGate, PauliEvolutionGate
from qiskit.synthesis.evolution import LieTrotter
from qiskit.transpiler.layout import Layout
from qiskit.circuit.library.standard_gates import PhaseGate
from qiskit.circuit.library.standard_gates import RZZGate
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

def _fast_walsh_hadamard(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).copy()
    n = v.size
    h = 1
    while h < n:
        for i in range(0, n, 2 * h):
            a = v[i:i + h].copy()
            b = v[i + h:i + 2 * h].copy()
            v[i:i + h] = a + b
            v[i + h:i + 2 * h] = a - b
        h *= 2
    return v

def walsh_coeffs(diag_vals: np.ndarray) -> np.ndarray:
    diag_vals = np.asarray(diag_vals, dtype=np.float64)
    nb = int(np.log2(diag_vals.size))
    coeffs = _fast_walsh_hadamard(diag_vals) / (2**nb)
    return coeffs

def diagonal_to_z_pauliop(diag_vals: np.ndarray, *, atol: float = 1e-12) -> SparsePauliOp:
    diag_vals = np.asarray(diag_vals, dtype=np.float64)
    dim = diag_vals.size
    nb = int(np.log2(dim))
    if 2**nb != dim:
        raise ValueError("diag_vals length must be a power of 2.")
    coeffs = _fast_walsh_hadamard(diag_vals) / (2**nb)

    labels, c = [], []
    for s, coeff in enumerate(coeffs):
        if abs(coeff) <= atol:
            continue
        bits = [(s >> i) & 1 for i in range(nb)]
        label = "".join("Z" if bits[nb - 1 - i] else "I" for i in range(nb))
        labels.append(label)
        c.append(coeff)

    if not labels:
        labels, c = ["I" * nb], [0.0]

    return SparsePauliOp(labels, coeffs=np.asarray(c, dtype=np.float64)).simplify(atol=atol)

def append_split_operator_evolution_old(qc, qf, qb, basis_info, t, n_steps):
    dt = t / n_steps
    nb = len(qb)

    k2  = np.asarray(basis_info["k2"],       dtype=np.float64)
    Wp2 = np.asarray(basis_info["Wp2_diag"], dtype=np.float64)
    Wpp = np.asarray(basis_info["Wpp_diag"], dtype=np.float64)

    # --- Keep boson-only diagonals as DiagonalGate (let Qiskit synthesize)
    V_wp2 = (-(dt / 4.0)) * Wp2
    V_Wpp_pos = (-(dt / 4.0)) * (+Wpp)
    V_bos = V_wp2 + V_Wpp_pos 
    gate_V_bos = DiagonalGate(np.exp(1j * V_bos).tolist())

    # --- Replace controlled diagonal with Pauli-diagonal evolution
    # Need exp[-i (dt/4) * (Z_f ⊗ Wpp_diag)]
    H_wpp_b = diagonal_to_z_pauliop(Wpp)
    H_wpp = extend_with_Z_front(H_wpp_b)

    evo_wpp = PauliEvolutionGate(H_wpp, time=(dt / 4.0), synthesis=LieTrotter(reps=1))

    # --- Kinetic diagonal in k-basis
    T = (-(dt / 2.0)) * k2
    gate_T = DiagonalGate(np.exp(1j * T).tolist())

    qft = QFTGate(nb)
    iqft = qft.inverse()

    for _ in range(n_steps):
        # V half-step
        qc.append(gate_V_bos, qb)
        qc.append(evo_wpp, [qf] + qb)

        # T full-step
        qc.append(qft, qb)
        qc.append(gate_T, qb)
        qc.append(iqft, qb)

        # V half-step
        qc.append(gate_V_bos, qb)
        qc.append(evo_wpp, [qf] + qb)

def spectrum_report(V_bos, top=25):
    coeffs = walsh_coeffs(V_bos)
    nb = int(np.log2(coeffs.size))

    mags = np.abs(coeffs)
    idx = np.argsort(mags)[::-1]

    print(f"nb={nb}, dim={coeffs.size}")
    print(f"coeff L1={mags.sum():.6g}, L2={np.sqrt((mags*mags).sum()):.6g}, max={mags[idx[0]]:.6g}")

    print("\nTop terms:")
    for rank in range(top):
        s = int(idx[rank])
        print(f"{rank:2d}: |c|={mags[s]:.3e}, weight={int(s.bit_count()):2d}, index={s}")

    # how many terms to capture 99% of L2 energy?
    l2 = mags*mags
    total = l2.sum()
    cumsum = np.cumsum(l2[idx])
    k99 = int(np.searchsorted(cumsum, 0.99*total) + 1)
    print(f"\nTerms to capture 99% of L2 energy: {k99}")

def topk_walsh_z_terms(diag_vals: np.ndarray, K: int, nb):
    diag_vals = np.asarray(diag_vals, dtype=np.float64)
    coeffs = _fast_walsh_hadamard(diag_vals) / (2**nb)

    mags = np.abs(coeffs)
    idx = np.argsort(mags)[::-1]

    out = []
    for s in idx:
        if s == 0:
            continue  # drop identity / global phase
        out.append((int(s), float(coeffs[int(s)])))
        if len(out) >= K:
            break
    return out

def apply_truncated_V_as_gadgets(qc: QuantumCircuit, qb, terms):
    """
    Apply U = exp(i * sum_s c_s Z_s) for Z-only terms.
    `terms` is list of (s, c_s) where s is bitmask (LSB=qubit0).
    """
    nb = len(qb)
    for s, c in terms:
        bits = [(s >> i) & 1 for i in range(nb)]
        qubits = [qb[i] for i, b in enumerate(bits) if b]
        w = len(qubits)

        if w == 1:
            # exp(i c Z) = RZ(theta) with theta = -2c
            qc.rz(-2.0 * c, qubits[0])

        elif w == 2:
            # exp(i c Z⊗Z) = RZZ(theta) with theta = -2c
            qc.append(RZZGate(-2.0 * c), qubits)

        else:
            # generic multi-Z phase gadget using CNOT ladder
            # exp(i c Z...Z)
            # Implement via: CNOT chain -> RZ on last -> uncompute
            for q in qubits[:-1]:
                qc.cx(q, qubits[-1])
            qc.rz(-2.0 * c, qubits[-1])
            for q in reversed(qubits[:-1]):
                qc.cx(q, qubits[-1])


def mask_to_label(mask: int, nb: int) -> str:
    # Qiskit Pauli label: rightmost char is qubit 0
    chars = ["I"] * nb
    for q in range(nb):
        if (mask >> q) & 1:
            chars[nb - 1 - q] = "Z"
    return "".join(chars)

def topk_diagonal_as_z_sparsepauliop(diag_vals: np.ndarray, K: int, nb: int, drop_identity: bool = True):
    diag_vals = np.asarray(diag_vals, dtype=np.float64)
    coeffs = _fast_walsh_hadamard(diag_vals) / (2**nb)

    mags = np.abs(coeffs)
    idx = np.argsort(mags)[::-1]

    labels = []
    out_coeffs = []
    for s in idx:
        s = int(s)
        if drop_identity and s == 0:
            continue
        c = float(coeffs[s])
        if c == 0.0:
            continue
        labels.append(mask_to_label(s, nb))
        out_coeffs.append(c)
        if len(labels) >= K:
            break

    if not labels:
        return SparsePauliOp(PauliList(["I"*nb]), coeffs=[0.0])

    return SparsePauliOp(PauliList(labels), coeffs=np.asarray(out_coeffs, dtype=np.float64))

def append_split_operator_evolution(qc, qf, qb, basis_info, t, n_steps):
    dt = t / n_steps
    nb = len(qb)

    # --- keep boson diagonal in x-basis as DiagonalGate (for now)
    k2 = np.asarray(basis_info["k2"], dtype=np.float64)
    Wp2 = np.asarray(basis_info["Wp2_diag"], dtype=np.float64)
    Wpp = np.asarray(basis_info["Wpp_diag"], dtype=np.float64)

    V_bos = (-(dt / 4.0)) * (Wp2)# + Wpp)
    #spectrum_report(V_bos, top=25)
    terms = topk_walsh_z_terms(V_bos, K=25, nb=nb)
    #print("Walsh", terms)

    phases = np.asarray(V_bos, dtype=np.float64).reshape(-1)
    H = np.diag(phases)
    op = SparsePauliOp.from_operator(H) 

    #print("SPO", op)

    Tphase = (-(dt/2.0)) * k2
    #spectrum_report(Tphase, top=25)
    terms_T = topk_walsh_z_terms(Tphase, K=25, nb=nb)

    H_V_trunc = topk_diagonal_as_z_sparsepauliop(V_bos, K=25, nb=nb, drop_identity=True)
    H_T_trunc = topk_diagonal_as_z_sparsepauliop(Tphase, K=25, nb=nb, drop_identity=True)

    # Z_f ⊗ Wpp
    H_wpp_b = diagonal_to_z_pauliop(Wpp)
    labels = ["Z" + p.to_label() for p in H_wpp_b.paulis]
    H_wpp = SparsePauliOp(labels, H_wpp_b.coeffs)

    evo_wpp = PauliEvolutionGate(H_wpp, time=(dt / 4.0), synthesis=LieTrotter(reps=1))

    qft = QFTGate(nb)
    iqft = qft.inverse()

    for _ in range(n_steps):
        #apply_truncated_V_as_gadgets(qc, qb, terms)
        qc.append(PauliEvolutionGate(H_V_trunc, time=-1.0, synthesis=LieTrotter(reps=1)),qb)
        qc.append(evo_wpp, [qf] + qb)

        qc.append(qft, qb)
        qc.append(PauliEvolutionGate(H_V_trunc, time=-1.0, synthesis=LieTrotter(reps=1)),qb)
        #apply_truncated_V_as_gadgets(qc, qb, terms_T)
        qc.append(iqft, qb)

        qc.append(PauliEvolutionGate(H_V_trunc, time=-1.0, synthesis=LieTrotter(reps=1)),qb)
        #apply_truncated_V_as_gadgets(qc, qb, terms)
        qc.append(evo_wpp, [qf] + qb)

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

def create_circuit(basis_info, backend, basis_state, optimization_level, cutoff, t, num_trotter_steps):

    num_qubits = 1 + int(np.log2(cutoff))
    qc = QuantumCircuit(num_qubits)

    # Your existing basis-state init (note: you reverse in your code)
    for q, bit in enumerate(reversed(basis_state)):
        if bit == 1:
            qc.x(q)

    qf = num_qubits - 1          # fermion = last qubit
    qb = list(range(num_qubits - 1))  # boson register = all earlier qubits

    append_split_operator_evolution(qc, qf, qb, basis_info, t, num_trotter_steps)

    qc.measure_all()

    #summarize_circuit(qc, "pre-transpile")
    pm = generate_preset_pass_manager(target=backend.target, optimization_level=optimization_level)#, routing_method="sabre", layout_method="sabre")
 
    try:
        circuit_isa = pm.run(qc)
    except:
        print(f"Error transpiling")
        pm = generate_preset_pass_manager(target=backend.target, optimization_level=optimization_level, translation_method="synthesis")
        circuit_isa = pm.run(qc)

    #summarize_circuit(circuit_isa, "post-transpile")
    #print_layout_info(circuit_isa)
    #print(circuit_isa.draw("text", fold=200))

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

def truncate_by_coeff_weight(pauli_coeffs, pauli_labels, keep_ratio=0.999, min_keep=64):

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
    return c[keep_idx], lab[keep_idx], info

def run_skqd(basis_info, H_pauli, H_info, pauli_terms, basis_state, backend_info, run_info, base_path, log_enabled):

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
        
        qc, circuit_cost = create_circuit(basis_info, backend, basis_state, backend_info["optimization_level"], H_info["cutoff"], t, n_steps)

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
    cutoff = 2

    #backend_name = 'ibm_kingston'
    #backend_name = 'ibm_torino'
    backend_name = "Aer"

    # Noise model options
    use_noise_model = 1
    gate_error=False
    readout_error=False  
    thermal_relaxation=False

    noise_model_options = {
        "gate_error":gate_error,
        "readout_error":readout_error,   
        "thermal_relaxation":thermal_relaxation
        }

    shots = 10000
    optimization_level = 3
    resilience_level = 2 # 1 = readout , 2 = readout + gate

    # trimming
    conserve_fermion = False
    keep_ratio = 1.0 

    x_max=8.0

    for potential in ['DW']:
        for cutoff in [2,4,8,16,32,64,128,256,512,1024]:

            print(f"Running for {potential} and cutoff {cutoff}")

            tags=["SKQD", f"shots:{shots}", f"{potential}", f"cutoff={cutoff}"]


            base_path = os.path.join(repo_path,r"SUSY\SUSY QM\Qiskit\SKQD\BasisTesting\RealTranspile\Position+QFT+NoDiag", potential)
            #base_path = os.path.join(repo_path,r"SUSY\SUSY QM\Qiskit\SKQD\BasisTesting\Aer\Position+QFT+NoDiag", potential)
            os.makedirs(base_path, exist_ok=True)
            log_path = os.path.join(base_path, f"logs_{str(cutoff)}")

            if log_enabled: 
                os.makedirs(log_path, exist_ok=True)
                log_path = os.path.join(log_path, f"vqe_run.log")
                logger = setup_logger(log_path, f"logger", enabled=log_enabled)

            if log_enabled: logger.info(f"Running for {potential} potential and cutoff {cutoff}")

            num_qubits = int(1+np.log2(cutoff))

            H_dense, basis_info = sqm.calculate_hamiltonian_position(cutoff, potential, x_max=x_max)
            H_pauli = SparsePauliOp.from_operator(H_dense)
            pauli_labels = H_pauli.paulis
            pauli_coeffs = H_pauli.coeffs
            pauli_terms = [(complex(c), p.to_label()) for c, p in zip(H_pauli.coeffs, H_pauli.paulis)]

            num_qubits = int(1 + np.log2(cutoff))
            dense_H_size = H_dense.shape
            eigenvalues = np.sort(np.linalg.eigvals(H_dense))[:4]
            min_eigenvalue = np.min(eigenvalues)

            if log_enabled: logger.info(f"min_eigenvalue: {min_eigenvalue}")

            if potential == "DW" and cutoff ==16:
                basis_state = [0,0]+[0]*(num_qubits-2) 
            else:
                basis_state = [1,1]+[0]*(num_qubits-2)

            fermion_qubits = [num_qubits-1]
            num_fermions = sum(basis_state[::-1][q] for q in fermion_qubits)

            
            dt=0.5
            max_k = 1
            tol = 1e-10

            trotter_patience = 2
            energy_patience = 3        
            max_n_steps = 3

            H_info = {"potential": potential,
                    "cutoff": cutoff,
                    "x_max": x_max,
                    "num_qubits": num_qubits,
                    "num_paulis": len(pauli_labels),
                    "dense_H_size": dense_H_size,
                    "eigenvalues": [float(e) for e in eigenvalues.real],
                    "basis": basis_state
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


            run_skqd(basis_info, H_pauli, H_info, pauli_terms, basis_state, backend_info, run_info, base_path, log_enabled)
    
        
