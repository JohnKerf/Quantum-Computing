import os, json, logging, time, dataclasses
import numpy as np
from collections import Counter
from datetime import datetime

from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister, Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import SamplerV2 as Sampler, QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import PauliList, SparsePauliOp
from scipy.sparse.linalg import eigsh

from multiprocessing import Pool

import wesszumino as wz

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




from pathlib import Path
from ast import literal_eval

def _parse_coeff(x):
    """
    Robust parser for coefficients if they are stored as floats OR strings like '(0.25+0j)'.
    """
    if isinstance(x, (int, float, complex, np.number)):
        return complex(x)
    if isinstance(x, str):
        return complex(literal_eval(x))
    raise TypeError(f"Unsupported coefficient type: {type(x)}")


def reconstruct_from_energy_contrib_threshold(
    json_path,
    threshold,
    *,
    strict=True,              # True => abs(contrib) > threshold, False => >=
    drop_identity=False,      # optionally remove identity term
    simplify_atol=0.0,        # optional simplify tolerance
    return_mask=False,
):
    """
    Reconstruct a SparsePauliOp Hamiltonian keeping only terms with
    abs(pauli_energy_contrib_ground) > threshold.

    Parameters
    ----------
    json_path : str or Path
        Path to your Hamiltonian JSON.
    threshold : float
        Threshold on abs(pauli_energy_contrib_ground).
    strict : bool
        If True, keep abs(contrib) > threshold. If False, keep abs(contrib) >= threshold.
    drop_identity : bool
        If True, drop the all-identity term (e.g. 'IIII...').
    simplify_atol : float
        Passed to SparsePauliOp.simplify(atol=...).
    return_mask : bool
        If True, also return the boolean mask used.

    Returns
    -------
    H_pruned : SparsePauliOp
    info : dict
        Diagnostics about kept/dropped terms.
    mask : np.ndarray (optional)
        Boolean mask of kept terms.
    """
    json_path = Path(json_path)
    with open(json_path, "r") as f:
        d = json.load(f)

    labels = d["pauli_labels"]
    coeffs = [_parse_coeff(c) for c in d["pauli_coeffs"]]
    contribs = np.asarray(d["pauli_energy_contrib_ground"], dtype=float)

    if not (len(labels) == len(coeffs) == len(contribs)):
        raise ValueError(
            f"Length mismatch: labels={len(labels)}, coeffs={len(coeffs)}, contribs={len(contribs)}"
        )

    # Threshold condition
    if strict:
        mask = np.abs(contribs) > float(threshold)
    else:
        mask = np.abs(contribs) >= float(threshold)

    # Optionally remove identity term
    if drop_identity:
        identity_mask = np.array([set(lbl) == {"I"} for lbl in labels], dtype=bool)
        mask = mask & (~identity_mask)

    kept_labels = [lbl for lbl, m in zip(labels, mask) if m]
    kept_coeffs = [c for c, m in zip(coeffs, mask) if m]

    if len(kept_labels) == 0:
        raise ValueError(
            "No Pauli terms survived the threshold. Lower the threshold or disable drop_identity."
        )

    H_pruned = SparsePauliOp(kept_labels, coeffs=np.asarray(kept_coeffs, dtype=complex))

    # Optional simplify (merges duplicate labels if any, removes near-zero coeffs)
    if simplify_atol is not None:
        H_pruned = H_pruned.simplify(atol=float(simplify_atol))

    info = {
        "json_path": str(json_path),
        "threshold": float(threshold),
        "strict": strict,
        "drop_identity": drop_identity,
        "num_terms_total": len(labels),
        "num_terms_kept_before_simplify": int(np.sum(mask)),
        "num_terms_dropped": int(len(labels) - np.sum(mask)),
        "num_terms_final": len(H_pruned.paulis),
        "kept_fraction": float(np.sum(mask) / len(labels)),
        "sum_abs_contrib_kept": float(np.sum(np.abs(contribs[mask]))),
        "sum_abs_contrib_dropped": float(np.sum(np.abs(contribs[~mask]))),
    }

    if return_mask:
        return H_pruned, info, mask
    return H_pruned, info

def _parse_complex_coeff(x):
    """Parse coefficients stored as strings like '(0.25+0j)' safely."""
    if isinstance(x, (int, float, complex, np.number)):
        return complex(x)
    if isinstance(x, str):
        return complex(literal_eval(x))
    raise TypeError(f"Unsupported coeff type: {type(x)}")

def _safe_get_nested(d, *keys, default=None):
    """Safely read nested dictionary keys."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _extract_metric(d, candidate_paths):
    """
    Try multiple possible nested paths to find a metric.
    Example path: ("circuit_cost", "num_2q_ops")
    """
    for path in candidate_paths:
        val = _safe_get_nested(d, *path, default=None)
        if val is not None:
            return val
    return None

def reconstruct_sparsepauliop_from_ablation_json(
    json_path,
    utility_tol=None,                   # keep drop-candidates with utility >= utility_tol
    require_positive_2q_benefit=False,  # if True, only drop terms that reduce 2Q metric
    gate_metric="num_2q_ops",
    prefer_depth2q=False,
    max_terms_to_drop=None,             # cap number of dropped terms
    max_cumulative_damage=None,         # cap total absolute damage sum of dropped terms
    drop_identity=False,
):
    """
    Reconstruct original SparsePauliOp from one-term-removal ablation JSON and prune
    terms using utility scoring only.

    Utility score (higher = better term to drop):
        utility = max(0, full_gate_metric - variant_gate_metric) / (abs_damage + eps)

    Returns
    -------
    H_pruned : SparsePauliOp
    info : dict
        Summary + kept/dropped term records.
    """
    eps = 1e-12

    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    variants = data["variants"]
    variants_sorted = sorted(variants, key=lambda v: int(v["removed_index"]))

    # ----------------------------
    # Reconstruct original term list
    # ----------------------------
    rows = []
    n_qubits = None

    for v in variants_sorted:
        idx = int(v["removed_index"])
        label = v["removed_label"]
        coeff = _parse_complex_coeff(v["removed_coeff"])
        diff = float(v.get("diff", 0.0))

        if n_qubits is None:
            n_qubits = len(label)

        rows.append(
            {
                "index": idx,
                "label": label,
                "coeff": coeff,
                "diff": diff,           # treated as damage
                "variant_raw": v,
            }
        )

    # ----------------------------
    # Full (unablated) gate metric
    # ----------------------------
    metric_candidates_root = [
        ("circuit_cost", gate_metric),
        ("initial_circuit_cost", gate_metric),
        (gate_metric,),
    ]

    full_gate_metric = None
    for path in metric_candidates_root:
        val = _safe_get_nested(data, *path, default=None)
        if val is not None:
            full_gate_metric = float(val)
            break

    # ----------------------------
    # Compute per-term utility
    # ----------------------------
    for r in rows:
        abs_damage = float(r["diff"])
        r["abs_damage"] = abs_damage

        v = r["variant_raw"]

        var_gate_metric = _extract_metric(
            v,
            [
                ("circuit_cost", gate_metric),
                (gate_metric,),
                ("transpile", gate_metric),
                ("cost", gate_metric),
                ("metrics", gate_metric),
            ],
        )

        if var_gate_metric is None and prefer_depth2q:
            var_gate_metric = _extract_metric(
                v,
                [
                    ("circuit_cost", "depth_2q"),
                    ("depth_2q",),
                    ("transpile", "depth_2q"),
                ],
            )

        r["variant_gate_metric"] = float(var_gate_metric) if var_gate_metric is not None else None

        if (full_gate_metric is not None) and (r["variant_gate_metric"] is not None):
            delta_gate = float(full_gate_metric - r["variant_gate_metric"])
        else:
            delta_gate = None

        r["delta_gate_metric"] = delta_gate

        # Higher utility = better term to drop
        if delta_gate is None:
            utility = None
        else:
            utility = max(0.0, delta_gate) / (abs_damage + eps)

        r["utility"] = utility

    # ----------------------------
    # Choose terms to drop
    # ----------------------------
    candidates = []
    for r in rows:
        utility = r["utility"]
        delta_gate = r["delta_gate_metric"]

        if utility is None:
            continue
        if require_positive_2q_benefit and (delta_gate is None or delta_gate <= 0):
            continue
        if utility_tol is not None and utility < utility_tol:
            continue

        candidates.append(r)

    # Best utility first
    candidates.sort(
        key=lambda rr: (rr["utility"], rr["delta_gate_metric"] if rr["delta_gate_metric"] is not None else 0.0),
        reverse=True,
    )

    drop_indices = set()
    cumulative_damage = 0.0
    dropped_count = 0

    for r in candidates:
        damage_for_budget = r["abs_damage"]

        if (max_terms_to_drop is not None) and (dropped_count >= int(max_terms_to_drop)):
            break

        if (max_cumulative_damage is not None) and (cumulative_damage + damage_for_budget > max_cumulative_damage):
            continue

        drop_indices.add(r["index"])
        cumulative_damage += damage_for_budget
        dropped_count += 1

    # Optional forced identity drop
    if drop_identity:
        for r in rows:
            if set(r["label"]) == {"I"}:
                drop_indices.add(r["index"])

    # ----------------------------
    # Build pruned Hamiltonian
    # ----------------------------
    labels_kept = []
    coeffs_kept = []
    kept_terms = []
    dropped_terms = []

    for r in rows:
        idx = r["index"]
        label = r["label"]
        coeff = r["coeff"]

        term_record = {
            "index": idx,
            "label": label,
            "coeff": coeff,
            "diff": r["diff"],
            "abs_damage": r["abs_damage"],
            "variant_gate_metric": r["variant_gate_metric"],
            "delta_gate_metric": r["delta_gate_metric"],
            "utility": r["utility"],
        }

        if idx in drop_indices:
            dropped_terms.append(term_record)
        else:
            labels_kept.append(label)
            coeffs_kept.append(coeff)
            kept_terms.append(term_record)

    if len(labels_kept) == 0:
        # explicit zero operator on correct qubit count
        H_pruned = SparsePauliOp.from_list([("I" * n_qubits, 0.0)])
    else:
        H_pruned = SparsePauliOp.from_list(list(zip(labels_kept, coeffs_kept))).simplify(atol=0.0, rtol=0.0)

    # ----------------------------
    # Summary info
    # ----------------------------
    dropped_abs_damage = float(sum(t["abs_damage"] for t in dropped_terms))
    dropped_positive_gate_benefit = float(sum(max(0.0, t["delta_gate_metric"] or 0.0) for t in dropped_terms))
    dropped_negative_gate_effect = float(sum(min(0.0, t["delta_gate_metric"] or 0.0) for t in dropped_terms))

    info = {
        "file": str(json_path),
        "potential": data.get("potential"),
        "cutoff": data.get("cutoff"),
        "N": data.get("N"),
        "basis": data.get("basis"),
        "n_qubits": n_qubits,
        "pruning_mode": "utility_only",
        "utility_tol": utility_tol,
        "require_positive_2q_benefit": require_positive_2q_benefit,
        "gate_metric": gate_metric,
        "full_gate_metric": full_gate_metric,
        "prefer_depth2q": prefer_depth2q,
        "max_terms_to_drop": max_terms_to_drop,
        "max_cumulative_damage": max_cumulative_damage,
        "drop_identity": drop_identity,
        "n_terms_original": len(rows),
        "n_terms_kept": len(kept_terms),
        "n_terms_dropped": len(dropped_terms),
        "dropped_abs_damage_sum": dropped_abs_damage,
        "dropped_positive_gate_benefit_sum": dropped_positive_gate_benefit,
        "dropped_negative_gate_effect_sum": dropped_negative_gate_effect,
        "kept_terms": kept_terms,
        "dropped_terms": dropped_terms,
    }

    return H_pruned, info

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

def create_circuit(target, avqe_circuit, basis_state, optimization_level, num_qubits, H_pauli, num_trotter_steps, transpiler_seed):
    
    qr = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qr)

    qc.append(avqe_circuit, qr)
    # for q, bit in enumerate(basis_state):
    #     if bit == 1:
    #         qc.x(q)

    t = Parameter("t")

    evol_gate = PauliEvolutionGate(H_pauli,time=t,synthesis=LieTrotter(reps=num_trotter_steps))
    #evol_gate = PauliEvolutionGate(H_pauli, time=t, synthesis=SuzukiTrotter(order=2, reps=num_trotter_steps))
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
    min_eigenvalue = np.min(H_info["eigenvalues"]) if H_info["eigenvalues"] is not None else None

    seed = (os.getpid() * int(time.time())) % 123456789 - run_idx

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
    transpile_data = []
    

    t1 = datetime.now()
    qc_template, circuit_cost = create_circuit(target, avqe_circuit, H_info["basis"], backend_info["optimization_level"], num_qubits, H_pauli, n_steps, backend_info["transpiler_seed"])
    transpile_time = datetime.now() - t1

    transpile_info = {"num_trotters": n_steps,
                      "transpile_time": str(transpile_time),
                      "circuit_cist": circuit_cost}
    
    if log_enabled: logger.info("[Run %d]\n%s",run_idx,json.dumps(transpile_info, indent=4, default=str))

    transpile_data.append(transpile_info)

    while not converged and k <= max_k:

        step_start = datetime.now()

        if log_enabled: logger.info(f"[Run {run_idx}] Running for Krylov dimension {k}")
        print(f"Running for Krylov dimension {k}")

        t = (dt*k) #/ n_steps
        
        qc = qc_template.assign_parameters({"t": t}, inplace=False)

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

            if log_enabled: logger.info(f"[Run {run_idx}] Job ID: {job_id} - QPU usage: {usage} - Total usage {QPU_usage}")

        # trim per Krylov step
        raw_counts = counts
        raw_unique = len(raw_counts)

        fermion_rejected = 0

        t1 = datetime.now()
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

        #H_reduced = wz.reduced_sparse_matrix_from_pauli_terms(pauli_terms, top_states)
        H_reduced = wz.reduced_sparse_matrix_from_pauli_terms_fast(pauli_terms, top_states)
        H_reduce_time = datetime.now() - t1

        
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

        step_time = datetime.now() - step_start

        row = { 
            "D": k,
            "t":t,
            "num_trotter_steps": n_steps,
            "step_time": step_time,
            "H_reduce_time": str(H_reduce_time),
            "circuit_time": str(Ct),
            "shot_processing": shot_processing,
            "new_basis_count": len(samples) - pre_samples,
            "total_basis_count": len(samples),
            "H_reduced_size": H_reduced.shape,
            "reduction": (1 - (H_reduced.shape[0] / dense_H_size[0]))*100,
            "H_reduced_e": me,
            "used_dense": used_dense,
            "eigenvalue_time": str(HRt),
            "diff": np.abs(min_eigenvalue-me) if min_eigenvalue is not None else None,
            "change_from_prev": None if diff_prev == np.inf else diff_prev
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

            print(f"increasing trotter")

            t1 = datetime.now()
            qc_template, circuit_cost = create_circuit(target, avqe_circuit, H_info["basis"], backend_info["optimization_level"], num_qubits, H_pauli, n_steps, backend_info["transpiler_seed"])
            transpile_time = datetime.now() - t1

            transpile_info = {
                "num_trotters": n_steps,
                "transpile_time": str(transpile_time),
                "circuit_cist": circuit_cost
                }
            
            if log_enabled: logger.info("[Run %d]\n%s",run_idx,json.dumps(transpile_info, indent=4, default=str))

            transpile_data.append(transpile_info)

    endtime = datetime.now()

    final_data = {
            "seed": seed,
            "starttime": starttime.strftime("%Y-%m-%d_%H-%M-%S"),
            "endtime": endtime.strftime("%Y-%m-%d_%H-%M-%S"),
            "time_taken": str(endtime-starttime),
            "H_info": H_info,
            "backend_info": backend_info,
            "Ansatz_info" : Ansatz_info,
            "transpile_data":transpile_data,
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


    #with open(os.path.join(base_path, f"{potential}_L{cutoff}_{run_idx}.json"), "w") as json_file:
    with open(os.path.join(base_path, f"{potential}_{cutoff}.json"), "w") as json_file:
        json.dump(final_data, json_file, indent=4, default=str)

    if log_enabled: logger.info(f"[Run {run_idx}] Done")
    print(f"Run {run_idx} done")

if __name__ == "__main__":

    run_idx = 0#int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

    starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_enabled = True

    N = 3
    a = 1.0
    c = -0.2
    potential = "linear"
    #potential="quadratic"
    boundary_condition = 'dirichlet'

    #backend_name = 'ibm_kingston'
    #backend_name = 'ibm_torino'
    backend_name = "Aer"

    # Noise model options
    use_noise_model = 1
    gate_error=False
    readout_error=False  
    thermal_relaxation=False
    transpiler_seed=42

    noise_model_options = {
        "gate_error":gate_error,
        "readout_error":readout_error,   
        "thermal_relaxation":thermal_relaxation
        }

    shots = 100000
    optimization_level = 3
    resilience_level = 0 # 1 = readout , 2 = readout + gate

    # trimming
    conserve_fermion = True
    keep_ratio = 0.999

    dt=1.0
    max_k = 10
    tol = 1e-8

    trotter_patience = 2
    energy_patience = 3        
    max_n_steps = 5

    for cutoff in [8]:

        print(f"Running for {potential} and cutoff {cutoff}")

        tags=["Open-plan", "SKQD", f"shots:{shots}", f"{boundary_condition}", f"{potential}", f"N={N}", f"cutoff={cutoff}"]


        if potential == 'quadratic':
            folder = 'C' + str(abs(c)) + '/' + 'N'+ str(N)
        else:
            folder = 'N'+ str(N)

        #base_path = os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\Wess-Zumino\Qiskit\SKQD\NQCC\Open Plan", "Suzuki2", boundary_condition, potential, folder, str(starttime))
        base_path = os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\Wess-Zumino\Qiskit\SKQD\test3", boundary_condition, potential, folder, str(shots))
        #base_path = os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\Wess-Zumino\Qiskit\SKQD\Files\Fock+Full", boundary_condition, potential, folder, str(shots))
        os.makedirs(base_path, exist_ok=True)
        log_path = os.path.join(base_path, f"logs_{str(cutoff)}")

        if log_enabled: 
            os.makedirs(log_path, exist_ok=True)
            log_path = os.path.join(log_path, f"vqe_run_{run_idx}.log")
            logger = setup_logger(log_path, f"logger", enabled=log_enabled)
        else:
            logger = None

        if log_enabled: logger.info(f"Running VQE for {potential} potential and cutoff {cutoff}")

        try:
            H_path = os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\Wess-Zumino\Analyses\Model Checks\HamiltonianData", boundary_condition, potential, folder, f"{potential}_{cutoff}.json")
            with open(H_path, 'r') as file:
                H_data = json.load(file)

            full_pauli_coeffs = H_data["pauli_coeffs"]
            full_pauli_labels = H_data["pauli_labels"]
            num_qubits = H_data['num_qubits']
            dense_H_size = H_data['H_size']
            eigenvalues = H_data['eigenvalues']
            min_eigenvalue = np.min(eigenvalues)
        except:
            print("Hamiltonian not on file... creating")
            H_pauli, num_qubits = wz.build_wz_hamiltonian(cutoff,N,a,c=c,m=1.0,potential=potential,boundary_condition=boundary_condition)
            full_pauli_coeffs = np.real(H_pauli.coeffs).astype(float).tolist()
            full_pauli_labels = H_pauli.paulis.to_labels()
            dense_H_size = [2**num_qubits, 2**num_qubits]
            eigenvalues = None

        
        pauli_terms = list(zip(full_pauli_coeffs, full_pauli_labels))

        #ml_path = os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\Wess-Zumino\Qiskit\SKQD\ML-Testing\ML-Testing", potential, folder, f"L{cutoff}_fock.json")
        #ml_path = r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\Wess-Zumino\Qiskit\SKQD\ML-Testing\linear_N3_L16_Fock.json"
        ml_path = os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\Wess-Zumino\Qiskit\SKQD\ML-Testing\ML-Testing2", potential, folder, f"L{cutoff}_fock.json")
        #ml_path = r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\Wess-Zumino\Qiskit\SKQD\ML-Testing\ML-Testing2\linear\N3\L32_Fock.json"


        # H_pauli, info = reconstruct_sparsepauliop_from_ablation_json(
        #     ml_path,
        #     require_positive_2q_benefit=True,
        #     gate_metric="num_2q_ops",
        #     #max_terms_to_drop=50,          # start here
        #     max_cumulative_damage=0.2,   # 2% of gap
        # )

        json_path = r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\Wess-Zumino\Analyses\Model Checks\GroundstatePauliContributions\dirichlet\linear\N3\linear_8.json"
        threshold = 1e-4

        H_pauli, info = reconstruct_from_energy_contrib_threshold(
            json_path,
            threshold=threshold,
            strict=True,         # abs(contrib) > threshold
            drop_identity=False, # set True if you want to force-remove constant shift
            simplify_atol=0.0
        )

        trunc_info = {}

        # H_pauli, _ = wz.build_wz_hamiltonian(
        #     cutoff,
        #     N,
        #     a,
        #     c=c,
        #     m=1.0,
        #     potential=potential,
        #     boundary_condition=boundary_condition,
        #     include_grad=True,
        #     include_q2=True,
        #     include_qq=True, 
        #     include_wpq=True
        # )

        pauli_coeffs = np.real(H_pauli.coeffs).astype(float).tolist()
        pauli_labels = H_pauli.paulis.to_labels()

        # kept_coeffs, kept_labels, keep_idx, trunc_info = truncate_by_coeff_weight(pauli_coeffs, pauli_labels, keep_ratio=keep_ratio)
        # keep_idx = np.sort(keep_idx)
        # kept_coeffs = np.asarray(pauli_coeffs)[keep_idx]
        # kept_labels = np.asarray(pauli_labels)[keep_idx]
        # if log_enabled:
        #     logger.info(f"Truncation: {trunc_info}")

        # H_pauli = SparsePauliOp(PauliList(kept_labels.tolist()), kept_coeffs.tolist())

        

        #if log_enabled: logger.info(f"min_eigenvalue: {min_eigenvalue}")

        qps = int(np.log2(cutoff)) + 1
        fermion_qubits = [(s + 1) * qps - 1 for s in range(N)] 

        basis = [0] * num_qubits
        basis[2*qps - 1 :: 2*qps] = [1] * (N // 2)
        basis_state = basis

        num_fermions = sum(basis_state[q] for q in fermion_qubits)

        include_basis=True
        include_rys=True
        include_xxyys=True
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
        
    

        run_skqd(run_idx, H_pauli, H_info, Ansatz_info, pauli_terms, avqe_circuit, backend_info, run_info, base_path, log_enabled, logger)
    
        
