import os, json, logging, time, dataclasses
import math, itertools
import numpy as np
from collections import Counter
from datetime import datetime

from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library import PauliEvolutionGate, XXPlusYYGate
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import SamplerV2 as Sampler, QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import PauliList, SparsePauliOp

from scipy.sparse.linalg import eigsh

import wesszumino as wz

import git
repo_path = git.Repo('.', search_parent_directories=True).working_tree_dir

path = os.path.join(repo_path, r"open-apikey.json")
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



def build_evolution_synthesis(synthesis_name: str, reps: int):
    """Return a Qiskit evolution synthesis object for PauliEvolutionGate.

    Supported:
      - "lie": first-order Lie-Trotter
      - "suzuki2": second-order Suzuki-Trotter
      - "suzuki4": fourth-order Suzuki-Trotter
    """
    reps = max(1, int(reps))
    name = synthesis_name.lower().strip()
    if name in ("lie", "lietrotter", "trotter", "first"):
        return LieTrotter(reps=reps)
    if name in ("suzuki2", "suzuki_2", "second", "order2", "o2"):
        return SuzukiTrotter(order=2, reps=reps)
    if name in ("suzuki4", "suzuki_4", "fourth", "order4", "o4"):
        return SuzukiTrotter(order=4, reps=reps)
    raise ValueError(f"Unknown synthesis_name: {synthesis_name!r}")


def reps_from_dt_slice_max(t: float, dt_slice_max: float) -> int:
    """Choose reps so that the product-formula slice time |t|/reps <= dt_slice_max."""
    dt_slice_max = float(dt_slice_max)
    if dt_slice_max <= 0:
        raise ValueError("dt_slice_max must be > 0")
    return max(1, int(math.ceil(abs(float(t)) / dt_slice_max)))


def circuit_cost_metrics(qc: QuantumCircuit) -> dict:
    """Cheap hardware-cost proxies from a transpiled circuit."""
    try:
        ops = qc.count_ops()
        ops_str = {str(k): int(v) for k, v in ops.items()}
    except Exception:
        ops_str = None

    # Count 2-qubit ops by qargs length (works across bases)
    try:
        n2q = sum(1 for inst, qargs, cargs in qc.data if len(qargs) == 2)
    except Exception:
        n2q = None

    return {
        "depth": qc.depth(),
        "size": qc.size(),
        "num_2q_ops": n2q,
        "count_ops": ops_str,
    }


def create_circuit(
    backend,
    avqe_circuit,
    optimization_level,
    basis_state,
    num_qubits,
    H_pauli,
    t,
    *,
    reps: int,
    synthesis_name: str = "lie",
):
    """Build + transpile the SBQKD circuit for evolution time t.

    Returns:
        circuit_isa (QuantumCircuit): transpiled circuit
        cost (dict): depth/size/2Q counts etc.
    """
    qr = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qr)

    # If you ever want to enforce a specific computational basis state directly:
    # for q, bit in enumerate(basis_state):
    #     if bit == 1:
    #         qc.x(q)

    qc.append(avqe_circuit, qr)

    synthesis = build_evolution_synthesis(synthesis_name, reps=reps)
    evol_gate = PauliEvolutionGate(H_pauli, time=t, synthesis=synthesis)
    qc.append(evol_gate, qr)

    qc.measure_all()

    pm = generate_preset_pass_manager(target=backend.target, optimization_level=optimization_level)
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




if __name__ == "__main__":
    starttime = datetime.now()

    # -----------------------------
    # User-run configuration
    # -----------------------------
    log_enabled = False
    base_seed = (os.getpid() * int(time.time())) % 123456789

    # Problem settings (edit as needed)
    N_values = [2]          # e.g. [2, 3, 4, 5, 6]
    a = 1.0
    c = -0.2
    potential_values = ["linear"]   # e.g. ["linear", "quadratic"]
    boundary_condition = "dirichlet"
    cutoff_values = [4]

    # Trimming / postselection
    CONSERVE_FERMIONS = True
    TRIM_STATES = False
    P_KEEP = 0.995

    # Backend / execution
    backend_name = "Aer"     # e.g. "Aer" or "ibm_torino"
    use_noise_model = True   # when backend_name == "Aer"
    shots = 10000
    optimization_level = 3
    resilience_level = 2  # 1 = readout, 2 = readout + gate twirling + DD

    # Noise model toggles (used when backend_name == "Aer" and use_noise_model True)
    gate_error = True
    readout_error = True
    thermal_relaxation = True
    noise_model_options = {
        "gate_error": gate_error,
        "readout_error": readout_error,
        "thermal_relaxation": thermal_relaxation,
    }

    # SBQKD stopping
    max_k = 40
    tol = 1e-12

    # -----------------------------
    # Sweep configuration (tuning)
    # -----------------------------
    SWEEP_ENABLED = True

    # dt controls how quickly Krylov vectors change: t = dt * k
    dt_values = [0.25, 0.5]#, 1.0, 1.5, 2.0, 3.0]

    # product-formula control: choose reps so that |t|/reps <= dt_slice_max
    dt_slice_max_values = [0.5, 0.25]#, 0.125]

    # synthesis options for PauliEvolutionGate
    synthesis_names = ["lie", "suzuki2"]  # try: ["lie", "suzuki2", "suzuki4"]

    # sampling randomness: repeat a few seeds for Aer/noise-model runs
    sweep_seeds = [base_seed]#, base_seed + 1, base_seed + 2]

    # -----------------------------
    # Repo path resolution
    # -----------------------------
    repo = git.Repo(".", search_parent_directories=True)
    repo_path = repo.working_tree_dir

    for potential in potential_values:
        for N in N_values:
            for cutoff in cutoff_values:

                # Tagging (kept for IBM Runtime runs)
                base_tags = [
                    "SBQKD",
                    f"shots:{shots}",
                    f"{boundary_condition}",
                    f"{potential}",
                    f"N={N}",
                    f"cutoff={cutoff}",
                ]

                # Output folder structure (Windows-safe)
                if potential == "quadratic":
                    folder = f"C{abs(c)}/N{N}"
                else:
                    folder = f"N{N}"

                base_path = os.path.join(
                    repo_path,
                    r"SUSY\Wess-Zumino\Qiskit\SBQKD\Files\Optimize",
                    boundary_condition,
                    potential,
                    folder,
                )
                os.makedirs(base_path, exist_ok=True)

                # Load Hamiltonian (Pauli terms + metadata)
                H_path = os.path.join(
                    repo_path,
                    r"SUSY\Wess-Zumino\Analyses\Model Checks\HamiltonianData",
                    boundary_condition,
                    potential,
                    folder,
                    f"{potential}_{cutoff}.json",
                )
                with open(H_path, "r") as file:
                    H_data = json.load(file)

                pauli_coeffs = H_data["pauli_coeffs"]
                pauli_labels = H_data["pauli_labels"]
                H_pauli = SparsePauliOp(PauliList(pauli_labels), pauli_coeffs)
                pauli_terms = list(zip(pauli_coeffs, pauli_labels))

                num_qubits = H_data["num_qubits"]
                dense_H_size = H_data["H_size"]
                eigenvalues = H_data["eigenvalues"]
                min_eigenvalue = float(np.min(eigenvalues))

                # Best computational basis seed state for the AVQE pattern ansatz
                basis_state = H_data["best_basis_state"][::-1]

                # Fermion parity postselection support
                qps = int(num_qubits / N)
                fermion_qubits = [(s + 1) * qps - 1 for s in range(N)]
                num_fermions = int(sum(basis_state[q] for q in fermion_qubits))

                # Build your initial ansatz (same across sweep settings)
                avqe_circuit = wz.build_avqe_pattern_ansatz(
                    N=N,
                    cutoff=cutoff,
                    include_basis=True,
                    include_rys=True,
                    include_xxyys=True,
                )

                # Determine which parameter sets to run
                if SWEEP_ENABLED:
                    param_sets = list(itertools.product(synthesis_names, dt_values, dt_slice_max_values, sweep_seeds))
                else:
                    # Backward-ish compatible single run (pick one)
                    param_sets = [("lie", 3.0, 3.0 / 2.0, base_seed)]  # dt_slice_max ≈ dt/n_steps (old n_steps=2)

                for synthesis_name, dt, dt_slice_max, seed in param_sets:

                    run_tag = f"synth={synthesis_name}_dt={dt}_dtslice={dt_slice_max}_seed={seed}"
                    tags = base_tags + [run_tag]

                    run_dir = os.path.join(base_path, "sweeps", run_tag)
                    os.makedirs(run_dir, exist_ok=True)

                    log_path = os.path.join(run_dir, "sbqkd_run.log")
                    logger = setup_logger(log_path, "logger", enabled=log_enabled)
                    if log_enabled:
                        logger.info(f"Run tag: {run_tag}")
                        logger.info(f"min_eigenvalue: {min_eigenvalue}")

                    # Backend + sampler (re-created per seed so Aer/noise is reproducible)
                    backend, sampler = get_backend(
                        backend_name,
                        use_noise_model,
                        noise_model_options,
                        resilience_level,
                        seed,
                        shots,
                        tags,
                    )

                    sampler_options = dataclasses.asdict(sampler.options)

                    converged = False
                    samples = Counter()
                    prev_energy = np.inf

                    all_data = []
                    all_counts = []
                    all_energies = []
                    job_info = {}

                    k = 1
                    while (not converged) and (k <= max_k):

                        if log_enabled:
                            logger.info(f"Running for Krylov dimension {k}")
                        print(f"[{run_tag}] Running for Krylov dimension {k}")

                        t = float(dt) * k
                        reps = reps_from_dt_slice_max(t, dt_slice_max)

                        qc, qc_cost = create_circuit(
                            backend,
                            avqe_circuit,
                            optimization_level,
                            basis_state,
                            num_qubits,
                            H_pauli,
                            t,
                            reps=reps,
                            synthesis_name=synthesis_name,
                        )

                        t1 = datetime.now()
                        counts, job_id, job_metrics = get_counts(sampler, qc, shots)
                        Ct = datetime.now() - t1

                        if backend_name != "Aer":
                            job_info[job_id] = job_metrics

                            # Optional: sum QPU seconds if present
                            try:
                                jobs = job_info.values()
                                QPU_usage = 0.0
                                for j in jobs:
                                    if j and "usage" in j and "quantum_seconds" in j["usage"]:
                                        QPU_usage += float(j["usage"]["quantum_seconds"])
                            except Exception:
                                QPU_usage = None
                        else:
                            QPU_usage = None

                        # ---------- postprocess counts ----------
                        raw_counts = dict(counts)
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
                            post_rejected += int(rej_w)

                        if TRIM_STATES:
                            counts, kept, rej = trim_counts(counts, p_keep=P_KEEP)
                            trim_rejected += int(rej)

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

                        # Update global sample counter with *kept* counts
                        samples.update(counts)

                        sorted_states = sorted(samples.items(), key=lambda x: x[1], reverse=True)
                        top_states = [s for s, _c in sorted_states]
                        all_counts.append(dict(sorted_states))

                        # Reduced Hamiltonian + eigen solve
                        H_reduced = wz.reduced_sparse_matrix_from_pauli_terms(pauli_terms, top_states)

                        t1 = datetime.now()
                        me = eigsh(H_reduced, k=1, which="SA", return_eigenvectors=False)[0].real
                        HRt = datetime.now() - t1

                        diff_prev = float(np.abs(prev_energy - me))

                        row = {
                            "D": k,
                            "t": t,
                            "dt": float(dt),
                            "dt_slice_max": float(dt_slice_max),
                            "reps": int(reps),
                            "synthesis": synthesis_name,
                            "circuit_time": str(Ct),
                            "circuit_cost": qc_cost,
                            "num_samples": len(samples),
                            "shot_processing": shot_processing,
                            "H_reduced_size": H_reduced.shape,
                            "reduction": (1 - (H_reduced.shape[0] / dense_H_size[0])) * 100,
                            "H_reduced_e": float(me),
                            "eigenvalue_time": str(HRt),
                            "diff_to_exact": float(np.abs(min_eigenvalue - me)),
                            "change_from_prev": None if prev_energy is np.inf else diff_prev,
                        }

                        all_data.append(row)
                        all_energies.append(float(me))

                        # Convergence criterion (same as your original): stop when energy stops changing
                        converged = True if diff_prev < tol else False

                        if (not converged) and (k == max_k):
                            print(f"[{run_tag}] max_k reached")
                            break
                        elif not converged:
                            prev_energy = float(me)
                            k += 1
                        else:
                            print(f"[{run_tag}] Converged")

                    endtime = datetime.now()

                    final_data = {
                        "starttime": starttime.strftime("%Y-%m-%d_%H-%M-%S"),
                        "endtime": endtime.strftime("%Y-%m-%d_%H-%M-%S"),
                        "potential": potential,
                        "boundary_condition": boundary_condition,
                        "cutoff": cutoff,
                        "N": N,
                        "a": a,
                        "c": c,
                        "num_qubits": num_qubits,
                        "dense_H_size": dense_H_size,
                        "min_eigenvalue": min_eigenvalue,
                        "exact_eigenvalues": eigenvalues,
                        "backend_name": backend_name,
                        "use_noise_model": use_noise_model if backend_name == "Aer" else None,
                        "noise_model_options": noise_model_options if (backend_name == "Aer" and use_noise_model) else None,
                        "shots": shots,
                        "optimization_level": optimization_level,
                        "resilience_level": resilience_level if backend_name != "Aer" else None,
                        "sweep_params": {
                            "synthesis": synthesis_name,
                            "dt": float(dt),
                            "dt_slice_max": float(dt_slice_max),
                            "seed": int(seed),
                            "max_k": int(max_k),
                            "tol": float(tol),
                        },
                        "sampler_options": sampler_options,
                        "num_jobs": len(job_info) if backend_name != "Aer" else None,
                        "QPU_usage": QPU_usage if backend_name != "Aer" else None,
                        "job_info": job_info,
                        "all_run_data": all_data,
                        "all_counts": all_counts,
                        "all_energies": all_energies,
                    }

                    out_path = os.path.join(run_dir, f"{potential}_{cutoff}.json")
                    with open(out_path, "w") as json_file:
                        json.dump(final_data, json_file, indent=4, default=str)

                    print(f"[{run_tag}] Done -> {out_path}")

