import numpy as np
import nevergrad as ng
import logging, json, time, os, dataclasses
from datetime import datetime

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_ibm_runtime import EstimatorV2 as Estimator, QiskitRuntimeService, Session
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_aer.noise import NoiseModel

from susy_qm import calculate_Hamiltonian, ansatze

import git
repo_path = git.Repo('.', search_parent_directories=True).working_tree_dir


#path = r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\open-apikey.json"
path = r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\apikey.json"
with open(path, encoding="utf-8") as f:
    api_key = json.load(f).get("apikey")

IBM_QUANTUM_API_KEY = api_key
#ibm_instance_crn = "crn:v1:bluemix:public:quantum-computing:us-east:a/3ff62345f67c45e48e47a7f57d2f39f5:83214c75-88ab-4e55-8a87-6502ecc7cc9b::" #US
ibm_instance_crn = "crn:v1:bluemix:public:quantum-computing:us-east:a/d4f95db0515b47b7ba61dba8a424f873:ed0704ac-ad7d-4366-9bcc-4217fb64abd1::"
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


def _tensor_power(op, k):
    """Return op^(⊗k); for k=0 return None."""
    if k == 0:
        return None
    out = op
    for _ in range(k - 1):
        out = out.tensor(op)
    return out


def build_multi_copy_circuit(num_blocks, num_qubits_single, ansatz):
    """
    Build a packed circuit with `num_blocks` disjoint copies of a single-site ansatz.

    Returns:
        full      : QuantumCircuit with num_blocks * num_qubits_single qubits
        num_params: number of parameters per copy
    """
    # One-copy ansatz block
    block = ansatze.pl_to_qiskit(ansatz, num_qubits=num_qubits_single, reverse_bits=True)
    block_params = list(block.parameters)
    num_params = len(block_params)

    param = ParameterVector("θ", num_blocks * num_params)
    full = QuantumCircuit(num_blocks * num_qubits_single)

    for j in range(num_blocks):
        start = j * num_params
        stop = (j + 1) * num_params
        param_block = param[start:stop]

        mapping = {p: param_block[i] for i, p in enumerate(block_params)}
        block_j = block.assign_parameters(mapping, inplace=False)

        qubits_j = list(range(j * num_qubits_single, (j + 1) * num_qubits_single))
        full.compose(block_j, qubits=qubits_j, inplace=True)

    return full, num_params


def build_multi_copy_observables(num_blocks, num_qubits_single, H_single_sp):
    """
    Given a single-copy SparsePauliOp H_single_sp, build [H0, ..., H_{num_blocks-1}]
    each acting on its own block of `num_qubits_single` in a num_blocks * num_qubits_single register.
    """
    I = SparsePauliOp.from_list([("I" * num_qubits_single, 1.0)])
    Hs = []

    for j in range(num_blocks):
        left_blocks = num_blocks - j - 1
        right_blocks = j

        left = _tensor_power(I, left_blocks)
        right = _tensor_power(I, right_blocks)

        pieces = [p for p in (left, H_single_sp, right) if p is not None]
        Hj = pieces[0]
        for p in pieces[1:]:
            Hj = Hj.tensor(p)
        Hs.append(Hj)

    return Hs


def apply_relu(energy, eps, lam, p):
    neg = max(0.0, -(energy + eps))
    return energy + lam * (neg ** p)

def evaluate_energies(full, Hs, estimator, param_flat, eps, lam, p, shots, backend_name):

    #print("Evaluating energies")

    precision = 0.0 if shots is None else 1.0/np.sqrt(shots)

    """
    One Estimator call:
        (full, Hs, [param_flat]) → energies for each active block.
    """
    pubs = [(full, Hs, [param_flat])]

    if backend_name == "Aer":
        job = estimator.run(pubs, precision=precision)
        job_id = job.job_id()
    else:
        job = estimator.run(pubs)
        job_status = job.status()
        job_id = job.job_id()

        if job_status == 'ERROR':
            logger.info(f"Job Id: {job_id} failes with status {job_status}, retrying")
            job = estimator.run(pubs)    
            job_id = job.job_id()
            job_status = job.status()

    #print("Estimator job id:", job_id)
    #print("Initial status:", job.status())

    result = job.result()
    Es = np.array(result[0].data.evs, dtype=float)

    Es = [apply_relu(E, eps, lam, p) for E in Es]

    try:
        job_metrics = job.metrics()
        usage = job_metrics.get("usage")
        QPU_usage = float(usage["seconds"])
        logger.info(f"Job Id: {job_id} used {QPU_usage}s QPU time")
    except AttributeError:
        #print("No usage/metrics available for this estimator type.")
        job_metrics = None

    return Es, job_id, job_metrics


# ---------------- Initial packed problem ----------------
def build_active_problem(H_single_sp, best_params, num_qubits_single, active_ids, ng_budget):
    """Build circuit, observables and optimizers for the current active set."""
    num_active = len(active_ids)
    full, num_params = build_multi_copy_circuit(num_active, num_qubits_single, ansatz)
    Hs_active = build_multi_copy_observables(num_active, num_qubits_single, H_single_sp)

    # One Nevergrad optimizer per active copy, starting from its best_param
    optimizers = []
    for cid in active_ids:
        x0_k = best_params[cid]
        low, high = 0.0, 2.0*np.pi
        parametrization = ng.p.Array(init=x0_k).set_bounds(low, high)


        # opt = ng.optimizers.MultiCobyla(
        #     parametrization=parametrization,
        #     budget=ng_budget,
        #     num_workers=1,
        # )
        opt = ng.optimizers.NGOpt(
            parametrization=parametrization,
            budget=ng_budget,  # upper bound on total tell() calls
            num_workers=1,
        )
        optimizers.append(opt)

    return full, Hs_active, optimizers, num_params


def create_isas(backend, full, Hs_active, optimization_level):

    if backend_name != "Aer" or use_noise_model:
        target = backend.target
        pm = generate_preset_pass_manager(target=target, optimization_level=optimization_level)
        ansatz_isa = pm.run(full)

        layout = getattr(ansatz_isa, "layout", None)
        hamiltonian_isa = [H.apply_layout(layout) for H in Hs_active] if layout else Hs_active

        #if log_enabled: logger.info(f"Hamiltonian: {hamiltonian_isa}")
    else:
        ansatz_isa = full
        hamiltonian_isa = Hs_active

    return ansatz_isa, hamiltonian_isa


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

            print(noise_model.noise_instructions)


        else:
            backend = AerSimulator(method="statevector")
    else:
        backend = service.backend(backend_name)


    if backend_name == "Aer":

        estimator = AerEstimator(
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
        estimator = Estimator(mode=backend)
        estimator.options.environment.job_tags = tags
        estimator.options.default_shots = shots
        estimator.options.resilience_level = resilience_level

    return backend, estimator


def run_vqe(*,
            backend_name, 
            use_noise_model, 
            noise_model_options,
            H_single_sp, 
            resilience_level, 
            optimization_level, 
            num_params_single, 
            num_qubits_single, 
            num_copies, 
            shots,
            cost_function,
            optimizer_info,
            tags
            ):
    


    min_iter, max_iter, ng_budget, improve_tol_abs, improve_tol_rel, patience = (optimizer_info[k]for k in ("min_iter", "max_iter", "ng_budget", "improve_tol_abs", "improve_tol_rel", "patience"))
    p, lam, eps = (cost_function[k]for k in ("p", "lam", "eps"))
    
    seed = (os.getpid() * int(time.time())) % 123456789
    run_info["seed"] = seed

    # One x0 per global copy
    rng = np.random.default_rng(seed)
    x0_global = [rng.random(size=num_params_single)*2*np.pi for _ in range(num_copies)]

    best_params = np.array(x0_global, dtype=float)          # shape (num_copies, num_params_single)
    best_E = np.full(num_copies, np.nan, dtype=float)  
    prev_E = np.full(num_copies, np.nan, dtype=float)
    stale_iters = np.zeros(num_copies, dtype=int)  # per-copy counters

    active_ids = list(range(num_copies))             # global indices of active copies
    iters_per_copy = np.zeros(num_copies, dtype=int)
    converged_flags = np.zeros(num_copies, dtype=bool)

    full, Hs_active, optimizers_active, num_params = build_active_problem(H_single_sp, best_params, num_qubits_single, active_ids, ng_budget)

    logger.info("Initial active copies: %s", active_ids)
    logger.info("Params per copy: %d", num_params)

    backend, estimator = get_backend(backend_name, use_noise_model, noise_model_options, resilience_level, seed, shots, tags)
    full_isa, Hs_active_isa = create_isas(backend, full, Hs_active, optimization_level)

    print(full_isa.draw("text"))

    estimator_options = dataclasses.asdict(estimator.options)
    if log_enabled: logger.info(json.dumps(estimator_options, indent=4, default=str))

    # ---------------- Optimization loop with shrinking ----------------
    print("Starting VQE")
    vqe_start = datetime.now()
    iter_idx = 0
    iter_times = []

    job_info = {}

    while iter_idx < max_iter and active_ids:

        iter_start = datetime.now()

        if iter_idx % 50 == 0:
            print(f"iter {iter_idx} started")

        num_active = len(active_ids)
        #print(f"  num_active = {num_active}")
        #print(f"  optimizers_active length = {len(optimizers_active)}")

        xs = []
        params = []
        for j, opt in enumerate(optimizers_active):
            #print(f"  [iter {iter_idx}] about to ask optimizer {j}")
            x = opt.ask()
            #print(f"  [iter {iter_idx}] got x from optimizer {j}: shape={np.shape(x.value)}")
            xs.append(x)
            param_j = np.asarray(x.value, dtype=float)
            #print(f"  [iter {iter_idx}] param_j shape={param_j.shape}")
            params.append(param_j)

        #print(f"  [iter {iter_idx}] finished all asks, len(params)={len(params)}")

        param_active = np.concatenate(params, axis=0)
        #print(f"  [iter {iter_idx}] param_active shape={param_active.shape}")

        Es_active, job_id, job_metrics = evaluate_energies(full_isa, Hs_active_isa, estimator, param_active, eps, lam, p, shots, backend_name)

        job_info[job_id] = job_metrics

        logger.info(
            f"[iter {iter_idx:03d}] active_ids={active_ids}, "
            f"Es_active={Es_active}"
        )

        # 3. per-copy updates and convergence check
        newly_converged_global = []

        for j, cid in enumerate(active_ids):
            E_k = float(Es_active[j])
            iters_per_copy[cid] += 1

            # --- first-time initialization for this copy ---
            if not np.isfinite(best_E[cid]):
                # First energy we've seen for this copy: just record it, no convergence logic yet
                best_E[cid] = E_k
                best_params[cid, :] = params[j]
                stale_iters[cid] = 0
                prev_E[cid] = E_k
                continue

            # --- has this copy improved its *best known* energy? ---
            # dynamic tolerance: absolute + relative
            # guard against nan/inf explicitly
            base = abs(best_E[cid]) if np.isfinite(best_E[cid]) else 0.0
            tol_k = improve_tol_abs + improve_tol_rel * base

            if E_k < best_E[cid] - tol_k:
                # significant improvement
                best_E[cid] = E_k
                best_params[cid, :] = params[j]
                stale_iters[cid] = 0
            else:
                # no significant improvement
                stale_iters[cid] += 1

            prev_E[cid] = E_k

            # --- convergence test for this copy ---
            if (iter_idx >= min_iter) and (stale_iters[cid] >= patience):
                newly_converged_global.append(cid)
                converged_flags[cid] = True
                logger.info(
                    f"  -> copy {cid} converged: "
                    f"best_E={best_E[cid]:.12f}, "
                    f"stale_iters={stale_iters[cid]}, "
                    f"tol_k={tol_k:.3e}"
                )


        # 4. tell each active optimizer its scalar loss
        for j, opt in enumerate(optimizers_active):
            opt.tell(xs[j], float(Es_active[j]))

        # 5. shrink the problem if some copies converged
        if newly_converged_global:
            # remove converged copies from the active set
            converged_set = set(newly_converged_global)
            active_ids = [cid for cid in active_ids if cid not in converged_set]

            logger.info("  shrinking active set, new active_ids: %s", active_ids)

            # rebuild packed circuit / observables / optimizers
            if active_ids:
                full, Hs_active, optimizers_active, num_params = build_active_problem(H_single_sp, best_params, num_qubits_single, active_ids, ng_budget)
                full_isa, Hs_active_isa = create_isas(backend, full, Hs_active, optimization_level)

        # 6. stop if no active copies remain
        if not active_ids:
            logger.info(f"All copies converged by iter {iter_idx}")
            break

        iter_idx += 1

        iter_end = datetime.now()
        iter_time = iter_end - iter_start
        iter_times.append(iter_time)

    vqe_end = datetime.now()
    vqe_time = vqe_end - vqe_start 

    # ---------------- Final verification ----------------
    logger.info("\n=== Final best parameters per global copy ===")
    for cid in range(num_copies):
        logger.info(f"Copy {cid}: best_E={best_E[cid]}, best_param={best_params[cid]}")

    # Total QPU usage
    if backend_name != "Aer":
        jobs = job_info.values()
        QPU_usage = 0.0
        for job in jobs:
            usage = job.get("usage")
            QPU_usage += float(usage["seconds"])

    logger.info(f"Job ID: {job_id} - QPU usage: {QPU_usage}")

    vqe_results = {
        "vqe_start": str(vqe_start),
        "vqe_end": str(vqe_end),
        "vqe_time": str(vqe_time),
        "energies": best_E.tolist(),
        "params": best_params.tolist(),
        "active_ids": active_ids,
        "iters": iters_per_copy.tolist(),
        "iter_times": [str(t) for t in iter_times],
        "converged": converged_flags.tolist(),
        "num_jobs": len(jobs) if backend_name != "Aer" else None,
        "QPU_usage": QPU_usage if backend_name != "Aer" else None,
        "job_info": job_info,
        "estimator_options": estimator_options
    }

    return vqe_results



    
if __name__ == "__main__":

    starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_enabled=True

    # ---------------- Problem setup ----------------
    potential = "DW"
    cutoff = 4
    num_copies = 30   # total copies (optimizers) you want initially

    #backend_name = 'ibm_kingston'
    #backend_name = 'ibm_fez'
    backend_name = 'ibm_torino'
    #backend_name = "ibm_strasbourg"
    #backend_name = "Aer"

    # Noise model options
    use_noise_model = 0
    gate_error=False
    readout_error=False  
    thermal_relaxation=False

    shots = 4096
    optimization_level = 3
    resilience_level = 2

    # ---------------- Global trackers per copy ----------------
    # Convergence hyperparameters
    improve_tol_abs = 1e-8      # absolute improvement threshold
    improve_tol_rel = 1e-3      # relative threshold (0.1% of |best_E|)
    patience        = 50        # how many iters of no improvement before freezing
    min_iter      = 50        # don't even consider convergence before this
    max_iter = 100
    ng_budget = 10000


    lam = 15
    p = 2

    if potential == "AHO":
        i = np.log2(cutoff)
        factor = 2**(((i-1)*i)/2)
        eps = 0.5 / factor
    else:
        eps = 0

    base_path = os.path.join(repo_path,r"SUSY\SUSY QM\Qiskit\NQCC Access\Q4_2025\Files\ResilienceLevelTesting\DW4\Level2", f"{potential}{cutoff}")
    os.makedirs(base_path, exist_ok=True)

    log_path = os.path.join(base_path, f"logs_{str(cutoff)}")

    if log_enabled: 
        os.makedirs(log_path, exist_ok=True)
        log_path = os.path.join(log_path, f"vqe_run.log")
        logger = setup_logger(log_path, f"logger", enabled=log_enabled)

    if log_enabled: logger.info(f"Running VQE for {potential} potential and cutoff {cutoff}")

    # Single-copy Hamiltonian (matrix) and convert to SparsePauliOp
    H_matrix = calculate_Hamiltonian(cutoff, potential)
    num_qubits_single = int(1 + np.log2(cutoff))
    H_single_sp = SparsePauliOp.from_operator(H_matrix)

    eigenvalues = np.sort(np.linalg.eigvals(H_matrix))[:4]

    # Single-copy ansatz object
    ansatze_type = "exact"
    if potential == "QHO":
        ansatz_name = f"CQAVQE_QHO_{ansatze_type}"
    elif (potential != "QHO") and (cutoff <= 64):
        ansatz_name = f"CQAVQE_{potential}{cutoff}_{ansatze_type}"
    else:
        ansatz_name = f"CQAVQE_{potential}16_{ansatze_type}"

    ansatz = ansatze.get(ansatz_name)
    num_params_single = ansatz.n_params

    #tags=["Open-access", ansatz_name, f"shots:{shots}", "Parallel Test", f"{num_copies} copies"]
    tags=["NQCC-Q4", ansatz_name, f"shots:{shots}", "Resilience Test", f"Resilience={resilience_level}", f"{num_copies} copies"]

    optimizer_info = {
            "name": "NGOpt",
            "min_iter": min_iter,
            "max_iter": max_iter,
            "ng_budget": ng_budget,
            "improve_tol_abs": improve_tol_abs,
            "improve_tol_rel": improve_tol_rel,
            "patience": patience
        }
    
    cost_function = {
            "type": "small negatives",
            "p":p,
            "lam":lam,
            "eps":eps
        }
    
    noise_model_options = {
        "gate_error":gate_error,
        "readout_error":readout_error,   
        "thermal_relaxation":thermal_relaxation
        }

    run_info = {"backend_name":backend_name,
                "use_noise_model": use_noise_model,
                "noise_model_options": noise_model_options if use_noise_model else None,
                "H_single_sp": H_single_sp,
                "num_qubits_single": num_qubits_single,
                "num_params_single": num_params_single,
                "num_copies": num_copies,
                "shots": shots,
                "resilience_level": resilience_level,
                "optimization_level": optimization_level,
                "cost_function": cost_function,
                "optimizer_info": optimizer_info,
                "tags":tags
                }
    
    print(json.dumps(run_info, indent=4, default=str))
    logger.info(json.dumps(run_info, indent=4, default=str))

    results = run_vqe(**run_info)

    end_time=datetime.now()

    # Save run
    run = {
        "starttime": starttime,
        "endtime": end_time.strftime("%Y-%m-%d_%H-%M-%S"),
        "potential": potential,
        "cutoff": cutoff,
        "exact_eigenvalues": [x.real.tolist() for x in eigenvalues],
        "ansatz": ansatz_name,
        "run_info": run_info,
        "results": results
    }

    # Save the variable to a JSON file
    path = os.path.join(base_path, "{}_{}.json".format(potential, cutoff))
    with open(path, "w") as json_file:
        json.dump(run, json_file, indent=4, default=str)

    print("Done")
