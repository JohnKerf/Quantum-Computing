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


path = r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\open-apikey.json"
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


def _tensor_power(op, k):
    """Return op^(⊗k); for k=0 return None."""
    if k == 0:
        return None
    out = op
    for _ in range(k - 1):
        out = out.tensor(op)
    return out


def build_multi_copy_circuit(num_blocks, num_qubits, ansatz):
    """
    Build a packed circuit with `num_blocks` disjoint copies of a single-site ansatz.

    Returns:
        full      : QuantumCircuit with num_blocks * num_qubits qubits
        num_params: number of parameters per copy
    """
    # One-copy ansatz block
    block = ansatze.pl_to_qiskit(ansatz, num_qubits=num_qubits, reverse_bits=True)
    block_params = list(block.parameters)
    num_params = len(block_params)

    theta = ParameterVector("θ", num_blocks * num_params)
    full = QuantumCircuit(num_blocks * num_qubits)

    for j in range(num_blocks):
        start = j * num_params
        stop = (j + 1) * num_params
        theta_block = theta[start:stop]

        mapping = {p: theta_block[i] for i, p in enumerate(block_params)}
        block_j = block.assign_parameters(mapping, inplace=False)

        qubits_j = list(range(j * num_qubits, (j + 1) * num_qubits))
        full.compose(block_j, qubits=qubits_j, inplace=True)

    return full, num_params


def build_multi_copy_observables(num_blocks, num_qubits, H_single_sp):
    """
    Given a single-copy SparsePauliOp H_single_sp, build [H0, ..., H_{num_blocks-1}]
    each acting on its own block of `num_qubits` in a num_blocks * num_qubits register.
    """
    I = SparsePauliOp.from_list([("I" * num_qubits, 1.0)])
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

def evaluate_energies(full, Hs, estimator, Theta_flat, eps, lam, p):

    print("Evaluating energies")

    """
    One Estimator call:
        (full, Hs, [Theta_flat]) → energies for each active block.
    """
    pubs = [(full, Hs, [Theta_flat])]
    job = estimator.run(pubs)

    print("Estimator job id:", job.job_id())
    print("Initial status:", job.status())

    result = job.result()
    Es = np.array(result[0].data.evs, dtype=float)

    Es = [apply_relu(E, eps, lam, p) for E in Es]
    
    return Es


def main():

    log_enabled=True

    # ---------------- Problem setup ----------------
    potential = "QHO"
    cutoff = 2
    num_total_copies = 50   # total copies (optimizers) you want initially

    #backend_name = 'ibm_kingston'
    #backend_name = 'ibm_fez'
    backend_name = 'ibm_torino'
    #backend_name = "ibm_strasbourg"
    #backend_name = "Aer"
    #backend_name = "SV-Estimator"

    use_noise_model = 0
    shots = 4096
    optimization_level = 3
    resilience_level = 0

    lam = 15
    p = 2

    if potential == "AHO":
        i = np.log2(cutoff)
        factor = 2**(((i-1)*i)/2)
        eps = 0.5 / factor
    else:
        eps = 0

    starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_path = os.path.join(repo_path,r"SUSY\SUSY QM\Qiskit\Paralleltesting", backend_name, potential, str(starttime))
    os.makedirs(base_path, exist_ok=True)

    log_path = os.path.join(base_path, f"logs_{str(cutoff)}")

    if log_enabled: 
        os.makedirs(log_path, exist_ok=True)
        log_path = os.path.join(log_path, f"vqe_run.log")
        logger = setup_logger(log_path, f"logger", enabled=log_enabled)

    # Single-copy Hamiltonian (matrix) and convert to SparsePauliOp
    H_matrix = calculate_Hamiltonian(cutoff, potential)
    num_qubits = int(1 + np.log2(cutoff))
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


    # ---------------- Global trackers per copy ----------------
    # Convergence hyperparameters
    improve_tol_abs = 1e-8      # absolute improvement threshold
    improve_tol_rel = 1e-3      # relative threshold (0.1% of |best_E|)
    patience        = 50        # how many rounds of no improvement before freezing
    min_rounds      = 30        # don't even consider convergence before this
    max_rounds = 200
    ng_budget = 200

    rng = np.random.default_rng(123)

    # One x0 per global copy
    num_params_single = ansatz.n_params
    x0_global = [rng.random(size=num_params_single)*2*np.pi for _ in range(num_total_copies)]

    best_theta = np.array(x0_global, dtype=float)          # shape (num_total_copies, num_params_single)
    best_E = np.full(num_total_copies, np.nan, dtype=float)  
    prev_E = np.full(num_total_copies, np.nan, dtype=float)
    stale_iters = np.zeros(num_total_copies, dtype=int)  # per-copy counters


    active_ids = list(range(num_total_copies))             # global indices of active copies


    tags=["Open-access", ansatz_name, f"shots:{shots}", "Parallel Test", f"{num_total_copies} copies"]

    run_info = {"backend":backend_name,
                "use_noise_model": use_noise_model,
                "Potential":potential,
                "cutoff": cutoff,
                "num_qubits": num_qubits,
                "num_params": num_params_single,
                "shots": shots,
                "optimization_level": optimization_level,
                "resilience_level": resilience_level,
                "lam": lam,
                "p":p,
                "eps":eps,
                "num_copies": num_total_copies,
                "min_rounds": min_rounds,
                "max_rounds": max_rounds,
                "ng_nudget": ng_budget,
                "patience": patience,
                "improve_tol_abs": improve_tol_abs,
                "improve_tol_rel": improve_tol_rel,
                "ansatz_name": ansatz_name,
                "path": base_path,
                "tags":tags
                }
    
    print(json.dumps(run_info, indent=4, default=str))
    logger.info(json.dumps(run_info, indent=4, default=str))



    # ---------------- Initial packed problem ----------------
    def build_active_problem(active_ids):
        """Build circuit, observables and optimizers for the current active set."""
        num_active = len(active_ids)
        full, num_params = build_multi_copy_circuit(num_active, num_qubits, ansatz)
        Hs_active = build_multi_copy_observables(num_active, num_qubits, H_single_sp)

        # One Nevergrad optimizer per active copy, starting from its best_theta
        optimizers = []
        for cid in active_ids:
            x0_k = best_theta[cid]
            low, high = 0.0, 2.0*np.pi
            parametrization = ng.p.Array(init=x0_k).set_bounds(low, high)
            opt = ng.optimizers.NGOpt(
                parametrization=parametrization,
                budget=ng_budget,  # upper bound on total tell() calls
                num_workers=1,
            )
            optimizers.append(opt)

        return full, Hs_active, optimizers, num_params

    full, Hs_active, optimizers_active, num_params = build_active_problem(active_ids)

    #logger.info(full.draw("text"))
    logger.info("Initial active copies: %s", active_ids)
    logger.info("Params per copy: %d", num_params)


    if backend_name == "Aer":

        if run_info['use_noise_model']:
            real_backend = service.backend("ibm_kingston")
            #real_backend = service.backend("ibm_strasbourg")
            noise_model = NoiseModel.from_backend(real_backend)
            backend = AerSimulator(noise_model=noise_model)
           
        else:
            backend = AerSimulator(method="statevector")

    elif backend_name == "SV-Estimator":
        pass

    else:
        backend = service.backend(backend_name)


    def create_isas(full, Hs_active):
        if (backend_name not in ["Aer"]) or use_noise_model:
            target = backend.target
            pm = generate_preset_pass_manager(target=target, optimization_level=run_info["optimization_level"])
            ansatz_isa = pm.run(full)

            layout = getattr(ansatz_isa, "layout", None)
            hamiltonian_isa = [H.apply_layout(layout) for H in Hs_active] if layout else Hs_active

            if log_enabled: logger.info(f"Hamiltonian: {hamiltonian_isa}")
        else:
            ansatz_isa = full
            hamiltonian_isa = Hs_active

        return ansatz_isa, hamiltonian_isa
    
    full, Hs_active = create_isas(full, Hs_active)

    print(full.draw("text"))

    seed = (os.getpid() * int(time.time())) % 123456789
    run_info["seed"] = seed


        
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
        estimator.options.resilience_level = run_info["resilience_level"]

    if log_enabled: logger.info(json.dumps(dataclasses.asdict(estimator.options), indent=4, default=str))

    # ---------------- Optimization loop with shrinking ----------------
    print("Starting VQE")
    vqe_start = datetime.now()
    round_idx = 0
    round_times = []
    while round_idx < max_rounds and active_ids:

        round_start = datetime.now()

        print(f"round {round_idx} started")

        num_active = len(active_ids)
        print(f"  num_active = {num_active}")
        print(f"  optimizers_active length = {len(optimizers_active)}")

        xs = []
        thetas = []
        for j, opt in enumerate(optimizers_active):
            #print(f"  [round {round_idx}] about to ask optimizer {j}")
            x = opt.ask()
            #print(f"  [round {round_idx}] got x from optimizer {j}: shape={np.shape(x.value)}")
            xs.append(x)
            theta_j = np.asarray(x.value, dtype=float)
            #print(f"  [round {round_idx}] theta_j shape={theta_j.shape}")
            thetas.append(theta_j)

        print(f"  [round {round_idx}] finished all asks, len(thetas)={len(thetas)}")

        Theta_active = np.concatenate(thetas, axis=0)
        print(f"  [round {round_idx}] Theta_active shape={Theta_active.shape}")

        Es_active = evaluate_energies(full, Hs_active, estimator, Theta_active, eps, lam, p)

        logger.info(
            f"[round {round_idx:03d}] active_ids={active_ids}, "
            f"Es_active={Es_active}"
        )

        # 3. per-copy updates and convergence check
        newly_converged_global = []

        for j, cid in enumerate(active_ids):
            E_k = float(Es_active[j])

            # --- first-time initialization for this copy ---
            if not np.isfinite(best_E[cid]):
                # First energy we've seen for this copy: just record it, no convergence logic yet
                best_E[cid] = E_k
                best_theta[cid, :] = thetas[j]
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
                best_theta[cid, :] = thetas[j]
                stale_iters[cid] = 0
            else:
                # no significant improvement
                stale_iters[cid] += 1

            prev_E[cid] = E_k

            # --- convergence test for this copy ---
            if (round_idx >= min_rounds) and (stale_iters[cid] >= patience):
                newly_converged_global.append(cid)
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
                full, Hs_active, optimizers_active, num_params = build_active_problem(
                    active_ids
                )
                full, Hs_active = create_isas(full, Hs_active)
                #logger.info(full.draw("text"))

        # 6. stop if no active copies remain
        if not active_ids:
            logger.info(f"All copies converged by round {round_idx}")
            break

        round_idx += 1

        round_end = datetime.now()
        round_time = round_end - round_start
        round_times.append(round_time)

    vqe_end = datetime.now()
    vqe_time = vqe_end - vqe_start 

    # ---------------- Final verification ----------------
    logger.info("\n=== Final best parameters per global copy ===")
    for cid in range(num_total_copies):
        logger.info(f"Copy {cid}: best_E={best_E[cid]}, best_theta={best_theta[cid]}")



    # Save run
    run = {
        "backend": backend_name,
        "session_id": None,
        "use_noise_model": use_noise_model,
        "starttime": starttime,
        "endtime": vqe_end.strftime("%Y-%m-%d_%H-%M-%S"),
        "potential": potential,
        "cutoff": cutoff,
        "exact_eigenvalues": [x.real.tolist() for x in eigenvalues],
        "ansatz": ansatz_name,
        "num_qubits": num_qubits,
        "num_params": num_params_single, 
        "num_copies": num_total_copies,
        "patience": patience,
        "shots": shots,
        "optimization_level": optimization_level,
        "resilience_level": resilience_level,
        "Optimizer": {
            "name": "NGOpt",
            "min_rounds": min_rounds,
            "max_rounds": max_rounds,
            "ng_budget": ng_budget,
            "improve_tol_abs": improve_tol_abs,
            "improve_tol_rel": improve_tol_rel,
        },
        "cost function":{
            "type": "small negatives",
            "p":p,
            "lam":lam,
            "eps":eps
        },
        "results": best_E.tolist(),
        "params": best_theta.tolist(),
        "num_iters": round_idx,
        "active_ids": active_ids,
        #"success": np.array(success, dtype=bool).tolist(),
        "round_times": [str(t) for t in round_times],
        "VQE_run_time": str(vqe_time),
        #"seeds": seeds,
        "session_info": None
    }

    # Save the variable to a JSON file
    path = os.path.join(base_path, "{}_{}.json".format(potential, cutoff))
    with open(path, "w") as json_file:
        json.dump(run, json_file, indent=4)

    print("Done")
    


if __name__ == "__main__":
    main()
