import numpy as np
import nevergrad as ng

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator

from susy_qm import calculate_Hamiltonian, ansatze


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


def evaluate_energies(full, Hs, estimator, Theta_flat):
    """
    One Estimator call:
        (full, Hs, [Theta_flat]) → energies for each active block.
    """
    pubs = [(full, Hs, [Theta_flat])]
    job = estimator.run(pubs)
    result = job.result()
    Es = np.array(result[0].data.evs, dtype=float)
    return Es


def main():
    # ---------------- Problem setup ----------------
    potential = "QHO"
    cutoff = 2
    num_total_copies = 2   # total copies (optimizers) you want initially

    # Single-copy Hamiltonian (matrix) and convert to SparsePauliOp
    H_matrix = calculate_Hamiltonian(cutoff, potential)
    num_qubits = int(1 + np.log2(cutoff))
    H_single_sp = SparsePauliOp.from_operator(H_matrix)

    min_eigenvalue = np.min(np.linalg.eigvals(H_matrix))

    # Single-copy ansatz object
    ansatze_type = "exact"
    if potential == "QHO":
        ansatz_name = f"CQAVQE_QHO_{ansatze_type}"
    elif (potential != "QHO") and (cutoff <= 64):
        ansatz_name = f"CQAVQE_{potential}{cutoff}_{ansatze_type}"
    else:
        ansatz_name = f"CQAVQE_{potential}16_{ansatze_type}"

    ansatz = ansatze.get(ansatz_name)

    estimator = StatevectorEstimator()

    # ---------------- Global trackers per copy ----------------
    # Convergence hyperparameters
    improve_tol_abs = 1e-8      # absolute improvement threshold
    improve_tol_rel = 1e-3      # relative threshold (0.1% of |best_E|)
    patience        = 50        # how many rounds of no improvement before freezing
    min_rounds      = 50        # don't even consider convergence before this
    max_rounds = 200
    stale_iters = np.zeros(num_total_copies, dtype=int)  # per-copy counters

    rng = np.random.default_rng(123)

    # One x0 per global copy
    num_params_single = ansatz.n_params
    x0_global = [rng.random(size=num_params_single)*2*np.pi for _ in range(num_total_copies)]

    best_theta = np.array(x0_global, dtype=float)          # shape (num_total_copies, num_params_single)
    best_E = np.full(num_total_copies, np.nan, dtype=float)  
    prev_E = np.full(num_total_copies, np.nan, dtype=float)



    active_ids = list(range(num_total_copies))             # global indices of active copies

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
                budget=max_rounds,  # upper bound on total tell() calls
                num_workers=1,
            )
            optimizers.append(opt)

        return full, Hs_active, optimizers, num_params

    full, Hs_active, optimizers_active, num_params = build_active_problem(active_ids)

    print(full.draw("text"))
    print("Initial active copies:", active_ids)
    print("Params per copy:", num_params)

    # ---------------- Optimization loop with shrinking ----------------
    round_idx = 0
    while round_idx < max_rounds and active_ids:
        num_active = len(active_ids)

        # 1. ask each active optimizer
        xs = []
        thetas = []
        for j, opt in enumerate(optimizers_active):
            x = opt.ask()
            xs.append(x)
            thetas.append(np.asarray(x.value, dtype=float))

        # 2. pack Θ_active and do ONE estimator call
        Theta_active = np.concatenate(thetas, axis=0)
        Es_active = evaluate_energies(full, Hs_active, estimator, Theta_active)

        print(
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
                print(
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

            print("  shrinking active set, new active_ids:", active_ids)

            # rebuild packed circuit / observables / optimizers
            if active_ids:
                full, Hs_active, optimizers_active, num_params = build_active_problem(
                    active_ids
                )
                #print(full.draw("text"))

        # 6. stop if no active copies remain
        if not active_ids:
            print(f"All copies converged by round {round_idx}")
            break

        round_idx += 1

    # ---------------- Final verification ----------------
    print(f"\nMin eigenvalue: {min_eigenvalue.real}")

    print("\n=== Final best parameters per global copy ===")
    for cid in range(num_total_copies):
        print(f"Copy {cid}: best_E={best_E[cid]}, best_theta={best_theta[cid]}")


if __name__ == "__main__":
    main()
