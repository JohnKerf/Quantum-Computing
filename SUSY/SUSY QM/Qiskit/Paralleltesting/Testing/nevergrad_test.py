import numpy as np

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from susy_qm import calculate_Hamiltonian, ansatze
import nevergrad as ng


import numpy as np


class QuantumScheduler:
    def __init__(self, full, Hs, estimator, optimizers):
        self.full = full
        self.Hs = Hs
        self.estimator = estimator
        self.optimizers = optimizers
        self.num_copies = len(optimizers)

    def _evaluate_energies(self, Theta_flat: np.ndarray) -> np.ndarray:
        pubs = [(self.full, self.Hs, [Theta_flat])]
        job = self.estimator.run(pubs)
        result = job.result()
        Es = np.array(result[0].data.evs, dtype=float)
        return Es

    def step(self):
        # 1. ask
        xs = [opt.ask() for opt in self.optimizers]
        thetas = [np.asarray(x.value, dtype=float) for x in xs]

        # 2. pack Θ
        Theta = np.concatenate(thetas, axis=0)

        # 3. one Estimator call
        Es = self._evaluate_energies(Theta)

        # 4. tell
        for k, opt in enumerate(self.optimizers):
            opt.tell(xs[k], Es[k])

        return thetas, Es






def _tensor_power(op, k):
    """Return op^(⊗k); for k=0 return None."""
    if k == 0:
        return None
    out = op
    for _ in range(k - 1):
        out = out.tensor(op)
    return out


def build_multi_copy_qho(num_copies=3, cutoff=2, potential="QHO"):
    """
    Build:
      - packed circuit `full` with `num_copies` disjoint blocks
      - list of observables Hs = [H0, ..., H_{num_copies-1}]
      - num_params = parameters per copy
    """
    # --- Single-copy Hamiltonian ---
    H = calculate_Hamiltonian(cutoff, potential)
    num_qubits = int(1 + np.log2(cutoff))  # your convention
    observable = SparsePauliOp.from_operator(H)

    # --- Observables for each copy ---
    I = SparsePauliOp.from_list([("I" * num_qubits, 1.0)])
    Hs = []

    for k in range(num_copies):
        left_blocks = num_copies - k - 1
        right_blocks = k
        left = _tensor_power(I, left_blocks)
        right = _tensor_power(I, right_blocks)

        pieces = [p for p in (left, observable, right) if p is not None]
        Hk = pieces[0]
        for p in pieces[1:]:
            Hk = Hk.tensor(p)
        Hs.append(Hk)

    # --- Ansatz selection (using your naming schema) ---
    ansatze_type = "exact"
    if potential == "QHO":
        ansatz_name = f"CQAVQE_QHO_{ansatze_type}"
    elif (potential != "QHO") and (cutoff <= 64):
        ansatz_name = f"CQAVQE_{potential}{cutoff}_{ansatze_type}"
    else:
        ansatz_name = f"CQAVQE_{potential}16_{ansatze_type}"

    ansatz = ansatze.get(ansatz_name)
    num_params = ansatz.n_params

    # --- Base block circuit for ONE copy ---
    block = ansatze.pl_to_qiskit(ansatz, num_qubits=num_qubits, reverse_bits=True)
    block_params = list(block.parameters)

    # --- Build packed circuit with num_copies blocks ---
    theta = ParameterVector("θ", num_copies * num_params)
    full = QuantumCircuit(num_copies * num_qubits)

    for k in range(num_copies):
        start = k * num_params
        stop = (k + 1) * num_params
        theta_block = theta[start:stop]  # slice of ParameterVector

        mapping = {p: theta_block[i] for i, p in enumerate(block_params)}
        block_k = block.assign_parameters(mapping, inplace=False)

        qubits_k = list(range(k * num_qubits, (k + 1) * num_qubits))
        full.compose(block_k, qubits=qubits_k, inplace=True)

    return full, Hs, num_params, num_copies


def evaluate_energies(full, Hs, estimator, Theta_flat):
    """
    One Estimator call:
      (full, Hs, [Theta_flat]) → energies for each copy.
    """
    pubs = [(full, Hs, [Theta_flat])]
    job = estimator.run(pubs)
    result = job.result()
    Es = np.array(result[0].data.evs, dtype=float)  # shape (num_copies,)
    return Es




def main():
    potential = "QHO"
    cutoff = 8
    num_copies = 3

    # 1. Build packed circuit + observables
    full, Hs, num_params, num_copies = build_multi_copy_qho(
        num_copies=num_copies, cutoff=cutoff, potential=potential
    )

    print(full.draw("text"))
    print("Number of copies:", num_copies)
    print("Params per copy:", num_params)

    estimator = StatevectorEstimator()

    # 2. Initial parameters (like your old x0_list)
    rng = np.random.default_rng(123)
    x0_list = [rng.uniform(-0.5, 0.5, size=num_params) for _ in range(num_copies)]
    # or set manually:
    # x0_list = [np.zeros(num_params) for _ in range(num_copies)]

    # 3. One Nevergrad optimizer per copy (with its own x0_k)
    budget = 200       # max number of rounds
    tol = 1e-8         # |ΔE| stopping tolerance

    optimizers = []
    for k in range(num_copies):
        x0_k = x0_list[k]
        parametrization = ng.p.Array(init=x0_k)
        opt = ng.optimizers.CMA(
            parametrization=parametrization,
            budget=budget,
            num_workers=1,
        )
        optimizers.append(opt)

    # 4. Scheduler (the “ask all → one circuit → tell all” engine)
    scheduler = QuantumScheduler(full, Hs, estimator, optimizers)

    # 5. Convergence loop: stop when |ΔE_k| < tol for all copies
    prev_E = None
    history_Es = []

    for it in range(budget):
        # Ask each optimizer
        xs = [opt.ask() for opt in optimizers]                    # list of ng.p.Array
        thetas = [np.asarray(x.value, dtype=float) for x in xs]   # list of (num_params,)

        # Pack Θ and do ONE Estimator call
        Theta = np.concatenate(thetas, axis=0)
        Es = scheduler._evaluate_energies(Theta)                  # shape (num_copies,)
        history_Es.append(Es.copy())

        if prev_E is not None:
            delta = np.abs(Es - prev_E)
            print(f"[iter {it:03d}] Es = {Es}, ΔE = {delta}")

            # convergence: all copies changed by less than tol
            if np.all(delta < tol):
                print(f"Converged at iter {it} with ΔE < {tol}")
                # still need to tell optimizers once at this point:
                for k, opt in enumerate(optimizers):
                    opt.tell(xs[k], Es[k])
                break
        else:
            print(f"[iter {it:03d}] Es = {Es} (no ΔE yet)")

        # Tell each optimizer its scalar loss
        for k, opt in enumerate(optimizers):
            opt.tell(xs[k], Es[k])

        prev_E = Es

    # 6. Extract best parameters per copy and verify once more
    best_params = []
    for k, opt in enumerate(optimizers):
        rec = opt.provide_recommendation()
        theta_k = np.asarray(rec.value, dtype=float)
        best_params.append(theta_k)
        print(f"Copy {k}: best theta = {theta_k}")

    Theta_star = np.concatenate(best_params, axis=0)
    final_Es = scheduler._evaluate_energies(Theta_star)

    print("\nEnergies from single Estimator call at Θ*:")
    for k, Ek in enumerate(final_Es):
        print(f"  Copy {k}: E = {Ek}")


if __name__ == "__main__":
    main()
