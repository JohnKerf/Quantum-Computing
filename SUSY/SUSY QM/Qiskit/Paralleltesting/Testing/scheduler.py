import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from susy_qm import calculate_Hamiltonian, ansatze
import nevergrad as ng
# ---------------------------------------------------------------------
#  Scheduler: single circuit / single Estimator call per "round"
# ---------------------------------------------------------------------

import threading
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor, as_completed


import threading
from concurrent.futures import Future
import numpy as np
import time


class SingleCircuitScheduler:
    """
    Scheduler that enforces ONE Estimator call per 'round':
    all copies submit their parameters, then we run

        (full, Hs, [Theta])

    where Theta is the concatenation of all copy params.
    Each optimizer k gets back its scalar energy E_k.
    """

    def __init__(self, full_circuit, observables, estimator, num_copies, num_params):
        self.full = full_circuit
        self.Hs = observables
        self.estimator = estimator
        self.num_copies = num_copies
        self.num_params = num_params

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

        # Pending requests for the current "round": copy_id -> (theta_small, future)
        self._pending = {}
        self._round = 0
        self._stopped = False

    def stop(self):
        with self._cond:
            print("[SCHED] stop() called, waking all waiting threads.")
            self._stopped = True
            self._cond.notify_all()

    def evaluate(self, copy_id: int, theta_small) -> float:
        """
        Called by optimizer k as its cost function.

        Blocks until all copies have called evaluate once in this round,
        then returns that copy's scalar energy.
        """
        theta_small = np.asarray(theta_small, dtype=float)
        if theta_small.shape != (self.num_params,):
            raise ValueError(
                f"theta_small must have shape ({self.num_params},), got {theta_small.shape}"
            )

        fut = Future()

        with self._cond:
            if self._stopped:
                raise RuntimeError("Scheduler stopped")

            if copy_id in self._pending:
                raise RuntimeError(
                    f"Copy {copy_id} already submitted params in round {self._round}"
                )

            print(f"[SCHED][round {self._round}] copy {copy_id} submitted params.")
            self._pending[copy_id] = (theta_small, fut)

            # If not all copies have submitted yet, wait until our future is filled
            if len(self._pending) < self.num_copies:
                missing = sorted(set(range(self.num_copies)) - set(self._pending.keys()))
                print(
                    f"[SCHED][round {self._round}] waiting for copies: {missing}"
                )
                # Wait in chunks so we can print if it takes too long
                while not fut.done() and not self._stopped:
                    self._cond.wait(timeout=10.0)
                    if not fut.done() and not self._stopped:
                        missing = sorted(
                            set(range(self.num_copies)) - set(self._pending.keys())
                        )
                        print(
                            f"[SCHED][round {self._round}] still waiting for copies: {missing}"
                        )

            else:
                # This thread becomes the 'leader' that runs the single circuit
                print(
                    f"[SCHED][round {self._round}] all copies submitted, building Theta."
                )
                t0 = time.time()
                try:
                    thetas_ordered = []
                    futures_ordered = []
                    for k in range(self.num_copies):
                        th_k, fut_k = self._pending[k]
                        thetas_ordered.append(th_k)
                        futures_ordered.append(fut_k)

                    Theta = np.concatenate(thetas_ordered, axis=0)

                    print(
                        f"[SCHED][round {self._round}] calling estimator.run(...)"
                    )
                    pubs = [(self.full, self.Hs, [Theta])]
                    job = self.estimator.run(pubs)
                    result = job.result()
                    Es = np.array(result[0].data.evs, dtype=float)  # (num_copies,)

                    dt = time.time() - t0
                    print(
                        f"[SCHED][round {self._round}] estimator returned in {dt:.3f}s."
                    )

                    for k, fut_k in enumerate(futures_ordered):
                        print(
                            f"[SCHED][round {self._round}] setting result for copy {k}: E={Es[k]}"
                        )
                        if not fut_k.done():
                            fut_k.set_result(float(Es[k]))

                except Exception as exc:
                    print(
                        f"[SCHED][round {self._round}] EXCEPTION during estimator call: {exc!r}"
                    )
                    for _, fut_k in self._pending.values():
                        if not fut_k.done():
                            fut_k.set_exception(exc)
                finally:
                    print(f"[SCHED][round {self._round}] round complete, resetting.")
                    self._pending.clear()
                    self._round += 1
                    self._cond.notify_all()

        # Outside the lock: return the scalar energy for this copy
        E = fut.result()
        print(
            f"[SCHED][round {self._round}] copy {copy_id} returning energy {E}"
        )
        return E







def _tensor_power(op, k):
    """Return op^(⊗k); for k=0 return None."""
    if k == 0:
        return None
    out = op
    for _ in range(k - 1):
        out = out.tensor(op)
    return out


def build_multi_copy_qho(num_copies=3, cutoff=2, potential="QHO"):
    # --- Single-copy Hamiltonian ---
    H = calculate_Hamiltonian(cutoff, potential)
    num_qubits = int(1 + np.log2(cutoff))  # your convention
    observable = SparsePauliOp.from_operator(H)

    # --- Observables for each copy ---
    I = SparsePauliOp.from_list([("I" * num_qubits, 1.0)])

    Hs = []
    for k in range(num_copies):
        # number of identity blocks to the "left" (higher qubits)
        left_blocks = num_copies - k - 1
        # number of identity blocks to the "right" (lower qubits)
        right_blocks = k

        left = _tensor_power(I, left_blocks)
        right = _tensor_power(I, right_blocks)

        pieces = [p for p in (left, observable, right) if p is not None]
        Hk = pieces[0]
        for p in pieces[1:]:
            Hk = Hk.tensor(p)

        Hs.append(Hk)

    # --- Ansatz selection (using your naming convention) ---
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
    block_params = list(block.parameters)  # typically length = num_params

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


# ---------------------------------------------------------------------
#  Running 3 COBYQA optimizers in parallel through the scheduler
# ---------------------------------------------------------------------

def main():
    # Set up problem: 3 copies of QHO with cutoff=2
    full, Hs, num_params, num_copies = build_multi_copy_qho(
        num_copies=3, cutoff=2, potential="QHO"
    )

    print(full.draw("text"))
    print("Number of copies:", num_copies)
    print("Number of params per copy:", num_params)
    print("Total params:", num_copies * num_params)

    # Estimator (statevector, exact expectation values)
    estimator = StatevectorEstimator()

    # Scheduler: single circuit / single Estimator call per round
    scheduler = SingleCircuitScheduler(
        full_circuit=full,
        observables=Hs,
        estimator=estimator,
        num_copies=num_copies,
        num_params=num_params,
    )

    # Cost function factory: one per copy
    def make_cost_fn(copy_id: int, scheduler: SingleCircuitScheduler):
        def cost(theta_k):
            print(f"[OPT {copy_id}] cost called with theta={theta_k}")
            E = scheduler.evaluate(copy_id, theta_k)
            print(f"[OPT {copy_id}] cost returning E={E}")
            return E
        return cost

    # Initial points for each optimizer
    rng = np.random.default_rng(123)
    x0_list = [rng.uniform(-0.5, 0.5, size=num_params) for _ in range(num_copies)]

    # COBYQA options (tune as you like)
    opt_options = {
        "maxiter": 50,
        "maxfev": 100,
        "initial_tr_radius": 0.5,
        "final_tr_radius": 1e-3,
        "scale": True,
        "disp": False,
    }

    def run_one_optimizer(copy_id: int):
        x0 = x0_list[copy_id]
        cost_fn = make_cost_fn(copy_id, scheduler)

        print(f"[OPT {copy_id}] starting minimize with x0={x0}")
        try:
            res = minimize(
                cost_fn,
                x0,
                method="COBYQA",
                options=opt_options,
            )
            print(f"[OPT {copy_id}] finished minimize, success={res.success}")
            return copy_id, res
        except Exception as e:
            print(f"[OPT {copy_id}] EXCEPTION in minimize: {e!r}")
            raise

    print("Running circuits")

    # Run 3 optimizers in parallel threads
    results = []
    with ThreadPoolExecutor(max_workers=num_copies) as ex:
        futures = {ex.submit(run_one_optimizer, k): k for k in range(num_copies)}
        for f in as_completed(futures):
            k = futures[f]
            try:
                res = f.result()
                print(f"[MAIN] Optimizer {k} finished.")
                results.append(res)
            except Exception as e:
                print(f"[MAIN] Optimizer {k} raised an exception: {e!r}")

    print("Finished running circuits.. stopping scheduler")
    # Stop the scheduler
    scheduler.stop()

    # Sort results by copy id
    results.sort(key=lambda x: x[0])

    print("\n=== Optimization results (per copy) ===")
    for copy_id, res in results:
        print(f"Copy {copy_id}:")
        print("  success:", res.success)
        print("  fun (energy):", res.fun)
        print("  x (params):", res.x)

    # Optional: verify final energies with one last Estimator call
    # Build Theta* from the optimized params
    theta_star_blocks = [res.x for _, res in results]
    Theta_star = np.concatenate(theta_star_blocks, axis=0)

    pubs = [(full, Hs, [Theta_star])]
    final_job = estimator.run(pubs)
    final_result = final_job.result()
    final_Es = np.array(final_result[0].data.evs, dtype=float)

    print("\nEnergies from a single Estimator call at Theta*:")
    for k, Ek in enumerate(final_Es):
        print(f"  Copy {k}: E = {Ek}")


if __name__ == "__main__":
    main()
