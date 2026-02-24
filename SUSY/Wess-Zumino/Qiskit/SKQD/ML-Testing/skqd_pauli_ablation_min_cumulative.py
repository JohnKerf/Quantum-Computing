import os
import json
import time
from datetime import datetime
from collections import Counter

import numpy as np
from scipy.sparse.linalg import eigsh

from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister, Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import PauliList, SparsePauliOp

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler

import wesszumino as wz


# =========================
# Small helpers
# =========================

target = AerSimulator(method="statevector").target

def truncate_by_coeff_weight(pauli_coeffs, pauli_labels, keep_ratio=0.999):
    c = np.asarray(pauli_coeffs)
    lab = np.asarray(pauli_labels)

    abs_c = np.abs(c)
    order = np.argsort(abs_c)[::-1]
    w = abs_c[order] ** 2
    cum = np.cumsum(w)
    total = float(cum[-1])
    target = keep_ratio * total
    m = int(np.searchsorted(cum, target, side="left") + 1)

    keep_idx = np.sort(order[:m])

    kept_coeffs = c[keep_idx]
    kept_labels = lab[keep_idx]
    trunc_info = {
        "m": int(m),
        "n": int(len(c)),
        "keep_frac_terms": float(m / len(c)),
        "keep_ratio": float(keep_ratio),
        "truncated": float(total - cum[m - 1]),
        "total_weight": float(total),
    }
    return kept_coeffs, kept_labels, keep_idx, trunc_info

def filter_counts_by_fermion_number(counts, fermion_qubits, num_fermions):
    kept = {}
    for key, c in counts.items():
        b = [int(ch) for ch in key[::-1]]
        if sum(b[i] for i in fermion_qubits) == num_fermions:
            kept[key] = c
    return kept

def create_circuit_template(avqe_circuit, num_qubits, H_pauli, n_steps, optimization_level, seed_transpiler):
    qr = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qr)
    qc.append(avqe_circuit, qr)

    t = Parameter("t")
    evol_gate = PauliEvolutionGate(H_pauli, time=t, synthesis=LieTrotter(reps=n_steps))
    qc.append(evol_gate, qr)
    qc.measure_all()

    
    pm = generate_preset_pass_manager(
        target=target,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )
    try:
        qc_isa = pm.run(qc)
    except Exception:
        pm = generate_preset_pass_manager(
            target=target,
            optimization_level=optimization_level,
            translation_method="synthesis",
            seed_transpiler=seed_transpiler,
        )
        qc_isa = pm.run(qc)
    return qc_isa

def get_counts(sampler, qc, shots):
    job = sampler.run([qc], shots=shots)
    return job.result()[0].data.meas.get_counts()

def run_skqd_compact(
    H_pauli,
    pauli_terms_full,
    avqe_circuit,
    H_info,
    run_info,
    backend_info,
    seed,
):
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
    optimization_level = backend_info["optimization_level"]
    transpiler_seed = backend_info["transpiler_seed"]

    sampler = AerSampler(
        options={
            "backend_options": {
                "method": "automatic",
                "seed_simulator": int(seed),
            },
            "run_options": {"shots": shots},
        }
    )

    num_qubits = H_info["num_qubits"]
    dense_dim = H_info["dense_H_size"][0]
    min_eigenvalue = np.min(H_info["eigenvalues"]) if H_info["eigenvalues"] is not None else None

    n_steps = 1
    k = 1
    patience_count = 0
    trotter_patience_count = 0
    converged = False
    prev_energy = None

    samples = Counter()
    energies = []

    t_start = time.time()

    qc_template = create_circuit_template(
        avqe_circuit=avqe_circuit,
        num_qubits=num_qubits,
        H_pauli=H_pauli,
        n_steps=n_steps,
        optimization_level=optimization_level,
        seed_transpiler=transpiler_seed,
    )

    while (not converged) and (k <= max_k):
        tval = dt * k
        qc = qc_template.assign_parameters({"t": tval}, inplace=False)

        counts = get_counts(sampler, qc, shots)
        if conserve_fermion:
            counts = filter_counts_by_fermion_number(counts, fermion_qubits, num_fermions)

        samples.update(counts)
        top_states = [s for s, _ in sorted(samples.items(), key=lambda x: x[1], reverse=True)]

        H_reduced = wz.reduced_sparse_matrix_from_pauli_terms_fast(pauli_terms_full, top_states)

        if H_reduced.shape[0] < 2000:
            me = float(np.min(np.linalg.eigvals(H_reduced.todense())).real)
            used_dense = True
        else:
            me = float(eigsh(H_reduced, k=1, which="SA", return_eigenvectors=False)[0].real)
            used_dense = False

        energies.append(me)

        if prev_energy is None:
            diff_prev = None
        else:
            diff_prev = float(abs(prev_energy - me))
        prev_energy = me

        if diff_prev is None:
            k += 1
            continue

        if diff_prev < tol:
            patience_count += 1
            trotter_patience_count += 1
        else:
            patience_count = 0

        if patience_count >= energy_patience:
            converged = True
            break

        if trotter_patience_count >= trotter_patience:
            n_steps += 1
            trotter_patience_count = 0
            if n_steps > max_n_steps:
                break
            qc_template = create_circuit_template(
                avqe_circuit=avqe_circuit,
                num_qubits=num_qubits,
                H_pauli=H_pauli,
                n_steps=n_steps,
                optimization_level=optimization_level,
                seed_transpiler=transpiler_seed,
            )

        k += 1

    t_end = time.time()

    return {
        "seed": int(seed),
        "converged": bool(converged),
        "final_k": int(len(energies)),
        "final_energy": float(energies[-1]) if energies else None,
        "energy_trace": [float(x) for x in energies],
        "final_num_trotter_steps": int(n_steps),
        "num_basis_states": int(len(samples)),
        "dense_dim": int(dense_dim),
        "used_dense_last": bool(used_dense) if energies else None,
        "diff_to_exact_ground_last": (float(abs(min_eigenvalue - energies[-1])) if (min_eigenvalue is not None and energies) else None),
        "runtime_s": float(t_end - t_start),
    }


# =========================
# Main experiment
# =========================

if __name__ == "__main__":
    # -------- user settings --------
    run_idx = 0

    N = 2
    a = 1.0
    c = -0.2
    potential = "linear"          # "linear" or "quadratic"
    boundary_condition = "dirichlet"
    cutoffs = [8]

    keep_ratio = 0.999
    conserve_fermion = True

    # SKQD controls
    dt = 1.0
    max_k = 10
    tol = 1e-8
    trotter_patience = 2
    energy_patience = 3
    max_n_steps = 5

    # Aer / transpilation
    shots = 100000
    optimization_level = 0
    transpiler_seed = 42

    # If you want a stored Hamiltonian json with pauli_expvals_ground / pauli_energy_contrib_ground:
    H_json_path = None
    # Example:
    H_json_path = r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\Wess-Zumino\Analyses\Model Checks\GroundstatePauliContributions\dirichlet\linear\N2\linear_16.json"

    # Output folder
    base_path = os.path.join(
        r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\Wess-Zumino\Qiskit\SKQD\ML-Testing\Ablation",
        boundary_condition,
        potential,
        f"N{N}",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(base_path, exist_ok=True)

    # -------- optional Hamiltonian metadata load (for exact eigvals + per-term ground metrics) --------
    H_data = None
    if H_json_path is not None:
        with open(H_json_path, "r") as f:
            H_data = json.load(f)

    print("[H_JSON] loaded:", H_json_path)
    print("[H_JSON] keys:", list(H_data.keys())[:20])
    print("[H_JSON] cutoff:", H_data.get("cutoff"))
    print("[H_JSON] len(pauli_labels):", None if H_data.get("pauli_labels") is None else len(H_data["pauli_labels"]))
    print("[H_JSON] len(pauli_expvals_ground):", None if H_data.get("pauli_expvals_ground") is None else len(H_data["pauli_expvals_ground"]))
    print("[H_JSON] len(pauli_energy_contrib_ground):", None if H_data.get("pauli_energy_contrib_ground") is None else len(H_data["pauli_energy_contrib_ground"]))

    metric_by_label = {}
    if H_data is not None:
        h_labels = H_data.get("pauli_labels")
        h_exp = H_data.get("pauli_expvals_ground")
        h_contrib = H_data.get("pauli_energy_contrib_ground")
        if h_labels is not None and h_exp is not None and h_contrib is not None:
            metric_by_label = {
                str(lbl): {
                    "pauli_expvals_ground": h_exp[i],
                    "pauli_energy_contrib_ground": h_contrib[i],
                }
                for i, lbl in enumerate(h_labels)
            }

    for cutoff in cutoffs:
        print(f"[START] cutoff={cutoff}")

        # Full Hamiltonian (for reduced-matrix diagonalization target)
        H_full, num_qubits = wz.build_wz_hamiltonian(
            cutoff, N, a, c=c, m=1.0, potential=potential, boundary_condition=boundary_condition
        )
        full_pauli_coeffs = np.real(H_full.coeffs).astype(float)
        full_pauli_labels = np.array(H_full.paulis.to_labels())
        pauli_terms_full = list(zip(full_pauli_coeffs.tolist(), full_pauli_labels.tolist()))

        dense_H_size = [2 ** num_qubits, 2 ** num_qubits]
        eigenvalues = H_data["eigenvalues"] if (H_data is not None and int(H_data.get("cutoff", cutoff)) == cutoff) else None

        # Truncate once
        kept_coeffs, kept_labels, keep_idx, trunc_info = truncate_by_coeff_weight(
            full_pauli_coeffs, full_pauli_labels, keep_ratio=keep_ratio
        )
        H_trunc = SparsePauliOp(PauliList(kept_labels.tolist()), kept_coeffs.tolist())

        # Basis / ansatz
        qps = int(np.log2(cutoff)) + 1
        fermion_qubits = [(s + 1) * qps - 1 for s in range(N)]
        basis = [0] * num_qubits
        basis[2 * qps - 1 :: 2 * qps] = [1] * (N // 2)
        num_fermions = sum(basis[q] for q in fermion_qubits)

        avqe_circuit = wz.build_avqe_pattern_ansatz(
            N=N, cutoff=cutoff, include_basis=True, include_rys=True, include_xxyys=True
        )

        H_info = {
            "potential": potential,
            "boundary_condition": boundary_condition,
            "cutoff": cutoff,
            "N": N,
            "a": a,
            "c": None if potential == "linear" else c,
            "num_qubits": num_qubits,
            "dense_H_size": dense_H_size,
            "eigenvalues": eigenvalues,
            "basis": basis,
            "trunc_info": trunc_info,
        }

        run_info = {
            "dt": dt,
            "max_k": max_k,
            "tol": tol,
            "conserve_fermion": conserve_fermion,
            "fermion_qubits": fermion_qubits,
            "num_fermions": num_fermions,
            "trotter_patience": trotter_patience,
            "energy_patience": energy_patience,
            "max_n_steps": max_n_steps,
        }

        backend_info = {
            "shots": shots,
            "optimization_level": optimization_level,
            "transpiler_seed": transpiler_seed,
        }

        # Baseline on truncated evolution Hamiltonian
        baseline_seed = 12345
        baseline = run_skqd_compact(
            H_pauli=H_trunc,
            pauli_terms_full=pauli_terms_full,
            avqe_circuit=avqe_circuit,
            H_info=H_info,
            run_info=run_info,
            backend_info=backend_info,
            seed=baseline_seed,
        )
        baseline_energy = baseline["final_energy"]
        print(f"[BASELINE] E_final = {baseline_energy}")

        # Optional per-term ground metrics from stored Hamiltonian file (aligned to full labels)
        expvals_ground = None
        energy_contrib_ground = None
        if H_data is not None and int(H_data.get("cutoff", cutoff)) == cutoff:
            expvals_ground = H_data.get("pauli_expvals_ground", None)
            energy_contrib_ground = H_data.get("pauli_energy_contrib_ground", None)

        # One-term ablations over kept terms
        rows = []
        n_kept = len(kept_labels)
        for j in range(n_kept):
            mask = np.ones(n_kept, dtype=bool)
            mask[j] = False

            ablated_labels = kept_labels[mask].tolist()
            ablated_coeffs = kept_coeffs[mask].tolist()
            H_abl = SparsePauliOp(PauliList(ablated_labels), ablated_coeffs)

            summary = run_skqd_compact(
                H_pauli=H_abl,
                pauli_terms_full=pauli_terms_full,
                avqe_circuit=avqe_circuit,
                H_info=H_info,
                run_info=run_info,
                backend_info=backend_info,
                seed=baseline_seed + j + 1,
            )

            removed_global_idx = int(keep_idx[j])
            m = metric_by_label.get(str(kept_labels[j]), {})
            row = {
                "removed_local_idx": int(j),
                "removed_global_idx": removed_global_idx,
                "removed_label": str(kept_labels[j]),
                "removed_coeff": float(np.real(kept_coeffs[j])),
                "removed_abs_coeff": float(abs(kept_coeffs[j])),
                "baseline_final_energy": baseline_energy,
                "ablated_final_energy": summary["final_energy"],
                "delta_vs_baseline": (
                    None if (summary["final_energy"] is None or baseline_energy is None)
                    else float(summary["final_energy"] - baseline_energy)
                ),
                "abs_delta_vs_baseline": (
                    None if (summary["final_energy"] is None or baseline_energy is None)
                    else float(abs(summary["final_energy"] - baseline_energy))
                ),
                "converged": summary["converged"],
                "final_k": summary["final_k"],
                "final_num_trotter_steps": summary["final_num_trotter_steps"],
                "num_basis_states": summary["num_basis_states"],
                "runtime_s": summary["runtime_s"],
                "pauli_expvals_ground": m.get("pauli_expvals_ground"),
                "pauli_energy_contrib_ground": m.get("pauli_energy_contrib_ground"),
            }
            rows.append(row)

            print(f"[{j+1}/{n_kept}] {row['removed_label']}  dE={row['delta_vs_baseline']}")

        rows_by_safest = sorted(
            rows,
            key=lambda r: r["abs_delta_vs_baseline"] if r["abs_delta_vs_baseline"] is not None else np.inf,
        )

        checkpoints = []
        if n_kept > 1:
            for v in [1, 2, 5, 10, 20, 30, 40, 50, 75, 100]:
                if v < n_kept:
                    checkpoints.append(v)
            if (n_kept - 1) not in checkpoints:
                checkpoints.append(n_kept - 1)
        checkpoints = sorted(set(checkpoints))

        batch_rows = []
        for n_remove in checkpoints:
            remove_local_idx = {int(r["removed_local_idx"]) for r in rows_by_safest[:n_remove]}
            mask = np.array([j not in remove_local_idx for j in range(n_kept)], dtype=bool)

            H_batch = SparsePauliOp(
                PauliList(kept_labels[mask].tolist()),
                kept_coeffs[mask].tolist()
            )

            summary = run_skqd_compact(
                H_pauli=H_batch,
                pauli_terms_full=pauli_terms_full,
                avqe_circuit=avqe_circuit,
                H_info=H_info,
                run_info=run_info,
                backend_info=backend_info,
                seed=baseline_seed + 100000 + n_remove,
            )

            e_final = summary["final_energy"]
            dE = None if (e_final is None or baseline_energy is None) else float(e_final - baseline_energy)

            batch_rows.append({
                "num_removed": int(n_remove),
                "num_kept": int(mask.sum()),
                "removed_labels": [str(r["removed_label"]) for r in rows_by_safest[:n_remove]],
                "final_energy": e_final,
                "delta_vs_baseline": dE,
                "abs_delta_vs_baseline": (None if dE is None else float(abs(dE))),
                "converged": summary["converged"],
                "final_k": summary["final_k"],
                "final_num_trotter_steps": summary["final_num_trotter_steps"],
                "num_basis_states": summary["num_basis_states"],
                "runtime_s": summary["runtime_s"],
            })

            print(f"[BATCH {n_remove}] dE={dE}")

        rows_by_effect_desc = sorted(
            rows,
            key=lambda r: r["abs_delta_vs_baseline"] if r["abs_delta_vs_baseline"] is not None else -1.0,
            reverse=True,
        )

        out = {
            "study": "one_term_ablation_after_truncation_with_cumulative_batch_pruning",
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "settings": {
                "potential": potential,
                "boundary_condition": boundary_condition,
                "N": N,
                "cutoff": cutoff,
                "a": a,
                "c": c,
                "keep_ratio": keep_ratio,
                "shots": shots,
                "dt": dt,
                "max_k": max_k,
                "tol": tol,
                "trotter_patience": trotter_patience,
                "energy_patience": energy_patience,
                "max_n_steps": max_n_steps,
                "optimization_level": optimization_level,
                "transpiler_seed": transpiler_seed,
                "conserve_fermion": conserve_fermion,
            },
            "trunc_info": trunc_info,
            "baseline": baseline,
            "counts": {
                "num_full_terms": int(len(full_pauli_labels)),
                "num_kept_terms": int(n_kept),
                "num_ablations": int(len(rows)),
                "num_batch_tests": int(len(batch_rows)),
            },
            "one_term_rows_by_effect_desc": rows_by_effect_desc,
            "one_term_rows_by_safest": rows_by_safest,
            "cumulative_batch_rows": batch_rows,
        }

        out_fp = os.path.join(base_path, f"{potential}_{cutoff}_pauli_ablation.json")
        with open(out_fp, "w") as f:
            json.dump(out, f, indent=2)

        print(f"[SAVED] {out_fp}")
