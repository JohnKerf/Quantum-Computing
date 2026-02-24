import os, json
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import SamplerV2 as Sampler, QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from datetime import datetime
import wesszumino as wz
from qiskit.quantum_info import SparsePauliOp, PauliList
from scipy.sparse.linalg import eigsh
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

# aer_backend = AerSimulator(method="statevector")
# compile_target = aer_backend.target


def sparsepauliop_remove_each_term(spo: SparsePauliOp, simplify=True, atol=1e-12):
    """
    Return a list of new SparsePauliOp objects, each with exactly one term removed.
    Also returns metadata about which term was removed.
    """
    labels = spo.paulis.to_labels()
    coeffs = np.asarray(spo.coeffs, dtype=complex)

    out = []
    for i in range(len(labels)):
        new_labels = labels[:i] + labels[i+1:]
        new_coeffs = np.concatenate([coeffs[:i], coeffs[i+1:]])

        new_spo = SparsePauliOp(new_labels, new_coeffs)

        if simplify:
            # combines duplicate Pauli strings, drops near-zero coeffs
            new_spo = new_spo.simplify(atol=atol)

        out.append(
            {
                "removed_index": i,
                "removed_label": labels[i],
                "removed_coeff": coeffs[i],
                "op": new_spo,
            }
        )
    return out

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

def _kq_only_depth(qc, k=2, ignore_ops=("barrier", "measure")):
    """Depth counting only k-qubit operations (ignoring selected ops)."""
    qc_k = QuantumCircuit(qc.num_qubits, qc.num_clbits)

    q_index = {q: i for i, q in enumerate(qc.qubits)}
    c_index = {c: i for i, c in enumerate(qc.clbits)}

    for inst in qc.data:
        op = inst.operation
        if op.name in ignore_ops:
            continue
        if op.num_qubits == k:
            qargs = [qc_k.qubits[q_index[q]] for q in inst.qubits]
            cargs = [qc_k.clbits[c_index[c]] for c in inst.clbits] if inst.clbits else []
            qc_k.append(op, qargs, cargs)

    return qc_k.depth()

def circuit_cost_metrics(qc):

    ops = qc.count_ops()
    ops_str = {str(k): int(v) for k, v in ops.items()}

    ignore_ops = {"barrier", "measure"}

    n2q = sum(1
        for inst in qc.data
        if inst.operation.num_qubits == 2 and inst.operation.name not in ignore_ops
    )

    n3q = sum(1
        for inst in qc.data
        if inst.operation.num_qubits == 3 and inst.operation.name not in ignore_ops
    )

    return {
        "depth": qc.depth(),
        "size": qc.size(),
        "num_2q_ops": n2q,
        "depth_2q": _kq_only_depth(qc, k=2),
        "num_3q_ops": n3q,
        "depth_3q": _kq_only_depth(qc, k=3),
        "count_ops": ops_str,
    }

def create_circuit(optimization_level, num_qubits, H_pauli, t_k, num_trotter_steps, seed=42):
    
    qr = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qr)

    #evol_gate = PauliEvolutionGate(H_pauli, time=t_k, synthesis=SuzukiTrotter(order=2, reps=num_trotter_steps))
    evol_gate = PauliEvolutionGate(H_pauli,time=t_k,synthesis=LieTrotter(reps=num_trotter_steps))
    qc.append(evol_gate, qr)


    qc.measure_all()

    target = compile_target
    pm = generate_preset_pass_manager(target=target, optimization_level=optimization_level, seed_transpiler=seed)
    circuit_isa = pm.run(qc)
    
    return circuit_isa, circuit_cost_metrics(circuit_isa)

def get_cost(N, cutoff, seed):

    a = 1.0
    c = -0.2
    potential = "linear"
    boundary_condition = 'dirichlet'
    optimization_level = 0
    keep_ratio=0.999

    num_qubits = N*int(1+np.log2(cutoff))

    H_pauli, num_qubits = wz.build_wz_hamiltonian(cutoff,N,a, c=c,m=1.0,potential=potential,boundary_condition=boundary_condition)

    pauli_coeffs = np.real(H_pauli.coeffs).astype(float).tolist()
    pauli_labels = H_pauli.paulis.to_labels()

    kept_coeffs, kept_labels, keep_idx, trunc_info = truncate_by_coeff_weight(pauli_coeffs, pauli_labels, keep_ratio=keep_ratio)
    keep_idx = np.sort(keep_idx)
    kept_coeffs = np.asarray(pauli_coeffs)[keep_idx]
    kept_labels = np.asarray(pauli_labels)[keep_idx]

    H_pauli = SparsePauliOp(PauliList(kept_labels.tolist()), kept_coeffs.tolist())

    qc, initial_circuit_cost = create_circuit(optimization_level, num_qubits, H_pauli, t_k=1.0, num_trotter_steps=1, seed=seed)


    H = H_pauli.to_matrix(sparse=True)
    min_eigenvalue = eigsh(H, k=1, which="SA", return_eigenvectors=False)[0].real

    variants = sparsepauliop_remove_each_term(H_pauli)
    num_variants = len(variants)

    print(f"{num_variants} different H varaints")

    for i, v in enumerate(variants):
        print(f"variant [{i+1}/{num_variants}]")
        qc, circuit_cost = create_circuit(optimization_level, num_qubits, v['op'], t_k=1.0, num_trotter_steps=1, seed=seed)

        H_v = v['op'].to_matrix(sparse=True)
        v_e = eigsh(H_v, k=1, which="SA", return_eigenvectors=False)[0].real
        diff = np.abs(min_eigenvalue-v_e)

        variants[i]["diff"] = float(diff)
        variants[i]["cost"] = circuit_cost
        variants[i].pop('op')

    return trunc_info, initial_circuit_cost, variants

# def job_grid():
#     Ns = [2]
#     cts = ["QFT_full"]#["QFT_full", "QFT_trunc", ]
#     cutoff_list = [32]
#     grid = [(N, x, c) for N in Ns for x in cts for c in cutoff_list]
#     return grid

def main():

    N=4
    cutoff=16
    basis="Fock"
    potential="linear"
    all_data = {}

    task_id = 0
    # grid = job_grid()
    # N, basis, cutoff = grid[task_id]
    seed = 42

    trunc_info, initial_circuit_cost, variants = get_cost(N, cutoff, seed)

    print(f"[{task_id}] - Running {potential} - N{N} - L{cutoff} - {basis}")

    all_data = {
        "loadtime": str(datetime.now()),
        "potential": potential,
        "cutoff": cutoff,
        "N": N,
        "basis": basis,
        "trunc_info": trunc_info,
        "initial_circuit_cost": initial_circuit_cost,
        "variants": variants,
        "seed": seed
    }

    print("Done")


    with open(os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\Wess-Zumino\Qiskit\SKQD\ML-Testing", f"{potential}_N{N}_L{cutoff}_{basis}.json"), "w") as json_file:
        json.dump(all_data, json_file, indent=4, default=str)

if __name__ == "__main__":
    main() 