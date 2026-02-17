import os, json
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import SamplerV2 as Sampler, QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from datetime import datetime
import wesszumino as wz
from qiskit.quantum_info import SparsePauliOp, PauliList

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

def create_circuit(optimization_level, num_qubits, H_pauli, t_k, num_trotter_steps, circuit_type, cutoff, N, basis_info=None, boundary_condition=None, seed=42):
    
    qr = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qr)

    #dt_step = t_k / num_trotter_steps

    if circuit_type == "QFT_trunc":
        keep_ratio_1site = 0.999
        keep_ratio_2site = 0.999
    else:
        keep_ratio_1site = 1.0
        keep_ratio_2site = 1.0

    if circuit_type in ["QFT_trunc", "QFT_full"]:
        nb = int(np.log2(cutoff))
        n_per_site = int(1+nb)

        qf_sites = []
        qb_sites = []
        for n in range(N):
            base = n * n_per_site
            qb = [qc.qubits[base + i] for i in range(nb)]
            qf = qc.qubits[base + nb]

            qf_sites.append(qf)
            qb_sites.append(qb)

        wz.append_wz_split_operator_evolution(qc, qf_sites, qb_sites, basis_info, t_k, num_trotter_steps, boundary_condition=boundary_condition, scheme='strang',
                                            keep_ratio_1site = keep_ratio_1site,
                                            keep_ratio_2site = keep_ratio_2site,
                                            min_keep_1site = 0,
                                            min_keep_2site = 0)
    else:
        evol_gate = PauliEvolutionGate(H_pauli,time=t_k,synthesis=LieTrotter(reps=num_trotter_steps))
        #evol_gate = PauliEvolutionGate(H_pauli, time=t_k, synthesis=SuzukiTrotter(order=2, reps=num_trotter_steps))
        qc.append(evol_gate, qr)


    qc.measure_all()

    target = compile_target
    pm = generate_preset_pass_manager(target=target, optimization_level=optimization_level, seed_transpiler=seed)
    circuit_isa = pm.run(qc)
    
    return circuit_isa, circuit_cost_metrics(circuit_isa)

def get_cost(circuit_type, N, cutoff, seed):

    a = 1.0
    c = -0.2
    potential = "linear"
    boundary_condition = 'dirichlet'

    x_max = 5.0
    num_qubits = N*int(1+np.log2(cutoff))


    if circuit_type == "Fock":
        H_pauli, num_qubits = wz.build_wz_hamiltonian(cutoff,N,a, c=c,m=1.0,potential=potential,boundary_condition=boundary_condition,remove_zero_terms=True)
        basis_info = None
    elif circuit_type == "Trunc":
        H_pauli, num_qubits = wz.build_wz_hamiltonian(cutoff,N,a, c=c,m=1.0,potential=potential,boundary_condition=boundary_condition,remove_zero_terms=True)
        pauli_coeffs = np.real(H_pauli.coeffs).astype(float).tolist() 
        pauli_labels = H_pauli.paulis.to_labels()
        kept_coeffs, kept_labels, keep_idx, trunc_info = truncate_by_coeff_weight(pauli_coeffs, pauli_labels, keep_ratio=0.999)

        keep_idx = np.sort(keep_idx)
        kept_coeffs = np.asarray(pauli_coeffs)[keep_idx]
        kept_labels = np.asarray(pauli_labels)[keep_idx]

        H_pauli = SparsePauliOp(PauliList(kept_labels.tolist()), kept_coeffs.tolist())
        basis_info = None
    else:
        H_pauli, n_qubits, basis_info = wz.build_wz_hamiltonian_with_qft_parts(cutoff=cutoff, N=N, a=a, potential=potential, boundary_condition=boundary_condition,c=c, x_max=x_max)

    qc, circuit_cost = create_circuit(3, num_qubits, H_pauli, t_k=1.0, num_trotter_steps=1, circuit_type=circuit_type, cutoff=cutoff, N=N, basis_info=basis_info, boundary_condition=boundary_condition, seed=seed)

    return circuit_cost['num_2q_ops']

def job_grid():
    Ns = [2]
    cts = ["QFT_full"]#["QFT_full", "QFT_trunc", ]
    cutoff_list = [32]
    grid = [(N, x, c) for N in Ns for x in cts for c in cutoff_list]
    return grid

def main():

    # N=2
    # cutoff=32
    # basis="Fock"
    potential="linear"
    all_data = {}

    task_id = 0
    grid = job_grid()
    N, basis, cutoff = grid[task_id]
    seed = 42

    cost = get_cost(basis, N, cutoff, seed)

    print(f"[{task_id}] - Running {potential} - N{N} - L{cutoff} - {basis}")

    all_data = {
        "loadtime": str(datetime.now()),
        "potential": potential,
        "cutoff": cutoff,
        "N": N,
        "basis": basis,
        "cost": cost,
        "seed": seed
    }

    print("Done")


    with open(os.path.join(r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\SUSY\Wess-Zumino\Qiskit\SKQD\GateScaling\BasisScaling\testing", f"{potential}_N{N}_L{cutoff}_{basis}.json"), "w") as json_file:
        json.dump(all_data, json_file, indent=4, default=str)

if __name__ == "__main__":
    main() 