import pennylane as qml
import pennylane.numpy as pnp
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import numpy as np

'''
Function to list gate operations within a circuit
'''
def gate_list_from_ansatz(ansatz_fn, params, num_qubits):
    with qml.tape.QuantumTape() as tape:
        ansatz_fn(params,num_qubits)
    return [
        {"gate": op.name, "wires": list(op.wires), "param": op.parameters}
        for op in tape.operations
    ]



def truncate_ansatz(ansatz_fn, params, num_qubits, max_gate):

    if max_gate is not None:
        gate_diff = ansatz_fn.n_params - max_gate if ansatz_fn.n_params > max_gate else 0

        dummy_params = np.append(params, [0.0]*gate_diff)
        with qml.tape.QuantumTape() as tape:
            ansatz_fn(dummy_params, num_qubits)

        ops = tape.operations

        if isinstance(ops[0],qml.BasisState):
            gate_list = ops[:max_gate+1]
        else:
            gate_list = ops[:max_gate]

    else:
        with qml.tape.QuantumTape() as tape:
            ansatz_fn(params, num_qubits)

        gate_list = tape.operations
   
    return gate_list



def pl_to_qiskit(ansatz_fn, params=None, num_qubits=None, num_layers=None, circular=None, reverse_bits=True):

    if callable(ansatz_fn.n_params):
        num_params = ansatz_fn.n_params(num_qubits,num_layers)
    else:
        num_params = ansatz_fn.n_params
    

    if params is None:
        param_objs = [Parameter(f"θ{i}") for i in range(num_params)]
    else:
        param_objs = params

    if circular is None:
        with qml.tape.QuantumTape() as tape:
            ansatz_fn(param_objs, num_qubits)
    else:
        with qml.tape.QuantumTape() as tape:
            ansatz_fn(param_objs, num_qubits, circular=circular)

    qc = QuantumCircuit(num_qubits)

    def one_qubit(op, apply):
        (q,) = [int(w) for w in op.wires]
        apply(q)

    def two_qubit(op, apply):
        q0, q1 = [int(w) for w in op.wires]
        apply(q0, q1)

    for op in tape.operations:
        name = op.name
        params = list(op.parameters)
        wires = [int(w) for w in op.wires]

        # ---- Initial basis state preparation ----
        if name == "BasisState":
          
            bits = np.array(params[0]).astype(int).ravel().tolist()
            for b, w in zip(bits, wires):
                if b == 1:
                    qc.x(w)

        elif name == "Barrier":
            qc.barrier()

        # ---- single-qubit rotations ----
        elif name == "RX":
            one_qubit(op, lambda q: qc.rx(params[0], q))
        elif name == "RY":
            one_qubit(op, lambda q: qc.ry(params[0], q))
        elif name == "RZ":
            one_qubit(op, lambda q: qc.rz(params[0], q))

        # ---- single-qubit gates ----
        elif name == "X":
            one_qubit(op, lambda q: qc.x(q))
        elif name == "Y":
            one_qubit(op, lambda q: qc.y(q))
        elif name == "Z":
            one_qubit(op, lambda q: qc.z(q))
        elif name == "H":
            one_qubit(op, lambda q: qc.h(q))

         # ---- Two-qubit rotations ----
        elif name in ("CRY"):
            two_qubit(op, lambda q0, q1: qc.cry(params[0], q0, q1))
        elif name == "CRZ":
            two_qubit(op, lambda q0, q1: qc.crz(params[0], q0, q1))


        # ---- Two-qubit gates ----
        elif name in ("CNOT", "CX"):
            two_qubit(op, lambda q0, q1: qc.cx(q0, q1))
        elif name == "CZ":
            two_qubit(op, lambda q0, q1: qc.cz(q0, q1))
        elif name == "SWAP":
            two_qubit(op, lambda q0, q1: qc.swap(q0, q1))

        # ---- Not yet supported ----
        else:
            raise NotImplementedError(
                f"Unsupported PennyLane op '{name}' with wires {wires} and params {params}. "
                "Add a mapping above to handle it."
            )

    if reverse_bits:
        qc = qc.reverse_bits()

    return qc



################### Real Amplitudes ####################    
# def real_amplitudes(params, num_qubits, circular=True, include_fermion=True):
    
#     n = num_qubits-1
#     wires = range(num_qubits)

#     for i, w in enumerate(wires):
#         qml.RY(params[i], wires=w)

#     if num_qubits > 1:
#         for i in range(1, num_qubits):
#             qml.CNOT(wires=[wires[i-1], wires[i]])

#         if circular: qml.CNOT(wires=[wires[-1], wires[0]])

#     for i, w in enumerate(wires):
#         qml.RY(params[n + i], wires=w)

# real_amplitudes.name = "real_amplitudes"

################### Real Amplitudes #################### 
def real_amplitudes(params, num_qubits, num_layers=1, circular=True):

    n=num_qubits-1
    wires = list(range(num_qubits))

    idx = 0

    def ry_layer():
        nonlocal idx
        for w in wires:
            qml.RY(params[idx], wires=w)
            idx += 1


    def entangle_layer():

        qml.Barrier()
    
        for i in range(n):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
      
        if circular and n > 2:
            qml.CNOT(wires=[wires[-1], wires[0]])

        qml.Barrier()


    for _ in range(num_layers):
        ry_layer()
        if n > 1:
            entangle_layer()
            
    ry_layer()


real_amplitudes.name = "real_amplitudes"
real_amplitudes.n_params = lambda num_qubits, num_layers=1, **_: num_qubits * (num_layers + 1)



################### efficientSU2 ####################    
def efficientSU2(params, num_qubits, circular=True, include_fermion=True):
    
    n = num_qubits-1
    wires = range(num_qubits)

    params_idx = 0
    for w in wires:
        qml.RY(params[params_idx], wires=w)
        params_idx +=1

    params_idx = 0
    for w in wires:
        qml.RZ(params[params_idx], wires=w)
        params_idx +=1
        

    if num_qubits > 1:
        for i in range(1, num_qubits):
            qml.CNOT(wires=[wires[i-1], wires[i]])

        if circular: qml.CNOT(wires=[wires[-1], wires[0]])

    for w in wires:
        qml.RY(params[params_idx], wires=w)
        params_idx +=1

    for w in wires:
        qml.RZ(params[params_idx], wires=w)
        params_idx +=1

efficientSU2.name = "efficientSU2"
efficientSU2.n_params = lambda num_qubits, num_layers=1, **_: 2*num_qubits*(num_layers+1)


def strongly_entangling_layers(params, num_qubits, include_fermion=True, num_layers=1):

   params = pnp.tensor(params)
   shape = qml.StronglyEntanglingLayers.shape(n_layers=num_layers, n_wires=num_qubits)
   weights = params.reshape(shape)

   qml.StronglyEntanglingLayers(weights=weights, wires=range(num_qubits))
   
strongly_entangling_layers.name = "strongly_entangling_layers"
strongly_entangling_layers.n_params = lambda num_qubits, num_layers=1, **_: 3*num_qubits*num_layers






################### COBYQA Adaptive-VQE Exact Ansatze ####################      
'''
Ansatze produced from running the COBYQA Adaptive-VQE with shots=None
'''
################### QHO ###################
def CQAVQE_QHO_exact(params, num_qubits, include_fermion=True):
    if include_fermion:
        basis = [1] + [0]*(num_qubits-1)
        qml.BasisState(basis, wires=range(num_qubits))
    qml.RY(params[0], wires=[0])

CQAVQE_QHO_exact.n_params = 1
CQAVQE_QHO_exact.name = "CQAVQE_QHO_exact"

################### DW ###################
def CQAVQE_DW2_exact(params, num_qubits, include_fermion=True):
    qml.RY(params[0], wires=[num_qubits-1])    

CQAVQE_DW2_exact.n_params = 1
CQAVQE_DW2_exact.name = "CQAVQE_DW2_exact"

def CQAVQE_DW4_exact(params, num_qubits, include_fermion=True):
    if include_fermion:
        basis = [1] + [0]*(num_qubits-1)
        qml.BasisState(basis, wires=range(num_qubits))
    qml.RY(params[0], wires=[num_qubits-2])
    qml.RY(params[1], wires=[num_qubits-1])
    qml.CRY(params[2], wires=[num_qubits-2,num_qubits-1])   

CQAVQE_DW4_exact.n_params = 3
CQAVQE_DW4_exact.name = "CQAVQE_DW4_exact" 

def CQAVQE_DW8_exact(params, num_qubits, include_fermion=True):
    qml.RY(params[0], wires=[num_qubits-1])
    qml.CRY(params[1], wires=[num_qubits-1, num_qubits-2])
    qml.RY(params[2], wires=[num_qubits-3])
    qml.RY(params[3], wires=[num_qubits-2])
    qml.CRY(params[4], wires=[num_qubits-1, num_qubits-3])
    qml.CRY(params[5], wires=[num_qubits-2, num_qubits-3])
    qml.RY(params[6], wires=[num_qubits-2])  

CQAVQE_DW8_exact.n_params = 7
CQAVQE_DW8_exact.name = "CQAVQE_DW8_exact"

def CQAVQE_DW16_exact(params, num_qubits, include_fermion=True):
    qml.RY(params[0], wires=[num_qubits-1])
    qml.CRY(params[1], wires=[num_qubits-1, num_qubits-2])
    qml.RY(params[2], wires=[num_qubits-3])
    qml.RY(params[3], wires=[num_qubits-2])
    qml.RY(params[4], wires=[num_qubits-4])
    qml.CRY(params[5], wires=[num_qubits-1, num_qubits-3])
    qml.CRY(params[6], wires=[num_qubits-1, num_qubits-4])
    qml.CRY(params[7], wires=[num_qubits-2, num_qubits-3])
    qml.RY(params[8], wires=[num_qubits-2])
    qml.CRY(params[9], wires=[num_qubits-2, num_qubits-4])
    qml.CRY(params[10], wires=[num_qubits-3, num_qubits-4])
    qml.RY(params[11], wires=[num_qubits-2])
    qml.CRY(params[12], wires=[num_qubits-3, num_qubits-2])
    qml.RY(params[13], wires=[num_qubits-3])
    qml.CRY(params[14], wires=[num_qubits-4, num_qubits-3])
    qml.RY(params[15], wires=[num_qubits-4])
    qml.CRY(params[16], wires=[num_qubits-4, num_qubits-2])
    qml.RY(params[17], wires=[num_qubits-3])

CQAVQE_DW16_exact.n_params = 18
CQAVQE_DW16_exact.name = "CQAVQE_DW16_exact"


def CQAVQE_DW32_exact(params, num_qubits, include_fermion=True):
    qml.RY(params[0], wires=[num_qubits-1])
    qml.CRY(params[1], wires=[num_qubits-1, num_qubits-2])
    qml.RY(params[2], wires=[num_qubits-3])
    qml.RY(params[3], wires=[num_qubits-2])
    qml.RY(params[4], wires=[num_qubits-4])
    qml.CRY(params[5], wires=[num_qubits-1, num_qubits-3])
    qml.CRY(params[6], wires=[num_qubits-1, num_qubits-4])
    qml.CRY(params[7], wires=[num_qubits-2, num_qubits-3])
    qml.RY(params[8], wires=[num_qubits-2])
    qml.CRY(params[9], wires=[num_qubits-2, num_qubits-4])
    qml.CRY(params[10], wires=[num_qubits-1, num_qubits-5])
    qml.RY(params[11], wires=[num_qubits-5])
    qml.CRY(params[12], wires=[num_qubits-3, num_qubits-4])
    qml.CRY(params[13], wires=[num_qubits-2, num_qubits-5])
    qml.RY(params[14], wires=[num_qubits-2])
    qml.CRY(params[15], wires=[num_qubits-3, num_qubits-5])
    qml.RY(params[16], wires=[num_qubits-3])
    qml.CRY(params[17], wires=[num_qubits-4, num_qubits-5])
    qml.RY(params[18], wires=[num_qubits-4])
    qml.CRY(params[19], wires=[num_qubits-4, num_qubits-3])
    qml.RY(params[20], wires=[num_qubits-4])
    qml.CRY(params[21], wires=[num_qubits-4, num_qubits-1])
    qml.RY(params[22], wires=[num_qubits-4])

CQAVQE_DW32_exact.n_params = 23
CQAVQE_DW32_exact.name = "CQAVQE_DW32_exact"


def CQAVQE_DW64_exact(params, num_qubits, include_fermion=True):
    qml.RY(params[0], wires=[num_qubits-1])
    qml.CRY(params[1], wires=[num_qubits-1, num_qubits-2])
    qml.RY(params[2], wires=[num_qubits-3])
    qml.RY(params[3], wires=[num_qubits-2])
    qml.RY(params[4], wires=[num_qubits-4])
    qml.CRY(params[5], wires=[num_qubits-1, num_qubits-3])
    qml.CRY(params[6], wires=[num_qubits-1, num_qubits-4])
    qml.CRY(params[7], wires=[num_qubits-2, num_qubits-3])
    qml.RY(params[8], wires=[num_qubits-2])
    qml.CRY(params[9], wires=[num_qubits-2, num_qubits-4])
    qml.CRY(params[10], wires=[num_qubits-1, num_qubits-5])
    qml.RY(params[11], wires=[num_qubits-5])
    qml.CRY(params[12], wires=[num_qubits-3, num_qubits-4])
    qml.CRY(params[13], wires=[num_qubits-2, num_qubits-5])
    qml.RY(params[14], wires=[num_qubits-2])
    qml.CRY(params[15], wires=[num_qubits-3, num_qubits-5])
    qml.CRY(params[16], wires=[num_qubits-1, num_qubits-6])
    qml.RY(params[17], wires=[num_qubits-3])
    qml.RY(params[18], wires=[num_qubits-6])
    qml.CRY(params[19], wires=[num_qubits-4, num_qubits-3])
    qml.CRY(params[20], wires=[num_qubits-4, num_qubits-5])
    qml.CRY(params[21], wires=[num_qubits-2, num_qubits-6])
    qml.RY(params[22], wires=[num_qubits-5])
    qml.RY(params[23], wires=[num_qubits-4])
    qml.CRY(params[24], wires=[num_qubits-4, num_qubits-1])
    qml.RY(params[25], wires=[num_qubits-4])
    qml.RY(params[26], wires=[num_qubits-6])
    

CQAVQE_DW64_exact.n_params = 27
CQAVQE_DW64_exact.name = "CQAVQE_DW64_exact"


################### AHO ###################
def CQAVQE_AHO2_exact(params, num_qubits, include_fermion=True):
    if include_fermion:
        basis = [1] + [0]*(num_qubits-1)
        qml.BasisState(basis, wires=range(num_qubits))
    qml.RY(params[0], wires=[0])  

CQAVQE_AHO2_exact.n_params = 1
CQAVQE_AHO2_exact.name = "CQAVQE_AHO2_exact"

def CQAVQE_AHO4_exact(params, num_qubits, include_fermion=True):
    if include_fermion:
        basis = [1] + [0]*(num_qubits-1)
        qml.BasisState(basis, wires=range(num_qubits))
    qml.RY(params[0], wires=[num_qubits-2])

CQAVQE_AHO4_exact.n_params = 1
CQAVQE_AHO4_exact.name = "CQAVQE_AHO4_exact"

def CQAVQE_AHO8_exact(params, num_qubits, include_fermion=True):
    if include_fermion:
        basis = [1] + [0]*(num_qubits-1)
        qml.BasisState(basis, wires=range(num_qubits))
    qml.RY(params[0], wires=[num_qubits-3])
    qml.RY(params[1], wires=[num_qubits-2])
    qml.CRY(params[2], wires=[num_qubits-2, num_qubits-3]) 

CQAVQE_AHO8_exact.n_params = 3
CQAVQE_AHO8_exact.name = "CQAVQE_AHO8_exact"

def CQAVQE_AHO16_exact(params, num_qubits, include_fermion=True):
    if include_fermion:
        basis = [1] + [0]*(num_qubits-1)
        qml.BasisState(basis, wires=range(num_qubits))
    qml.RY(params[0], wires=[num_qubits-3])
    qml.RY(params[1], wires=[num_qubits-4])
    qml.RY(params[2], wires=[num_qubits-2])
    qml.CRY(params[3], wires=[num_qubits-2, num_qubits-3])
    qml.CRY(params[4], wires=[num_qubits-2, num_qubits-4])
    qml.CRY(params[5], wires=[num_qubits-3, num_qubits-4])
    qml.CRY(params[6], wires=[num_qubits-4, num_qubits-3])  

CQAVQE_AHO16_exact.n_params = 7
CQAVQE_AHO16_exact.name = "CQAVQE_AHO16_exact"

def CQAVQE_AHO32_exact(params, num_qubits, include_fermion=True):
    if include_fermion:
        basis = [1] + [0]*(num_qubits-1)
        qml.BasisState(basis, wires=range(num_qubits))
    qml.RY(params[0], wires=[num_qubits-3])
    qml.RY(params[1], wires=[num_qubits-4])
    qml.RY(params[2], wires=[num_qubits-2])
    qml.CRY(params[3], wires=[num_qubits-2, num_qubits-3])
    qml.CRY(params[4], wires=[num_qubits-2, num_qubits-4])
    qml.RY(params[5], wires=[num_qubits-5])
    qml.CRY(params[6], wires=[num_qubits-3, num_qubits-4])  
    qml.CRY(params[7], wires=[num_qubits-2, num_qubits-5])
    qml.CRY(params[8], wires=[num_qubits-3, num_qubits-5])  
    qml.CRY(params[9], wires=[num_qubits-4, num_qubits-5])
    qml.RY(params[10], wires=[num_qubits-3])
    qml.CRY(params[11], wires=[num_qubits-4, num_qubits-3])  
    qml.RY(params[12], wires=[num_qubits-4])
    qml.CRY(params[13], wires=[num_qubits-5, num_qubits-3]) 
    qml.RY(params[14], wires=[num_qubits-5])
    qml.CRY(params[15], wires=[num_qubits-5, num_qubits-4]) 
    qml.RY(params[16], wires=[num_qubits-5])


CQAVQE_AHO32_exact.n_params = 17
CQAVQE_AHO32_exact.name = "CQAVQE_AHO32_exact"


def CQAVQE_AHO64_exact(params, num_qubits, include_fermion=True):
    if include_fermion:
        basis = [1] + [0]*(num_qubits-1)
        qml.BasisState(basis, wires=range(num_qubits))
    qml.RY(params[0], wires=[num_qubits-3])
    qml.RY(params[1], wires=[num_qubits-4])
    qml.RY(params[2], wires=[num_qubits-2])
    qml.CRY(params[3], wires=[num_qubits-2, num_qubits-3])
    qml.CRY(params[4], wires=[num_qubits-2, num_qubits-4])
    qml.RY(params[5], wires=[num_qubits-5])
    qml.CRY(params[6], wires=[num_qubits-3, num_qubits-4])  
    qml.CRY(params[7], wires=[num_qubits-2, num_qubits-5])
    qml.RY(params[8], wires=[num_qubits-4])
    qml.CRY(params[9], wires=[num_qubits-3, num_qubits-5])  
    qml.CRY(params[10], wires=[num_qubits-4, num_qubits-5])
    qml.CRY(params[11], wires=[num_qubits-4, num_qubits-5])
    qml.RY(params[12], wires=[num_qubits-3])
    qml.RY(params[13], wires=[num_qubits-3])
    qml.CRY(params[14], wires=[num_qubits-4, num_qubits-3])  
    qml.CRY(params[15], wires=[num_qubits-4, num_qubits-3])  
    qml.CRY(params[16], wires=[num_qubits-4, num_qubits-3])  
    qml.RY(params[17], wires=[num_qubits-4])
    qml.RY(params[18], wires=[num_qubits-4])
    qml.CRY(params[19], wires=[num_qubits-5, num_qubits-3]) 
    qml.RY(params[20], wires=[num_qubits-5])
    qml.CRY(params[21], wires=[num_qubits-5, num_qubits-4]) 
    qml.RY(params[22], wires=[num_qubits-5])
    qml.CRY(params[23], wires=[num_qubits-5, num_qubits-4]) 
    qml.RY(params[24], wires=[num_qubits-5])
    qml.RY(params[25], wires=[num_qubits-5])


CQAVQE_AHO64_exact.n_params = 26
CQAVQE_AHO64_exact.name = "CQAVQE_AHO64_exact"


def CQAVQE_AHO128_exact(params, num_qubits):
    basis = [1] + [0]*(num_qubits-1)
    qml.BasisState(basis, wires=range(num_qubits))
    qml.RY(params[0], wires=[num_qubits-3])
    qml.RY(params[1], wires=[num_qubits-4])
    qml.RY(params[2], wires=[num_qubits-2])
    qml.CRY(params[3], wires=[num_qubits-2, num_qubits-3])
    qml.CRY(params[4], wires=[num_qubits-2, num_qubits-4])
    qml.RY(params[5], wires=[num_qubits-5])
    qml.CRY(params[6], wires=[num_qubits-3, num_qubits-4])  
    qml.CRY(params[7], wires=[num_qubits-2, num_qubits-5])
    qml.RY(params[8], wires=[num_qubits-6])
    qml.CRY(params[9], wires=[num_qubits-3, num_qubits-5])  
    qml.CRY(params[10], wires=[num_qubits-2, num_qubits-6])
    qml.CRY(params[11], wires=[num_qubits-4, num_qubits-5])
    qml.RY(params[12], wires=[num_qubits-3])
    qml.CRY(params[13], wires=[num_qubits-3, num_qubits-6])  
    qml.CRY(params[14], wires=[num_qubits-4, num_qubits-6])  
    qml.RY(params[15], wires=[num_qubits-7])
    qml.CRY(params[16], wires=[num_qubits-4, num_qubits-3])  
    qml.RY(params[17], wires=[num_qubits-4])
    qml.RY(params[18], wires=[num_qubits-6])
    qml.RY(params[19], wires=[num_qubits-5])
    qml.CRY(params[20], wires=[num_qubits-3, num_qubits-2])  
    qml.CRY(params[21], wires=[num_qubits-4, num_qubits-2])  
    qml.RY(params[22], wires=[num_qubits-3])
    qml.CRY(params[23], wires=[num_qubits-5, num_qubits-3])  
    qml.RY(params[24], wires=[num_qubits-5])

    
CQAVQE_AHO128_exact.n_params = 25
CQAVQE_AHO128_exact.name = "CQAVQE_AHO128_exact"



################### COBYQA Adaptive-VQE Reduced Ansatze ####################      
'''
Reduced Ansatze produced from running the COBYQA Adaptive-VQE with shots=None
'''
################### QHO ###################
def CQAVQE_QHO_Reduced(params, num_qubits, include_fermion=True):
    if include_fermion:
        basis = [1] + [0]*(num_qubits-1)
        qml.BasisState(basis, wires=range(num_qubits))
    qml.RY(params[0], wires=[0])

CQAVQE_QHO_Reduced.n_params = 1
CQAVQE_QHO_Reduced.name = "CQAVQE_QHO_Reduced"

################### DW ###################
def CQAVQE_DW2_Reduced(params, num_qubits, include_fermion=True):
    qml.RY(params[0], wires=[num_qubits-1])     

CQAVQE_DW2_Reduced.n_params = 1
CQAVQE_DW2_Reduced.name = "CQAVQE_DW2_Reduced"

def CQAVQE_DW4_Reduced(params, num_qubits, include_fermion=True):
    if include_fermion:
        basis = [1] + [0]*(num_qubits-1)
        qml.BasisState(basis, wires=range(num_qubits))
    qml.RY(params[0], wires=[num_qubits-2])
    qml.RY(params[1], wires=[num_qubits-1])
    qml.CRY(params[2], wires=[num_qubits-2,num_qubits-1])     

CQAVQE_DW4_Reduced.n_params = 3
CQAVQE_DW4_Reduced.name = "CQAVQE_DW4_Reduced" 

def CQAVQE_DW8_Reduced(params, num_qubits, include_fermion=True):
    qml.RY(params[0], wires=[num_qubits-1])
    qml.CRY(params[1], wires=[num_qubits-1, num_qubits-2])
    qml.RY(params[2], wires=[num_qubits-3])
    qml.RY(params[3], wires=[num_qubits-2])


CQAVQE_DW8_Reduced.n_params = 4
CQAVQE_DW8_Reduced.name = "CQAVQE_DW8_Reduced"

def CQAVQE_DW16_Reduced(params, num_qubits, include_fermion=True):
    qml.RY(params[0], wires=[num_qubits-1])
    qml.CRY(params[1], wires=[num_qubits-1, num_qubits-2])
    qml.RY(params[2], wires=[num_qubits-3])
    qml.RY(params[3], wires=[num_qubits-2])
    

CQAVQE_DW16_Reduced.n_params = 4
CQAVQE_DW16_Reduced.name = "CQAVQE_DW16_Reduced"

################### AHO ###################
def CQAVQE_AHO2_Reduced(params, num_qubits, include_fermion=True):
    if include_fermion:
        basis = [1] + [0]*(num_qubits-1)
        qml.BasisState(basis, wires=range(num_qubits))
    qml.RY(params[0], wires=[0])  

CQAVQE_AHO2_Reduced.n_params = 1
CQAVQE_AHO2_Reduced.name = "CQAVQE_AHO2_Reduced"

def CQAVQE_AHO4_Reduced(params, num_qubits, include_fermion=True):
    if include_fermion:
        basis = [1] + [0]*(num_qubits-1)
        qml.BasisState(basis, wires=range(num_qubits))
    qml.RY(params[0], wires=[num_qubits-2])

CQAVQE_AHO4_Reduced.n_params = 1
CQAVQE_AHO4_Reduced.name = "CQAVQE_AHO4_Reduced"

def CQAVQE_AHO8_Reduced(params, num_qubits, include_fermion=True):
    if include_fermion:
        basis = [1] + [0]*(num_qubits-1)
        qml.BasisState(basis, wires=range(num_qubits))
    qml.RY(params[0], wires=[num_qubits-3])
    qml.RY(params[1], wires=[num_qubits-2])
    qml.CRY(params[2], wires=[num_qubits-2, num_qubits-3]) 

CQAVQE_AHO8_Reduced.n_params = 3
CQAVQE_AHO8_Reduced.name = "CQAVQE_AHO8_Reduced"

def CQAVQE_AHO16_Reduced(params, num_qubits, include_fermion=True):
    if include_fermion:
        basis = [1] + [0]*(num_qubits-1)
        qml.BasisState(basis, wires=range(num_qubits))
    qml.RY(params[0], wires=[num_qubits-2])
    qml.RY(params[1], wires=[num_qubits-3])
    qml.RY(params[2], wires=[num_qubits-4])
    qml.CRY(params[3], wires=[num_qubits-2, num_qubits-3])

CQAVQE_AHO16_Reduced.n_params = 4
CQAVQE_AHO16_Reduced.name = "CQAVQE_AHO16_Reduced"




################### COBYQA Adaptive-VQE circuit depth Ansatze ####################      
'''
Reduced Ansatze by limiting circuit depth
'''
################### DW ###################
def CQAVQE_DW16_CD3(params, num_qubits):
    #CD1
    qml.RY(params[0], wires=[num_qubits-1])
    qml.RY(params[1], wires=[num_qubits-3])
    qml.RY(params[2], wires=[num_qubits-4])
    #CD2
    qml.CRY(params[3], wires=[num_qubits-1, num_qubits-2])
    #CD3
    qml.RY(params[4], wires=[num_qubits-2])
    qml.CRY(params[5], wires=[num_qubits-1, num_qubits-3])

CQAVQE_DW16_CD3.n_params = 6
CQAVQE_DW16_CD3.name = "CQAVQE_DW16_CD3"


def CQAVQE_DW32_CD3(params, num_qubits):
    #CD1
    qml.RY(params[0], wires=[num_qubits-1])
    qml.RY(params[1], wires=[num_qubits-3])
    qml.RY(params[2], wires=[num_qubits-4])
    #CD2
    qml.CRY(params[3], wires=[num_qubits-1, num_qubits-2])
    #CD3
    qml.RY(params[4], wires=[num_qubits-2])
    qml.CRY(params[5], wires=[num_qubits-1, num_qubits-3])

CQAVQE_DW32_CD3.n_params = 6
CQAVQE_DW32_CD3.name = "CQAVQE_DW32_CD3"


def CQAVQE_DW64_CD3(params, num_qubits):
    #CD1
    qml.RY(params[0], wires=[num_qubits-1])
    qml.RY(params[1], wires=[num_qubits-3])
    qml.RY(params[2], wires=[num_qubits-4])
    #CD2
    qml.CRY(params[3], wires=[num_qubits-1, num_qubits-2])
    #CD3
    qml.RY(params[4], wires=[num_qubits-2])
    qml.CRY(params[5], wires=[num_qubits-1, num_qubits-3])

CQAVQE_DW64_CD3.n_params = 6
CQAVQE_DW64_CD3.name = "CQAVQE_DW64_CD3"


################### AHO ###################

def CQAVQE_AHO16_CD3(params, num_qubits):
    basis = [1] + [0]*(num_qubits-1)
    qml.BasisState(basis, wires=range(num_qubits))
    #CD1
    qml.RY(params[0], wires=[num_qubits-2])
    qml.RY(params[1], wires=[num_qubits-3])
    qml.RY(params[2], wires=[num_qubits-4])
    #CD2
    qml.CRY(params[3], wires=[num_qubits-2, num_qubits-3])
    #CD3
    qml.CRY(params[4], wires=[num_qubits-2, num_qubits-4])

CQAVQE_AHO16_CD3.n_params = 5
CQAVQE_AHO16_CD3.name = "CQAVQE_AHO16_CD3"


def CQAVQE_AHO32_CD3(params, num_qubits):
    basis = [1] + [0]*(num_qubits-1)
    qml.BasisState(basis, wires=range(num_qubits))
    #CD1
    qml.RY(params[0], wires=[num_qubits-2])
    qml.RY(params[1], wires=[num_qubits-3])
    qml.RY(params[2], wires=[num_qubits-4])
    qml.RY(params[3], wires=[num_qubits-5])
    #CD2
    qml.CRY(params[4], wires=[num_qubits-2, num_qubits-3])
    #CD3
    qml.CRY(params[5], wires=[num_qubits-2, num_qubits-4])

CQAVQE_AHO32_CD3.n_params = 6
CQAVQE_AHO32_CD3.name = "CQAVQE_AHO32_CD3"


def CQAVQE_AHO64_CD3(params, num_qubits):
    basis = [1] + [0]*(num_qubits-1)
    qml.BasisState(basis, wires=range(num_qubits))
    #CD1
    qml.RY(params[0], wires=[num_qubits-2])
    qml.RY(params[1], wires=[num_qubits-3])
    qml.RY(params[2], wires=[num_qubits-4])
    qml.RY(params[3], wires=[num_qubits-5])
    qml.RY(params[4], wires=[num_qubits-6])
    #CD2
    qml.CRY(params[5], wires=[num_qubits-2, num_qubits-3])
    #CD3
    qml.CRY(params[6], wires=[num_qubits-2, num_qubits-4])

CQAVQE_AHO64_CD3.n_params = 7
CQAVQE_AHO64_CD3.name = "CQAVQE_AHO64_CD3"

        
    
'''
Getter for ansatze
'''   
ANSATZE = {
    obj.name: obj
    for obj in globals().values()
    if callable(obj) and hasattr(obj, "name")
}

def get(name):
    try:
        return ANSATZE[name]
    except KeyError:
        raise ValueError(f"Ansatz '{name}' not found. Available: {list(ANSATZE)}")    






        