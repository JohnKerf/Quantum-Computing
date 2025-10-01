import pennylane as qml

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



def truncate_ansatz(ansatz_fn, params, num_qubits, max_gates):
    with qml.tape.QuantumTape() as tape:
        ansatz_fn(params, num_qubits)

    reduced = tape.operations[:max_gates+1]

    return reduced




################### Real Amplitudes ####################    
def real_amplitudes(params, num_qubits, circular=True):
    
    n = num_qubits-1
    wires = range(num_qubits)

    for i, w in enumerate(wires):
        qml.RY(params[i], wires=w)

    for i in range(1, num_qubits):
        qml.CNOT(wires=[wires[i-1], wires[i]])

    if circular: qml.CNOT(wires=[wires[-1], wires[0]])

    for i, w in enumerate(wires):
        qml.RY(params[n + i], wires=w)

    real_amplitudes.n_params = 2*num_qubits
    real_amplitudes.name = "real_amplitudes"



################### COBYQA Adaptive-VQE Exact Ansatze ####################      
'''
Ansatze produced from running the COBYQA Adaptive-VQE with shots=None
'''
################### QHO ###################
def CQAVQE_QHO_exact(params, num_qubits):
    basis = [1] + [0]*(num_qubits-1)
    qml.BasisState(basis, wires=range(num_qubits))
    qml.RY(params[0], wires=[0])

CQAVQE_QHO_exact.n_params = 1
CQAVQE_QHO_exact.name = "CQAVQE_QHO_exact"

################### DW ###################
def CQAVQE_DW2_exact(params, num_qubits):
    qml.RY(params[0], wires=[1])    

CQAVQE_DW2_exact.n_params = 1
CQAVQE_DW2_exact.name = "CQAVQE_DW2_exact"

def CQAVQE_DW4_exact(params, num_qubits):
    basis = [1] + [0]*(num_qubits-1)
    qml.BasisState(basis, wires=range(num_qubits))
    qml.RY(params[0], wires=[1])
    qml.RY(params[1], wires=[2])
    qml.CRY(params[2], wires=[1,2])   

CQAVQE_DW4_exact.n_params = 3
CQAVQE_DW4_exact.name = "CQAVQE_DW4_exact" 

def CQAVQE_DW8_exact(params, num_qubits):
    qml.RY(params[0], wires=[num_qubits-1])
    qml.CRY(params[1], wires=[num_qubits-1, num_qubits-2])
    qml.RY(params[2], wires=[num_qubits-3])
    qml.RY(params[3], wires=[num_qubits-2])
    qml.CRY(params[4], wires=[num_qubits-1, num_qubits-3])
    qml.CRY(params[5], wires=[num_qubits-2, num_qubits-3])
    qml.RY(params[6], wires=[num_qubits-2])  

CQAVQE_DW8_exact.n_params = 7
CQAVQE_DW8_exact.name = "CQAVQE_DW8_exact"

def CQAVQE_DW16_exact(params, num_qubits):
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

################### AHO ###################
def CQAVQE_AHO2_exact(params, num_qubits):
    basis = [1] + [0]*(num_qubits-1)
    qml.BasisState(basis, wires=range(num_qubits))
    qml.RY(params[0], wires=[0])  

CQAVQE_AHO2_exact.n_params = 1
CQAVQE_AHO2_exact.name = "CQAVQE_AHO2_exact"

def CQAVQE_AHO4_exact(params, num_qubits):
    basis = [1] + [0]*(num_qubits-1)
    qml.BasisState(basis, wires=range(num_qubits))
    qml.RY(params[0], wires=[1])

CQAVQE_AHO4_exact.n_params = 1
CQAVQE_AHO4_exact.name = "CQAVQE_AHO4_exact"

def CQAVQE_AHO8_exact(params, num_qubits):
    basis = [1] + [0]*(num_qubits-1)
    qml.BasisState(basis, wires=range(num_qubits))
    qml.RY(params[0], wires=[num_qubits-3])
    qml.RY(params[1], wires=[num_qubits-2])
    qml.CRY(params[2], wires=[num_qubits-2, num_qubits-3]) 

CQAVQE_AHO8_exact.n_params = 3
CQAVQE_AHO8_exact.name = "CQAVQE_AHO8_exact"

def CQAVQE_AHO16_exact(params, num_qubits):
    basis = [1] + [0]*(num_qubits-1)
    qml.BasisState(basis, wires=range(num_qubits))
    qml.RY(params[0], wires=[num_qubits-2])
    qml.RY(params[1], wires=[num_qubits-3])
    qml.RY(params[2], wires=[num_qubits-4])
    qml.CRY(params[3], wires=[num_qubits-2, num_qubits-3])
    qml.CRY(params[4], wires=[num_qubits-2, num_qubits-4])
    qml.CRY(params[5], wires=[num_qubits-3, num_qubits-4])
    qml.CRY(params[6], wires=[num_qubits-4, num_qubits-3])  

CQAVQE_AHO16_exact.n_params = 7
CQAVQE_AHO16_exact.name = "CQAVQE_AHO16_exact"



################### COBYQA Adaptive-VQE Reduced Ansatze ####################      
'''
Reduced Ansatze produced from running the COBYQA Adaptive-VQE with shots=None
'''
################### QHO ###################
def CQAVQE_QHO_Reduced(params, num_qubits):
    basis = [1] + [0]*(num_qubits-1)
    qml.BasisState(basis, wires=range(num_qubits))
    qml.RY(params[0], wires=[0])

CQAVQE_QHO_Reduced.n_params = 1
CQAVQE_QHO_Reduced.name = "CQAVQE_QHO_Reduced"

################### DW ###################
def CQAVQE_DW2_Reduced(params, num_qubits):
    qml.RY(params[0], wires=[1])    

CQAVQE_DW2_Reduced.n_params = 1
CQAVQE_DW2_Reduced.name = "CQAVQE_DW2_Reduced"

def CQAVQE_DW4_Reduced(params, num_qubits):
    basis = [1] + [0]*(num_qubits-1)
    qml.BasisState(basis, wires=range(num_qubits))
    qml.RY(params[0], wires=[1])
    qml.RY(params[1], wires=[2])
    qml.CRY(params[2], wires=[1,2])   

CQAVQE_DW4_Reduced.n_params = 3
CQAVQE_DW4_Reduced.name = "CQAVQE_DW4_Reduced" 

def CQAVQE_DW8_Reduced(params, num_qubits):
    qml.RY(params[0], wires=[num_qubits-1])
    qml.CRY(params[1], wires=[num_qubits-1, num_qubits-2])
    qml.RY(params[2], wires=[num_qubits-3])
    qml.RY(params[3], wires=[num_qubits-2])


CQAVQE_DW8_Reduced.n_params = 4
CQAVQE_DW8_Reduced.name = "CQAVQE_DW8_Reduced"

def CQAVQE_DW16_Reduced(params, num_qubits):
    qml.RY(params[0], wires=[num_qubits-1])
    qml.CRY(params[1], wires=[num_qubits-1, num_qubits-2])
    qml.RY(params[2], wires=[num_qubits-3])
    qml.RY(params[3], wires=[num_qubits-2])
    

CQAVQE_DW16_Reduced.n_params = 4
CQAVQE_DW16_Reduced.name = "CQAVQE_DW16_Reduced"

################### AHO ###################
def CQAVQE_AHO2_Reduced(params, num_qubits):
    basis = [1] + [0]*(num_qubits-1)
    qml.BasisState(basis, wires=range(num_qubits))
    qml.RY(params[0], wires=[0])  

CQAVQE_AHO2_Reduced.n_params = 1
CQAVQE_AHO2_Reduced.name = "CQAVQE_AHO2_Reduced"

def CQAVQE_AHO4_Reduced(params, num_qubits):
    basis = [1] + [0]*(num_qubits-1)
    qml.BasisState(basis, wires=range(num_qubits))
    qml.RY(params[0], wires=[1])

CQAVQE_AHO4_Reduced.n_params = 1
CQAVQE_AHO4_Reduced.name = "CQAVQE_AHO4_Reduced"

def CQAVQE_AHO8_Reduced(params, num_qubits):
    basis = [1] + [0]*(num_qubits-1)
    qml.BasisState(basis, wires=range(num_qubits))
    qml.RY(params[0], wires=[num_qubits-3])
    qml.RY(params[1], wires=[num_qubits-2])
    qml.CRY(params[2], wires=[num_qubits-2, num_qubits-3]) 

CQAVQE_AHO8_Reduced.n_params = 3
CQAVQE_AHO8_Reduced.name = "CQAVQE_AHO8_Reduced"

def CQAVQE_AHO16_Reduced(params, num_qubits):
    basis = [1] + [0]*(num_qubits-1)
    qml.BasisState(basis, wires=range(num_qubits))
    qml.RY(params[0], wires=[num_qubits-2])
    qml.RY(params[1], wires=[num_qubits-3])
    qml.RY(params[2], wires=[num_qubits-4])
    qml.CRY(params[3], wires=[num_qubits-2, num_qubits-3])
  
CQAVQE_AHO16_Reduced.n_params = 4
CQAVQE_AHO16_Reduced.name = "CQAVQE_AHO16_Reduced"

        
    
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






        