# PennyLane imports
import pennylane as qml
from pennylane import numpy as pnp

# General imports
import numpy as np

from qiskit.quantum_info import SparsePauliOp

# custom module
from susy_qm import calculate_Hamiltonian

from scipy.optimize import minimize

class qm_ansatz:

    def __init__(self, potential, cutoff, include_cnot=False):
        
        self.potential = potential
        self.cutoff = cutoff
        self.H = calculate_Hamiltonian(self.cutoff, self.potential)
        hamiltonian = SparsePauliOp.from_operator(self.H)
        self.num_qubits = hamiltonian.num_qubits
        self.num_params = 0

        cnot_pool = []
        cz_pool = []

        for control in range(self.num_qubits):
                for target in range(self.num_qubits):
                    if control != target:
                        cnot_pool.append(qml.CNOT(wires=[control, target]))
                        cz_pool.append(qml.CZ(wires=[control, target]))

        rot_pool = [qml.Rot(0.0, 0.0, 0.0, wires=x) for x in range(self.num_qubits)]
        self.operator_pool = rot_pool +  cz_pool
        if include_cnot: self.operator_pool += cnot_pool


    
    def create_circuit(self, params, trial_op, op_list):

        dev = qml.device("default.qubit", wires=self.num_qubits)

        @qml.qnode(dev)
        def circuit():

            if len(op_list) > 0:
                for o, p, w, _ in op_list:
                    if (o == qml.CNOT) | (o == qml.CZ):
                        o(wires=w)
                    else:
                        o(*p, wires=w)

            op = type(trial_op)

            if (type(trial_op) == qml.CNOT) | (type(trial_op) == qml.CZ):
                op(wires=trial_op.wires)
            else:
                op(*params, wires=trial_op.wires)

            return qml.expval(qml.Hermitian(self.H, wires=range(self.num_qubits)))
        
        return circuit()
    
    
    def cost_function(self, params, trial_op, op_list):

        params = pnp.tensor(params, requires_grad=True)
        energy = self.create_circuit(params, trial_op, op_list)

        return energy
    
    
    def run_adapt_vqe(self, num_steps, o_iters = 1000, o_tol=1e-6, con_tol=1e-4):

        x0 = np.random.uniform(0, 2 * np.pi, size=3)

        op_list = []

        for i in range(num_steps):

            print(f"step: {i}")

            energies = []
            e_params = []

            for trial_op in self.operator_pool:

                res = minimize(
                        self.cost_function,
                        x0,
                        args=(trial_op, op_list),
                        method= "COBYLA",
                        options= {'maxiter':o_iters, 'tol': o_tol}
                    )
                
                energies.append(res.fun)
                e_params.append(res.x)

            min_arg = np.argmin(energies)
            min_energy = energies[min_arg]
            print(f"Min energy: {min_energy}")

            min_op = type(self.operator_pool[min_arg])
            min_wires = self.operator_pool[min_arg].wires
            min_params = e_params[min_arg]

            if (i != 0):
                if np.abs(min_energy - op_list[i-1][3]) < con_tol:
                    print("Converged")
                    break

            op_list.append((min_op, min_params, min_wires, min_energy))

        return op_list
        
        
    def reduce_op_list(self, op_list):

        last_operator = {}
        reduced_op_list = []
        num_params = 0

        for o, p, w, _ in op_list:

            if o == qml.CNOT:
                last_operator[w[0]] = o
                last_operator[w[1]] = o
                reduced_op_list.append(("CNOT", w.tolist()))

            elif o == qml.CZ:
                last_operator[w[0]] = o
                last_operator[w[1]] = o
                reduced_op_list.append(("CZ",w.tolist()))

            elif w[0] in last_operator.keys():
                if last_operator[w[0]] == o:
                    continue
                else:
                    last_operator[w[0]] = o
                    reduced_op_list.append(("Rot",w.tolist()))
                    num_params = num_params + 3
            else:
                last_operator[w[0]] = o
                reduced_op_list.append(("Rot",w.tolist()))
                num_params = num_params + 3

        self.num_params = num_params

        return reduced_op_list
    


    def construct_ansatz(self, reduced_op_list, params):

        dev = qml.device("default.qubit", wires=self.num_qubits)
        @qml.qnode(dev)
        def ansatz():

            params_index = 0

            for o, w in reduced_op_list:
                if o == "CNOT":
                    qml.CNOT(wires=w)
                elif o == "CZ":
                    qml.CZ(wires=w)
                else:
                    num_gate_params = qml.Rot.num_params
                    qml.Rot(*params[params_index:(params_index + num_gate_params)], wires=w)
                    params_index = params_index + num_gate_params

            return qml.expval(qml.Hermitian(self.H, wires=range(self.num_qubits)))

        return ansatz
    
    

        