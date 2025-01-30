# PennyLane imports
import pennylane as qml
from pennylane import numpy as pnp

# General imports
import numpy as np

from qiskit.quantum_info import SparsePauliOp

# custom module
from susy_qm import calculate_Hamiltonian, calculate_wz_hamiltonian

from scipy.optimize import minimize

import os
import json

class adaptive_ansatz:

    def __init__(self, potential, cutoff, num_steps=10, type='qm', include_cnot=False, **kwargs):
        
        self.potential = potential
        self.cutoff = cutoff
        self.type = type
        self.num_steps = num_steps

        if type == 'qm':
            self.H = calculate_Hamiltonian(cutoff, potential)

        elif type == 'wz':

            self.N = kwargs.get("N") 
            self.a = kwargs.get("a")
            self.c = kwargs.get("c")
            self.bc = kwargs.get("bc")

            self.H = calculate_wz_hamiltonian(cutoff, self.N, self.a, potential, self.bc, self.c)

        hamiltonian = SparsePauliOp.from_operator(self.H)
        self.num_qubits = hamiltonian.num_qubits
        self.num_params = 0

        eigenvalues, eigenvectors = np.linalg.eig(self.H)
        min_index = np.argmin(eigenvalues)
        self.min_eigenvalue = eigenvalues[min_index]
        self.min_eigenvector = np.asarray(eigenvectors[:, min_index])
       
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
    
    
    def run_adapt_vqe(self, num_steps=None, o_iters = 1000, o_tol=1e-6, con_tol=1e-6):

        print("Running ADAPT VQE")
        num_steps = num_steps if num_steps is not None else self.num_steps

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

            if ((self.type == 'qm') & (i != 0)) | ((self.type == 'wz') & (i > 15)):
                if np.abs(min_energy - op_list[i-1][3]) < con_tol:
                    print("Converged")
                    break

            op_list.append((min_op, min_params, min_wires, min_energy))

        self.op_list = op_list

        print("Done")

        return op_list
        
        
    def reduce_op_list(self, op_list=None):

        print("Reducing op list")

        op_list = op_list if op_list is not None else self.op_list

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
        self.reduced_op_list = reduced_op_list

        print("Done")

        return reduced_op_list
    


    def construct_ansatz(self, params, reduced_op_list=None, type='expval', Hamiltonian=None, num_qubits=None):

        rol = reduced_op_list if reduced_op_list is not None else self.reduced_op_list
        H = Hamiltonian if Hamiltonian is not None else self.H
        nq = num_qubits if num_qubits is not None else self.num_qubits

        dev = qml.device("default.qubit", wires=nq)
        @qml.qnode(dev)
        def ansatz():

            params_index = 0

            for o, w in rol:
                if o == "CNOT":
                    qml.CNOT(wires=w)
                elif o == "CZ":
                    qml.CZ(wires=w)
                else:
                    num_gate_params = qml.Rot.num_params
                    qml.Rot(*params[params_index:(params_index + num_gate_params)], wires=w)
                    params_index = params_index + num_gate_params

            if type == 'expval':
                return qml.expval(qml.Hermitian(H, wires=range(nq)))
            else:
                return qml.state()

        return ansatz
    

        
    def run_overlap_test(self, reduced_op_list=None,  min_eigenvector=None, num_params=None, o_iters = 10000, o_tol=1e-8):

        print("Running overlap test")

        nump = num_params if num_params is not None else self.num_params
        me = min_eigenvector if min_eigenvector is not None else self.min_eigenvector

        def overlap_function(params, type):

            params = pnp.tensor(params, requires_grad=True)
            ansatz_state = self.construct_ansatz(params, reduced_op_list, type='state')()
            
            if type == 'overlap':
                overlap = np.vdot(me, ansatz_state)
                cost = np.abs(overlap)**2  
            
            else:
                min_eigenvector_prob = np.abs(me)**2
                ansatz_prob = np.abs(ansatz_state)**2
                cost = np.sum(np.sqrt(min_eigenvector_prob * ansatz_prob))

            return (1 - cost)


        x0 = np.random.uniform(0, 2 * np.pi, size=nump)

        print("Running for overlap")

        overlap_res = minimize(
            overlap_function,
            x0,
            args=('overlap'),
            method= "COBYLA",
            options= {'maxiter':o_iters, 'tol': o_tol}
        )

        overlap = overlap_res.fun
        self.overlap = overlap

        print(f"Overlap: {overlap}")

        print("Running for hellinger")

        hellinger_res = minimize(
            overlap_function,
            x0,
            args=('hellinger'),
            method= "COBYLA",
            options= {'maxiter':o_iters, 'tol': o_tol}
        )

        hellinger_fidelity = hellinger_res.fun
        self.hellinger_fidelity = hellinger_fidelity

        print(f"Hellinger fidelity: {hellinger_fidelity}")

        print("Done")

        return overlap, hellinger_fidelity


    def save_data(self, base_path):

        print("Saving data")

        params = np.random.uniform(0, 2 * np.pi, size=self.num_params)
        fig, ax = qml.draw_mpl(self.construct_ansatz(params))()

        if self.type == 'qm':

            data = {"potential": self.potential,
                "cutoff": self.cutoff,
                "optimizer": "COBYLA",
                "num steps": self.num_steps,
                "op_list": [str(x) for x in self.op_list],
                "reduced_op_list": self.reduced_op_list,
                "num_params": self.num_params,
                "overlap": self.overlap,
                "hellinger_fidelity": self.hellinger_fidelity}

            base_path = os.path.join(base_path, self.potential)
            os.makedirs(base_path, exist_ok=True)
            path = os.path.join(base_path, "{}_{}.json".format(self.potential, self.cutoff))

            print(f"Saving to: {path}")

            with open(path, 'w') as json_file:
                json.dump(data, json_file, indent=4)

            print("Data saved")

            print("Saving circuit image")
            pic_path = os.path.join(base_path, "{}_{}_circuit_diagram.png".format(self.potential, self.cutoff))
            fig.savefig(pic_path)
    
        elif self.type == 'wz':

            data = {"potential": self.potential,
                "cutoff": self.cutoff,
                "bc": self.bc,
                'N': self.N,
                'a': self.a,
                'c': None if self.potential == "linear" else self.c,
                "optimizer": "COBYLA",
                "num steps": self.num_steps,
                "op_list": [str(x) for x in self.op_list],
                "reduced_op_list": self.reduced_op_list,
                "num_params": self.num_params,
                "overlap": self.overlap,
                "hellinger_fidelity": self.hellinger_fidelity}

            if self.potential == 'quadratic':
                folder = 'C' + str(abs(self.c)) + '/' + 'N'+ str(self.N)
            else:
                folder = 'N'+ str(self.N)

            base_path = os.path.join(base_path, self.bc, self.potential, folder)
            os.makedirs(base_path, exist_ok=True)
            path = os.path.join(base_path, "{}_{}.json".format(self.potential, self.cutoff))

            print(f"Saving to: {path}")

            with open(path, 'w') as json_file:
                json.dump(data, json_file, indent=4)

            print("Saving circuit image")
            pic_path = os.path.join(base_path, "{}_{}_circuit_diagram.png".format(self.potential, self.cutoff))
            fig.savefig(pic_path)

        print("Data saved")
        

    def run_all(self, path):
        self.run_adapt_vqe()
        self.reduce_op_list()
        self.run_overlap_test()
        self.save_data(path)

        
        
    

        