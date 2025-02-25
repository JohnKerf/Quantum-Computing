import pennylane as qml
from scipy.optimize import minimize
import numpy as np
from datetime import datetime, timedelta

class cobyla_vqd:

    def __init__(self, dev, num_energy_levels, beta=2.0, num_swap_tests=1):
        
        self.dev = dev
        self.num_energy_levels = num_energy_levels
        self.beta = beta
        self.num_swap_tests = num_swap_tests

    

    def cost_function(ansatz, dev, params, prev_param_list, H, num_qubits, beta, num_swap_tests=1):
    
        #Swap test to calculate overlap
        @qml.qnode(dev)
        def swap_test(params, prev_params):

            ansatz(params)
            ansatz(prev_params, prev=True)

            qml.Barrier()
            for i in range(num_qubits):
                qml.CNOT(wires=[i, i+num_qubits])    
                qml.Hadamard(wires=i)      

            prob = qml.probs(wires=range(2*num_qubits))

            return prob
        

        @qml.qnode(dev)
        def expected_value(params):

            ansatz(params)
            exval = qml.expval(qml.Hermitian(H, wires=range(num_qubits)))

            return exval
        
        
        def overlap(params, prev_params):

            probs = swap_test(params, prev_params)

            overlap = 0
            for idx, p in enumerate(probs):

                bitstring = format(idx, '0{}b'.format(2*num_qubits))

                counter_11 = 0
                for i in range(num_qubits):
                    a = int(bitstring[i])
                    b = int(bitstring[i+num_qubits])
                    if (a == 1 and b == 1):
                        counter_11 +=1

                overlap += p*(-1)**counter_11

            return overlap
        
        
        def multi_swap_test(params, prev_params):
        
            results = []
            for _ in range(num_swap_tests):

                ol = overlap(params, prev_params)
                results.append(ol)
            
            avg_ol = sum(results) / num_swap_tests

            return avg_ol
        

        def loss_f(params):

            energy = expected_value(params)
            penalty = 0

            if len(prev_param_list) != 0:
                for prev_param in prev_param_list:
                    ol = multi_swap_test(params,prev_param)
                    penalty += (beta*ol)

            return energy + (penalty)

        return loss_f(params)


    def run_vqd(x0, max_iter, tol, H, num_qubits, shots):
        
        all_energies = []
        prev_param_list = []
        all_success = []
        all_num_iters = []
        all_evaluations = []
        all_dev_times = []

        device_time = timedelta()

        for _ in range(num_energy_levels):
            
            # Differential Evolution optimization
            res = minimize(
                    cost_function,
                    x0=x0,
                    args=(prev_param_list, H, num_qubits, shots, beta, num_swap_tests),
                    method= "COBYLA",
                    options= {'maxiter':10000, 'tol': 1e-8}
                )
            res = differential_evolution(
                cost_function(prev_param_list, H, num_qubits, shots, beta, num_swap_tests),
                bounds,
                maxiter=max_iter,
                tol=tol,
                atol=abs_tol,
                strategy=strategy,
                popsize=popsize,
                init=scaled_samples,
                seed=seed
            )


            all_energies.append(res.fun)
            prev_param_list.append(res.x)
            all_success.append(res.success)
            all_num_iters.append(res.nit)
            all_evaluations.append(res.nfev)

        run_end = datetime.now()
        run_time = run_end - run_start
        

        return {
            "seed": seed,
            "energies": all_energies,
            "params": prev_param_list,
            "success": all_success,
            "num_iters": all_num_iters,
            "evaluations": all_evaluations,
            "run_time": run_time,
            "device_time": device_time
        }