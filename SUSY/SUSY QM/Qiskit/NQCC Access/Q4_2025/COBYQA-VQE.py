import os, json, time, logging, dataclasses
from datetime import datetime
import numpy as np

from scipy.optimize import minimize

from qiskit.quantum_info import SparsePauliOp

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_aer.noise import NoiseModel

from qiskit_ibm_runtime import EstimatorV2 as Estimator, QiskitRuntimeService, Session
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.primitives import StatevectorEstimator

from susy_qm import calculate_Hamiltonian, ansatze


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import git
repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_tree_dir

#service = QiskitRuntimeService(name="NQCC-Q3")
path = r"C:\Users\Johnk\Documents\PhD\Quantum Computing Code\Quantum-Computing\apikey.json"
with open(path, encoding="utf-8") as f:
    api_key = json.load(f).get("apikey")

NQCC_IBM_QUANTUM_API_KEY = api_key
ibm_instance_crn = "crn:v1:bluemix:public:quantum-computing:us-east:a/d4f95db0515b47b7ba61dba8a424f873:ed0704ac-ad7d-4366-9bcc-4217fb64abd1::" #US
#ibm_instance_crn = "" #Frankfurt
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=NQCC_IBM_QUANTUM_API_KEY, instance=ibm_instance_crn)
    
def setup_logger(logfile_path, name, enabled=True):
    if not enabled:
        
        logger = logging.getLogger(f"{name}_disabled")
        logger.handlers = []               
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        logger.setLevel(logging.CRITICAL)
        return logger

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(logfile_path)
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def run_vqe(i, H, log_dir, log_enabled, ansatz, run_info):

    backend_name = run_info['backend']
    num_params = run_info['num_params']

    if backend_name == "Aer":

        if run_info['use_noise_model']:
            real_backend = service.backend("ibm_kingston")
            #real_backend = service.backend("ibm_strasbourg")
            noise_model = NoiseModel.from_backend(real_backend)
            backend = AerSimulator(noise_model=noise_model)
           
        else:
            backend = AerSimulator(method="statevector")

    elif backend_name == "SV-Estimator":
        pass

    else:
        backend = service.backend(backend_name)

    if log_enabled: os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"vqe_run_{i}.log")
    logger = setup_logger(log_path, f"logger_{i}", enabled=log_enabled)

    seed = (os.getpid() * int(time.time())) % 123456789
    run_info["seed"] = seed
    #seed = 41540533

    run_start = datetime.now()

    observable = SparsePauliOp.from_operator(H)

    #param_objs = [Parameter(f"θ{i}") for i in range(num_params)]
    #qc = QuantumCircuit(num_qubits)
    #n = num_qubits-1
    #qc.x(n)
    #qc.ry(param_objs[0], n)

    qc = ansatze.pl_to_qiskit(ansatz, num_qubits=num_qubits, reverse_bits=True)


    if (backend_name in ["ibm_kingston", "ibm_strasbourg"]) or use_noise_model:
        target = backend.target
        pm = generate_preset_pass_manager(target=target, optimization_level=run_info["optimization_level"])
        ansatz_isa = pm.run(qc)

        layout = getattr(ansatz_isa, "layout", None)
        hamiltonian_isa = observable.apply_layout(layout) if layout else observable

        if log_enabled: logger.info(f"Hamiltonian: {hamiltonian_isa}")
    else:
        ansatz_isa = qc
        hamiltonian_isa = observable

    last_energy = None
    iteration_count = 0
    def cost_function(params, ansatz_isa, hamiltonian_isa, estimator):
        nonlocal last_energy, iteration_count

        #params = [
        #    0.3966509377256218,
        #    5.951916764806598,
        #    6.008466886069231
        #]

        pub = (ansatz_isa, [hamiltonian_isa], params)
        result = estimator.run(pubs=[pub]).result()
        energy = result[0].data.evs[0]

        last_energy = energy
        iteration_count +=1

        neg = max(0.0, -(energy + eps))
        
        return energy + lam * (neg ** p)
    
    
    def callback(xk, convergence=None):
        if log_enabled:
            logger.info(f"Iteration {iteration_count}: Energy = {last_energy:.8f}")

    if log_enabled: logger.info(json.dumps(run_info, indent=4, default=str))
    if log_enabled: logger.info(f"Starting VQE run {i} (seed={seed})")
    #if log_enabled: logger.info(qc)

    np.random.seed(seed)
    x0 = np.random.random(size=num_params)*2*np.pi
    bounds = [(0, 2 * np.pi) for _ in range(num_params)]

    if backend_name == "SV-Estimator":

        sesh_id = 'N/A'

        print("Running for SV Estimator")
        estimator = StatevectorEstimator()

        res = minimize(
            cost_function,
            x0,
            args=(qc,observable,estimator),
            #bounds=bounds,
            method= "COBYQA",
            options= {
                'maxiter':max_iter, 
                'maxfev':2*max_iter, 
                'initial_tr_radius':initial_tr_radius, 
                'final_tr_radius':final_tr_radius, 
                'scale':True, 
                'disp':False},
            callback=callback
        )

    else:
        print("Running with session")
        
        with Session(backend=backend) as session:

            sesh_id = session.session_id

            print("Session ID:", sesh_id)
            if log_enabled: logger.info(f"Session ID: {sesh_id}")

            if backend_name == "Aer":

                estimator = AerEstimator(
                    options={
                        "backend_options": {
                            "method": "automatic",
                            "noise_model": noise_model if use_noise_model else None,
                            "seed_simulator": seed
                        },
                        "run_options": {
                            "shots": shots
                        }
                    }
                )
            else:
                estimator = Estimator(mode=session)
                estimator.options.environment.job_tags = tags
                estimator.options.default_shots = shots
                estimator.options.resilience_level = run_info["resilience_level"]
                
            if log_enabled: logger.info(json.dumps(dataclasses.asdict(estimator.options), indent=4, default=str))

            res = minimize(
                cost_function,
                x0,
                args=(ansatz_isa,hamiltonian_isa,estimator),
                #bounds=bounds,
                method= "COBYQA",
                options= {
                    'maxiter':max_iter, 
                    'maxfev':2*max_iter, 
                    'initial_tr_radius':initial_tr_radius, 
                    'final_tr_radius':final_tr_radius, 
                    'scale':True, 
                    'disp':False},
                callback=callback
            )

    
    run_end = datetime.now()

    results_data = {
        "seed": seed,
        "session_id": sesh_id,
        "energy": res.fun,
        "params": res.x.tolist(),
        "success": res.success,
        "num_iters": int(res.nfev),
        "run_time": run_end - run_start
    }


    if log_enabled: logger.info(f"Completed VQE run {i}: Energy = {res.fun:.6f}")
    if log_enabled: logger.info(f"optimizer message: {res.message}")
    if log_enabled: logger.info(json.dumps(results_data, indent=4, default=str))
        
    return results_data

if __name__ == "__main__":

    log_enabled = True

    backend_name = 'ibm_kingston'
    #backend_name = "ibm_strasbourg"
    #backend_name = "Aer"
    #backend_name = "SV-Estimator"

    use_noise_model = 0
    shots = 4096
    optimization_level = 3
    resilience_level = 2

    potential = "QHO"
    cutoff = 8

    ansatze_type = 'exact' #exact, Reduced, CD3

    if potential == "QHO":
        ansatz_name = f"CQAVQE_QHO_{ansatze_type}"
    elif (potential != "QHO") and (cutoff <= 64):
        ansatz_name = f"CQAVQE_{potential}{cutoff}_{ansatze_type}"
    else:
        ansatz_name = f"CQAVQE_{potential}16_{ansatze_type}"

    #ansatz_name = f"CQAVQE_{potential}{cutoff}_CD3"
    #ansatz_name = "real_amplitudes"

    ansatz = ansatze.get(ansatz_name)

    num_vqe_runs = 1
    max_iter = 200
    initial_tr_radius = 0.3
    final_tr_radius = 1e-8

    lam = 15
    p = 2

    if potential == "AHO":
        i = np.log2(cutoff)
        factor = 2**(((i-1)*i)/2)
        eps = 0.5 / factor
    else:
        eps = 0

    print(f"Running for {potential} potential with cutoff {cutoff} and shots {shots}")

    starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #base_path = os.path.join(repo_path,r"SUSY\SUSY QM\Qiskit\NQCC Access\Q4_2025\Files\{}".format(backend_name), str(shots), potential, str(starttime))
    base_path = os.path.join(repo_path,r"SUSY\SUSY QM\Qiskit\NQCC Access\Q4_2025\Files\ResilienceLevelTesting\QHO8\Level2")
    os.makedirs(base_path, exist_ok=True)

    log_path = os.path.join(base_path, f"logs_{str(cutoff)}")
    
    H = calculate_Hamiltonian(cutoff, potential)
    eigenvalues = np.sort(np.linalg.eigvals(H))[:4]
    num_qubits = int(1 + np.log2(cutoff))

    if ansatz_name == 'real_amplitudes':
        num_params = 2*num_qubits
    else:
        num_params = ansatz.n_params


    tags=["NQCC-Q4", ansatz_name, f"shots:{shots}", "Resilience Test", f"Resilience={resilience_level}"]

    run_info = {"backend":backend_name,
                "use_noise_model": use_noise_model,
                "Potential":potential,
                "cutoff": cutoff,
                "num_qubits": num_qubits,
                "num_params": num_params,
                "shots": shots,
                "optimization_level": optimization_level,
                "resilience_level": resilience_level,
                "lam": lam,
                "p":p,
                "eps":eps,
                "num_vqe_runs": num_vqe_runs,
                "max_iter": max_iter,
                "initial_tr_radius": initial_tr_radius,
                "final_tr_radius": final_tr_radius,
                "ansatz_name": ansatz_name,
                "path": base_path,
                "tags":tags
                }

    print(json.dumps(run_info, indent=4, default=str))

    vqe_starttime = datetime.now()

    i=1
    vqe_results =  run_vqe(i, H, log_path, log_enabled, ansatz, run_info)

    # Collect results
    seeds = vqe_results["seed"]
    session_id = vqe_results["session_id"]
    energies = vqe_results["energy"]
    x_values = vqe_results["params"]
    success = vqe_results["success"]
    num_iters = vqe_results["num_iters"]
    run_time = str(vqe_results["run_time"])

    vqe_end = datetime.now()
    vqe_time = vqe_end - vqe_starttime

    # Save run
    run = {
        "backend": backend_name,
        "session_id": session_id,
        "use_noise_model": use_noise_model,
        "starttime": starttime,
        "endtime": vqe_end.strftime("%Y-%m-%d_%H-%M-%S"),
        "potential": potential,
        "cutoff": cutoff,
        "exact_eigenvalues": [x.real.tolist() for x in eigenvalues],
        "ansatz": ansatz_name,
        "num_VQE": num_vqe_runs,
        "shots": shots,
        "optimization_level": optimization_level,
        "resilience_level": resilience_level,
        "Optimizer": {
            "name": "COBYQA",
            "maxiter": max_iter,
            'maxfev': 2*max_iter,
            "initial_tr_radius": initial_tr_radius,
            "final_tr_radius": final_tr_radius
        },
        "cost function":{
            "type": "small negatives",
            "p":p,
            "lam":lam,
            "eps":eps
        },
        "results": energies,
        "params": x_values,
        "num_iters": num_iters,
        "success": np.array(success, dtype=bool).tolist(),
        "run_time": run_time,
        "VQE_run_time": str(vqe_time),
        "seeds": seeds,
    }

    # Save the variable to a JSON file
    path = os.path.join(base_path, "{}_{}.json".format(potential, cutoff))
    with open(path, "w") as json_file:
        json.dump(run, json_file, indent=4)

    print("Done")
