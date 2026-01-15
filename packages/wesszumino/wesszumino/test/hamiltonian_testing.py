from __future__ import annotations
import logging
import numpy as np
from typing import Tuple
from scipy.sparse import coo_matrix
from qiskit.quantum_info import Operator, SparsePauliOp, PauliList

from hamiltonian_logging import get_file_logger, log_matrix, log_paulis

# create a 0 sparse pauli op
def zero_sparse_pauliop(n_qubits):
    return SparsePauliOp(PauliList(["I" * n_qubits]), [0.0], ignore_pauli_phase=True)


# =============================================================================
# 1) Single-site HO matrices (boson) and fermion operators
# =============================================================================

def create_matrix(cutoff, kind, m = 1.0):
    """ q & p in the HO Fock basis up to cutoff."""
    mat = np.zeros((cutoff, cutoff), dtype=np.complex128)
    for i in range(cutoff):
        if i > 0:
            if kind == "q":
                mat[i, i - 1] = (1.0 / np.sqrt(2.0 * m)) * np.sqrt(i)
            elif kind == "p":
                mat[i, i - 1] = 1j * np.sqrt(m / 2.0) * np.sqrt(i)
        if i < cutoff - 1:
            if kind == "q":
                mat[i, i + 1] = (1.0 / np.sqrt(2.0 * m)) * np.sqrt(i + 1)
            elif kind == "p":
                mat[i, i + 1] = -1j * np.sqrt(m / 2.0) * np.sqrt(i + 1)
    return mat


def single_site_operators(cutoff, m = 1.0):
    
    q_b = create_matrix(cutoff, "q", m=m)
    p_b = create_matrix(cutoff, "p", m=m)
    I_b = np.eye(cutoff, dtype=np.complex128)

    chi = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    chi_dag = np.array([[0, 0], [1, 0]], dtype=np.complex128)
    I_f = np.eye(2, dtype=np.complex128)

    # site Hilbert space: fermion x boson
    q_site = np.kron(I_f, q_b)
    p_site = np.kron(I_f, p_b)
    chi_site = np.kron(chi, I_b)
    chi_dag_site = np.kron(chi_dag, I_b)

    return q_site, p_site, chi_site, chi_dag_site


# =============================================================================
# 2) Local dense blocks (1-site, 2-site)
# =============================================================================

def build_onsite_blocks(cutoff, a, potential="linear", c=0.0, m=1.0):
    """
    Build single-site dense blocks:

    H_bos_onsite = p^2/(2a) + (a/2) W'(q)^2
    H_bf_loc     = (n_f - 1/2 I) W''(q)
    """
    q_site, p_site, chi_site, chi_dag_site = single_site_operators(cutoff, m=m)
    D = q_site.shape[0]
    I_site = np.eye(D, dtype=np.complex128)

    # Fermion number on site
    n_f_site = chi_dag_site @ chi_site  # acts on H_site

    if potential == "linear":
        # W'(q)=q, W''(q)=1
        W_prime_site = q_site
        W_pp_site = np.eye(D, dtype=np.complex128)
    elif potential == "quadratic":
        # W'(q)=c+q^2, W''(q)=2q
        W_prime_site = c * I_site + q_site @ q_site
        W_pp_site = 2.0 * q_site
    else:
        raise ValueError("potential must be 'linear' or 'quadratic'")

    H_bos_kin = (p_site @ p_site) / (2.0 * a)
    H_bos_pot = 0.5 * a * (W_prime_site @ W_prime_site)
    H_bos_onsite = H_bos_kin + H_bos_pot

    H_bf_loc = (n_f_site - 0.5 * I_site) @ W_pp_site # no (-1)^n here

    return H_bos_onsite, H_bf_loc


def build_hopping_block(cutoff, a=1.0, m=1.0):
    """
    Two-site fermion hopping block on H_site⊗H_site:

    H_hop = 0.5 (chi_ndag chi_{n+1} + chi_{n+1}dag chi_n)

    No anti-periodic sign here apply -1 later.
    """
    q_site, p_site, chi_site, chi_dag_site = single_site_operators(cutoff, m=m)
    D = q_site.shape[0]
    I_site = np.eye(D, dtype=np.complex128)

    chi_n = np.kron(chi_site, I_site)
    chi_np1 = np.kron(I_site, chi_site)
    chi_dag_n = np.kron(chi_dag_site, I_site)
    chi_dag_np1 = np.kron(I_site, chi_dag_site)

    H_hop = 0.5 * (chi_dag_n @ chi_np1 + chi_dag_np1 @ chi_n)
    return H_hop


# =============================================================================
# 3) Dense -> SparsePauliOp + embedding on lattice sites
# =============================================================================

def dense_to_sparse_pauliop(H_dense, atol = 1e-12):
    op = Operator(H_dense)
    sp = SparsePauliOp.from_operator(op, atol=atol)
    return sp


def embed_sparse_pauliop_on_sites(H_block, sites,n_site,N):
    """
    Embed a local SparsePauliOp onto a full lattice with N sites and n_site qubits per site.

    qiskit qubit ordering means rightmost char acts on q0 and leftmost qn
    e.g. ZIX, q2=Z, q1=I and q0=X

    sites examples:
    [n] (1-site), [n, n+1] (2-site)

    Global qubit index convention:
    global_qubit = site*n_site + local_in_site
    """
    n_total = N * n_site
    n_block = len(sites) * n_site

    labels = H_block.paulis.to_labels()
    coeffs = H_block.coeffs

    out_labels = []
    out_coeffs = []

    for lab, c in zip(labels, coeffs):
        chars = ["I"] * n_total

        # local qubit -> global qubit map
        for k, site in enumerate(sites):
            for j in range(n_site):
                local_qubit = k * n_site + j
                global_qubit = site * n_site + j

                ch = lab[len(lab) - 1 - local_qubit] #get local qubit char based on qiskit ordering
                chars[n_total - 1 - global_qubit] = ch

        out_labels.append("".join(chars))
        out_coeffs.append(c)

    return SparsePauliOp(PauliList(out_labels), out_coeffs, ignore_pauli_phase=True)



# =============================================================================
# 4) WZ Hamiltonian (periodic/dirichlet BC)
# =============================================================================

def build_wz_hamiltonian(
                         cutoff, N, a, c=0.0, m=1.0, 
                         potential="linear", boundary_condition="periodic", 
                         atol_decompose=1e-12, atol_simplify=1e-12, remove_zero_terms=True, 
                         log_path=None, log_pauli_terms=True,  log_level= logging.INFO, log_mode="w", pauli_max_terms=120, pauli_sort_by_abs = False, log_running_total = False
                         ):
    """
    Build the Wess–Zumino Hamiltonian as a Qiskit SparsePauliOp using
    only local blocks (1-site and 2-site).
    """

    logger = None
    if log_path is not None:
        logger = get_file_logger(log_path, level=log_level, mode=log_mode)
        logger.info("=== build_wz_hamiltonian start ===")
        logger.info(f"N={N}, cutoff={cutoff}, a={a}, c={c}, m={m}, potential={potential}, bc={boundary_condition}")

    # single-site blocks
    H_bos_onsite_dense, H_bf_loc_dense = build_onsite_blocks(cutoff, a, potential=potential, c=c, m=m)
    D_site = H_bos_onsite_dense.shape[0]
    n_site = int(np.log2(D_site))
    n_total_qubits = N * n_site

    # if logger is not None:
    #     q_site, p_site, chi_site, chi_dag_site = single_site_operators(cutoff, m=m)
    #     log_matrix(logger, "q_site", q_site)
    #     log_matrix(logger, "p_site", p_site)
    #     log_matrix(logger, "chi_site", chi_site)
    #     log_matrix(logger, "chi_dag_site", chi_dag_site)

    #     log_matrix(logger, "H_bos_onsite_dense", H_bos_onsite_dense)
    #     log_matrix(logger, "H_bf_loc_dense", H_bf_loc_dense)

    H_bos_onsite = dense_to_sparse_pauliop(H_bos_onsite_dense, atol=atol_decompose)
    H_bf_loc = dense_to_sparse_pauliop(H_bf_loc_dense, atol=atol_decompose)

    if logger is not None and logger.isEnabledFor(logging.DEBUG) and log_pauli_terms:
        log_paulis(logger, "H_bos_onsite (Pauli decomposition)", H_bos_onsite,
                    max_terms=pauli_max_terms, sort_by_abs=pauli_sort_by_abs)
        log_paulis(logger, "H_bf_loc (Pauli decomposition)", H_bf_loc,
                    max_terms=pauli_max_terms, sort_by_abs=pauli_sort_by_abs)

    # two-site hopping
    H_hop_dense = build_hopping_block(cutoff, a=a, m=m)
    H_hop = dense_to_sparse_pauliop(H_hop_dense, atol=atol_decompose)
    
    # if logger is not None:
    #     log_matrix(logger, "H_hop_dense (2-site)", H_hop_dense)
    if logger is not None and logger.isEnabledFor(logging.DEBUG) and log_pauli_terms:
        log_paulis(logger, "H_hop (2-site Pauli decomposition)", H_hop,
                    max_terms=pauli_max_terms, sort_by_abs=pauli_sort_by_abs)

    # bosonic gradient + potential-gradient building blocks
    q_site, _, _, _ = single_site_operators(cutoff, m=m)
    I_site = np.eye(D_site, dtype=np.complex128)

    if potential == "linear":
        W_prime_site = q_site
    elif potential == "quadratic":
        W_prime_site = c * I_site + (q_site @ q_site)

    q2_site = q_site @ q_site
    qq_dense = np.kron(q_site, q_site)              # q X q
    Wp_q_dense = np.kron(W_prime_site, q_site)      # W'(q) X q

    # if logger is not None:
    #     log_matrix(logger, "q2_site (dense)", q2_site)
    #     log_matrix(logger, "qq_dense (2-site)", qq_dense)
    #     log_matrix(logger, "Wp_q_dense (2-site)", Wp_q_dense)

    H_q2 = dense_to_sparse_pauliop(q2_site, atol=atol_decompose)
    H_qq = dense_to_sparse_pauliop(qq_dense, atol=atol_decompose)
    H_Wp_q = dense_to_sparse_pauliop(Wp_q_dense, atol=atol_decompose)
    
    if logger is not None and logger.isEnabledFor(logging.DEBUG) and log_pauli_terms:
        log_paulis(logger, "H_q2 (Pauli decomposition)", H_q2,
                    max_terms=pauli_max_terms, sort_by_abs=pauli_sort_by_abs)
        log_paulis(logger, "H_qq (2-site Pauli decomposition)", H_qq,
                    max_terms=pauli_max_terms, sort_by_abs=pauli_sort_by_abs)
        log_paulis(logger, "H_Wp_q (2-site Pauli decomposition)", H_Wp_q,
                    max_terms=pauli_max_terms, sort_by_abs=pauli_sort_by_abs)

    # assemble for all sites
    H_total = zero_sparse_pauliop(n_total_qubits) # initiate full H as 0.0 x III...I

    def add_embedded(step_name: str, local_op: SparsePauliOp, sites: list[int], scale: float):
        nonlocal H_total
        embedded = embed_sparse_pauliop_on_sites(local_op, sites, n_site, N)

        if logger is not None:
            logger.info(f"[ADD] {step_name}: sites={sites}, scale={scale:+g}")
            if logger.isEnabledFor(logging.DEBUG) and log_pauli_terms:
                log_paulis(logger, f"Embedded {step_name} -> sites={sites}", embedded,
                            max_terms=pauli_max_terms, sort_by_abs=pauli_sort_by_abs)

        H_total = H_total + (scale * embedded)

        if logger is not None and log_running_total and logger.isEnabledFor(logging.DEBUG) and log_pauli_terms:
            log_paulis(logger, f"Running H_total after {step_name} sites={sites}", H_total,
                        max_terms=pauli_max_terms, sort_by_abs=pauli_sort_by_abs)

    # On-site: sum_n [ H_bos_onsite(n) + (-1)^n H_bf(n) ]
    for n in range(N):
        add_embedded("H_bos_onsite", H_bos_onsite, [n], scale=1.0)
        add_embedded("H_bf_loc * (-1)^n", H_bf_loc, [n], scale=float((-1) ** n))

    # Hopping
    if boundary_condition == "periodic":
        min_N_for_grad = 1
        for n in range(N):
            n_next = (n + 1) % N
            sign = -1.0 if (n == N - 1) else 1.0  # anti-periodic on the wrap link only
            add_embedded("H_hop", H_hop, [n, n_next], scale=float(sign))
    else:  # dirichlet
        min_N_for_grad = 2
        for n in range(N - 1):
            add_embedded("H_hop", H_hop, [n, n + 1], scale=1.0)

    # Gradient + potential-gradient
    if N >= min_N_for_grad:
        def neighbors(idx: int):
            if boundary_condition == "periodic":
                return (idx - 1) % N, (idx + 1) % N
            # dirichlet
            nm1 = idx - 1 if idx > 0 else None
            np1 = idx + 1 if idx < N - 1 else None
            return nm1, np1

        for n in range(N):
            nm1, np1 = neighbors(n)

            # (a/2) g_n^2 = 1/(8a)(q_{n+1}^2 + q_{n-1}^2 - 2 q_{n+1} q_{n-1})
            if np1 is not None:
                add_embedded("g^2: + q^2(np1)", H_q2, [np1], scale=float(1.0 / (8.0 * a)))
            if nm1 is not None:
                add_embedded("g^2: + q^2(nm1)", H_q2, [nm1], scale=float(1.0 / (8.0 * a)))
            if (np1 is not None) and (nm1 is not None):
                add_embedded("g^2: - 2 q(nm1)q(np1)", H_qq, [nm1, np1], scale=float(-1.0 / (4.0 * a)))

            # a W'(q_n) g_n = 0.5( W'(q_n) q_{n+1} - W'(q_n) q_{n-1} )
            if np1 is not None:
                add_embedded("Wp*g: + Wp(n) q(np1)", H_Wp_q, [n, np1], scale=0.5)
            if nm1 is not None:
                add_embedded("Wp*g: - Wp(n) q(nm1)", H_Wp_q, [n, nm1], scale=-0.5)

    # Simplify
    if logger is not None:
        log_paulis(logger, "H before simplify", H_total,
                        max_terms=pauli_max_terms, sort_by_abs=pauli_sort_by_abs)
        logger.info("[SIMPLIFY] simplifying H_total")

    H_total = H_total.simplify(atol=atol_simplify)

    if logger is not None:
        log_paulis(logger, "Final H", H_total,
                        max_terms=pauli_max_terms, sort_by_abs=pauli_sort_by_abs)

    if logger is not None:
        logger.info("=== build_wz_hamiltonian end ===")

    return H_total, n_total_qubits

