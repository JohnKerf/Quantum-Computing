import numpy as np
from typing import Tuple
from scipy.sparse import coo_matrix
from qiskit.quantum_info import Operator, SparsePauliOp, PauliList

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

def build_wz_hamiltonian(cutoff, N, a, c=0.0, m=1.0, 
                         potential="linear", boundary_condition="periodic", 
                         atol_decompose=1e-12, atol_simplify=1e-12, include_grad=True, include_qq=True, include_wpq=True, include_q2=True):
    """
    Build the Wess–Zumino Hamiltonian as a Qiskit SparsePauliOp using
    only local blocks (1-site and 2-site).
    """

    # single-site blocks
    H_bos_onsite_dense, H_bf_loc_dense = build_onsite_blocks(cutoff, a, potential=potential, c=c, m=m)
    D_site = H_bos_onsite_dense.shape[0]
    n_site = int(np.log2(D_site))
    n_total_qubits = N * n_site

    H_bos_onsite = dense_to_sparse_pauliop(H_bos_onsite_dense, atol=atol_decompose)
    #print(H_bos_onsite)
    H_bf_loc = dense_to_sparse_pauliop(H_bf_loc_dense, atol=atol_decompose)
    #print(H_bf_loc)

    # two-site hopping
    H_hop_dense = build_hopping_block(cutoff, a=a, m=m)
    H_hop = dense_to_sparse_pauliop(H_hop_dense, atol=atol_decompose)
    #print(H_hop)

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

    H_q2 = dense_to_sparse_pauliop(q2_site, atol=atol_decompose)
    #print(H_q2)
    H_qq = dense_to_sparse_pauliop(qq_dense, atol=atol_decompose)
    #print(H_qq)
    H_Wp_q = dense_to_sparse_pauliop(Wp_q_dense, atol=atol_decompose)
    #print(H_Wp_q)

    # assemble for all sites
    H_total = zero_sparse_pauliop(n_total_qubits) # initiate full H as 0.0 x III...I
    #print(H_total)

    # On-site: sum_n [ H_bos_onsite(n) + (-1)^n H_bf(n) ]
    for n in range(N):
        #print(n)
        H_total = H_total + embed_sparse_pauliop_on_sites(H_bos_onsite, [n], n_site, N)
        #print(H_total)
        H_total = H_total + ((-1) ** n) * embed_sparse_pauliop_on_sites(H_bf_loc, [n], n_site, N)
        #print(H_total)

    # Hopping
    if boundary_condition == "periodic":
        min_N_for_grad = 3
        for n in range(N):
            n_next = (n + 1) % N
            sign = -1.0 if (n == N - 1) else 1.0
            H_total = H_total + sign * embed_sparse_pauliop_on_sites(H_hop, [n, n_next], n_site, N)
        
    else:  # dirichlet
        min_N_for_grad = 2
        #print(H_total)
        for n in range(N - 1):
            #print(n)
            H_total = H_total + embed_sparse_pauliop_on_sites(H_hop, [n, n + 1], n_site, N)
            #print(H_total)

    # Gradient + potential-gradient
    if include_grad:
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
                if include_q2:
                    if np1 is not None:
                        H_total = H_total + (1.0 / (8.0 * a)) * embed_sparse_pauliop_on_sites(H_q2, [np1], n_site, N)
                    if nm1 is not None:
                        H_total = H_total + (1.0 / (8.0 * a)) * embed_sparse_pauliop_on_sites(H_q2, [nm1], n_site, N)

                if include_qq:
                    if (np1 is not None) and (nm1 is not None):
                        H_total = H_total + (-1.0 / (4.0 * a)) * embed_sparse_pauliop_on_sites(H_qq, [nm1, np1], n_site, N)

                # a W'(q_n) g_n = 0.5( W'(q_n) q_{n+1} - W'(q_n) q_{n-1} )
                if include_wpq:
                    if np1 is not None:
                        H_total = H_total + 0.5 * embed_sparse_pauliop_on_sites(H_Wp_q, [n, np1], n_site, N)
                    if nm1 is not None:
                        H_total = H_total + (-0.5) * embed_sparse_pauliop_on_sites(H_Wp_q, [n, nm1], n_site, N)


    # Simplify
    H_total = H_total.simplify(atol=atol_simplify)

    return H_total, n_total_qubits




############################################################################################################################################################
#  Create reduced sparse matrix from pauli terms and bitstrings
############################################################################################################################################################
def apply_pauli_to_bitstring(pauli_str, bitstring):
    """
    Qiskit convention:
      - rightmost char of pauli_str acts on qubit 0
      - rightmost bit of bitstring is qubit 0
    """

    phase = 1.0 + 0.0j
    out = list(bitstring)
    n = len(bitstring)

    # qubit q corresponds to index -(q+1)
    for q in range(n):
        idx = n - 1 - q
        p = pauli_str[idx]
        b = 1 if bitstring[idx] == "1" else 0

        if p == "I":
            continue
        elif p == "X":
            out[idx] = "0" if b else "1"
        elif p == "Z":
            if b:
                phase *= -1
        elif p == "Y":
            out[idx] = "0" if b else "1"
            phase *= (1j if b == 0 else -1j)  # Y|0>=i|1>, Y|1>=-i|0>
        else:
            raise ValueError(f"Bad Pauli char '{p}' at qubit {q}")

    return phase, "".join(out)

def reduced_sparse_matrix_from_pauli_terms(pauli_terms, basis_states):
    """
    Build reduced Hamiltonian as a sparse matrix from Pauli terms.

    pauli_terms: list of (coeff, pauli_str)
        e.g. [(0.5, "ZIIII"), (-1.2, "XXIYZ"), ...]
    basis_states: list of bitstrings, all same length n
        e.g. ["10001","00001"]
    """

    m = len(basis_states)

    # Map bitstring -> basis index
    idx = {s: i for i, s in enumerate(basis_states)}

    # Lists for COO data
    rows = []
    cols = []
    data = []

    for coeff, pstr in pauli_terms:
        for ket in basis_states:
            phase, out_state = apply_pauli_to_bitstring(pstr, ket)
            if out_state in idx:
                i = idx[out_state]   # row index  (bra = out_state)
                j = idx[ket]         # column index (ket)
                value = coeff * phase

                # Store non-zero contribution
                if value != 0:
                    rows.append(i)
                    cols.append(j)
                    data.append(value)

    # Build sparse matrix
    H_red_coo = coo_matrix((data, (rows, cols)),
                           shape=(m, m),
                           dtype=np.complex128)
    H_red = H_red_coo.tocsr()

    return H_red


############################################################################################################################################################
#  Create reduced sparse matrix from pauli terms and bitstrings - FAST Version
############################################################################################################################################################

def _compile_pauli_masks(pauli_str):
    """
    pauli_str is length n, leftmost acts on qubit n-1, rightmost on qubit 0 (Qiskit convention).
    Returns (xmask, zmask, ny, base_i_factor).
    """
    xmask = 0
    zmask = 0
    ny = 0

    # qubit 0 is rightmost char
    # so for qubit q, look at pauli_str[-(q+1)]
    for q, ch in enumerate(reversed(pauli_str)):
        if ch == "I":
            continue
        if ch == "X":
            xmask |= (1 << q)
        elif ch == "Z":
            zmask |= (1 << q)
        elif ch == "Y":
            xmask |= (1 << q)
            zmask |= (1 << q)
            ny += 1
        else:
            raise ValueError(f"Bad Pauli char '{ch}'")

    i_factor = (1j) ** ny  # global factor from all Y’s
    return xmask, zmask, i_factor

def reduced_sparse_matrix_from_pauli_terms_fast(pauli_terms, basis_states):
    """
    pauli_terms: list[(coeff, pauli_str)]
    basis_states: list[str] bitstrings, all same length.

    Returns CSR sparse matrix (via COO->CSR). Duplicates are fine; CSR will sum them.
    """
    m = len(basis_states)
    if m == 0:
        return coo_matrix((0, 0), dtype=np.complex128).tocsr()

    # Convert basis to ints once
    basis_ints = [int(s, 2) for s in basis_states]
    idx = {s_int: i for i, s_int in enumerate(basis_ints)}

    rows = []
    cols = []
    data = []

    # Localize for speed
    idx_get = idx.get
    basis_ints_local = basis_ints

    for coeff, pstr in pauli_terms:
        if coeff == 0:
            continue

        xmask, zmask, i_factor = _compile_pauli_masks(pstr)
        coeff_i = coeff * i_factor

        # loop over ket states
        for j, s in enumerate(basis_ints_local):
            out = s ^ xmask
            i = idx_get(out)
            if i is None:
                continue

            # (-1)^{popcount(s & zmask)}
            sign = -1.0 if ((s & zmask).bit_count() & 1) else 1.0

            val = coeff_i * sign
            # val can be complex; keep it
            rows.append(i)
            cols.append(j)
            data.append(val)

    H_red = coo_matrix((np.asarray(data, dtype=np.complex128),
                        (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
                       shape=(m, m),
                       dtype=np.complex128).tocsr()
    return H_red


