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

#############################################################################
                                # Position QFT
#############################################################################
def fourier_matrix(n):
    """DFT matrix F with F[j,k] = exp(-2π i j k / n)/sqrt(n)."""
    j = np.arange(n)[:, None]
    k = np.arange(n)[None, :]
    return np.exp(-2j * np.pi * j * k / n) / np.sqrt(n)


def create_position_operators(cutoff, x_max, m=1.0):
  
    N = cutoff
    L = 2.0 * x_max
    dx = L / N
    x = -x_max + dx * np.arange(N)  # [-x_max, x_max)

    q = np.diag(x)

    # Fourier wavenumbers: k = 2π * n / L
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    k2 = (k * k)

    F = fourier_matrix(N)
    Fdg = F.conj().T

    p = Fdg @ np.diag(k) @ F
    p2 = Fdg @ np.diag(k2) @ F

    return q, p, p2


def single_site_operators(cutoff, m = 1.0, x_max=8.0):
    
    q_b, p_b, p2_b = create_position_operators(cutoff, x_max=x_max, m=m)
    I_b = np.eye(cutoff, dtype=np.complex128)

    chi = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    chi_dag = np.array([[0, 0], [1, 0]], dtype=np.complex128)
    I_f = np.eye(2, dtype=np.complex128)

    # site Hilbert space: fermion x boson
    q_site = np.kron(I_f, q_b)
    p_site = np.kron(I_f, p_b)
    p2_site = np.kron(I_f, p2_b)
    chi_site = np.kron(chi, I_b)
    chi_dag_site = np.kron(chi_dag, I_b)

    return q_site, p_site, p2_site, chi_site, chi_dag_site


# =============================================================================
# 2) Local dense blocks (1-site, 2-site)
# =============================================================================

def build_onsite_blocks(cutoff, a, potential="linear", c=0.0, m=1.0, x_max=8.0):
    """
    Build single-site dense blocks:

    H_bos_onsite = p^2/(2a) + (a/2) W'(q)^2
    H_bf_loc     = (n_f - 1/2 I) W''(q)
    """
    q_site, p_site, p2_site, chi_site, chi_dag_site = single_site_operators(cutoff, m=m, x_max=x_max)
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

    H_bos_kin = p2_site  / (2.0 * a)
    H_bos_pot = 0.5 * a * (W_prime_site @ W_prime_site)
    H_bos_onsite = H_bos_kin + H_bos_pot

    H_bf_loc = (n_f_site - 0.5 * I_site) @ W_pp_site # no (-1)^n here

    return H_bos_onsite, H_bf_loc


def build_hopping_block(cutoff, a=1.0, m=1.0, x_max=8.0):
    """
    Two-site fermion hopping block on H_site⊗H_site:

    H_hop = 0.5 (chi_ndag chi_{n+1} + chi_{n+1}dag chi_n)

    No anti-periodic sign here apply -1 later.
    """
    q_site, p_site, p2_site, chi_site, chi_dag_site = single_site_operators(cutoff, m=m, x_max=x_max)
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
    H_dense = 0.5 * (H_dense + H_dense.conj().T)
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

def build_wz_hamiltonian_position(cutoff, N, a, c=0.0, m=1.0, 
                         potential="linear", boundary_condition="periodic", 
                         atol_decompose=1e-12, atol_simplify=1e-12, remove_zero_terms=True, x_max=8.0):
    """
    Build the Wess–Zumino Hamiltonian as a Qiskit SparsePauliOp using
    only local blocks (1-site and 2-site).
    """

    # single-site blocks
    H_bos_onsite_dense, H_bf_loc_dense = build_onsite_blocks(cutoff, a, potential=potential, c=c, m=m, x_max=x_max)
    D_site = H_bos_onsite_dense.shape[0]
    n_site = int(np.log2(D_site))
    n_total_qubits = N * n_site

    H_bos_onsite = dense_to_sparse_pauliop(H_bos_onsite_dense, atol=atol_decompose)
    #print(H_bos_onsite)
    H_bf_loc = dense_to_sparse_pauliop(H_bf_loc_dense, atol=atol_decompose)
    #print(H_bf_loc)

    # two-site hopping
    H_hop_dense = build_hopping_block(cutoff, a=a, m=m, x_max=x_max)
    H_hop = dense_to_sparse_pauliop(H_hop_dense, atol=atol_decompose)
    #print(H_hop)

    # bosonic gradient + potential-gradient building blocks
    q_site, _, _, _, _ = single_site_operators(cutoff, m=m, x_max=x_max)
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
        min_N_for_grad = 3
        #print(H_total)
        for n in range(N - 1):
            #print(n)
            H_total = H_total + embed_sparse_pauliop_on_sites(H_hop, [n, n + 1], n_site, N)
            #print(H_total)

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
                H_total = H_total + (1.0 / (8.0 * a)) * embed_sparse_pauliop_on_sites(H_q2, [np1], n_site, N)
            if nm1 is not None:
                H_total = H_total + (1.0 / (8.0 * a)) * embed_sparse_pauliop_on_sites(H_q2, [nm1], n_site, N)
            if (np1 is not None) and (nm1 is not None):
                H_total = H_total + (-1.0 / (4.0 * a)) * embed_sparse_pauliop_on_sites(H_qq, [nm1, np1], n_site, N)

            # a W'(q_n) g_n = 0.5( W'(q_n) q_{n+1} - W'(q_n) q_{n-1} )
            if np1 is not None:
                H_total = H_total + 0.5 * embed_sparse_pauliop_on_sites(H_Wp_q, [n, np1], n_site, N)
            if nm1 is not None:
                H_total = H_total + (-0.5) * embed_sparse_pauliop_on_sites(H_Wp_q, [n, nm1], n_site, N)

    # Simplify
    H_total = H_total.simplify(atol=atol_simplify)

    return H_total, n_total_qubits


def wz_position_grid_data(
    cutoff: int,
    *,
    potential: str = "linear",
    c: float = 0.0,
    x_max: float = 8.0,
):
    """
    Return the diagonal data needed for QFT-based evolution of the WZ bosons.

    Boson grid is x in [-x_max, x_max), with k from FFT frequencies.

    For your WZ conventions (linear/quadratic superpotential):
      - linear:   W'(q)=q,      W''(q)=1
      - quadratic:W'(q)=c+q^2,  W''(q)=2q
    """
    N = cutoff
    L = 2.0 * x_max
    dx = L / N
    x = -x_max + dx * np.arange(N)  # [-x_max, x_max)

    # Fourier wavenumbers
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    k2 = (k * k).astype(np.float64)

    if potential == "linear":
        Wp = x
        Wpp = np.ones_like(x)
    elif potential == "quadratic":
        Wp = c + x**2
        Wpp = 2.0 * x
    else:
        raise ValueError("potential must be 'linear' or 'quadratic'")

    return {
        "x": x.astype(np.float64),
        "dx": float(dx),
        "L": float(L),
        "k": k.astype(np.float64),
        "k2": k2,
        "Wp_diag": Wp.astype(np.float64),
        "Wp2_diag": (Wp * Wp).astype(np.float64),
        "Wpp_diag": Wpp.astype(np.float64),
    }



def build_wz_hamiltonian_with_qft_parts(
    cutoff: int,
    N: int,
    a: float,
    c: float = 0.0,
    m: float = 1.0,
    potential: str = "linear",
    boundary_condition: str = "periodic",
    x_max: float = 8.0,
    atol_decompose: float = 1e-12,
    atol_simplify: float = 1e-12,
):
    """
    Returns:
      H_total, n_total_qubits, parts

    - H_total is your existing SparsePauliOp (still useful for checks/debugging).
    - parts is a structured description of the *diagonal pieces* needed for QFT-based evolution.
    """
    # Keep your existing Pauli Hamiltonian (HO/Fock version) if you want,
    # OR (better) once you switch your site operators to position basis,
    # this will match the position discretization too.
    H_total, n_qubits = build_wz_hamiltonian_position(
        cutoff, N, a,
        c=c, m=m,
        potential=potential,
        boundary_condition=boundary_condition,
        atol_decompose=atol_decompose,
        atol_simplify=atol_simplify,
        x_max=x_max
    )  # 

    # Grid data for the bosons (this is what you need for QFT evolution)
    basis_info = wz_position_grid_data(cutoff, potential=potential, c=c, x_max=x_max)

    return H_total.simplify(atol=1e-12), n_qubits, basis_info




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

