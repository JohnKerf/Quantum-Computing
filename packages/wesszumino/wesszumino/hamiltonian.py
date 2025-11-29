import numpy as np
import pennylane as qml
from math import log2
import ast
import re
from scipy.sparse import coo_matrix



def pauli_str_to_op(term: str):
    term = term.strip()

    # Global / multi-wire identity like "I([0, 2, 1, 3])"
    if term.startswith("I("):
        inside = term[term.find("(")+1:term.rfind(")")]
        wires = ast.literal_eval(inside)  # works for [..] or a single int
        return qml.Identity(wires=wires)

    # Tensor products like "X(0) @ X(2)"
    factors = [f.strip() for f in term.split("@")]
    ops = []
    for f in factors:
        m = re.match(r"([IXYZ])\(([^)]+)\)", f)
        if not m:
            raise ValueError(f"Can't parse factor: {f}")
        pauli, wstr = m.group(1), m.group(2).strip()

        # wires could be "0" or "[0,2,...]"
        wires = ast.literal_eval(wstr) if wstr[0] in "[(" else int(wstr)

        if pauli == "I":
            ops.append(qml.Identity(wires=wires))
        elif pauli == "X":
            ops.append(qml.PauliX(wires))
        elif pauli == "Y":
            ops.append(qml.PauliY(wires))
        elif pauli == "Z":
            ops.append(qml.PauliZ(wires))

    # build the tensor product
    op = ops[0]
    for o in ops[1:]:
        op = op @ o
    return op




############################################################################################################################################################
#  Create WZ Hamiltonian by pauli terms
############################################################################################################################################################
##############################################################################
# 1. Single-site HO matrices (boson) and fermion operators
##############################################################################

def create_matrix(cutoff, type, m=1.0):
    """Return q or p in the HO Fock basis up to 'cutoff'."""
    mat = np.zeros((cutoff, cutoff), dtype=np.complex128)
    for i in range(cutoff):
        if i > 0:
            if type == "q":
                mat[i, i-1] = (1.0/np.sqrt(2.0*m)) * np.sqrt(i)
            elif type == "p":
                mat[i, i-1] = 1j*np.sqrt(m/2.0)*np.sqrt(i)
        if i < cutoff - 1:
            if type == "q":
                mat[i, i+1] = (1.0/np.sqrt(2.0*m)) * np.sqrt(i+1)
            elif type == "p":
                mat[i, i+1] = -1j*np.sqrt(m/2.0)*np.sqrt(i+1)
    return mat

def single_site_operators(cutoff, m=1.0):
    """Return q_site, p_site, chi_site, chi_dag_site on H_site = H_f ⊗ H_b."""
    q_b = create_matrix(cutoff, "q", m=m)
    p_b = create_matrix(cutoff, "p", m=m)
    I_b = np.eye(cutoff, dtype=np.complex128)

    chi = np.array([[0, 1],[0, 0]], dtype=np.complex128)
    chi_dag = np.array([[0, 0],[1, 0]], dtype=np.complex128)
    I_f = np.eye(2, dtype=np.complex128)

    # site Hilbert space: fermion ⊗ boson
    q_site   = np.kron(I_f, q_b)
    p_site   = np.kron(I_f, p_b)
    chi_site = np.kron(chi, I_b)
    chi_dag_site = np.kron(chi_dag, I_b)

    return q_site, p_site, chi_site, chi_dag_site

##############################################################################
# 2. Local dense blocks (1-site, 2-site, 3-site)
##############################################################################

def build_onsite_blocks(cutoff, a, potential="linear", c=0.0, m=1.0):
    """
    Build single-site dense blocks:
      H_bos_onsite = p^2/(2a) + (a/2) W'(q)^2  on H_site
      H_bf_loc     = (n_f - 1/2 I_f) W''(q)    on H_site  (WITHOUT (-1)^n)
    """
    q_site, p_site, chi_site, chi_dag_site = single_site_operators(cutoff, m=m)
    D = q_site.shape[0]
    I_site = np.eye(D, dtype=np.complex128)

    # Fermion number on site
    n_f_site = chi_dag_site @ chi_site   # acts on H_site

    # Decide W'(q), W''(q) on H_site using q acting only on boson part:
    # q_site = I_f ⊗ q_b, so we can treat W'(q_site) directly
    if potential == "linear":
        # W'(q) = q, W''(q) = 1
        W_prime_site = q_site
        W_pp_site = np.eye(D, dtype=np.complex128)
    elif potential == "quadratic":
        # W'(q) = c + q^2, W''(q) = 2q
        W_prime_site = c * I_site + q_site @ q_site
        W_pp_site = 2.0 * q_site
    else:
        raise ValueError("potential must be 'linear' or 'quadratic'")

    # Bosonic kinetic + potential
    H_bos_kin = (p_site @ p_site) / (2.0*a)
    H_bos_pot = 0.5 * a * (W_prime_site @ W_prime_site)
    H_bos_onsite = H_bos_kin + H_bos_pot

    # Boson-fermion commutator factor (no (-1)^n yet)
    I_f_site = I_site  # already includes fermion identity
    H_bf_loc = (n_f_site - 0.5*I_f_site) @ W_pp_site

    return H_bos_onsite, H_bf_loc


def build_hopping_block(cutoff, a=1.0, m=1.0):
    """
    Two-site fermion hopping block on H_site⊗H_site:

      H_hop = 0.5 (chi_n^† chi_{n+1} + chi_{n+1}^† chi_n)

    (No periodic sign here; we apply -1 on the wrap link during assembly.)
    """
    q_site, p_site, chi_site, chi_dag_site = single_site_operators(cutoff, m=m)
    D = q_site.shape[0]
    I_site = np.eye(D, dtype=np.complex128)

    chi_n      = np.kron(chi_site, I_site)
    chi_np1    = np.kron(I_site, chi_site)
    chi_dag_n  = np.kron(chi_dag_site, I_site)
    chi_dag_np1= np.kron(I_site, chi_dag_site)

    H_hop = 0.5 * (chi_dag_n @ chi_np1 + chi_dag_np1 @ chi_n)
    return H_hop


def build_grad_pg_block(cutoff, a, potential="linear", c=0.0, m=1.0):
    """
    Three-site block for:

      (a/2) g_n^2 + a W'(q_n) g_n

    where g_n = (q_{n+1} - q_{n-1})/(2a),
    on sites (n-1, n, n+1) in order [left, center, right].

    This block acts on H_site⊗H_site⊗H_site (dimension D^3).
    """
    q_site, p_site, chi_site, chi_dag_site = single_site_operators(cutoff, m=m)
    D = q_site.shape[0]
    I_site = np.eye(D, dtype=np.complex128)

    # 3-site Hilbert: left (L), center (C), right (R)
    # orders: L ⊗ C ⊗ R
    q_L = np.kron(np.kron(q_site, I_site), I_site)
    q_C = np.kron(np.kron(I_site, q_site), I_site)
    q_R = np.kron(np.kron(I_site, I_site), q_site)

    # W'(q_C) on center site
    if potential == "linear":
        W_prime_C = q_C
    elif potential == "quadratic":
        W_prime_C = c * np.kron(np.kron(I_site, I_site), I_site) + q_C @ q_C
    else:
        raise ValueError("potential must be 'linear' or 'quadratic'")

    # gradient g_n = (q_R - q_L)/(2a)
    g_n = (q_R - q_L) / (2.0*a)
    grad_term = 0.5 * a * (g_n @ g_n)
    pg_term = a * (W_prime_C @ g_n)

    H_grad_pg = grad_term + pg_term
    return H_grad_pg

##############################################################################
# 3. Matrix -> Pauli (local) and embedding onto chosen sites
##############################################################################

def embed_dense_to_qubits(H_dense, n_qubits=None):
    """Embed dense (d x d) into 2^n x 2^n, top-left block."""
    d = H_dense.shape[0]
    if n_qubits is None:
        n_qubits = int(log2(d))
    dim_q = 2**n_qubits
    H_q = np.zeros((dim_q, dim_q), dtype=np.complex128)
    H_q[:d, :d] = H_dense
    return H_q, n_qubits

def pauli_decompose_dense(H_dense, n_qubits=None):
    """Convenience: dense -> qubit matrix -> qml.Hamiltonian."""
    H_q, n_q = embed_dense_to_qubits(H_dense, n_qubits=n_qubits)
    H_pl = qml.pauli_decompose(H_q)
    return H_pl, n_q

def embed_pauli_block_on_sites(H_block, sites, n_site, N):
    """
    Embed a local qml.Hamiltonian H_block (with wires 0..n_block-1)
    onto the full lattice with N sites and n_site qubits per site.
    'sites' is a list like [n] (1-site), [n, n+1] (2-site), [n-1,n,n+1] (3-site).
    """
    # build wire map: local_block_wire -> global_wire
    wire_map = {}
    for k, site in enumerate(sites):
        for j in range(n_site):
            local_wire = k*n_site + j
            global_wire = site*n_site + j
            wire_map[local_wire] = global_wire

    # apply map_wires to each term
    mapped_ops = []
    mapped_coeffs = []

    for coeff, op in zip(H_block.coeffs, H_block.ops):
        op_mapped = qml.map_wires(op, wire_map)
        mapped_ops.append(op_mapped)
        mapped_coeffs.append(coeff)

    return qml.Hamiltonian(mapped_coeffs, mapped_ops)

##############################################################################
# 4. Full scalable WZ Hamiltonian (periodic BC, N >= 3 recommended)
##############################################################################

def strip_zero_terms(H, tol=1e-12):
    new_coeffs = []
    new_ops    = []
    for c, o in zip(H.coeffs, H.ops):
        if abs(c) > tol:
            new_coeffs.append(c)
            new_ops.append(o)
    return qml.Hamiltonian(new_coeffs, new_ops)


def build_wz_hamiltonian(
    cutoff,
    N,
    a,
    c=0.0,
    m=1.0,
    potential="linear",
    boundary_condition="periodic",
    remove_zero_terms=True
):
    """
    Build the Wess–Zumino Hamiltonian as a PennyLane qml.Hamiltonian using
    only local blocks (1-site and 2-site). This avoids constructing any
    3-site dense matrices, so memory scales much better with the bosonic cutoff.

    Bosonic part per site n:
      p_n^2/(2a) + (a/2) W'(q_n)^2
    plus gradient and potential-gradient:
      (a/2) g_n^2 + a W'(q_n) g_n

    with
      g_n = (q_{n+1} - q_{n-1})/(2a)

    For boundary="dirichlet", missing neighbors are set to zero:
      - n = 0:    g_0     = (q_1 - 0)/(2a)
      - n = N-1:  g_{N-1} = (0 - q_{N-2})/(2a)

    For boundary="periodic", indices are taken modulo N.

    The gradient + potential-gradient pieces are expanded into 1- and 2-site
    operators, so no 3-site blocks are ever built.
    """
    if boundary_condition not in ("periodic", "dirichlet"):
        raise ValueError("boundary must be 'periodic' or 'dirichlet'.")

    # ------------------------------------------------------------------
    #  Single-site blocks: bosonic onsite + boson-fermion interaction
    # ------------------------------------------------------------------
    H_bos_onsite, H_bf_loc = build_onsite_blocks(
        cutoff, a, potential, c=c, m=m
    )

    D_site = H_bos_onsite.shape[0]
    n_site = int(log2(D_site))

    # Pauli-decompose single-site blocks
    H_bos_onsite_pl, _ = pauli_decompose_dense(H_bos_onsite, n_qubits=n_site)
    H_bf_pl, _         = pauli_decompose_dense(H_bf_loc,     n_qubits=n_site)

    # ------------------------------------------------------------------
    #  Two-site fermion hopping block
    # ------------------------------------------------------------------
    H_hop_dense = build_hopping_block(cutoff, a=a, m=m)
    H_hop_pl, _ = pauli_decompose_dense(H_hop_dense)  # 2 * n_site qubits

    # ------------------------------------------------------------------
    #  Building blocks for gradient + potential-gradient (bosons only)
    #
    #  We will use the identities (for interior n):
    #
    #    g_n = (q_{n+1} - q_{n-1})/(2a)
    #
    #    (a/2) g_n^2
    #      = 1/(8a) (q_{n+1}^2 + q_{n-1}^2 - 2 q_{n+1} q_{n-1})
    #
    #    a W'(q_n) g_n
    #      = 0.5 ( W'(q_n) q_{n+1} - W'(q_n) q_{n-1} )
    #
    #  This only involves:
    #    - 1-site operators: q^2, W'(q) on a single site
    #    - 2-site operators: q_i q_j, W'(q_i) q_j
    #
    # ------------------------------------------------------------------
    # Get single-site q and W'(q)
    q_site, p_site, chi_site, chi_dag_site = single_site_operators(cutoff, m=m)
    I_site = np.eye(D_site, dtype=np.complex128)

    if potential == "linear":
        # W'(q) = q
        W_prime_site = q_site
    elif potential == "quadratic":
        # W'(q) = c + q^2
        W_prime_site = c * I_site + (q_site @ q_site)
    else:
        raise ValueError("potential must be 'linear' or 'quadratic'")

    # 1-site: q^2
    q2_site = q_site @ q_site

    # 2-site: q ⊗ q
    qq_dense = np.kron(q_site, q_site)

    # 2-site: W'(q) ⊗ q  (central site ⊗ neighbor)
    Wp_q_dense = np.kron(W_prime_site, q_site)

    # Pauli-decompose these basic blocks
    H_q2_pl, _   = pauli_decompose_dense(q2_site,    n_qubits=n_site)
    H_qq_pl, _   = pauli_decompose_dense(qq_dense)        # 2 * n_site
    H_Wp_q_pl, _ = pauli_decompose_dense(Wp_q_dense)      # 2 * n_site

    # ------------------------------------------------------------------
    #  Assemble full Hamiltonian
    # ------------------------------------------------------------------
    n_total_qubits = N * n_site
    H_total = qml.Hamiltonian([], [])

    # On-site terms: sum_n [ H_bos_onsite(n) + (-1)^n H_bf(n) ]
    for n in range(N):
        # bosonic on-site
        H_total = H_total + embed_pauli_block_on_sites(H_bos_onsite_pl, [n], n_site, N)
        # boson-fermion term with staggered sign
        H_total = H_total + ((-1) ** n) * embed_pauli_block_on_sites(
            H_bf_pl, [n], n_site, N
        )

    # ------------------------------------------------------------------
    #  Fermion hopping
    # ------------------------------------------------------------------
    if boundary_condition == "periodic":
        min_N_for_grad = 1
        # links (n, n+1 mod N), minus sign on wrap link (N-1 -> 0)
        for n in range(N):
            n_next = (n + 1) % N
            sign = -1.0 if (n == N - 1) else 1.0
            H_total = H_total + sign * embed_pauli_block_on_sites(
                H_hop_pl, [n, n_next], n_site, N
            )
    else:  # 'dirichlet'
        min_N_for_grad = 2
        # only links (0,1), (1,2), ..., (N-2, N-1), no wrap, no extra sign
        for n in range(N - 1):
            H_total = H_total + embed_pauli_block_on_sites(
                H_hop_pl, [n, n + 1], n_site, N
            )

    # ------------------------------------------------------------------
    #  Gradient + potential-gradient (bosons only)
    #
    #  H_grad_pg = sum_n [ (a/2) g_n^2 + a W'(q_n) g_n ]
    #
    #  Implemented using:
    #    - H_q2_pl:   q^2 on one site
    #    - H_qq_pl:   q_i q_j on two sites
    #    - H_Wp_q_pl: W'(q_i) q_j on two sites
    #
    # ------------------------------------------------------------------
    if N >= min_N_for_grad:
        def neighbors(n):
            if boundary_condition == "periodic":
                nm1 = (n - 1) % N
                np1 = (n + 1) % N
                return nm1, np1
            else:  # 'dirichlet'
                nm1 = n - 1 if n > 0     else None
                np1 = n + 1 if n < N - 1 else None
                return nm1, np1

        for n in range(N):
            nm1, np1 = neighbors(n)

            if boundary_condition == "periodic" and (nm1 == np1):
                continue

            # ---------- (a/2) g_n^2 piece ----------
            if np1 is not None:
                H_total = H_total + (1.0 / (8.0 * a)) * embed_pauli_block_on_sites(
                    H_q2_pl, [np1], n_site, N
                )
            if nm1 is not None:
                H_total = H_total + (1.0 / (8.0 * a)) * embed_pauli_block_on_sites(
                    H_q2_pl, [nm1], n_site, N
                )
            if (np1 is not None) and (nm1 is not None):
                H_total = H_total + (-1.0 / (4.0 * a)) * embed_pauli_block_on_sites(
                    H_qq_pl, [nm1, np1], n_site, N
                )

            # ---------- a W'(q_n) g_n piece ----------
            if np1 is not None:
                H_total = H_total + 0.5 * embed_pauli_block_on_sites(
                    H_Wp_q_pl, [n, np1], n_site, N
                )
            if nm1 is not None:
                H_total = H_total + (-0.5) * embed_pauli_block_on_sites(
                    H_Wp_q_pl, [n, nm1], n_site, N
                )

    # ------------------------------------------------------------------
    #  Simplify and return
    # ------------------------------------------------------------------
    H_total = H_total.simplify()

    if remove_zero_terms:
        H_total = strip_zero_terms(H_total)

    return H_total, n_total_qubits



############################################################################################################################################################
#  Create reduced sparse matrix from pauli terms and bitstrings
############################################################################################################################################################
def apply_pauli_to_bitstring(pauli_str, bitstring):
    """
    Apply a Pauli string (like "IXYZ...") to a computational basis bitstring.

    Returns:
        phase (complex), out_bitstring (str)
    """
    phase = 1.0 + 0.0j
    out = list(bitstring)

    for q, (p, b_char) in enumerate(zip(pauli_str.upper(), bitstring)):
        b = 1 if b_char == "1" else 0

        if p == "I":
            continue
        elif p == "X":
            out[q] = "0" if b else "1"
        elif p == "Z":
            if b:
                phase *= -1
        elif p == "Y":
            out[q] = "0" if b else "1"
            phase *= (1j if b == 0 else -1j)  # i * (-1)^b
        else:
            raise ValueError(f"Bad Pauli char '{p}' at qubit {q}")

    return phase, "".join(out)


def reduced_sparse_matrix_from_pauli_terms(pauli_terms, basis_states):
    """
    Build reduced Hamiltonian as a sparse matrix from explicit Pauli terms.

    pauli_terms: list of (coeff, pauli_str)
        e.g. [(0.5, "ZIIII"), (-1.2, "XXIYZ"), ...]
    basis_states: list of bitstrings, all same length n
        e.g. top_states from counts

    Returns
    -------
    H_red : scipy.sparse.csr_matrix (complex)
        Reduced Hamiltonian in the basis given by basis_states
    """
    # Clean basis states
    basis_states = [s.strip() for s in basis_states]
    n = len(basis_states[0])
    m = len(basis_states)

    # Map bitstring -> basis index
    idx = {s: i for i, s in enumerate(basis_states)}

    # Lists for COO data
    rows = []
    cols = []
    data = []

    for coeff, pstr in pauli_terms:
        pstr = pstr.strip().upper()
        # Act on each basis state |ket>
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

    # Build sparse matrix in COO then convert to CSR
    H_red_coo = coo_matrix((data, (rows, cols)),
                           shape=(m, m),
                           dtype=np.complex128)
    H_red = H_red_coo.tocsr()  # sums duplicate (i,j) entries

    return H_red



############################################################################################################################################################
#  Return pauli strings from operator
############################################################################################################################################################
def op_to_full_pauli_string(op, wire_order):
    """
    Convert a single PennyLane Pauli word into a full string over wire_order.
    Example output: "IIXZY"
    """
    wire_pos = {w: i for i, w in enumerate(wire_order)}
    n = len(wire_order)
    chars = ["I"] * n

    def walk(o):
        # Products / tensors expose operands
        if hasattr(o, "operands") and o.operands is not None:
            for sub in o.operands:
                walk(sub)
            return

        name = o.name
        if name in ("PauliX", "X"):
            c = "X"
        elif name in ("PauliY", "Y"):
            c = "Y"
        elif name in ("PauliZ", "Z"):
            c = "Z"
        elif name in ("Identity", "I"):
            c = "I"
        else:
            raise ValueError(f"Unexpected operator in Pauli word: {name}")

        for w in o.wires:
            chars[wire_pos[w]] = c

    walk(op)
    return "".join(chars)


def pauli_terms_from_operator(H_pauli, wire_order):
    """
    Extract list of (coeff, pauli_str) from a PennyLane Hamiltonian / Sum.
    """
    if not hasattr(H_pauli, "terms"):
        raise TypeError("H_pauli has no .terms(); pass explicit pauli_terms instead.")

    coeffs, ops = H_pauli.terms()
    terms = []
    for c, o in zip(coeffs, ops):
        pstr = op_to_full_pauli_string(o, wire_order)
        terms.append((complex(c), pstr))
    return terms


