import numpy as np
from scipy.sparse import eye, kron, coo_matrix
from functools import reduce

#############################################################################
                           #p and q in HO basis
#############################################################################

def create_matrix(cut_off, type, m=1):
    # Initialize a zero matrix
    matrix = np.zeros((cut_off, cut_off), dtype=np.complex128)
    
    # Fill the off-diagonal values
    for i in range(cut_off):
        if i > 0:  # Fill left off-diagonal
            if type == 'q':
                matrix[i][i - 1] = (1/np.sqrt(2*m)) * np.sqrt(i)
            else:
                matrix[i][i - 1] = (1j*np.sqrt(m/2)) * np.sqrt(i)

        if i < cut_off - 1:  # Fill right off-diagonal
            if type == 'q':
                matrix[i][i + 1] = (1/np.sqrt(2*m)) * np.sqrt(i + 1)
            else:
                matrix[i][i + 1] = (-1j*np.sqrt(m/2)) * np.sqrt(i + 1)

    return matrix


#############################################################################
                                #SUSY QM
#############################################################################
# Fermion x Boson     
# Function to calculate the Hamiltonian
def calculate_Hamiltonian(cut_off, potential, m=1, g=1, u=1):

    # Generate the position (q) and momentum (p) matrices
    q = create_matrix(cut_off, 'q')  # q matrix
    p = create_matrix(cut_off, 'p')  # p matrix
    p2 = np.matmul(p, p)

    # Calculate q^2 and q^3 for potential terms
    q2 = np.matmul(q, q)
    q3 = np.matmul(q2, q)

    #fermionic identity
    I_f = np.eye(2)

    #bosonic identity
    I_b = np.eye(cut_off)

    # Superpotential derivatives
    if potential == 'QHO':
        W_prime = m*q  # W'(q) = mq
        W_double_prime = m*I_b #W''(q) = m

    elif potential == 'AHO':
        W_prime = m*q + g*q3  # W'(q) = mq + gq^3
        W_double_prime = m*I_b + 3*g*q2  # W''(q) = m + 3gq^2

    elif potential == 'DW':
        W_prime = m*q + g*q2 + g*u**2*I_b  # W'(q) = mq + gq^2 + gu^2
        W_double_prime = m*I_b + 2*g*q  # W''(q) = m + 2gq

    else:
        print("Not a valid potential")
        raise

    # Kinetic term: p^2

    # Commutator term [b^†, b] = -Z
    Z = np.array([[1, 0], [0, -1]])  # Pauli Z matrix for fermion number
    commutator_term = np.kron(Z, W_double_prime)

    # Construct the block-diagonal kinetic term (bosonic and fermionic parts)
    # Bosonic part is the same for both, hence we use kron with the identity matrix
    kinetic_term = np.kron(I_f, p2)

    # Potential term (W' contribution)
    potential_term = np.kron(I_f, np.matmul(W_prime, W_prime))

    # Construct the full Hamiltonian
    H_SQM = 0.5 * (kinetic_term + potential_term + commutator_term)
    H_SQM[np.abs(H_SQM) < 10e-12] = 0
    
    return H_SQM


#############################################################################
                                # Position QFT
#############################################################################

def fourier_matrix(n):
    """DFT matrix F with F[j,k] = exp(-2π i j k / n)/sqrt(n)."""
    j = np.arange(n)[:, None] # column vector
    k = np.arange(n)[None, :] # row vector
    # j*k is a nxn matrix
    return np.exp(-2j * np.pi * j * k / n) / np.sqrt(n)

def calculate_hamiltonian_position(cutoff, potential, x_max=8.0, m=1.0, g=1.0, u=1.0):
    
    # grid: N points on length L = 2*x_max, periodic
    N = cutoff
    L = 2.0 * x_max
    dx = L / N
    x = -x_max + dx * np.arange(N)
    #print("x: ", x)

    # This didnt work as well
    # x = np.linspace(-x_max, x_max, N)  
    # dx = 2.0 * x_max / (N - 1)
    # x  = -x_max + dx * np.arange(N)

    q = np.diag(x)

    # Fourier wavenumbers: k = 2π * n / L
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    #print("k: ",k)
    k2 = (k * k)

    F = fourier_matrix(N)     
    #print("F: ",F)               
    Fdg = F.conj().T
    p2 = Fdg @ np.diag(k2) @ F
    
    
    if potential == "QHO":
        Wp  = m * x
        Wpp = m * np.ones_like(x)

    elif potential == "AHO":
        Wp  = m * x + g * x**3
        Wpp = m + 3 * g * x**2

    elif potential == "DW":
        Wp  = m * x + g * x**2 + g * u**2
        Wpp = m + 2 * g * x

    out = {
        "x": x, 
        "dx": dx, 
        "L": L,
        "k": k,
        "k2": (k * k).astype(np.float64),
        "Wp2_diag": (Wp * Wp).astype(np.float64),
        "Wpp_diag": (Wpp).astype(np.float64),
    }

    # Fermion Z operator
    Z_f = np.array([[1, 0], [0, -1]])

    Wp2_mat = np.diag((Wp**2))
    Wpp_mat = np.diag((Wpp)) 

    #fermionic identity
    I_f = np.eye(2)

    H = 0.5 * (np.kron(I_f, p2) + np.kron(I_f, Wp2_mat) + np.kron(Z_f, Wpp_mat))
    H[np.abs(H) < 1e-12] = 0.0
    H = 0.5 * (H + H.conj().T)

    return H, out





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
    Build reduced Hamiltonian as a sparse matrix from explicit Pauli terms.

    pauli_terms: list of (coeff, pauli_str)
        e.g. [(0.5, "ZIIII"), (-1.2, "XXIYZ"), ...]
    basis_states: list of bitstrings, all same length n
        e.g. top_states from counts
    """
    # Clean basis states
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
   
   
