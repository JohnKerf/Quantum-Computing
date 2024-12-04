import numpy as np

#############################################################################
                           #p and q in HO basis
#############################################################################

def create_matrix(cut_off, type):
    # Initialize a zero matrix of the specified size
    matrix = np.zeros((cut_off, cut_off), dtype=np.complex128)
    
    # Fill the off-diagonal values with square roots of integers
    for i in range(cut_off):
        if i > 0:  # Fill left off-diagonal
            if type == 'q':
                matrix[i][i - 1] = (1/np.sqrt(2)) * np.sqrt(i)  # sqrt(i) for left off-diagonal
            else:
                matrix[i][i - 1] = (1j/np.sqrt(2)) * np.sqrt(i)

        if i < cut_off - 1:  # Fill right off-diagonal
            if type == 'q':
                matrix[i][i + 1] = (1/np.sqrt(2)) * np.sqrt(i + 1)  # sqrt(i + 1) for right off-diagonal
            else:
                matrix[i][i + 1] = (-1j/np.sqrt(2)) * np.sqrt(i + 1)

    return matrix


#############################################################################
                                #SUSY QM
#############################################################################
    
# Function to calculate the Hamiltonian
def calculate_Hamiltonian(cut_off, potential):
    # Generate the position (q) and momentum (p) matrices
    q = create_matrix(cut_off, 'q')  # q matrix
    p = create_matrix(cut_off, 'p')  # p matrix

    # Calculate q^2 and q^3 for potential terms
    q2 = np.matmul(q, q)
    q3 = np.matmul(q2, q)

    #fermionic identity
    I_f = np.eye(2)

    #bosonic identity
    I_b = np.eye(cut_off)

    # Superpotential derivatives
    if potential == 'QHO':
        W_prime = q  # W'(q) = q
        W_double_prime = I_b #W''(q) = 1

    elif potential == 'AHO':
        W_prime = q + q3  # W'(q) = q + q^3
        W_double_prime = I_b + 3 * q2  # W''(q) = 1 + 3q^2

    elif potential == 'DW':
        W_prime = q + q2 + I_b  # W'(q) = q + q^2 + 1
        W_double_prime = I_b + 2 * q  # W''(q) = 1 + 2q

    else:
        print("Not a valid potential")
        raise

    # Kinetic term: p^2
    p2 = np.matmul(p, p)

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

#################################################################################
                                #Wess-Zumino
#################################################################################

from scipy.sparse import eye, kron, coo_matrix
from functools import reduce

def kron_tensor(size, site, total_sites, operator):

    I = eye(size, format='coo', dtype=np.complex128)
    operators = [I] * total_sites
    operators[site] = operator
  
    return reduce(kron, operators) 


def calculate_wz_hamiltonian(cutoff, N, a, potential, boundary_condition, c=0):

    I_b = eye(cutoff ** N, format='coo')
    I_f = eye(2 ** N, format='coo')
    dim = I_b.size * I_f.size

    zero_qop = coo_matrix((I_b.size,I_b.size), dtype=np.complex128)
    zero_cop = coo_matrix((I_f.size,I_f.size), dtype=np.complex128)

    H_b = coo_matrix((I_b.size,I_b.size), dtype=np.complex128)
    H_f = coo_matrix((I_f.size,I_f.size), dtype=np.complex128)
    H_bi = coo_matrix((dim, dim), dtype=np.complex128)

    H = coo_matrix((dim, dim), dtype=np.complex128)

    q = coo_matrix(create_matrix(cutoff, 'q'))
    p = coo_matrix(create_matrix(cutoff, 'p'))
    chi = coo_matrix([[0, 1], [0, 0]], dtype=np.complex128)
    chidag = coo_matrix([[0, 0], [1, 0]], dtype=np.complex128)

    for n in range(N):

        q_n = kron_tensor(cutoff, n, N, q)
        p_n = kron_tensor(cutoff, n, N, p)

        chi_n = kron_tensor(2, n, N, chi)
        chidag_n = kron_tensor(2, n, N, chidag)

        # Boson terms
        # Kinetic term
        p2 = coo_matrix(p_n @ p_n / (2 * a))  

        # Potential term
        if potential == "linear":
            W_prime = q_n  # W'(q) = q
            W_double_prime = I_b  # W''(q) = 1

        elif potential == "quadratic":
            W_prime = c * I_b + coo_matrix(q_n @ q_n)  # W'(q) = c + q^2
            W_double_prime = 2*q_n  # W''(q) = 2q

        potential_term = (a / 2) * coo_matrix(W_prime @ W_prime)

        if boundary_condition == 'dirichlet':
            if n == 0:
                q_nm1 = zero_qop
                q_np1 = kron_tensor(cutoff, (n + 1), N, q)
                chi_np1 = kron_tensor(2, (n + 1), N, chi)
                chidag_np1 = kron_tensor(2, (n + 1) , N, chidag)

            elif n == N-1:
                q_nm1 = kron_tensor(cutoff, (n - 1), N, q)
                q_np1 = zero_qop
                chi_np1 = zero_cop
                chidag_np1 = zero_cop

            else:
                q_np1 = kron_tensor(cutoff, (n + 1), N, q)
                chi_np1 = kron_tensor(2, (n + 1), N, chi)
                chidag_np1 = kron_tensor(2, (n + 1) , N, chidag)
                q_nm1 = kron_tensor(cutoff, (n - 1), N, q)

        elif boundary_condition == 'periodic':
            
            q_np1 = kron_tensor(cutoff, (n + 1) % N, N, q)
            q_nm1 = kron_tensor(cutoff, (n - 1) % N, N, q)

            if n == N-1:
                chi_np1 = kron_tensor(2, 0, N, -1*chi)
                chidag_np1 = kron_tensor(2, 0, N, -1*chidag)

            else:
                chi_np1 = kron_tensor(2, (n + 1), N, chi)
                chidag_np1 = kron_tensor(2, (n + 1) , N, chidag)
    
            
        gradient = coo_matrix((q_np1 - q_nm1) / (2 * a))
        gradient_term = (a / 2) * (gradient @ gradient)

        potential_gradient_term = a * (W_prime @ gradient)

        H_b += (p2 + potential_term + gradient_term + potential_gradient_term)

        # Boson-Fermion term
        commutator_term = kron(W_double_prime, ((-1) ** n) * coo_matrix(chidag_n @ chi_n - 0.5 * I_f), format='coo')
        H_bi += commutator_term

        #Fermion term
        fermion_hopping = 0.5*(chidag_n @ chi_np1 + chidag_np1 @ chi_n)
        H_f += (fermion_hopping)

    H = kron(H_b, I_f, format='coo') + kron(I_b, H_f, format='coo') + H_bi

    return H.todense()
   
   
