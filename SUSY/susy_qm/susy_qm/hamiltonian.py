import numpy as np

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
    
   
