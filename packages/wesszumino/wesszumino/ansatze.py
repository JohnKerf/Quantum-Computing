import numpy as np
from qiskit.circuit.library import XXPlusYYGate
from qiskit import QuantumCircuit

def build_avqe_pattern_ansatz(
    N: int,
    cutoff: int,
    *,
    beta: float = -np.pi / 2,
    bond0_sign: int = -1,  # +1 => +pi/8 on bond 0, -1 => -pi/8 on bond 0
    include_basis: bool =True,
    include_rys: bool =True,
    include_xxyys: bool =True
) -> QuantumCircuit:
    """
    Pattern-only pruned ansatz inferred from your diagrams.

    XXPlusYY:
      - acts on last qubit of each site
      - nearest-neighbour chain
      - theta alternates +/- pi/8 along bonds

    RY:
      cutoff=2: none
      cutoff=4: local=1 only
        - edge sites: -0.0784
        - inner sites: -0.14
      cutoff=8: locals=1,2
        - edge sites: (-0.0784, +0.0038)
        - inner sites: (-0.14,   +0.0125)
      cutoff=16: same RY pattern as cutoff=8, but applied to locals=1,2
        (i.e., still only 2 RYs per site even though qps=5)
    """


    qps = int((1 + np.log2(cutoff)))
    n_qubits = N * qps
    qc = QuantumCircuit(n_qubits)

    def q(site: int, local: int) -> int:
        return site * qps + local

    def last(site: int) -> int:
        return q(site, qps - 1)

    def is_edge(site: int) -> bool:
        return site == 0 or site == N - 1
    
    # -------------------
    # Basis state
    # -------------------
    if include_basis:
        for s in range(N):
            want_odd = (s % 2 == 1)
            if want_odd:
                qc.x(last(s))

    # -------------------
    # RY pattern by cutoff
    # -------------------
    if include_rys:
        if cutoff == 4:
            for s in range(N):
                theta = -0.0784 if is_edge(s) else -0.14
                qc.ry(theta, q(s, 1))  # only local=1

        elif cutoff in (8, 16):
            # 2 RYs per site on locals 1 and 2
            for s in range(N):
                if is_edge(s):
                    thetas = (-0.0784, 0.0038)
                else:
                    #continue
                    thetas = (-0.14, 0.0125)
                qc.ry(thetas[0], q(s, 1))
                qc.ry(thetas[1], q(s, 2))

    # cutoff == 2: no RYs

    # -------------------
    # XXPlusYY chain
    # -------------------
    if include_xxyys:
        theta0 = np.pi / 8
        for b in range(N - 1):
            theta = bond0_sign * ((-1) ** b) * theta0
            qc.append(XXPlusYYGate(theta, beta), [last(b), last(b + 1)])

    return qc
