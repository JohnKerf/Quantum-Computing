import numpy as np
from qiskit.circuit.library import XXPlusYYGate, QFTGate, DiagonalGate, PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, PauliList
from qiskit.synthesis import LieTrotter
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


# -----------------------------------------------------------------------------
# Walsh -> Z-only SparsePauliOp for diagonal vectors (length 2**nb)
# -----------------------------------------------------------------------------
def _fast_walsh_hadamard(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).copy()
    n = v.size
    h = 1
    while h < n:
        for i in range(0, n, 2 * h):
            a = v[i:i + h].copy()
            b = v[i + h:i + 2 * h].copy()
            v[i:i + h] = a + b
            v[i + h:i + 2 * h] = a - b
        h *= 2
    return v

def diagonal_to_z_sparsepauliop(
    diag_vals: np.ndarray,
    nb: int,
    *,
    atol: float = 1e-12,
    drop_identity: bool = False,
) -> SparsePauliOp:
    diag_vals = np.asarray(diag_vals, dtype=np.float64).reshape(-1)
    if diag_vals.size != (1 << nb):
        raise ValueError(f"diag length {diag_vals.size} != 2**nb {1<<nb}")

    coeffs = _fast_walsh_hadamard(diag_vals) / (1 << nb)

    labels = []
    out = []
    for s, c in enumerate(coeffs):
        if drop_identity and s == 0:
            continue
        if abs(c) <= atol:
            continue

        # Qiskit label convention: rightmost char is qubit-0 of the *given qubit list*
        chars = ["I"] * nb
        for q in range(nb):
            if (s >> q) & 1:
                chars[nb - 1 - q] = "Z"
        labels.append("".join(chars))
        out.append(float(c))

    if not labels:
        return SparsePauliOp(PauliList(["I" * nb]), coeffs=[0.0])

    return SparsePauliOp(PauliList(labels), coeffs=np.asarray(out, dtype=np.float64)).simplify(atol=atol)

def truncate_spo_by_weight(
    H: SparsePauliOp,
    *,
    keep_ratio: float = 1.0,
    min_keep: int = 0,
    weight_power: float = 2.0,
    atol: float = 1e-12,
    reorder_to_original: bool = True,   # <-- add this
):
    """
    Truncate terms by coefficient weight, optionally restoring original term order.
    Returns: (H_truncated, info)
    """
    coeffs = np.asarray(H.coeffs)
    paulis = H.paulis  # PauliList

    abs_c = np.abs(coeffs)
    order = np.argsort(abs_c)[::-1]                 # pick largest first
    w = (abs_c[order] ** weight_power)
    cum = np.cumsum(w)
    total = float(cum[-1]) if len(cum) else 0.0
    target = keep_ratio * total if total else 0.0

    m = int(np.searchsorted(cum, target, side="left") + 1) if len(cum) else 0
    m = max(m, int(min_keep))
    m = min(m, len(coeffs))

    keep_idx = order[:m]                             # indices into ORIGINAL H
    keep_idx = np.asarray(keep_idx, dtype=int)

    if reorder_to_original:
        keep_idx = np.sort(keep_idx)                # <-- SQM-style fix

    H_tr = SparsePauliOp(paulis[keep_idx], coeffs[keep_idx]).simplify(atol=atol)

    info = {
        "m": m,
        "n": len(coeffs),
        "keep_frac_terms": (m / len(coeffs)) if len(coeffs) else 0.0,
        "keep_ratio": keep_ratio,
        "keep_idx": keep_idx,                        # keep for debugging
        "total_weight": total,
        "truncated_weight": float(total - (cum[m-1] if m > 0 else 0.0)) if total else 0.0,
    }
    return H_tr, info

# -----------------------------------------------------------------------------
# WZ split operator using Walsh-truncated Z expansions (single-site & two-site)
# -----------------------------------------------------------------------------
def append_wz_split_operator_evolution(
    qc,
    qf_sites,                 # list of fermion qubits, length N
    qb_sites,                 # list of lists: qb_sites[n] is boson qubits for site n (length nb)
    basis_info,               # dict: x, k2, Wp_diag, Wp2_diag, Wpp_diag
    t_k: float,               # total evolution time for this Krylov step
    n_steps: int,             # number of outer split steps
    *,
    a: float = 1.0,
    boundary_condition: str = "periodic",   # "periodic" or "dirichlet"
    boson_indexing: str = "lsb_first",      # "lsb_first" or "msb_first"
    include_fermion_hopping: bool = True,
    scheme: str = "strang",                 # "strang" or "lie"
    # truncation knobs
    keep_ratio_1site: float = 1.0,
    keep_ratio_2site: float = 1.0,
    min_keep_1site: int = 0,
    min_keep_2site: int = 0,
    weight_power: float = 2.0,
    atol: float = 1e-12,
):
    """
    Custom split-operator evolution for Wess–Zumino.

    - scheme="strang": 2nd-order Strang splitting over blocks: V(dt/2) T(dt) hop(dt) V(dt/2)
    - scheme="lie":    1st-order Lie-Trotter over blocks:      V(dt)   T(dt) hop(dt)

    Uses Walsh-Hadamard Z-string expansions for diagonal operators (truncated by coeff-weight).
    """
    scheme = scheme.lower()
    if scheme not in ("strang", "lie"):
        raise ValueError("scheme must be 'strang' or 'lie'")

    if n_steps <= 0:
        raise ValueError("n_steps must be >= 1")
    dt = float(t_k) / float(n_steps)

    N = len(qb_sites)
    if len(qf_sites) != N:
        raise ValueError("qf_sites and qb_sites must have the same number of sites.")
    if N == 0:
        return

    nb = len(qb_sites[0])
    if any(len(qb_sites[n]) != nb for n in range(N)):
        raise ValueError("All qb_sites[n] must have the same length (nb).")

    if boundary_condition not in ("periodic", "dirichlet"):
        raise ValueError("boundary_condition must be 'periodic' or 'dirichlet'.")
    if boson_indexing not in ("lsb_first", "msb_first"):
        raise ValueError("boson_indexing must be 'lsb_first' or 'msb_first'.")

    # diagonals
    x   = np.asarray(basis_info["x"], dtype=np.float64)
    k2  = np.asarray(basis_info["k2"], dtype=np.float64)
    Wp  = np.asarray(basis_info["Wp_diag"], dtype=np.float64)
    Wp2 = np.asarray(basis_info["Wp2_diag"], dtype=np.float64)
    Wpp = np.asarray(basis_info["Wpp_diag"], dtype=np.float64)
    x2 = x * x

    # qubit order helper (must match how diag values are indexed)
    def qb_order(qb_list):
        return list(qb_list) if boson_indexing == "lsb_first" else list(qb_list)[::-1]

    # build / truncate helpers for 1-site and 2-site diagonals
    def _make_1site_op(diag_vals: np.ndarray, *, drop_I: bool = True):
        H_full = diagonal_to_z_sparsepauliop(diag_vals, nb, atol=atol, drop_identity=drop_I)
        H_tr, info = truncate_spo_by_weight(
            H_full,
            keep_ratio=keep_ratio_1site,
            min_keep=min_keep_1site,
            weight_power=weight_power,
            atol=atol,
        )
        return H_tr, info

    def _make_2site_op(diag_vals_2site: np.ndarray, *, drop_I: bool = True):
        diag_flat = np.asarray(diag_vals_2site, dtype=np.float64).reshape(-1)
        H_full = diagonal_to_z_sparsepauliop(diag_flat, 2 * nb, atol=atol, drop_identity=drop_I)
        H_tr, info = truncate_spo_by_weight(
            H_full,
            keep_ratio=keep_ratio_2site,
            min_keep=min_keep_2site,
            weight_power=weight_power,
            atol=atol,
        )
        return H_tr, info

    # -------------------------------------------------------------------------
    # Precompute truncated operators and PauliEvolutionGates (half + full where needed)
    # -------------------------------------------------------------------------

    # (a/2) W'(q)^2
    H_Wp2, _info_Wp2 = _make_1site_op(Wp2, drop_I=True)
    evo_Wp2_half = PauliEvolutionGate(H_Wp2, time=(a * dt / 4.0), synthesis=LieTrotter(reps=1))
    evo_Wp2_full = PauliEvolutionGate(H_Wp2, time=(a * dt / 2.0), synthesis=LieTrotter(reps=1))

    # Boson kinetic p^2/(2a) in k basis (QFT sandwich)
    H_k2, _info_k2 = _make_1site_op(k2, drop_I=True)
    evo_T_full = PauliEvolutionGate(H_k2, time=(dt / (2.0 * a)), synthesis=LieTrotter(reps=1))

    # bf term: Z_f ⊗ Wpp_op  (note: Wpp_op may include identity if you keep it)
    H_Wpp_b, _info_Wpp = _make_1site_op(Wpp, drop_I=False)
    labels = ["Z" + p.to_label() for p in H_Wpp_b.paulis]
    H_ZfWpp = SparsePauliOp(labels, H_Wpp_b.coeffs).simplify(atol=atol)

    # half-step bf: time = -((-1)^n) * dt/4
    evo_bf_even_half = PauliEvolutionGate(H_ZfWpp, time=-(+1.0) * dt / 4.0, synthesis=LieTrotter(reps=1))
    evo_bf_odd_half  = PauliEvolutionGate(H_ZfWpp, time=-(-1.0) * dt / 4.0, synthesis=LieTrotter(reps=1))
    # full-step bf: double
    evo_bf_even_full = PauliEvolutionGate(H_ZfWpp, time=-(+1.0) * dt / 2.0, synthesis=LieTrotter(reps=1))
    evo_bf_odd_full  = PauliEvolutionGate(H_ZfWpp, time=-(-1.0) * dt / 2.0, synthesis=LieTrotter(reps=1))

    # Gradient + pgrad terms (always included, as requested)
    # q^2 term
    H_x2, _info_x2 = _make_1site_op(x2, drop_I=True)

    # half-step times
    evo_grad_q2_all_half      = PauliEvolutionGate(H_x2, time=(dt / (8.0 * a)),  synthesis=LieTrotter(reps=1))
    evo_grad_q2_edge_half     = PauliEvolutionGate(H_x2, time=(dt / (16.0 * a)), synthesis=LieTrotter(reps=1))
    evo_grad_q2_interior_half = PauliEvolutionGate(H_x2, time=(dt / (8.0 * a)),  synthesis=LieTrotter(reps=1))

    # full-step times (double)
    evo_grad_q2_all_full      = PauliEvolutionGate(H_x2, time=(dt / (4.0 * a)),  synthesis=LieTrotter(reps=1))
    evo_grad_q2_edge_full     = PauliEvolutionGate(H_x2, time=(dt / (8.0 * a)),  synthesis=LieTrotter(reps=1))
    evo_grad_q2_interior_full = PauliEvolutionGate(H_x2, time=(dt / (4.0 * a)),  synthesis=LieTrotter(reps=1))

    # qq between (n-1, n+1)
    vals_grad_qq = np.outer(x, x)
    H_grad_qq_2, _info_gradqq = _make_2site_op(vals_grad_qq, drop_I=True)
    evo_grad_qq_half = PauliEvolutionGate(H_grad_qq_2, time=(-dt / (8.0 * a)), synthesis=LieTrotter(reps=1))
    evo_grad_qq_full = PauliEvolutionGate(H_grad_qq_2, time=(-dt / (4.0 * a)), synthesis=LieTrotter(reps=1))

    # potential-gradient
    if boundary_condition == "periodic":
        vals_bond = 0.5 * (np.outer(Wp, x) - np.outer(x, Wp))
        H_pgrad_2, _info_pgrad = _make_2site_op(vals_bond, drop_I=True)
        evo_pgrad_bond_half = PauliEvolutionGate(H_pgrad_2, time=(dt / 2.0), synthesis=LieTrotter(reps=1))
        evo_pgrad_bond_full = PauliEvolutionGate(H_pgrad_2, time=(dt),       synthesis=LieTrotter(reps=1))
        evo_pgrad_forward_half = evo_pgrad_backward_half = None
        evo_pgrad_forward_full = evo_pgrad_backward_full = None
    else:
        vals_forward = np.outer(Wp, x)
        H_pgrad_dir_2, _info_pgrad = _make_2site_op(vals_forward, drop_I=True)
        evo_pgrad_forward_half  = PauliEvolutionGate(H_pgrad_dir_2, time=(+dt / 4.0), synthesis=LieTrotter(reps=1))
        evo_pgrad_backward_half = PauliEvolutionGate(H_pgrad_dir_2, time=(-dt / 4.0), synthesis=LieTrotter(reps=1))
        evo_pgrad_forward_full  = PauliEvolutionGate(H_pgrad_dir_2, time=(+dt / 2.0), synthesis=LieTrotter(reps=1))
        evo_pgrad_backward_full = PauliEvolutionGate(H_pgrad_dir_2, time=(-dt / 2.0), synthesis=LieTrotter(reps=1))
        evo_pgrad_bond_half = evo_pgrad_bond_full = None

    # QFT sandwich for kinetic
    qft = QFTGate(nb)
    iqft = qft.inverse()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def neighbors(idx: int):
        if boundary_condition == "periodic":
            return (idx - 1) % N, (idx + 1) % N
        nm1 = idx - 1 if idx > 0 else None
        np1 = idx + 1 if idx < N - 1 else None
        return nm1, np1

    def apply_V(kind: str):
        """Apply V block with either 'half' or 'full' step scaling."""
        if kind == "half":
            evo_Wp2 = evo_Wp2_half
            bf_even = evo_bf_even_half
            bf_odd  = evo_bf_odd_half

            grad_q2_all      = evo_grad_q2_all_half
            grad_q2_edge     = evo_grad_q2_edge_half
            grad_q2_interior = evo_grad_q2_interior_half
            grad_qq          = evo_grad_qq_half

            pgrad_bond   = evo_pgrad_bond_half
            pgrad_fwd    = evo_pgrad_forward_half
            pgrad_bwd    = evo_pgrad_backward_half

        elif kind == "full":
            evo_Wp2 = evo_Wp2_full
            bf_even = evo_bf_even_full
            bf_odd  = evo_bf_odd_full

            grad_q2_all      = evo_grad_q2_all_full
            grad_q2_edge     = evo_grad_q2_edge_full
            grad_q2_interior = evo_grad_q2_interior_full
            grad_qq          = evo_grad_qq_full

            pgrad_bond   = evo_pgrad_bond_full
            pgrad_fwd    = evo_pgrad_forward_full
            pgrad_bwd    = evo_pgrad_backward_full

        else:
            raise ValueError("kind must be 'half' or 'full'")

        # onsite bosonic potential + bf
        for n in range(N):
            qb = qb_order(qb_sites[n])
            qf = qf_sites[n]

            qc.append(evo_Wp2, qb)

            if (n % 2) == 0:
                qc.append(bf_even, [qf] + qb)
            else:
                qc.append(bf_odd,  [qf] + qb)

        # (1) gradient q^2
        if boundary_condition == "periodic":
            for j in range(N):
                qc.append(grad_q2_all, qb_order(qb_sites[j]))
        else:
            for j in range(N):
                qb = qb_order(qb_sites[j])
                if j == 0 or j == N - 1:
                    qc.append(grad_q2_edge, qb)
                else:
                    qc.append(grad_q2_interior, qb)

        # (2) gradient qq between (n-1, n+1)
        for n in range(N):
            nm1, np1 = neighbors(n)
            if (nm1 is None) or (np1 is None):
                continue
            qb_pair = qb_order(qb_sites[nm1]) + qb_order(qb_sites[np1])
            qc.append(grad_qq, qb_pair)

        # (3) potential-gradient
        if boundary_condition == "periodic":
            for n in range(N):
                np1 = (n + 1) % N
                qb_pair = qb_order(qb_sites[n]) + qb_order(qb_sites[np1])
                qc.append(pgrad_bond, qb_pair)
        else:
            for n in range(N):
                nm1, np1 = neighbors(n)
                if np1 is not None:
                    qb_pair = qb_order(qb_sites[n]) + qb_order(qb_sites[np1])
                    qc.append(pgrad_fwd, qb_pair)
                if nm1 is not None:
                    qb_pair = qb_order(qb_sites[n]) + qb_order(qb_sites[nm1])
                    qc.append(pgrad_bwd, qb_pair)

    def apply_hop_full():
        if (not include_fermion_hopping) or (N < 2):
            return

        if boundary_condition == "periodic":
            for n in range(N):
                n_next = (n + 1) % N
                sign = -1.0 if (n == N - 1) else 1.0
                theta = sign * (dt / 2.0)  # dt already per-step
                qc.append(XXPlusYYGate(theta, 0.0), [qf_sites[n], qf_sites[n_next]])
        else:
            for n in range(N - 1):
                theta = dt / 2.0
                qc.append(XXPlusYYGate(theta, 0.0), [qf_sites[n], qf_sites[n + 1]])

    # -------------------------------------------------------------------------
    # Main loop: toggle between Strang and Lie-Trotter over blocks
    # -------------------------------------------------------------------------
    for _ in range(n_steps):
        if scheme == "strang":
            apply_V("half")
        else:
            apply_V("full")

        # kinetic per site: QFT -> evo_T_full -> iQFT
        for n in range(N):
            qb = qb_order(qb_sites[n])
            qc.append(qft, qb)
            qc.append(evo_T_full, qb)
            qc.append(iqft, qb)

        apply_hop_full()

        if scheme == "strang":
            apply_V("half")

    # Optional: return truncation summaries if you want to log them
    # return {
    #     "Wp2": info_Wp2,
    #     "k2": info_k2,
    #     "Wpp": info_Wpp,
    #     "grad_q2": info_x2,
    #     "grad_qq": info_gradqq,
    #     "pgrad": (info_pgrad if include_grad_and_pgrad and N >= min_N_for_grad else None),
    # }


