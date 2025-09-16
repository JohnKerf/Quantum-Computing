from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix

Complex = np.complex128
Potential1D = Literal["QHO", "AHO", "DW"]

# ----------------- helpers -----------------
def _qp_matrix(cutoff: int, kind: Literal["q","p"], m: float = 1.0, dtype=Complex) -> NDArray[Complex]:
    mat = np.zeros((cutoff, cutoff), dtype=dtype)
    for i in range(cutoff):
        if i > 0:
            if kind == "q":
                mat[i, i-1] = (1/np.sqrt(2*m)) * np.sqrt(i)
            else:
                mat[i, i-1] = (1j*np.sqrt(m/2)) * np.sqrt(i)
        if i < cutoff - 1:
            if kind == "q":
                mat[i, i+1] = (1/np.sqrt(2*m)) * np.sqrt(i+1)
            else:
                mat[i, i+1] = (-1j*np.sqrt(m/2)) * np.sqrt(i+1)
    return mat

# ----------------- SUSY QM dataclass -----------------
@dataclass(slots=True)
class SUSY1D:
    cutoff: int
    potential: Potential1D
    m: float = 1.0
    g: float = 1.0
    u: float = 1.0
    dtype: np.dtype = Complex
    sparse: bool = False 

    # cache for q, p
    _q: Optional[NDArray[Complex]] = field(default=None, init=False, repr=False)
    _p: Optional[NDArray[Complex]] = field(default=None, init=False, repr=False)

    @property
    def q(self) -> NDArray[Complex]:
        if self._q is None:
            self._q = _qp_matrix(self.cutoff, "q", self.m, self.dtype)
        return self._q

    @property
    def p(self) -> NDArray[Complex]:
        if self._p is None:
            self._p = _qp_matrix(self.cutoff, "p", self.m, self.dtype)
        return self._p

    @property
    def I_f(self) -> NDArray[Complex]:
        return np.eye(2, dtype=self.dtype)

    @property
    def I_b(self) -> NDArray[Complex]:
        return np.eye(self.cutoff, dtype=self.dtype)

    @property
    def Z(self) -> NDArray[Complex]:
        return np.array([[1, 0], [0, -1]], dtype=self.dtype)

    # ---- superpotential derivatives ----
    def W_prime(self) -> NDArray[Complex]:
        q = self.q
        if self.potential == "QHO":
            return self.m * q
        elif self.potential == "AHO":
            q2, q3 = q @ q, (q @ q) @ q
            return self.m * q + self.g * q3
        elif self.potential == "DW":
            q2 = q @ q
            return self.m * q + self.g * q2 + self.g * (self.u**2) * self.I_b
        raise ValueError(f"Unknown potential {self.potential}")

    def W_double_prime(self) -> NDArray[Complex]:
        q = self.q
        if self.potential == "QHO":
            return self.m * self.I_b
        elif self.potential == "AHO":
            return self.m * self.I_b + 3 * self.g * (q @ q)
        elif self.potential == "DW":
            return self.m * self.I_b + 2 * self.g * q
        raise ValueError(f"Unknown potential {self.potential}")

    # ---- Hamiltonian ----
    def hamiltonian(self) -> NDArray[Complex] | coo_matrix:
        p2 = self.p @ self.p
        Wp = self.W_prime()
        Wpp = self.W_double_prime()

        comm = np.kron(self.Z, Wpp)
        kin = np.kron(self.I_f, p2)
        pot = np.kron(self.I_f, Wp @ Wp)

        H = 0.5 * (kin + pot + comm)
        H[np.abs(H) < 1e-11] = 0
        return H if not self.sparse else coo_matrix(H)
