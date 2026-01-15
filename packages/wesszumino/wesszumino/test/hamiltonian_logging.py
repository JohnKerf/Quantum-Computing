# hamiltonian_logging.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Iterable, Tuple

import numpy as np
from qiskit.quantum_info import SparsePauliOp


def get_file_logger(
    logfile: str | Path,
    *,
    name: str = "hamiltonian.build",
    level: int = logging.INFO,
    mode: str = "w",
) -> logging.Logger:
    """
    File-only logger. No terminal output.

    - mode="w" overwrites each run, mode="a" appends.
    """
    logfile = Path(logfile)
    logfile.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Critical: prevent printing via root logger handlers.
    logger.propagate = False

    # Remove any existing handlers (prevents duplicate logs if re-imported)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fh = logging.FileHandler(logfile, mode=mode, encoding="utf-8")
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    fh.setLevel(level)
    logger.addHandler(fh)

    return logger


def format_matrix(M: np.ndarray, *, name: str = "", max_dim: int = 16, precision: int = 6) -> str:
    M = np.asarray(M)
    lines = []
    if name:
        lines.append(f"{name}: shape={M.shape}, dtype={M.dtype}")
    if M.ndim != 2:
        lines.append(np.array2string(M, precision=precision, suppress_small=True))
        return "\n".join(lines)

    r, c = M.shape
    if r <= max_dim and c <= max_dim:
        lines.append(np.array2string(M, precision=precision, suppress_small=True))
    else:
        rr, cc = min(max_dim, r), min(max_dim, c)
        lines.append(f"(showing top-left {rr}x{cc})")
        lines.append(np.array2string(M[:rr, :cc], precision=precision, suppress_small=True))
    return "\n".join(lines)


def iter_pauli_terms(op: SparsePauliOp) -> Iterable[Tuple[str, complex]]:
    labels = op.paulis.to_labels()
    coeffs = op.coeffs
    for lab, coeff in zip(labels, coeffs):
        yield lab, complex(coeff)


def format_paulis(
    op: SparsePauliOp,
    *,
    title: str = "",
    max_terms: Optional[int] = 50,
    precision: int = 10,
    sort_by_abs: bool = False,
) -> str:
    terms = list(iter_pauli_terms(op))
    if sort_by_abs:
        terms.sort(key=lambda t: abs(t[1]), reverse=True)

    head = terms if (max_terms is None or len(terms) <= max_terms) else terms[:max_terms]
    tail_note = None if (max_terms is None or len(terms) <= max_terms) else f"... ({len(terms) - max_terms} more)"

    lines = []
    if title:
        lines.append(f"{title}: {len(terms)} terms")
    for lab, coeff in head:
        re = np.round(coeff.real, precision)
        im = np.round(coeff.imag, precision)
        if abs(im) < 10 ** (-precision):
            lines.append(f"{re} * {lab}")
        elif abs(re) < 10 ** (-precision):
            lines.append(f"{im}j * {lab}")
        else:
            lines.append(f"({re} + {im}j) * {lab}")
    if tail_note:
        lines.append(tail_note)
    return "\n".join(lines)


def log_matrix(logger: logging.Logger, name: str, M: np.ndarray) -> None:
    logger.info(format_matrix(M, name=name))


def log_paulis(
    logger: logging.Logger,
    name: str,
    op: SparsePauliOp,
    *,
    max_terms: int = 50,
    sort_by_abs: bool = False,
) -> None:
    logger.debug(format_paulis(op, title=name, max_terms=max_terms, sort_by_abs=sort_by_abs))
