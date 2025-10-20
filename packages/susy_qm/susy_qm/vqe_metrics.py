# susy_qm/metrics/vqe_summary.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Dict, Any, List, Tuple, Optional, Literal
import json
import logging
import os
import numpy as np
import pandas as pd

import git
repo_path = git.Repo('.', search_parent_directories=True).working_tree_dir

__all__ = ["VQESummary"]

logger = logging.getLogger(__name__)


# ---------- small helpers ----------
def _safe_total_seconds(x: Any) -> float:
    """Convert timedelta-like (str/pandas Timedelta) to total seconds; NaN on failure."""
    try:
        return pd.to_timedelta(x).total_seconds()
    except Exception:  # noqa: BLE001
        return float("nan")


def _split_std_around_mean(values: np.ndarray, mean: float) -> Tuple[float, float]:
    """
    Lower/upper std around mean (ddof=1). Returns (lower_std, upper_std).
    If a side has <2 samples, return 0.0 for that side.
    """
    lower = values[values <= mean]
    upper = values[values >= mean]

    def _std(a: np.ndarray) -> float:
        return float(np.std(a, ddof=1)) if a.size >= 2 else 0.0

    return _std(lower), _std(upper)


def _load_json(fp: Path) -> Dict[str, Any]:
    with fp.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------- main container ----------
@dataclass(slots=True)
class VQESummary:
    """
    Container for tidy per-(potential, cutoff) aggregates + raw eval distributions.

    Access patterns:
      - Whole table: summary.df
      - Single field: summary.get("evals_dist", "DW", 16)
      - Dict of distributions: summary.evals_dist_map[("DW", 16)]
      - Plot-friendly pivots: summary.times, summary.time_bars, summary.evals_stats, summary.delta_e
    """
    df: pd.DataFrame  # columns: potential, cutoff, mean_time_s, lower_std_s, upper_std_s, ...

    # ----------------- Constructors -----------------
    @classmethod
    def from_path(
        cls,
        path: str | Path,
        cutoffs: Iterable[int],
        potentials: Iterable[str],
        *,
        converged_only: bool = True,
        on_missing: Literal["error", "skip"] = "error",
        debug: bool = False,
    ) -> "VQESummary":
        """
        Build summary from a directory layout: {path}/{potential}/{potential}_{cutoff}.json
        Always expects an "num_evaluations" field in JSON.
        """
      
        rows: List[Dict[str, Any]] = []

        for potential in potentials:
            for cutoff in cutoffs:
                fp = Path(os.path.join(repo_path, path, potential, f"{potential}_{cutoff}.json"))
                if not fp.exists():
                    msg = f"Missing file: {fp}"
                    if on_missing == "skip":
                        if debug:
                            logger.debug("[skip] %s", msg)
                        continue
                    raise FileNotFoundError(msg)

                data = _load_json(fp)

                # --- success/convergence
                success = np.asarray(data.get("success", []), dtype=bool)
                n_runs = int(success.size)
                converged_idx = np.where(success)[0]
                n_conv = int(converged_idx.size)

                # --- times
                t_raw = data.get("run_times", [])
                t_sec = np.array([_safe_total_seconds(t) for t in t_raw], dtype=float)
                t_sec = t_sec[~np.isnan(t_sec)]
                if t_sec.size:
                    mean_t = float(np.mean(t_sec))
                    lstd, ustd = _split_std_around_mean(t_sec, mean_t)
                else:
                    mean_t = lstd = ustd = 0.0

                # --- evals distribution (always num_evaluations)
                evals_d = np.asarray(data.get("num_evaluations", []), dtype=float)
                if evals_d.size:
                    mean_evals = float(np.mean(evals_d))
                    min_evals = float(np.min(evals_d))
                    max_evals = float(np.max(evals_d))
                    evals_list = evals_d.tolist()
                else:
                    mean_evals = min_evals = max_evals = 0.0
                    evals_list = []

                # --- energies
                exact_vals = np.asarray(data.get("exact_eigenvalues", []), dtype=float)
                exact_min = float(np.min(exact_vals)) if exact_vals.size else float("nan")

                results = np.asarray(data.get("results", []), dtype=float)
                if results.size:
                    if converged_only and n_conv > 0:
                        sel = results[converged_idx]
                    elif converged_only and n_conv == 0:
                        sel = np.array([np.nan])
                    else:
                        sel = results
                    median_e = float(np.nanmedian(sel))
                    min_e = float(np.nanmin(sel))
                else:
                    median_e = float("nan")
                    min_e = float("nan")

                delta_median = (
                    float(np.abs(exact_min - median_e))
                    if not np.isnan(exact_min)
                    else float("nan")
                )

                delta_min = (
                    float(np.abs(exact_min - min_e))
                    if not np.isnan(exact_min)
                    else float("nan")
                )

                if debug:
                    logger.debug("%s cutoff=%s ΔE_med=%s", potential, cutoff, delta_median)

                rows.append(
                    {
                        "potential": potential,
                        "cutoff": int(cutoff),
                        "mean_time_s": mean_t,
                        "lower_std_s": lstd,
                        "upper_std_s": ustd,
                        "mean_evals": mean_evals,
                        "min_evals": min_evals,
                        "max_evals": max_evals,
                        "delta_median_e": delta_median,
                        "delta_min_e": delta_min,
                        "n_runs": n_runs,
                        "n_converged": n_conv,
                        "evals_dist": evals_list,
                        "_converged_only": bool(converged_only),
                        "_source_file": str(fp),
                    }
                )

        df = pd.DataFrame(rows).sort_values(["potential", "cutoff"]).reset_index(drop=True)
        return cls(df=df)

    # ----------------- Getters -----------------
    def get(self, field: str, potential: str, cutoff: int) -> Any:
        """Fetch a single field (column) for a (potential, cutoff) row."""
        row = self.df[(self.df.potential == potential) & (self.df.cutoff == int(cutoff))]
        if row.empty:
            raise KeyError(f"No row for ({potential}, {cutoff})")
        return row.iloc[0][field]

    @property
    def evals_dist_map(self) -> Dict[Tuple[str, int], List[float]]:
        """Dict keyed by (potential, cutoff) -> evals distribution list."""
        return {(r.potential, int(r.cutoff)): r.evals_dist for r in self.df.itertuples(index=False)}

    @property
    def times(self) -> pd.DataFrame:
        """Pivot: mean run time (s) with index=cutoff, columns=potential."""
        return self.df.pivot(index="cutoff", columns="potential", values="mean_time_s")

    @property
    def time_bars(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """(lower_std, upper_std) pivots for asymmetric error bars."""
        lower = self.df.pivot(index="cutoff", columns="potential", values="lower_std_s")
        upper = self.df.pivot(index="cutoff", columns="potential", values="upper_std_s")
        return lower, upper

    @property
    def evals_stats(self) -> Dict[str, pd.DataFrame]:
        """Mean/min/max evals pivots, index=cutoff, columns=potential."""
        return {
            "mean": self.df.pivot(index="cutoff", columns="potential", values="mean_evals"),
            "min":  self.df.pivot(index="cutoff", columns="potential", values="min_evals"),
            "max":  self.df.pivot(index="cutoff", columns="potential", values="max_evals"),
        }

    @property
    def delta_median_e(self) -> pd.DataFrame:
        """Pivot of |median(result) - min(exact)|."""
        return self.df.pivot(index="cutoff", columns="potential", values="delta_median_e")
    
    @property
    def delta_min_e(self) -> pd.DataFrame:
        """Pivot of |min(result) - min(exact)|."""
        return self.df.pivot(index="cutoff", columns="potential", values="delta_min_e")

    # ----------------- Utilities -----------------
    def filter(self, *, potential: Optional[str] = None, cutoff: Optional[int] = None) -> "VQESummary":
        """Return a new VQESummary filtered by potential and/or cutoff."""
        m = pd.Series(True, index=self.df.index)
        if potential is not None:
            m &= self.df["potential"] == potential
        if cutoff is not None:
            m &= self.df["cutoff"] == int(cutoff)
        return VQESummary(self.df[m].reset_index(drop=True))

    def to_csv(self, path: str | Path, **kwargs) -> None:
        """Write the tidy table to CSV for later analysis."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path, index=False, **kwargs)


    def get_sel(
        self,
        potential: str,
        cutoff: int,
        *,
        converged_only: bool = True,
    ) -> np.ndarray:
        """
        Return the selected result array (`sel`) for a given (potential, cutoff),
        matching the same logic used in from_path().
        """
        # Locate the corresponding source file
        row = self.df[(self.df.potential == potential) & (self.df.cutoff == int(cutoff))]
        if row.empty:
            raise KeyError(f"No entry found for ({potential}, {cutoff})")

        fp = Path(row.iloc[0]["_source_file"])
        if not fp.exists():
            raise FileNotFoundError(f"Missing source file: {fp}")

        data = _load_json(fp)

        results = np.asarray(data.get("results", []), dtype=float)
        success = np.asarray(data.get("success", []), dtype=bool)
        converged_idx = np.where(success)[0]
        n_conv = int(converged_idx.size)

        if results.size:
            if converged_only and n_conv > 0:
                sel = results[converged_idx]
            elif converged_only and n_conv == 0:
                sel = np.array([np.nan])
            else:
                sel = results
        else:
            sel = np.array([], dtype=float)

        return sel

