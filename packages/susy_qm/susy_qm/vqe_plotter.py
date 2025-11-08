# susy_qm/plots/vqe_plots.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging
import itertools
from pathlib import Path
from typing import Iterable, Dict, Any, List, Tuple, Optional, Literal
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import LogLocator, LogFormatterMathtext, FuncFormatter, SymmetricalLogLocator, MaxNLocator, NullFormatter
from matplotlib.patches import Patch

from susy_qm.vqe_metrics import VQESummary

import git
repo_path = git.Repo('.', search_parent_directories=True).working_tree_dir

logger = logging.getLogger(__name__)

LabelPath = Tuple[str, str]

def _load_json(fp: Path) -> Dict[str, Any]:
    fp = Path(fp)
    with fp.open("r", encoding="utf-8") as f:
        return json.load(f)
    

def format_axis(
    ax, *,
    scale: str = "linear",
    linthresh: float = 1e-1,
    base: int = 10,
    show_zero: bool = True
):
    ax.tick_params(axis='both', which='both', direction='out', width=1, labelsize=8)

    ax.xaxis.get_offset_text().set_visible(False)

    if scale == "linear":
        ax.set_xscale("linear")

        # def fmt_linear(x, _):

        #     if x == 0:
        #         return "0"
            
        #     if abs(x) >= 1e3 or abs(x) < 1e-2:
        #         exp = int(np.floor(np.log10(abs(x))))
        #         coeff = x / (10**exp)
        #         sgn = "-" if x < 0 else ""

        #         if np.isclose(abs(coeff), 1.0, atol=1e-8):
        #             return rf"${sgn}10^{{{exp}}}$"
                
        #         c = f"{coeff:.1f}".rstrip("0").rstrip(".")

        #         return rf"${c}\times10^{{{exp}}}$"
            
        #     return f"{x:.4g}".rstrip("0").rstrip(".")
        
        # ax.xaxis.set_major_formatter(FuncFormatter(fmt_linear))

    elif scale == "symlog":

        ax.set_xscale("symlog", linthresh=linthresh, base=base)
        ax.xaxis.set_major_locator(SymmetricalLogLocator(base=base, linthresh=linthresh))

        def fmt_symlog(x, _):
            if x == 0:
                return "0" if show_zero else ""
            
            a = abs(x)
            if a < linthresh:
                s = f"{x:.3g}".rstrip("0").rstrip(".")
                return s or "0"
            
            exp = int(np.round(np.log10(a)))

            return rf"$-10^{{{exp}}}$" if x < 0 else rf"$10^{{{exp}}}$"
        
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_symlog))

    elif scale == "log":
        ax.set_xscale("log", base=base)
        ax.xaxis.set_major_locator(LogLocator(base=base))
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda x, _: rf"$10^{{{int(np.log10(x))}}}$" if x > 0 else ""
        ))

    ax.grid(True, axis='x', which='both', linestyle='--', alpha=0.35)



@dataclass(slots=True)
class BoxPlotter:
    potentials: List[str]
    cutoffs: List[int]
    shots_list: list[int]
    converged_only: bool = True
    on_missing: str = "skip"  # "skip" | "nan"
    debug: bool = False

    # ---------- loading ----------
    def _load(self, data_path: str, potential: str, cutoff: int, shots: int) -> Tuple[np.ndarray, float]:
        d_path = os.path.join(
            repo_path, data_path, str(shots), potential, f"{potential}_{cutoff}.json"
        )

        # Always return (energies, min_eigenvalue)
        if not os.path.exists(d_path):
            if self.debug:
                print(f"[missing] {d_path}")
            if self.on_missing == "skip":
                return np.array([], dtype=float), float("nan")
            else:  # "nan"
                return np.array([np.nan], dtype=float), float("nan")

        data = _load_json(d_path)

        # exact eigenvalues (robust if missing/empty)
        exact = data.get("exact_eigenvalues", [])
        if exact is None or len(exact) == 0:
            min_eigenvalue = float("nan")
        else:
            exact_eigenvalues = np.asarray(exact, dtype=float)
            # use nanmin in case any NaNs are present
            min_eigenvalue = float(np.nanmin(exact_eigenvalues))

        results = np.asarray(data.get("results", []), dtype=float)
        success = np.asarray(data.get("success", []), dtype=bool)

        if results.size:
            if self.converged_only:
                converged_idx = np.where(success)[0]
                if converged_idx.size > 0:
                    energies = results[converged_idx]
                else:
                    energies = np.array([np.nan], dtype=float)
            else:
                energies = results
        else:
            energies = np.array([], dtype=float)

        return energies, min_eigenvalue

    # ---------- axes helpers ----------
    @staticmethod
    def _as_2d_axes(axs, nr: int, nc: int):
        """Return axes as a 2D ndarray with shape (nr, nc)."""
        # Single Axes
        if hasattr(axs, "plot"):  # isinstance(axs, plt.Axes) but robust
            return np.array([[axs]])
        arr = np.asarray(axs)
        if arr.ndim == 0:  # single Axes wrapped oddly
            return np.array([[arr.item()]])
        if arr.ndim == 1:
            # If we asked for (nr, nc), decide orientation
            if nr == 1 and arr.size == nc:
                return arr.reshape(1, nc)
            if nc == 1 and arr.size == nr:
                return arr.reshape(nr, 1)
            # Fallback: assume row vector
            return arr.reshape(1, -1)
        return arr  # already 2D

    def _ensure_axes_grid(
        self,
        ncols: Optional[int] = None,
        nrows: Optional[int] = None,
        *,
        sharex: bool = False,
        sharey: bool = False,
        figsize=(12, 4),
        existing_axes=None,
    ):
        """Return (fig, axes_2d) with shape (nrows, ncols) == (len(potentials), len(shots_list))."""
        nc = ncols if ncols is not None else len(self.shots_list)
        nr = nrows if nrows is not None else len(self.potentials)

        if existing_axes is not None:
            fig = existing_axes.figure if hasattr(existing_axes, "figure") else plt.gcf()
            return fig, self._as_2d_axes(existing_axes, nr, nc)

        fig, axes = plt.subplots(nr, nc, figsize=figsize, sharex=sharex, sharey=sharey)
        axes2d = self._as_2d_axes(axes, nr, nc)
        return fig, axes2d

    # ---------- plots ----------
    def plot_energy_boxplot(
        self,
        *,
        data_paths: Optional[List[Tuple[str, str]]],
        shots=None,
        axes=None,
        box_width: float = 0.6,
        showfliers: bool = False,
        alpha: float = 0.6,
        show_legend: bool = True,
        scale_thresh: float = 1
    ):
        """
        Boxplots of energy distributions per cutoff, for each (potential, shots).
        - y-axis: cutoff labels
        - panels: one per potential × shots combination
        - COLORS: assigned by cutoff and kept consistent across subplots
        """
        by_dataset = False if type(data_paths)==str else True
        ncols = len(data_paths) if by_dataset else len(self.shots_list)  
        cols = data_paths if by_dataset else self.shots_list

        fig, axes_arr = self._ensure_axes_grid(ncols=ncols, existing_axes=axes, sharey=True, figsize=(12, 8))

        # consistent colors per cutoff across ALL subplots
        palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        cutoff_to_color = {c: palette[i % len(palette)] for i, c in enumerate(self.cutoffs)}


        for i, pot in enumerate(self.potentials):
            for j, col in enumerate(cols):

                if by_dataset: 
                    title, data_path = col
                else:
                    shots=col
                    data_path = data_paths

                ax = axes_arr[i, j]

                # gather energy arrays per cutoff (ensure non-empty arrays so boxplot won't crash)
                energies_per_cutoff = []
                min_eigenvalues = []
                for cutoff in self.cutoffs:
                    vals, min_eigenvalue = self._load(data_path, pot, cutoff, shots)
                    if vals.size == 0:
                        vals = np.array([np.nan])

                    energies_per_cutoff.append(np.asarray(vals, dtype=float))
                    min_eigenvalues.append(min_eigenvalue)

                min_val = np.min([np.min(es) for es in energies_per_cutoff])
                max_val = np.max([np.max(es) for es in energies_per_cutoff])
                val_diff = np.abs(max_val-min_val)

                all_vals = np.concatenate(energies_per_cutoff)

                if val_diff < scale_thresh:
                    scale = "linear"  
                elif np.all(all_vals > 0):
                    scale = "log"
                else:
                    scale = "symlog"

                #scale="linear"
                    
                # draw horizontal boxplots
                bp = ax.boxplot(
                    energies_per_cutoff,
                    labels=[str(c) for c in self.cutoffs],
                    vert=False,
                    widths=box_width,
                    patch_artist=True,
                    showfliers=showfliers,
                    medianprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.3),
                    capprops=dict(linewidth=1.3),
                    boxprops=dict(linewidth=1.3),
                    flierprops=dict(marker='o', markersize=4, linestyle='none', markeredgewidth=0.8),
                )

                for label in ax.get_yticklabels():
                    label.set_fontweight('normal')

                # color each box (and its whiskers/caps/median) by cutoff
                n = len(self.cutoffs)
                for k, cutoff in enumerate(self.cutoffs):
                    colour = cutoff_to_color[cutoff]
                    # box
                    box = bp["boxes"][k]
                    box.set_facecolor(colour)
                    box.set_edgecolor(colour)
                    box.set_alpha(alpha)
                    # whiskers: two per box (2k, 2k+1)
                    for w in (bp["whiskers"][2*k], bp["whiskers"][2*k + 1]):
                        w.set_color(colour)
                    # caps: two per box (2k, 2k+1)
                    for c in (bp["caps"][2*k], bp["caps"][2*k + 1]):
                        c.set_color(colour)
                    # median: one per box (k)
                    bp["medians"][k].set_color(colour)

                    if k < len(bp["fliers"]):
                        f = bp["fliers"][k]
                        f.set_markerfacecolor(colour)
                        f.set_markeredgecolor(colour)
                        f.set_alpha(alpha)

                    ax.axvline(x=min_eigenvalues[k], color=colour, linestyle="--", linewidth=1.0, alpha=0.7, label=f"$\\Lambda$={cutoff}")

                format_axis(ax, scale=scale)
  
                # labels/titles
                if j == 0:
                    ax.set_ylabel(pot)
                else:
                    ax.set_ylabel("")
                    ax.tick_params(axis="y", left=False, labelleft=False)

                if i == 0:
                    if by_dataset:
                        ax.set_title(title)
                    else:
                        if shots is None:
                            ax.set_title(f"Statevector")
                        else:
                            ax.set_title(f"{shots} shots")

                if i == (len(self.potentials)-1):
                    ax.set_xlabel("Energy")

                #ax.set_xscale("symlog")#, linthresh=1e-1)
                ax.grid(True, axis="x", linestyle="--", alpha=0.4)

        
        if show_legend:
            #axes_arr[0, 0].legend(loc="upper right", fontsize=8, ncol=len(self.cutoffs))
            axes_arr[0, 0].legend(loc="upper left", fontsize=12, ncol=1)

        fig.tight_layout(pad=0.6)
        return fig, axes_arr



@dataclass(slots=True)
class VQEPlotter:
    """
    Helper for plotting VQE summary metrics from one or more data sources.
    """
    data_paths: List[LabelPath]
    potentials: List[str]
    cutoffs: List[int]
    converged_only: bool = True
    on_missing: str = "skip"
    debug: bool = False

    # ---------- loading ----------
    def _load(self, path: str) -> VQESummary:
        return VQESummary.from_path(
            path,
            self.cutoffs,
            self.potentials,
            converged_only=self.converged_only,
            on_missing=self.on_missing,
            debug=self.debug,
        )

    def _ensure_axes_grid(
        self,
        ncols: Optional[int] = None,
        nrows: Optional[int] = 1,
        *,
        sharex: bool = True,
        sharey: bool = False,
        figsize=(12, 4),
        existing_axes=None,
    ):
        """Return (fig, axes_array) with shape (ncols, )."""
        n = ncols if ncols is not None else len(self.potentials)
        if existing_axes is not None:
            return existing_axes.figure, np.asarray(existing_axes)

        fig, axes = plt.subplots(nrows, n, figsize=figsize, sharex=sharex, sharey=sharey)
        if n == 1:
            axes = np.array([axes])
        return fig, axes

    def _style_x_cutoffs(self, ax):
        ax.set_xscale("log")
        ax.set_xticks(self.cutoffs)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(ticker.NullLocator())


    def plot_delta_e_vs_cutoff_line(self, *, figsize=(12, 4), axes=None, linewidth=1.0, marker="^", markersize=4.0, metric='median', scale="symlog", linthresh=1.0, sharey=True):

        markers = ["o", "s", "^", "D", "v", "P", "*", "X"]
        #marker_cycle = itertools.cycle(markers)
        fig, axes_arr = self._ensure_axes_grid(figsize=figsize, existing_axes=axes, sharey=sharey)

        for ax in axes_arr:
            if scale == "symlog":
                ax.set_yscale("symlog", linthresh=linthresh)
            else:
                ax.set_yscale(scale)
            #ax.yaxis.set_major_locator(LogLocator(base=10.0))        
            #ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))

        for i, pot in enumerate(self.potentials):
            ax = axes_arr[i]
            for j, (label, path) in enumerate(self.data_paths):
                summary = self._load(path)
                if metric == 'median':
                    y = summary.delta_median_e[pot].reindex(self.cutoffs)
                else:
                    y = summary.delta_min_e[pot].reindex(self.cutoffs)
                marker = markers[j]
                #marker = next(marker_cycle)
                ax.plot(self.cutoffs, y, linewidth=linewidth, marker=marker, markersize=markersize, label=label)
            ax.set_title(pot)
            ax.grid(True)
            self._style_x_cutoffs(ax)
            if i == 0:
                ax.set_ylabel(r"|$E_{\mathrm{exact}} - E_{\mathrm{median}}$|")
            elif sharey:
                ax.tick_params(axis="y", left=False, labelleft=False)

        def fmt_sigfig(x, _):
            return f"{x:.3g}"
        formatter = FuncFormatter(fmt_sigfig)

        #for ax in axes_arr:
        #    yticks = ax.get_yticks()
        #    ax.set_yticks([tick for tick in yticks if tick >= 0])
        #    ax.yaxis.set_major_formatter(formatter)
       

        axes_arr[len(self.potentials) // 2].set_xlabel(r"$\Lambda$")
        axes_arr[0].legend(loc="upper left", fontsize=12)
        fig.tight_layout(pad=0.6)
        return fig, axes_arr
    

    def plot_evals_vs_cutoff_box(
        self,
        *,
        axes=None,
        box_width: float = 0.25,
        group_spacing: float = 1.8,
        showfliers: bool = False,
        alpha: float = 0.6,
        show_legend: bool = True,
        show_title: bool = True
    ):
        """
        Grouped boxplots of num_evaluations per cutoff.
        - x-axis: cutoffs (Λ)
        - groups: data_paths (labels) offset around each cutoff
        - panels: one per potential
        """
        fig, axes_arr = self._ensure_axes_grid(existing_axes=axes, sharey=True, figsize=(12, 4))

        x = np.arange(len(self.cutoffs))*group_spacing
        labels = [lab for lab, _ in self.data_paths]
        n_labels = len(labels)

        # consistent colors per label, from Matplotlib default cycle
        palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        label_to_color = {lab: palette[i % len(palette)] for i, lab in enumerate(labels)}

        for ax, pot in zip(axes_arr, self.potentials):
            for i, (lab, path) in enumerate(self.data_paths):
                offset = (i - (n_labels - 1) / 2.0) * box_width
                positions = x + offset

                # gather distributions per cutoff in order
                summary = self._load(path)
                data_per_cutoff = []
                for c in self.cutoffs:
                    dist = summary.get("evals_dist", pot, c)
                    # ensure it's a list/array (may be empty)
                    data_per_cutoff.append(np.asarray(dist, dtype=float))

                bp = ax.boxplot(
                    data_per_cutoff,
                    positions=positions,
                    widths=box_width * 0.85,
                    patch_artist=True,
                    showfliers=showfliers,
                    medianprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.3),
                    capprops=dict(linewidth=1.3),
                    boxprops=dict(linewidth=1.3),
                )

                color = label_to_color[lab]
                for box in bp["boxes"]:
                    box.set_facecolor(color)
                    box.set_edgecolor(color)
                    box.set_alpha(alpha)
                for part in ("whiskers", "caps", "medians"):
                    for artist in bp[part]:
                        artist.set_color(color)

            if show_title: ax.set_title(pot)
            ax.set_xticks(x, self.cutoffs)
            ax.grid(True, axis="y")
            ax.set_yscale("log") 

        for i, ax in enumerate(axes_arr):
            if i > 0:
                ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

        axes_arr[0].set_ylabel(r"$N_{evals}$")
        axes_arr[len(self.potentials) // 2].set_xlabel(r"$\Lambda$")
        
        legend_patches = [
            Patch(facecolor=label_to_color[lab], edgecolor=label_to_color[lab], alpha=alpha, label=lab)
            for lab in labels
        ]
        if show_legend: axes_arr[0].legend(handles=legend_patches, loc="upper left", fontsize=12)

        fig.tight_layout(pad=0.6)
        return fig, axes_arr


        
