# susy_qm/plots/vqe_plots.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

from susy_qm.vqe_metrics import VQESummary

logger = logging.getLogger(__name__)

LabelPath = Tuple[str, str]

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

        fig, axes = plt.subplots(1, n, figsize=figsize, sharex=sharex, sharey=sharey)
        if n == 1:
            axes = np.array([axes])
        return fig, axes

    def _style_x_cutoffs(self, ax):
        ax.set_xscale("log")
        ax.set_xticks(self.cutoffs)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(ticker.NullLocator())

    # ---------- plots ----------
    def plot_delta_e_vs_cutoff_line(self, *, axes=None, marker="^"):
        """|E_exact - E_median| vs cutoff, one panel per potential."""
        fig, axes_arr = self._ensure_axes_grid(existing_axes=axes, sharey=True)
        for i, pot in enumerate(self.potentials):
            ax = axes_arr[i]
            for label, path in self.data_paths:
                summary = self._load(path)
                y = summary.delta_e[pot].reindex(self.cutoffs)
                ax.plot(self.cutoffs, y, marker=marker, label=label)
            ax.set_title(pot)
            ax.set_yscale("symlog", linthresh=1.0)
            ax.grid(True)
            self._style_x_cutoffs(ax)
            if i == 0:
                ax.set_ylabel(r"|$E_{\mathrm{exact}} - E_{\mathrm{median}}$|")
            else:
                ax.tick_params(axis="y", left=False, labelleft=False)

        axes_arr[len(self.potentials) // 2].set_xlabel(r"$\Lambda$")
        axes_arr[0].legend(loc="upper left", fontsize=8)
        fig.tight_layout(pad=0.6)
        return fig, axes_arr
    

    def plot_evals_vs_cutoff_box(
        self,
        *,
        axes=None,
        box_width: float = 0.25,
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

        x = np.arange(len(self.cutoffs))
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

        axes_arr[0].set_ylabel(r"$N_{Evals}$")
        axes_arr[len(self.potentials) // 2].set_xlabel(r"$\Lambda$")
        
        legend_patches = [
            Patch(facecolor=label_to_color[lab], edgecolor=label_to_color[lab], alpha=alpha, label=lab)
            for lab in labels
        ]
        if show_legend: axes_arr[0].legend(handles=legend_patches, loc="upper left", fontsize=8)

        fig.tight_layout(pad=0.6)
        return fig, axes_arr


        
