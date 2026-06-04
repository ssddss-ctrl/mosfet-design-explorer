"""
plotting.py
-----------
Shared matplotlib style and figure helpers for the MOSFET Design Explorer.

Matches the dark theme used in the simulation notebooks (sim1-sim4),
so exploration notebooks and portfolio demos look consistent.

Usage:
    from mosfet_explorer.plotting import apply_style, new_fig, label_axes, save_fig
"""

import matplotlib.pyplot as plt
import matplotlib as mpl


def apply_style():
    """
    Apply the project-wide matplotlib style.
    Dark background matching the simulation notebooks.
    Call once at the top of every notebook or script.
    """
    mpl.rcParams.update({
        # Figure
        "figure.facecolor":     "#1a1a1a",
        "figure.dpi":           120,
        "figure.autolayout":    True,

        # Axes
        "axes.facecolor":       "#0f0f0f",
        "axes.edgecolor":       "#444444",
        "axes.labelcolor":      "white",
        "axes.titlecolor":      "white",
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.grid":            True,
        "axes.labelsize":       12,
        "axes.titlesize":       13,
        "axes.prop_cycle":      mpl.cycler(color=[
            "#00bfff",   # cyan       — electrons / n-type
            "#ff4444",   # tomato     — holes / p-type
            "#32cd32",   # limegreen  — Fermi level
            "#ffd700",   # gold       — ni / intrinsic
            "#bb86fc",   # violet     — secondary
            "#ff8c00",   # orange     — tertiary
        ]),

        # Grid
        "grid.color":           "#444444",
        "grid.alpha":           0.2,
        "grid.linewidth":       0.8,

        # Lines
        "lines.linewidth":      2.2,

        # Ticks
        "xtick.color":          "white",
        "ytick.color":          "white",
        "xtick.labelsize":      10,
        "ytick.labelsize":      10,
        "xtick.direction":      "in",
        "ytick.direction":      "in",

        # Legend
        "legend.fontsize":      9,
        "legend.facecolor":     "#222222",
        "legend.edgecolor":     "#444444",
        "legend.labelcolor":    "white",
        "legend.framealpha":    0.8,

        # Text
        "text.color":           "white",
        "font.family":          "sans-serif",
        "font.size":            11,
    })


def new_fig(nrows=1, ncols=1, **kwargs):
    """
    Create a new figure and axes with project style applied.

    Parameters
    ----------
    nrows, ncols : int
        Subplot grid dimensions.
    **kwargs
        Passed to plt.subplots().

    Returns
    -------
    fig, ax
    """
    apply_style()
    return plt.subplots(nrows, ncols, **kwargs)


def label_axes(ax, title="", xlabel="", ylabel="", legend=False):
    """
    Set title, axis labels, and optionally show legend.
    """
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if legend:
        ax.legend()


def annotate_vline(ax, x, label, color="gray"):
    """
    Draw a vertical dashed reference line with a text label.
    Useful for marking VT, VGS, etc. on I-V plots.
    """
    ax.axvline(x, color=color, linestyle="--", linewidth=1.2, alpha=0.7)
    ymax = ax.get_ylim()[1]
    ax.text(x, ymax * 0.92, f" {label}", color=color, fontsize=9, va="top")


def save_fig(fig, path, dpi=150):
    """
    Save figure to file.

    Parameters
    ----------
    fig : matplotlib Figure
    path : str
        Output path, e.g. 'assets/week1_ni_vs_T.png'
    dpi : int
    """
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {path}")