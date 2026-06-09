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


def apply_style() -> None:
    """
    Apply the project-wide matplotlib dark theme.

    Patches matplotlib.rcParams in-place. Call once at the top of every
    notebook or script before creating any figures.

    Notes
    -----
    Uses ``grid.alpha`` (not ``axes.grid.alpha``) for compatibility with
    matplotlib >= 3.6. Avoid LaTeX in labels — use unicode directly
    (µ, Ω, cm⁻³, etc.) to prevent rendering overhead with this dark theme.
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


def new_fig(nrows: int = 1, ncols: int = 1, **kwargs) -> tuple:
    """
    Create a new figure and axes with the project style applied.

    Parameters
    ----------
    nrows : int
        Number of subplot rows.
    ncols : int
        Number of subplot columns.
    **kwargs
        Passed directly to plt.subplots().

    Returns
    -------
    fig, ax : Figure and Axes (or array of Axes if nrows/ncols > 1)
    """
    apply_style()
    return plt.subplots(nrows, ncols, **kwargs)


def label_axes(
    ax,
    title:  str  = "",
    xlabel: str  = "",
    ylabel: str  = "",
    legend: bool = False,
) -> None:
    """
    Set title, axis labels, and optionally show legend.

    Parameters
    ----------
    ax : matplotlib Axes
    title : str
        Axes title.
    xlabel : str
        x-axis label.
    ylabel : str
        y-axis label.
    legend : bool
        If True, call ax.legend().
    """
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if legend:
        ax.legend()


def annotate_vline(
    ax,
    x:     float,
    label: str,
    color: str = "gray",
) -> None:
    """
    Draw a vertical dashed reference line with a text label.

    Useful for marking Vt, Vgs, pinch-off, etc. on I-V plots.

    Parameters
    ----------
    ax : matplotlib Axes
    x : float
        x-position of the vertical line.
    label : str
        Text to display at the top of the line.
    color : str
        Line and label colour. Default 'gray'.
    """
    ax.axvline(x, color=color, linestyle="--", linewidth=1.2, alpha=0.7)
    ymax = ax.get_ylim()[1]
    ax.text(x, ymax * 0.92, f" {label}", color=color, fontsize=9, va="top")


def save_fig(fig, path: str, dpi: int = 150) -> None:
    """
    Save a figure to disk.

    Parameters
    ----------
    fig : matplotlib Figure
    path : str
        Output path, e.g. 'assets/week2_ni_vs_T.png'.
    dpi : int
        Resolution. Default 150.
    """
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {path}")