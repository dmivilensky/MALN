"""Shared figure style (print, light surface)."""
import logging
import matplotlib as mpl

logging.getLogger("fontTools").setLevel(logging.ERROR)

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def apply():
    mpl.rcParams.update({
        "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9,
        "axes.labelsize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.5, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 1.6, "lines.markersize": 4.5,
        "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 300,
        "mathtext.fontset": "cm", "pdf.fonttype": 42,
    })
