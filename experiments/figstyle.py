"""
Shared figure style for the publication plots (IEEE IoT Journal submission).

Rules kept in sync with the paper's figure specification:
- all text in the 9 to 9.5 pt band (9.5 pt titles, 9 pt everything else);
- figures are drawn at their true print width so text is never scaled down
  in LaTeX (IEEE single column 3.5 in, double column 7.16 in);
- flat, colour-blind-safe palette, white background, 300 dpi export;
- a series keeps the same colour in every figure (proposed = blue,
  comparator/naive = orange, CO2/secondary = green, neutral = grey).
"""
import matplotlib as mpl

# True print widths in inches.
COL_W = 3.5     # IEEE single-column figure width
FULL_W = 7.16   # IEEE double-column figure width

# Palette (fixed per series across all figures).
BLUE = "#2563EB"    # proposed method / renormalised series
ORANGE = "#E8710A"  # baseline / naive comparator
GREEN = "#059669"   # CO2 / secondary measured series
GREY = "#6B7280"    # neutral reference (persistence, diagonals)
TEXT = "#111827"
GRID = "#E5E7EB"


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": TEXT,
        "axes.labelcolor": TEXT,
        "text.color": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "legend.frameon": False,
        "figure.constrained_layout.use": True,
        "savefig.bbox": "tight",
    })
