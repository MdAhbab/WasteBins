"""
Shared figure style for the publication plots.

Rules kept in sync with the paper's figure specification:
- all text in the 9 to 9.5 pt band (9.5 pt titles, 9 pt everything else);
- figures are drawn at their true print width so text is never scaled down
  in LaTeX;
- flat, colour-blind-safe palette, white background, 300 dpi export;
- a series keeps the same colour in every figure (proposed = blue,
  comparator/naive = orange, CO2/secondary = green, neutral = grey).

The layout is single column.  The previous submission was to a two-column IEEE
journal, where a figure was drawn either at 3.5 in to sit inside one column or at
7.16 in to span both.  Neither width is right here.  A single-column page has one
text width, so a full-width figure is 6.0 in and a half-width figure that shares a
row with another is 2.95 in.  Redrawing at the correct width matters because text
inside a figure is not rescaled by LaTeX only when the figure is placed at the
width it was drawn for; a 7.16 in figure squeezed into a 6.0 in column drops its
9 pt labels to about 7.5 pt, below the floor the specification sets.
"""
import matplotlib as mpl

# True print widths in inches, for a single-column layout.
COL_W = 2.95    # half width, for two figures side by side
FULL_W = 6.0    # full text width

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
