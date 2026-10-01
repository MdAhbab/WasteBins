"""
Shared figure style for the publication plots.

Rules kept in sync with the paper's figure specification:
- titles and axis labels at 9 to 9.5 pt, ticks, legends and annotations at
  8.5 pt, nothing below 8 pt;
- figures are drawn at their true print width so text is never scaled down
  in LaTeX;
- flat, colour-blind-safe palette, white background, 300 dpi export, fonts
  embedded as TrueType;
- a series keeps the same colour in every figure (proposed rule = blue,
  comparator = orange, secondary measured series = green, neutral = grey), and
  a marker or line style as well, so no reading depends on hue alone.

The layout is the single-column Elsevier page of the cas-sc template, whose text
block is 164.6 mm wide.  A figure is drawn either at that full width or at half
of it, for two panels set side by side.  Drawing at the placement width matters
because LaTeX scales the text inside a figure with the figure.
"""
import matplotlib as mpl

# True print widths in inches for the cas-sc single-column page.
FULL_W = 6.48   # the text width
HALF_W = 3.2    # one of two figures side by side
COL_W = HALF_W  # kept for scripts written against the earlier layout

# Palette (fixed per series across all figures).
BLUE = "#2563EB"    # proposed rule
ORANGE = "#E8710A"  # comparator
GREEN = "#059669"   # secondary measured series
GREY = "#6B7280"    # neutral reference
TEXT = "#111827"
GRID = "#E5E7EB"


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9.0,
        "axes.titlesize": 9.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.titlesize": 10.0,
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
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.5,
        "lines.markersize": 5.0,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.constrained_layout.use": True,
        "savefig.bbox": "tight",
    })
