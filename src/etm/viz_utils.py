"""
Shared visualization utilities and style configuration.
Ensures consistent styling across all figure generation scripts.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

# --- STYLE CONFIGURATION ---
BASE_FONT_SIZE = mpl.rcParams.get("font.size", 10) + 2
TITLE_FONT_SIZE = BASE_FONT_SIZE + 2
LEGEND_FONT_SIZE = max(BASE_FONT_SIZE - 2, 10)
DARK_EDGE_COLOR = "#1f2937"
DARK_TEXT_COLOR = "#0f172a"
MAJOR_GRID_STYLE = {"color": "#c7ccd6", "linewidth": 0.9, "alpha": 0.7}

# --- COLORS & MARKERS ---
# User requirement:
# - Static dots: Red (0), Yellow (1), Blue (2)
# - Rotated dots: Light Red (0), Light Yellow (1), Light Blue (2)
# - Markers: Same for static/rotated.

# Using Gold (#FFD700) for Yellow to ensure visibility on white background,
# as pure Yellow (#FFFF00) is often too bright.
CLASS_COLORS = ["#FF0000", "#FFD700", "#0000FF"]     # Red, Gold, Blue
LIGHT_COLORS = ["#FFCCCC", "#FFFFE0", "#CCCCFF"]     # Light Red, Light Yellow, Light Blue
CLASS_MARKERS = ["o", "s", "^"]                      # Circle, Square, Triangle

def configure_style() -> None:
    """Apply global matplotlib style settings."""
    mpl.rcParams.update(
        {
            "font.size": BASE_FONT_SIZE,
            "font.weight": "bold",
            "axes.labelsize": BASE_FONT_SIZE,
            "axes.labelweight": "bold",
            "axes.titlesize": TITLE_FONT_SIZE,
            "axes.titleweight": "bold",
            "xtick.labelsize": BASE_FONT_SIZE,
            "ytick.labelsize": BASE_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "legend.framealpha": 0.92,
            "axes.edgecolor": DARK_EDGE_COLOR,
            "axes.labelcolor": DARK_TEXT_COLOR,
            "axes.titlecolor": DARK_TEXT_COLOR,
            "axes.linewidth": 1.1,
            "grid.color": MAJOR_GRID_STYLE["color"],
            "grid.linewidth": MAJOR_GRID_STYLE["linewidth"],
            "grid.alpha": MAJOR_GRID_STYLE["alpha"],
            "savefig.dpi": 300,
        }
    )

def enforce_bold_text(ax: mpl.axes.Axes) -> None:
    """Ensure all text elements in the axes are bold."""
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for text in ax.texts:
        text.set_fontweight("bold")
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontweight("bold")

def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    """Save figure in PDF, SVG, and PNG formats with 300 DPI."""
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    for ext in ("pdf", "svg", "png"):
        fig.savefig(figures_dir / f"{stem}.{ext}", format=ext, bbox_inches="tight", dpi=300)
    plt.close(fig)
