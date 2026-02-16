"""
Shared visualization utilities and style configuration.
Ensures consistent styling across all figure generation scripts.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

# --- STYLE CONFIGURATION ---
BASE_FONT_SIZE = 20  # reduced from 22
TITLE_FONT_SIZE = BASE_FONT_SIZE + 4
LEGEND_FONT_SIZE = BASE_FONT_SIZE - 2
DARK_EDGE_COLOR = "#1f2937"
DARK_TEXT_COLOR = "#0f172a"
MAJOR_GRID_STYLE = {"color": "#c7ccd6", "linewidth": 1.2, "alpha": 0.7}

# --- COLORS & MARKERS ---
# User requirement:
# - Static dots: Red (0), Yellow (1), Blue (2)
# - Rotated dots: Light Red (0), Light Yellow (1), Light Blue (2)
# - Markers: Same for static/rotated.
# - Fonts: 2x larger. Markers: 1x larger (meaning 2x size).

# Using Dark Gold (#D4AC0D) for Yellow to satisfy "darker and sharper".
# Using DodgerBlue (#1E90FF) for Blue to satisfy "lighter and sharper".
CLASS_COLORS = ["#FF0000", "#D4AC0D", "#1E90FF"]     # Red, Dark Gold, DodgerBlue
LIGHT_COLORS = ["#FFCCCC", "#FFFFE0", "#CCCCFF"]     # Light versions
CLASS_MARKERS = ["o", "s", "^"]                      # Circle, Square, Triangle

# --- MARKER SIZES ---
# Reduced by user request (-2 points scale approx)
MARKER_SIZE_LARGE = 200 # Was 240
MARKER_SIZE_MEDIUM = 100 # Was 120
MARKER_SIZE_SMALL = 60  # Was 80
LINE_MARKER_SIZE = 10   # Was 12

def configure_style() -> None:
    """Apply global matplotlib style settings."""
    mpl.rcParams.update(
        {
            "font.family": "Palatino Linotype",
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
        legend.set_title(legend.get_title().get_text(), prop={"weight": "bold"})

def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    """Save figure in PNG (300 dpi), PDF, and SVG formats in separate subfolders."""
    # Define formats and their specific kwargs
    formats = {
        "png": {"dpi": 300},
        "pdf": {},
        "svg": {}
    }

    base_figures_dir = out_dir / "figures"

    for fmt, kwargs in formats.items():
        # Create specific subfolder for each format
        fmt_dir = base_figures_dir / fmt
        fmt_dir.mkdir(parents=True, exist_ok=True)

        # Save figure
        fig.savefig(fmt_dir / f"{stem}.{fmt}", format=fmt, bbox_inches="tight", **kwargs)

    plt.close(fig)
