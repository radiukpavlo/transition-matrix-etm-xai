"""
Shared visualization utilities and style configuration for MNIST figures.
Ensures consistent styling across all figure generation scripts.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

# --- STYLE CONFIGURATION ---
BASE_FONT_SIZE = 20
TITLE_FONT_SIZE = BASE_FONT_SIZE + 4
LEGEND_FONT_SIZE = BASE_FONT_SIZE - 2
DARK_EDGE_COLOR = "#1f2937"
DARK_TEXT_COLOR = "#0f172a"
MAJOR_GRID_STYLE = {"color": "#c7ccd6", "linewidth": 1.2, "alpha": 0.7}

# --- COLORS ---
# Standard tab10 for digits 0-9
COLOR_CYCLE = mpl.colormaps["tab10"].colors
CLASS_MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"] # Extended for more classes if needed, or just basic ones.


def configure_style() -> None:
    """Apply global matplotlib style settings."""
    mpl.rcParams.update(
        {
            "font.size": BASE_FONT_SIZE,
            "font.weight": "normal",
            "axes.labelsize": BASE_FONT_SIZE,
            "axes.labelweight": "normal",
            "axes.titlesize": TITLE_FONT_SIZE,
            "axes.titleweight": "normal",
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
            "axes.prop_cycle": mpl.cycler(color=COLOR_CYCLE),
        }
    )

def enforce_bold_text(ax: mpl.axes.Axes) -> None:
    """Ensure all text elements in the axes are normal weight (not bold), as per user request."""
    # Explicitly set to normal just in case style defaults are otherwise
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("normal")
    for text in ax.texts:
        text.set_fontweight("normal")
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontweight("normal")

def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    """Save figure in PNG format with 300 DPI."""
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # User requested PNG specifically with 300 dpi.
    # We can also save PDF/SVG for completeness if desired, 
    # but strictly user emphasized "saving all figures in PNG".
    for ext in ["png"]:
        fig.savefig(figures_dir / f"{stem}.{ext}", format=ext, bbox_inches="tight", dpi=300)
    plt.close(fig)
