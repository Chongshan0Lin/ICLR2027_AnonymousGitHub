#!/usr/bin/env python3
"""Shared figure style for every main-paper figure (Figures 1-6).

Anchored on the palette and typography measured from the original Figure 1
schematic of the submitted PDF:

  text block width   396 pt = 5.50 in   (page text extent, excluding the
                                         line-number column)
  palette            sampled from a 400 dpi render of the schematic page:
                       #2b6897  blue      (5.9k px)
                       #ac6935  orange    (5.0k px)
                       #c3c3c3  neutral bars
                       #fafafa  panel background

The single most important consequence: a figure must be authored at the text
width so LaTeX places it at 1:1. A figure authored 14.68 in wide and scaled by
0.375 turns 7.5 pt legend text into 2.8 pt on the page. Authoring at 5.5 in
makes the number in the code the number on the page, and makes the target body
heights in inches mean the same thing here and in the compiled paper.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

TEXTW = 5.5                      # in, ICLR 2027 text block

# ---- palette (unchanged from the current figures) ------------------------
BLUE = "#2b6897"                 # primary / observed
ORANGE = "#ac6935"               # contrast / alternative model
GREY_FILL = "#c3c3c3"            # null, floor, control -- fills
GREY_LINE = "#8a8a8a"            # null, floor, control -- lines
INK = "#222222"                  # ground-truth observed, and text
BG = "#fafafa"                   # inactive panel background
SUBTLE = "#555555"               # secondary text (subtitles, annotations)

SEMANTIC = {"observed": BLUE, "ground_truth": INK, "contrast": ORANGE,
            "null": GREY_LINE, "null_fill": GREY_FILL}

# ---- the single set of type sizes used by every figure -------------------
FS_TITLE = 8.0                   # (a) panel label + model, bold
FS_SUB = 6.4                     # second title line, lighter
FS_LABEL = 8.0                   # axis labels
FS_TICK = 7.0                    # tick labels
FS_LEG = 7.2                     # legend entries
FS_ANNOT = 6.8                   # in-panel annotations

# ---- line / marker / band vocabulary -------------------------------------
LW_MAIN = 1.5                    # primary observed curve
LW_REF = 0.8                     # null / floor / zero reference (thinner)
MS_MAIN = 3.0                    # primary marker
BAND_ALPHA = 0.22                # bootstrap CI band, every figure
NULL_ALPHA = 0.40                # shuffled-time / scalar null band


def apply():
    plt.rcParams.update({
        # Nimbus Sans is the family the original figures embed
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial",
                            "Liberation Sans", "DejaVu Sans"],
        # Type 3 is the matplotlib default and is rejected by many venues
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.compression": 6,
        "svg.fonttype": "none",

        # mathtext must use the same family as the text, or $Z$ and $M(lag)$
        # silently render in DejaVu while every label renders in Nimbus Sans
        "mathtext.fontset": "custom",
        "mathtext.rm": "Nimbus Sans",
        "mathtext.it": "Nimbus Sans:italic",
        "mathtext.bf": "Nimbus Sans:bold",
        "mathtext.default": "it",

        "font.size": FS_LABEL,
        "axes.titlesize": FS_TITLE,
        "axes.titleweight": "bold",
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEG,

        "axes.linewidth": 0.6,
        "axes.edgecolor": "#444444",
        "axes.grid": True,
        "grid.color": "#e2e2e2",          # fainter than before
        "grid.linewidth": 0.4,
        "grid.alpha": 0.9,
        "axes.axisbelow": True,

        "lines.linewidth": LW_MAIN,
        "lines.markersize": MS_MAIN,
        "lines.markeredgewidth": 0.0,

        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.4,
        "legend.borderpad": 0.0,

        "figure.dpi": 200,
        "savefig.dpi": 200,
        # NOT "tight": a tight bbox trims a different amount from each figure,
        # so each lands at a different scale inside \textwidth and the effective
        # font size drifts between figures. Explicit margins + exact figsize
        # means every figure is placed 1:1 and 8 pt in the code is 8 pt on the
        # page -- which is also what makes the inch targets meaningful.
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
    })


# --------------------------------------------------------------- helpers
def clean(ax, hide=("top", "right")):
    for s in hide:
        ax.spines[s].set_visible(False)


def _text_width_in(fig, s, size, weight="normal"):
    """Rendered width of a string, in inches, for the current backend."""
    t = fig.text(0, 0, s, fontsize=size, fontweight=weight)
    fig.canvas.draw()
    w = t.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


def wrap_to(fig, text, size, avail, max_lines=2):
    """Greedy word wrap so a descriptor line fits the panel it belongs to."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if cur and _text_width_in(fig, t, size) > avail and len(lines) < max_lines - 1:
            lines.append(cur)
            cur = w
        else:
            cur = t
    lines.append(cur)
    return "\n".join(lines)


def panel_title(fig, ax, letter, main, sub=None, gutter_frac=0.0):
    """`(a) Model` bold, with an optional lighter descriptor beneath.

    The same convention in every quantitative figure. The bold line is measured
    and stepped down only if it genuinely cannot fit; the descriptor wraps
    instead of shrinking, so the bold size stays identical across panels.
    `gutter_frac` is extra inches a left-aligned title may overhang into the
    inter-panel gap without touching its neighbour.
    """
    head = f"({letter}) {main}"
    avail = ax.get_position().width * fig.get_figwidth()
    size = FS_TITLE
    for cand in (FS_TITLE, 7.6, 7.2, 6.9, 6.6):
        size = cand
        if _text_width_in(fig, head, cand, "bold") <= avail + gutter_frac:
            break
    nlines = 1
    ssize = None
    if sub:
        ssize = FS_SUB
        wrapped = wrap_to(fig, sub, ssize, avail)
        nlines = wrapped.count("\n") + 1
        ax.text(0.0, 1.02, wrapped, transform=ax.transAxes, fontsize=ssize,
                color=SUBTLE, va="bottom", ha="left", linespacing=1.25)
    ax.set_title(head, loc="left", fontsize=size, fontweight="bold",
                 pad=3.0 + 8.0 * nlines if sub else 3.0)
    return size, ssize


def strip(n, height, left=0.070, right=0.995, top=None, bottom=0.255,
          wspace=0.16, title_in=0.42):
    """A 1 x n horizontal small-multiple strip at exactly the text width.

    `top` is derived from the height so the panel-title block always gets the
    same ABSOLUTE space (`title_in`) whatever the body height is; that is what
    keeps Figures 2 and 4 pixel-aligned with each other.
    """
    fig = plt.figure(figsize=(TEXTW, height))
    if top is None:
        top = 1.0 - title_in / height
    gs = GridSpec(1, n, figure=fig, wspace=wspace,
                  left=left, right=right, top=top, bottom=bottom)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
    return fig, axes


def shared_labels(fig, axes, xlabel=None, ylabel=None, xy=(0.5, 0.012),
                  yx=0.008):
    """One figure-level x/y label instead of repeating it under every panel."""
    if xlabel:
        fig.text(xy[0], xy[1], xlabel, ha="center", va="bottom",
                 fontsize=FS_LABEL)
    if ylabel:
        pos = [a.get_position() for a in axes]
        ymid = (min(p.y0 for p in pos) + max(p.y1 for p in pos)) / 2
        fig.text(yx, ymid, ylabel, ha="left", va="center", rotation=90,
                 fontsize=FS_LABEL)


def strip_legend(fig, handles, labels, y=0.005, ncol=None):
    """One compact horizontal legend under all panels, lowest vertical cost."""
    return fig.legend(handles=handles, labels=labels, loc="lower center",
                      bbox_to_anchor=(0.5, y), ncol=ncol or len(labels),
                      frameon=False, fontsize=FS_LEG, handlelength=1.8,
                      handletextpad=0.5, columnspacing=1.5, borderpad=0.0)


def save(fig, stem, figdir):
    for ext in ("pdf", "png"):
        fig.savefig(figdir / f"{stem}.{ext}")
    plt.close(fig)
