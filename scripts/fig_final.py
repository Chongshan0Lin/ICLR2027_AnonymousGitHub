#!/usr/bin/env python3
"""Main-paper figures 2-6 (+ the appendix displacement figure).

Every number is read from the already-exported CSVs. No quantity is recomputed,
no statistic is re-derived and no scientific definition changes; this script
only re-lays-out and re-styles.

Paper figure numbering, read from the includegraphics order in main.tex:
    Fig 1  branch_feedback_schematic          (fig1_schematic.py)
    Fig 2  meta_reweighting_5way              1 x 5 strip
    Fig 3  meta_controlled_stochastic_3way    1 x 3 strip
    Fig 4  meta_temporal_memory_5way          1 x 5 strip, matched to Fig 2
    Fig 5  residual_structure_forecast        asymmetric 1 x 2
    Fig 6  prefork_instability                asymmetric 1 x 2
    App.   meta_behavioral_displacement_5way  1 x 5 strip, same language
"""
import csv, hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style as PS

# --- artifact-repo path resolution (added for the anonymous release) -------
# ARTIFACT_ROOT defaults to the repository this file lives in; override with
# the ARTIFACT_ROOT environment variable to point at a different export.
import os as _os
from pathlib import Path as _P
_REPO = _P(_os.environ.get("ARTIFACT_ROOT",
                           _P(__file__).resolve().parents[1]))
ART = _REPO / "artifacts"
OUTPUTS = _P(_os.environ.get("ARTIFACT_OUTPUTS", _REPO / "outputs"))
OUTPUTS.mkdir(parents=True, exist_ok=True)
FROZEN = ART / "round11_3b_trunk" / "analysis_todo_closure_v1"


def _mkout(d):
    """Create an output directory and the sub-directories the scripts write."""
    for _s in ("", "tables", "figure_data", "figures", "audit"):
        (d / _s).mkdir(parents=True, exist_ok=True)
    return d


WOUT = _mkout(OUTPUTS / "paper_main_figures")
# ---------------------------------------------------------------------------


D3 = ART
FIG, RES, REP = D3 / "figures", D3 / "results", D3 / "reports"
DAT = FIG / "meta_data"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rd = lambda p: list(csv.DictReader(Path(p).open()))
f_ = lambda x: float(x)
PS.apply()
FITS = {}

# Five panels across 5.5 in leaves 0.894 in each, which cannot hold a 20-char
# bold title at 8 pt. "public" and "Code" therefore live on the descriptor line
# (which wraps) rather than the bold line, so the bold size stays 8 pt in every
# panel of every figure. No information is dropped.
PANELS = [("a", "A", "Qwen2.5-3B", "Countdown, full-state fork"),
          ("b", "C", "Qwen2.5-7B", "Countdown, independent seeds"),
          ("c", "D", "Qwen2.5-3B", "Countdown, public dense continuation"),
          ("d", "E", "Qwen-1.5B", "HumanEval Code"),
          ("e", "F", "OLMo-3.1-7B", "HumanEval Code")]

# body heights in inches; the figure is authored at exactly the text width so
# these are also the heights the compiled paper sees
H_STRIP5, H_STRIP3, H_RESID, H_INSTAB = 1.78, 1.82, 2.20, 1.86


def audit_overlap(fig, stem):
    """Fail loudly if any two pieces of figure-level text collide."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = [(t.get_text()[:26], t.get_window_extent(renderer=r))
             for t in fig.texts if t.get_text().strip()]
    for lg in fig.legends:
        items.append(("<legend>", lg.get_window_extent(renderer=r)))
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_text().strip():
                items.append((t.get_text()[:26], t.get_window_extent(renderer=r)))
        if ax.get_legend() is not None:
            items.append(("<ax legend>",
                          ax.get_legend().get_window_extent(renderer=r)))
        for tl in (ax.get_xticklabels() + ax.get_yticklabels()
                   if ax.axison else []):
            if tl.get_text().strip() and tl.get_visible():
                items.append((f"tick {tl.get_text()[:8]}",
                              tl.get_window_extent(renderer=r)))
    bad = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (na, a), (nb, b) = items[i], items[j]
            if a.overlaps(b) and a.intersection(a, b).width > 1.0 \
                    and a.intersection(a, b).height > 1.0:
                bad.append(f"{na!r} x {nb!r}")
    if bad:
        print(f"  !! {stem}: {len(bad)} text overlap(s): {bad[:4]}")
    return len(bad)


def finish(fig, stem, note):
    nbad = audit_overlap(fig, stem)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{stem}.{ext}")
    plt.close(fig)
    w, h = fig.get_size_inches()
    print(f"  {stem:38s} {w:.2f} x {h:.2f} in   {note}"
          + ("" if not nbad else f"   [{nbad} OVERLAP]"))
    return sha(FIG / f"{stem}.pdf")


def titles(fig, axes, panels):
    """Apply the shared two-line panel-title convention and record the sizes."""
    for ax, (lt, _, main, sub) in zip(axes, panels):
        s, ss = PS.panel_title(fig, ax, lt, main, sub)
        FITS.setdefault(fig.get_label() or "fig", []).append((lt, s, ss))


# ======================================================== Figure 2  (1 x 5)
def fig2():
    rows = rd(DAT / "meta1_reweighting.csv") + rd(DAT / "meta1_code_panels.csv")
    by = {t: sorted([r for r in rows if r["track"] == t],
                    key=lambda r: f_(r["x_normalised"])) for _, t, _, _ in PANELS}
    lo = min(f_(r["z_lo"]) for v in by.values() for r in v)
    hi = max(f_(r["z_hi"]) for v in by.values() for r in v)
    ylim = (min(-0.6, lo - 0.4), hi * 1.10)          # unchanged from before
    fig, axes = PS.strip(5, H_STRIP5, left=0.105, bottom=0.287, title_in=0.42)
    fig.set_label("fig2")
    for ax, (lt, tr, main, sub) in zip(axes, PANELS):
        s = by[tr]
        est = [r for r in s if r.get("is_reference_anchor", "False") != "True"]
        ax.axhline(0, color=PS.GREY_LINE, lw=PS.LW_REF, zorder=1)
        ax.plot([f_(r["x_normalised"]) for r in s],
                [f_(r["z_reweight_mean"]) for r in s], "-", color=PS.BLUE,
                lw=PS.LW_MAIN, zorder=3)
        ax.plot([f_(r["x_normalised"]) for r in est],
                [f_(r["z_reweight_mean"]) for r in est], "o", color=PS.BLUE,
                ms=PS.MS_MAIN, zorder=4)
        ax.fill_between([f_(r["x_normalised"]) for r in est],
                        [f_(r["z_lo"]) for r in est], [f_(r["z_hi"]) for r in est],
                        color=PS.BLUE, alpha=PS.BAND_ALPHA, lw=0, zorder=2)
        ax.plot([0], [0], "o", mfc="white", mec=PS.BLUE, mew=1.0, ms=4.0, zorder=5)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(*ylim)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", ".5", "1"])
        PS.clean(ax)
    titles(fig, axes, PANELS)
    for a in axes[1:]:
        a.tick_params(labelleft=False)
    PS.shared_labels(fig, axes, "normalised training progress",
                     "standardised excess $Z$", xy=(0.55, 0.115), yx=0.016)
    PS.strip_legend(fig, [
        Line2D([], [], color=PS.BLUE, lw=PS.LW_MAIN, marker="o", ms=PS.MS_MAIN),
        Patch(facecolor=PS.BLUE, alpha=PS.BAND_ALPHA),
        Line2D([], [], color=PS.BLUE, lw=0, marker="o", mfc="white",
               mec=PS.BLUE, mew=1.0, ms=4.0),
        Line2D([], [], color=PS.GREY_LINE, lw=PS.LW_REF)],
        ["observed", "95% question bootstrap", "reference checkpoint ($Z=0$)",
         "scalar-sharpening null"], y=0.012)
    return finish(fig, "meta_reweighting_5way", "Fig 2  1x5 strip")


# ======================================================== Figure 3  (1 x 3)
def fig3():
    rows = rd(DAT / "meta2_stochastic_futures.csv")
    tracks = [p for p in PANELS if any(r["track"] == p[1] for r in rows)]
    by = {t: sorted([r for r in rows if r["track"] == t],
                    key=lambda r: f_(r["x_normalised"])) for _, t, _, _ in tracks}
    lo = min(f_(r["excess_js_lo"]) for v in by.values() for r in v)
    hi = max(f_(r["excess_js_hi"]) for v in by.values() for r in v)
    fig, axes = PS.strip(3, H_STRIP3, left=0.118, bottom=0.280, wspace=0.13, title_in=0.34)
    fig.set_label("fig3")
    for ax, (lt, tr, main, sub) in zip(axes, tracks):
        s = by[tr]
        x = [f_(r["x_normalised"]) for r in s]
        ax.axhline(0, color=PS.GREY_LINE, lw=PS.LW_REF, zorder=1)
        ax.plot(x, [f_(r["excess_js_mean"]) for r in s], "-o", color=PS.BLUE,
                lw=PS.LW_MAIN, ms=PS.MS_MAIN, zorder=3)
        ax.fill_between(x, [f_(r["excess_js_lo"]) for r in s],
                        [f_(r["excess_js_hi"]) for r in s], color=PS.BLUE,
                        alpha=PS.BAND_ALPHA, lw=0, zorder=2)
        # one consistent annotation slot in every panel, top-right
        ax.text(0.975, 0.955,
                f"modal agr. = {f_(s[-1]['modal_agreement_mean']):.2f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=PS.FS_ANNOT, color=PS.SUBTLE)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(min(0, lo) - 0.002, hi * 1.30)
        ax.set_xticks([0, 0.5, 1.0])
        PS.clean(ax)
    titles(fig, axes, tracks)
    for a in axes[1:]:
        a.tick_params(labelleft=False)
    PS.shared_labels(fig, axes, "normalised horizon", "excess pairwise JS",
                     xy=(0.56, 0.112), yx=0.016)
    PS.strip_legend(fig, [
        Line2D([], [], color=PS.BLUE, lw=PS.LW_MAIN, marker="o", ms=PS.MS_MAIN),
        Patch(facecolor=PS.BLUE, alpha=PS.BAND_ALPHA),
        Line2D([], [], color=PS.GREY_LINE, lw=PS.LW_REF)],
        ["observed", "95% question bootstrap",
         "rollout-matched measurement floor"], y=0.011)
    return finish(fig, "meta_controlled_stochastic_3way", "Fig 3  1x3 strip")


# ======================================================== Figure 4  (1 x 5)
def fig4():
    rows = rd(DAT / "meta3_temporal_memory.csv") + rd(DAT / "meta3_code_panels.csv")
    by = {t: sorted([r for r in rows if r["track"] == t],
                    key=lambda r: int(r["lag_intervals"])) for _, t, _, _ in PANELS}
    lagmax = max(int(r["lag_intervals"]) for v in by.values() for r in v)
    # identical geometry to Figure 2 -- they must read as a matched pair
    fig, axes = PS.strip(5, H_STRIP5, left=0.105, bottom=0.287, title_in=0.42)
    fig.set_label("fig4")
    for ax, (lt, tr, main, sub) in zip(axes, PANELS):
        s = by[tr]
        x = [int(r["lag_intervals"]) for r in s]
        ax.axhline(0, color=PS.GREY_LINE, lw=PS.LW_REF, zorder=1)
        ax.fill_between(x, [f_(r["null_lo"]) for r in s],
                        [f_(r["null_hi"]) for r in s], color=PS.GREY_FILL,
                        alpha=PS.NULL_ALPHA, lw=0, zorder=2)
        ax.plot(x, [f_(r["memory_mean"]) for r in s], "-o", color=PS.BLUE,
                lw=PS.LW_MAIN, ms=PS.MS_MAIN, zorder=4)
        ax.fill_between(x, [f_(r["memory_lo"]) for r in s],
                        [f_(r["memory_hi"]) for r in s], color=PS.BLUE,
                        alpha=PS.BAND_ALPHA, lw=0, zorder=3)
        ax.set_xticks(range(1, lagmax + 1))
        ax.set_xlim(0.5, lagmax + 0.5)
        ax.set_ylim(-0.55, 0.35)                      # unchanged
        PS.clean(ax)
    titles(fig, axes, PANELS)
    for a in axes[1:]:
        a.tick_params(labelleft=False)
    PS.shared_labels(fig, axes, "lag (observation intervals)",
                     "directional memory $M(\\mathrm{lag})$",
                     xy=(0.55, 0.115), yx=0.016)
    PS.strip_legend(fig, [
        Line2D([], [], color=PS.BLUE, lw=PS.LW_MAIN, marker="o", ms=PS.MS_MAIN),
        Patch(facecolor=PS.BLUE, alpha=PS.BAND_ALPHA),
        Patch(facecolor=PS.GREY_FILL, alpha=PS.NULL_ALPHA),
        Line2D([], [], color=PS.GREY_LINE, lw=PS.LW_REF)],
        ["observed", "95% question bootstrap", "shuffled-time null (95%)",
         "zero memory"], y=0.012)
    return finish(fig, "meta_temporal_memory_5way", "Fig 4  1x5 strip (matches Fig 2)")


# =============================================== Figure 5  (asymmetric 1 x 2)
DECOMP = [("measurement", PS.GREY_FILL), ("shared", PS.BLUE),
          ("continuation-specific", PS.ORANGE)]
SER = [("Observed", "observed_js", PS.INK, "-", 1.9, "o", 3.4, 6),
       ("S2  shared + specific", "S2", PS.BLUE, "-", 1.5, "s", 2.9, 5),
       ("S1$'$A  sharing permuted", "S1pA", PS.ORANGE, "--", 1.1, None, 0, 3),
       ("S1$'$B  sharing sign-flipped", "S1pB", PS.ORANGE, ":", 1.1, None, 0, 3),
       ("S1  iid full residual", "S1", PS.ORANGE, "-", 1.4, "^", 2.9, 4),
       ("S0  drift only", "S0", PS.GREY_LINE, "-.", 1.0, None, 0, 2)]


def fig5():
    rows = rd(RES / "residual_structure_forecast.csv")
    dec = {r["label"]: f_(r["percent_of_energy"]) for r in rows if r["panel"] == "a"}
    order = ["finite-rollout measurement", "shared across continuations",
             "continuation-specific"]
    H = sorted({int(r["horizon"]) for r in rows if r["panel"] == "b"})
    get = lambda k: {int(r["horizon"]): f_(r["mean_pairwise_js"]) for r in rows
                     if r["panel"] == "b" and r["series"] == k}
    fig = plt.figure(figsize=(PS.TEXTW, H_RESID))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[0.33, 0.67],
                  height_ratios=[1.0, 0.185], wspace=0.34, hspace=0.62,
                  left=0.088, right=0.995, top=1 - 0.26 / H_RESID, bottom=0.055)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    lax = fig.add_subplot(gs[1, :])
    lax.axis("off")

    # (a) one 0-100% stacked bar; percentages inside, names listed beneath so
    #     a 0.37 in segment never has to carry a 1.15 in word
    left = 0.0
    for (short, col), key in zip(DECOMP, order):
        pc = dec[key]
        a.barh([0.72], [pc], left=left, height=0.34, color=col,
               edgecolor="white", lw=0.7)
        a.text(left + pc / 2, 0.72, f"{pc:.1f}%", ha="center", va="center",
               fontsize=PS.FS_ANNOT,
               color="white" if col != PS.GREY_FILL else "#333333")
        left += pc
    for i, ((short, col), key) in enumerate(zip(DECOMP, order)):
        y = 0.30 - i * 0.26
        a.add_patch(plt.Rectangle((2, y - 0.055), 7, 0.11, color=col,
                                  clip_on=False))
        a.text(12, y, short, va="center", ha="left", fontsize=PS.FS_ANNOT,
               color=PS.SUBTLE)
    a.set_xlim(0, 100)
    a.set_ylim(-0.42, 0.98)
    a.set_yticks([])
    a.set_xticks([0, 50, 100])
    a.set_xlabel("share of residual energy (%)")
    a.grid(False)
    PS.panel_title(fig, a, "a", "one-step residual energy")
    PS.clean(a, ("left", "right", "top"))

    # (b) the main panel keeps two thirds of the width
    for lbl, key, col, ls, lw, mk, ms, z in SER:
        y = get(key)
        b.plot(H, [y[h] for h in H], ls=ls, color=col, lw=lw, marker=mk, ms=ms,
               zorder=z, label=lbl)
    b.set_xticks(H)
    b.set_xlim(3.5, 26.5)
    b.set_xlabel("horizon after the fork (training steps)")
    b.set_ylabel("mean pairwise JS")
    PS.panel_title(fig, b, "b", "recursive divergence forecast")
    PS.clean(b)
    b.text(0.030, 0.97, "shared structure explains 66\u201372% of\n"
           "the S1$\\rightarrow$S2 error reduction", transform=b.transAxes,
           va="top", ha="left", fontsize=PS.FS_ANNOT, color=PS.SUBTLE,
           linespacing=1.3)
    h, l = b.get_legend_handles_labels()
    lax.legend(h, l, loc="center", ncol=3, frameon=False, fontsize=6.8,
               handlelength=2.0, handletextpad=0.5, columnspacing=1.4,
               labelspacing=0.45, borderpad=0.0)
    return finish(fig, "residual_structure_forecast", "Fig 5  asymmetric 1x2 (33/67) + legend row")


# =============================================== Figure 6  (asymmetric 1 x 2)
def fig6():
    rows = rd(RES / "prefork_instability.csv")
    au = [r for r in rows if r["panel"] == "a"]
    sx = [r for r in rows if r["panel"] == "b" and r["metric"] == "modal_switch_rate"]
    base = f_(next(r for r in rows if r["metric"] == "switch_base_rate")["value"])
    aur = next(r for r in rows if r["metric"] == "AUROC_neg_margin")
    fig = plt.figure(figsize=(PS.TEXTW, H_INSTAB))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[0.42, 0.58], wspace=0.40,
                  left=0.175, right=0.988, top=1 - 0.26 / H_INSTAB, bottom=0.175)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ys = np.arange(len(au))[::-1]
    for y, r in zip(ys, au):
        v, lo, hi = f_(r["value"]), f_(r["ci_low"]), f_(r["ci_high"])
        a.plot([lo, hi], [y, y], color=PS.BLUE, lw=1.4, solid_capstyle="butt",
               zorder=3)
        a.plot([v], [y], "o", color=PS.BLUE, ms=4.0, zorder=4)
        a.text(hi + 0.02, y, f"{v:.3f}", va="center", fontsize=PS.FS_ANNOT,
               color="#333333")
    a.axvline(0.5, color=PS.GREY_LINE, lw=PS.LW_REF, ls="--", zorder=2)
    a.text(0.5, len(au) - 0.42, "chance", ha="center", va="top",
           fontsize=PS.FS_ANNOT, color=PS.GREY_LINE)
    a.set_yticks(ys)
    a.set_yticklabels([r["row"] for r in au])
    a.set_xlim(0.30, 1.06)
    a.set_xticks([0.4, 0.6, 0.8, 1.0])
    a.set_ylim(-0.6, len(au) - 0.25)
    a.set_xlabel("AUROC (future modal disagreement)")
    a.grid(axis="y", visible=False)
    PS.panel_title(fig, a, "a", "pre-fork state predicts disagreement")
    PS.clean(a, ("left", "right", "top"))

    x = [int(r["row"].split()[-1]) for r in sx]
    y = [f_(r["value"]) for r in sx]
    b.errorbar(x, y, yerr=[[f_(r["value"]) - f_(r["ci_low"]) for r in sx],
                           [f_(r["ci_high"]) - f_(r["value"]) for r in sx]],
               fmt="o-", color=PS.BLUE, lw=PS.LW_MAIN, ms=3.4, capsize=2.0,
               elinewidth=1.0, zorder=4)
    b.axhline(base, color=PS.GREY_LINE, lw=PS.LW_REF, ls="--", zorder=2)
    b.text(6.45, base + 0.004, f"base rate {base:.4f}", ha="right", va="bottom",
           fontsize=PS.FS_ANNOT, color=PS.GREY_LINE)
    b.set_xticks(x)
    b.set_xticklabels(["lowest", "2", "3", "4", "5", "highest"])
    b.set_xlim(0.5, 6.5)
    b.set_ylabel("modal-switch rate at $t{+}5$")
    b.set_xlabel("branch margin sextile")
    PS.panel_title(fig, b, "b", "low margin predicts switching")
    PS.clean(b)
    b.text(0.97, 0.95, f"90.8% of switches in lowest sextile\n"
           f"AUROC($-$margin) = {f_(aur['value']):.3f}", transform=b.transAxes,
           ha="right", va="top", fontsize=PS.FS_ANNOT, color=PS.SUBTLE,
           linespacing=1.3)
    return finish(fig, "prefork_instability", "Fig 6  asymmetric 1x2 (42/58)")


# ============================== appendix: behavioural displacement (1 x 5)
def fig_app_displacement():
    rows = rd(DAT / "meta4_behavioral_displacement.csv")
    by = {t: sorted([r for r in rows if r["track"] == t],
                    key=lambda r: f_(r["x_normalised"])) for _, t, _, _ in PANELS}
    hi = max(f_(r["d_move_hi"]) for v in by.values() for r in v)
    fig, axes = PS.strip(5, H_STRIP5, left=0.105, bottom=0.287, title_in=0.42)
    fig.set_label("app_disp")
    has_floor = False
    for ax, (lt, tr, main, sub) in zip(axes, PANELS):
        s = by[tr]
        x = [f_(r["x_normalised"]) for r in s]
        ax.plot(x, [f_(r["d_move_mean"]) for r in s], "-o", color=PS.BLUE,
                lw=PS.LW_MAIN, ms=PS.MS_MAIN, zorder=4)
        ax.fill_between(x, [f_(r["d_move_lo"]) for r in s],
                        [f_(r["d_move_hi"]) for r in s], color=PS.BLUE,
                        alpha=PS.BAND_ALPHA, lw=0, zorder=3)
        fl = [r for r in s if r.get("floor_mean") not in (None, "", "nan")]
        if fl:
            has_floor = True
            ax.plot([f_(r["x_normalised"]) for r in fl],
                    [f_(r["floor_mean"]) for r in fl], "--", color=PS.GREY_LINE,
                    lw=1.0, zorder=2)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.02 * hi, hi * 1.12)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", ".5", "1"])
        PS.clean(ax)
    titles(fig, axes, PANELS)
    for a in axes[1:]:
        a.tick_params(labelleft=False)
    PS.shared_labels(fig, axes, "normalised training progress",
                     "displacement $D_{\\mathrm{move}}$ (JS)",
                     xy=(0.55, 0.115), yx=0.016)
    hs = [Line2D([], [], color=PS.BLUE, lw=PS.LW_MAIN, marker="o", ms=PS.MS_MAIN),
          Patch(facecolor=PS.BLUE, alpha=PS.BAND_ALPHA)]
    ls = ["observed", "95% question bootstrap"]
    if has_floor:
        hs.append(Line2D([], [], color=PS.GREY_LINE, lw=1.0, ls="--"))
        ls.append("measurement floor")
    PS.strip_legend(fig, hs, ls, y=0.012)
    return finish(fig, "meta_behavioral_displacement_5way",
                  "App.  1x5 strip (same language)")


def main():
    t0 = time.time()
    print(f"rebuilding main figures at text width {PS.TEXTW} in, Nimbus Sans, "
          "pdf.fonttype=42\n")
    out = {"Figure 2": fig2(), "Figure 3": fig3(), "Figure 4": fig4(),
           "Figure 5": fig5(), "Figure 6": fig6(),
           "Appendix displacement": fig_app_displacement()}
    man = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "refactor": "geometry + typography only; no data, statistic, CI, "
                       "baseline or panel was changed",
           "text_width_in": PS.TEXTW,
           "body_heights_in": {"fig2": H_STRIP5, "fig3": H_STRIP3,
                               "fig4": H_STRIP5, "fig5": H_RESID,
                               "fig6": H_INSTAB, "appendix": H_STRIP5},
           "type_sizes": {"title": PS.FS_TITLE, "subtitle": PS.FS_SUB,
                          "label": PS.FS_LABEL, "tick": PS.FS_TICK,
                          "legend": PS.FS_LEG, "annotation": PS.FS_ANNOT},
           "line_marker": {"lw_main": PS.LW_MAIN, "lw_ref": PS.LW_REF,
                           "ms_main": PS.MS_MAIN, "band_alpha": PS.BAND_ALPHA,
                           "null_alpha": PS.NULL_ALPHA},
           "palette": {"blue": PS.BLUE, "orange": PS.ORANGE,
                       "grey_fill": PS.GREY_FILL, "grey_line": PS.GREY_LINE,
                       "ink": PS.INK},
           "fitted_title_sizes": {k: v for k, v in FITS.items()},
           "pdf_sha256": out,
           "code_sha256": {"<scripts>/fig_final.py": sha(__file__),
                           "<scripts>/paper_style.py":
                               sha("<scripts>/paper_style.py")},
           "git_commit": subprocess.run(["git", "-C", "<artifacts>",
                                         "rev-parse", "HEAD"],
                                        capture_output=True, text=True,
                                        timeout=30).stdout.strip(),
           "wall_sec": round(time.time() - t0, 1)}
    (RES / "figure_style_manifest.json").write_text(json.dumps(man, indent=1))
    print(f"\n  manifest -> results/figure_style_manifest.json  "
          f"({man['wall_sec']} s)")


if __name__ == "__main__":
    main()
