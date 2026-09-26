#!/usr/bin/env python3
"""Item 4, patch 3 — the raw-margin result, with no model and no folds.

The predictor IS the winner margin, so the cleanest final number ranks cells by
the margin itself. Nothing is fitted, so there is no cross-fitting, no fold
intercept and no fold scale: the only remaining source of uncertainty is which
questions are in the sample, which the question-cluster bootstrap handles.

Sign convention, fixed a priori by the monotone direction of each feature and not
chosen by what scored best (AUROC(x) = 1 - AUROC(-x), so the choice only flips the
number):
    margin, top-1 mass   -> higher means MORE persistent
    entropy              -> higher means LESS persistent
Reported as AUROC(-score, "the winner switches"), which is identically equal to
AUROC(score, "the winner persists"); the identity is asserted numerically below.
"""
import csv, hashlib, json, platform, subprocess, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

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


OUT = _mkout(OUTPUTS / "round11_3b_trunk" / "analysis_todo_closure_v1")
R9 = ART / "round9_markov_test"
BOOT, BOOT_SEED = 2000, 20260929
TIE_NEAR = 0.01
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
FEATS = {"margin": +1, "top1": +1, "entropy": -1}       # +1: higher -> more persistent


def ci(v, lo=.025, hi=.975):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [round(float(np.quantile(v, lo)), 6), round(float(np.quantile(v, hi)), 6)]


def main():
    t0 = time.time()
    rows = list(csv.DictReader((FROZEN / "item4_lockin_predictions_v2.csv").open()))
    fold = json.load((R9 / "fold_assignment.json").open())["assignment"]
    qs_all = sorted({r["question_id"] for r in rows})
    brng = np.random.default_rng(BOOT_SEED)
    bidx = brng.integers(0, len(qs_all), size=(BOOT, len(qs_all)))

    res = {}
    for delta in ("5", "10"):
        R = [r for r in rows if r["delta"] == delta]
        y = np.array([int(r["y"]) for r in R])            # 1 = winner persists
        sw = 1 - y                                        # 1 = winner switches
        F = {k: np.array([float(r[k]) for r in R]) for k in FEATS}
        pm = np.array([float(r["p_margin"]) for r in R])  # cross-fitted, for comparison
        fv = np.array([r["fold"] for r in R])
        cid = np.array([r["continuation_id"] for r in R])
        by_q = defaultdict(list)
        for i, r in enumerate(R):
            by_q[r["question_id"]].append(i)
        cache = [np.array([i for j in bidx[b] for i in by_q.get(qs_all[j], ())], int)
                 for b in range(BOOT)]

        def score(k):
            return FEATS[k] * F[k]

        out = {"n_cells": len(R), "n_questions": len({r["question_id"] for r in R}),
               "n_switches": int(sw.sum()),
               "switch_base_rate": round(float(sw.mean()), 6),
               "persistence_base_rate": round(float(y.mean()), 6),
               "features": {}}

        for k in FEATS:
            s = score(k)
            a_sw = float(roc_auc_score(sw, -s))
            a_ps = float(roc_auc_score(y, s))
            ap = float(average_precision_score(sw, -s))
            ba = np.array([roc_auc_score(sw[i], -s[i]) if len(set(sw[i])) > 1 else np.nan
                           for i in cache])
            bp = np.array([average_precision_score(sw[i], -s[i]) if len(set(sw[i])) > 1 else np.nan
                           for i in cache])
            out["features"][k] = {
                "sign": FEATS[k],
                "auroc_switch": round(a_sw, 6), "auroc_switch_ci": ci(ba),
                "auroc_persist_identity_check": round(a_ps, 6),
                "identity_holds": bool(abs(a_sw - a_ps) < 1e-12),
                "auprc_switch": round(ap, 6), "auprc_switch_ci": ci(bp),
                "auprc_switch_lift": round(ap / float(sw.mean()), 4)}
            out["features"][k]["_boot_auroc"] = ba
            out["features"][k]["_boot_auprc"] = bp

        # cross-fitted margin model, same resampling, for the paired comparison
        ba_cf = np.array([roc_auc_score(sw[i], -pm[i]) if len(set(sw[i])) > 1 else np.nan
                          for i in cache])
        a_cf = float(roc_auc_score(sw, -pm))
        out["cross_fitted_margin_model"] = {"auroc_switch": round(a_cf, 6),
                                            "auroc_switch_ci": ci(ba_cf)}
        d = out["features"]["margin"]["_boot_auroc"] - ba_cf
        out["raw_minus_cross_fitted"] = {
            "difference": round(out["features"]["margin"]["auroc_switch"] - a_cf, 6),
            "ci": ci(d), "p_gt_0": round(float((d[np.isfinite(d)] > 0).mean()), 4),
            "interpretation": "a paired question-cluster bootstrap of the SAME quantity computed "
                              "two ways; a CI straddling zero means cross-fitting neither helped "
                              "nor hurt, i.e. the fold intercepts and scales were not distorting "
                              "the headline"}

        # paired raw contrasts
        pair = {}
        for a, b in (("margin", "top1"), ("margin", "entropy")):
            for nm, key in (("auroc", "_boot_auroc"), ("auprc", "_boot_auprc")):
                dd = out["features"][a][key] - out["features"][b][key]
                pt = (out["features"][a][f"{nm}_switch"] - out["features"][b][f"{nm}_switch"])
                pair[f"{a}-{b}|{nm}"] = {"difference": round(pt, 6), "ci": ci(dd),
                                         "p_gt_0": round(float((dd[np.isfinite(dd)] > 0).mean()), 4)}
        out["paired_raw_contrasts"] = pair

        # per-fold raw AUROC: there is no fold effect to worry about, shown not asserted
        out["auroc_switch_by_fold"] = {}
        s = score("margin")
        for f in sorted(set(fv)):
            m = fv == f
            if len(set(sw[m])) > 1:
                out["auroc_switch_by_fold"][f] = {
                    "n": int(m.sum()), "n_switches": int(sw[m].sum()),
                    "auroc": round(float(roc_auc_score(sw[m], -s[m])), 6)}

        # ties and leave-one-continuation-out, on the raw score
        mg = F["margin"]
        out["tie_sensitivity"] = {}
        for nm, keep in (("all_cells", np.ones(len(R), bool)),
                         ("exclude_near_ties_margin_lt_0.01", mg >= TIE_NEAR)):
            b = np.array([roc_auc_score(sw[i[keep[i]]], -s[i[keep[i]]])
                          if len(set(sw[i[keep[i]]])) > 1 else np.nan for i in cache])
            out["tie_sensitivity"][nm] = {
                "n": int(keep.sum()), "n_switches": int(sw[keep].sum()),
                "auroc_switch": round(float(roc_auc_score(sw[keep], -s[keep])), 6),
                "ci": ci(b)}
        out["leave_one_continuation_out"] = [
            {"dropped": c, "n": int((cid != c).sum()),
             "n_switches": int(sw[cid != c].sum()),
             "auroc_switch": round(float(roc_auc_score(sw[cid != c], -s[cid != c])), 6)}
            for c in sorted(set(cid)) if len(set(sw[cid != c])) > 1]

        # where the switches actually are
        qe = np.quantile(mg, np.linspace(0, 1, 7))
        qe[0] -= 1e-9; qe[-1] += 1e-9
        sext = []
        for i in range(6):
            m = (mg > qe[i]) & (mg <= qe[i + 1])
            sext.append({"sextile": i + 1, "n": int(m.sum()),
                         "n_switches": int(sw[m].sum()),
                         "share_of_all_switches": round(float(sw[m].sum() / sw.sum()), 4),
                         "switch_rate": round(float(sw[m].mean()), 6)})
        out["switch_concentration_by_margin_sextile"] = sext
        out["share_of_switches_in_lowest_sextile"] = sext[0]["share_of_all_switches"]

        for k in FEATS:
            out["features"][k].pop("_boot_auroc")
            out["features"][k].pop("_boot_auprc")
        res[f"delta_{delta}"] = out

    blob = {"what_this_is":
            "the raw-margin result: AUROC(-margin, 'the winner switches'), computed directly "
            "from the winner margin with no model, no cross-fitting and no folds. This is the "
            "headline number; the cross-fitted logistic, its coefficient and its Brier score "
            "become supplementary.",
            "why": "the predictor is the margin itself, so fitting a per-fold logistic can only "
                   "add fold-specific intercepts and scales to the pooled ranking. Removing the "
                   "model removes that concern entirely.",
            "sign_convention": {k: ("higher -> more persistent" if v > 0
                                    else "higher -> less persistent") for k, v in FEATS.items()},
            "sign_convention_note": "fixed a priori by each feature's monotone direction, not "
                                    "chosen by what scored best; AUROC(x) = 1 - AUROC(-x), so the "
                                    "choice only flips the number",
            "no_fitting": True, "no_cross_validation_needed": True,
            "protocol": {"file": "item4_lockin_protocol.json",
                         "sha256": sha(FROZEN / "item4_lockin_protocol.json")},
            "inputs_sha256": {"item4_lockin_predictions_v2.csv":
                              sha(FROZEN / "item4_lockin_predictions_v2.csv"),
                              "fold_assignment.json": sha(R9 / "fold_assignment.json")},
            "code_sha256": {str(Path(__file__).name): sha(Path(__file__))},
            "git_commit": subprocess.run(["git", "-C", "<artifacts>", "rev-parse", "HEAD"],
                                         capture_output=True, text=True, timeout=20).stdout.strip(),
            "command": "python scripts/tc_item4_raw.py",
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "host": platform.node()},
            "seeds": {"bootstrap": BOOT_SEED}, "n_bootstrap": BOOT,
            "resampling_unit": "question", "results": res,
            "wall_sec": round(time.time() - t0, 1)}
    (OUT / "item4_raw_margin.json").write_text(json.dumps(blob, indent=1))

    for d in ("5", "10"):
        r = res[f"delta_{d}"]
        m = r["features"]["margin"]
        print(f"  delta={d}: {r['n_cells']:,} cells, {r['n_questions']} questions, "
              f"{r['n_switches']} switches (base rate {r['switch_base_rate']:.4f})")
        print(f"      RAW margin  AUROC {m['auroc_switch']:.4f} "
              f"[{m['auroc_switch_ci'][0]:.4f}, {m['auroc_switch_ci'][1]:.4f}]   "
              f"switch-AUPRC {m['auprc_switch']:.4f} "
              f"[{m['auprc_switch_ci'][0]:.4f}, {m['auprc_switch_ci'][1]:.4f}] "
              f"= {m['auprc_switch_lift']:.2f}x")
        print(f"      identity AUROC(-margin, switch) == AUROC(margin, persist): "
              f"{m['identity_holds']}")
        cf = r["cross_fitted_margin_model"]
        dd = r["raw_minus_cross_fitted"]
        print(f"      cross-fitted {cf['auroc_switch']:.4f} "
              f"[{cf['auroc_switch_ci'][0]:.4f}, {cf['auroc_switch_ci'][1]:.4f}];  "
              f"raw - cross-fitted {dd['difference']:+.4f} [{dd['ci'][0]:+.4f}, {dd['ci'][1]:+.4f}]")
        print(f"      raw top1 {r['features']['top1']['auroc_switch']:.4f}, "
              f"raw entropy {r['features']['entropy']['auroc_switch']:.4f}")
        print(f"      per-fold AUROC: "
              + ", ".join(f"{v['auroc']:.4f}" for v in r["auroc_switch_by_fold"].values()))
        print(f"      switches in the lowest-margin sextile: "
              f"{r['switch_concentration_by_margin_sextile'][0]['n_switches']}/{r['n_switches']} "
              f"= {r['share_of_switches_in_lowest_sextile']:.1%}")
    print(f"  item4_raw_margin.json sha256 {sha(OUT / 'item4_raw_margin.json')}")


if __name__ == "__main__":
    main()
