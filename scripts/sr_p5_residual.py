#!/usr/bin/env python3
"""P5: one-step residual decomposition re-audited at 1024 rollouts.

The old manuscript split one-step residual variation into measurement noise, a
component shared across continuations, and continuation-specific innovation,
using a 256-rollout analysis. Those numbers are not carried over; everything
here is recomputed on the current canonical dense five-continuation trajectory
at 1024 rollouts per prompt.

Transition models, both reported, neither selected on the answer it gives:

    R1  F(z_t)              + branch controls      (frozen r9_models M1)
    R2  F(z_t, v_t)         + branch controls      (frozen r9_models M2)

Residual, out of fold, in CLR space:

    r_{q,t,s} = clr(p_observed at t+Delta) - clr(p_predicted by F)

Split, with the small-S bias handled explicitly. With S continuations the naive
mean absorbs 1/S of the innovation energy, so the unbiased estimates are

    E_within  = mean_s || r_s - mean_s r_s ||^2          (biased low by (S-1)/S)
    E_eta     = S/(S-1) * E_within
    E_mu      = || mean_s r_s ||^2 - E_eta / S

Measurement noise at t+Delta is independent across continuations, so it lands
entirely in eta and is subtracted there, never from the shared component.
"""
import csv, hashlib, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
import r9_models as RM
from r6_lib import fit, predict

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


PMF = ART / "paper_main_figures"
R9 = ART / "round9_markov_test"
OUT = _mkout(OUTPUTS / "section4_stakeholder_resolution")
ALPHA, N_BOOT = 0.5, 10000
SEED = 20260924
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def clr(p):
    lp = np.log(np.maximum(p, 1e-300))
    return lp - lp.mean()


def oof_residuals(rows, feats, fold):
    """out-of-fold CLR residuals, keyed by (question, step_t, seed)."""
    units = RM.units_for(rows, feats)
    res = {}
    for k in sorted(set(fold.values())):
        tr = [u for u in units if fold[u["question_id"]] != k]
        te = [u for u in units if fold[u["question_id"]] == k]
        if not tr or not te:
            continue
        beta, sd = fit(tr, feats, RM.L2)
        for u in te:
            p_pred = predict(u, beta, sd)
            y = np.asarray(u["y"], float)
            n = y.sum()
            if n <= 0:
                continue
            p_obs = (y + ALPHA) / (n + ALPHA * len(y))
            res[(u["question_id"], u["step"], u["seed"])] = {
                "r": clr(p_obs) - clr(p_pred), "n_target": float(n), "K": len(y),
                "p_obs": p_obs}
    return res


def measurement_energy(p_obs, n, K, rng, reps=32):
    """E||z_a - z_b||^2 / 2 for two independent redraws of size n from p_obs."""
    v = []
    for _ in range(reps):
        a = rng.multinomial(int(n), p_obs); b = rng.multinomial(int(n), p_obs)
        za = clr((a + ALPHA) / (a.sum() + ALPHA * K))
        zb = clr((b + ALPHA) / (b.sum() + ALPHA * K))
        v.append(float(np.sum((za - zb) ** 2) / 2.0))
    return float(np.mean(v))


def decompose(res, rng):
    """per (question, transition): unbiased shared / seed-specific split."""
    by = defaultdict(dict)
    for (q, t, s), d in res.items():
        by[(q, t)][s] = d
    long_rows, cell = [], {}
    for (q, t), D in sorted(by.items()):
        seeds = sorted(D)
        S = len(seeds)
        if S < 3:
            continue
        Rm = np.array([D[s]["r"] for s in seeds])
        m = Rm.mean(0)
        e_raw = float(np.mean((Rm ** 2).sum(1)))
        e_within = float(np.mean(((Rm - m) ** 2).sum(1)))
        e_eta = e_within * S / (S - 1)
        e_mu = float((m ** 2).sum()) - e_eta / S
        e_meas = float(np.mean([measurement_energy(D[s]["p_obs"], D[s]["n_target"],
                                                   D[s]["K"], rng) for s in seeds]))
        e_eta_corr = e_eta - e_meas
        cell[(q, t)] = {"raw": e_raw, "meas": e_meas, "mu": e_mu, "eta": e_eta,
                        "eta_corr": e_eta_corr, "S": S, "K": D[seeds[0]]["K"]}
        for s in seeds:
            loo = np.array([D[u]["r"] for u in seeds if u != s]).mean(0)
            eta_s = D[s]["r"] - loo
            long_rows.append({"question_id": q, "transition_start": int(t),
                              "transition_end": int(t) + 5, "continuation_id": s,
                              "raw_residual_energy": round(float((D[s]["r"] ** 2).sum()), 8),
                              "measurement_energy": round(e_meas, 8),
                              "shared_component_energy": round(e_mu, 8),
                              "seed_specific_energy": round(e_eta, 8),
                              "measurement_corrected_seed_specific_energy": round(e_eta_corr, 8),
                              "leave_one_continuation_out_eta_energy":
                                  round(float((eta_s ** 2).sum()) * (S - 1) / S, 8),
                              "valid": True, "notes": ""})
    return cell, long_rows


def summarise(cell, rng, scope, model, qsel=None, tsel=None):
    keys = [k for k in cell if (qsel is None or k[0] in qsel) and (tsel is None or k[1] in tsel)]
    if not keys:
        return []
    raw = np.array([cell[k]["raw"] for k in keys])
    meas = np.array([cell[k]["meas"] for k in keys])
    mu = np.array([cell[k]["mu"] for k in keys])
    eta = np.array([cell[k]["eta"] for k in keys])
    etac = np.array([cell[k]["eta_corr"] for k in keys])
    qs = sorted({k[0] for k in keys})
    qi = defaultdict(list)
    for i, k in enumerate(keys):
        qi[k[0]].append(i)
    idx = rng.integers(0, len(qs), (N_BOOT, len(qs)))
    tot = float(raw.mean())
    rows = []
    for name, arr in (("raw_residual", raw), ("measurement", meas), ("shared_mu", mu),
                      ("seed_specific_eta", eta),
                      ("seed_specific_eta_measurement_corrected", etac)):
        bs = []
        for b in idx[:2000]:
            sel = [j for i in b for j in qi[qs[i]]]
            bs.append(arr[sel].mean())
        bs = np.array(bs)
        rows.append({"transition_model": model, "analysis_scope": scope, "component": name,
                     "absolute_energy": round(float(arr.mean()), 8),
                     "share": round(float(arr.mean()) / tot, 6),
                     "ci_low": round(float(np.quantile(bs, .025)), 8),
                     "ci_high": round(float(np.quantile(bs, .975)), 8),
                     "num_questions": len(qs), "num_transitions": len({k[1] for k in keys}),
                     "num_continuations": int(np.median([cell[k]["S"] for k in keys]))})
    return rows


def main():
    tab = PMF / "tables" / "transition_table_trackD.csv"
    rows = RM.load(tab)
    fold = json.load((R9 / "fold_assignment.json").open())["assignment"]
    rng = np.random.default_rng(SEED)
    sets = {"R1": [r for r in rows if np.isfinite(r["z_t"])],
            "R2": [r for r in rows if np.isfinite(r["velocity_t"])]}
    feats = {"R1": RM.MODELS["M1"], "R2": RM.MODELS["M2"]}

    all_long, all_summary, verd = [], [], {}
    for model in ("R1", "R2"):
        res = oof_residuals(sets[model], feats[model], fold)
        cell, long_rows = decompose(res, rng)
        for r in long_rows:
            r["transition_model"] = model
        all_long += long_rows
        all_summary += summarise(cell, rng, "all transitions", model)
        for t in sorted({k[1] for k in cell}):
            all_summary += summarise(cell, rng, f"transition {int(t)}->{int(t)+5}", model,
                                     tsel={t})
        verd[model] = cell
        S = summarise(cell, rng, "all transitions", model)
        d = {r["component"]: r for r in S}
        print(f"  {model}: raw {d['raw_residual']['absolute_energy']:.4f} | "
              f"measurement {d['measurement']['share']:.3f} | shared {d['shared_mu']['share']:.3f} | "
              f"eta {d['seed_specific_eta']['share']:.3f} | eta-corrected "
              f"{d['seed_specific_eta_measurement_corrected']['share']:.3f} "
              f"[{d['seed_specific_eta_measurement_corrected']['ci_low']:.4f}, "
              f"{d['seed_specific_eta_measurement_corrected']['ci_high']:.4f}]  "
              f"({d['raw_residual']['num_questions']} q x "
              f"{d['raw_residual']['num_transitions']} transitions)")

    # leave-one-continuation-out robustness on the primary model
    for model in ("R1", "R2"):
        cell = verd[model]
        for drop in sorted({s for (q, t, s) in
                            [(k[0], k[1], c) for k in cell for c in ["seed%d" % i for i in range(1, 6)]]}):
            sub = [r for r in sets[model] if r["seed"] != drop]
            res2 = oof_residuals(sub, feats[model], fold)
            c2, _ = decompose(res2, rng)
            all_summary += summarise(c2, rng, f"leave_out_{drop}", model)

    ord_ = ["transition_model", "analysis_scope", "component", "absolute_energy", "share",
            "ci_low", "ci_high", "num_questions", "num_transitions", "num_continuations"]
    with (OUT / "tables" / "residual_decomposition_1024_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ord_); w.writeheader(); w.writerows(all_summary)
    lord = ["question_id", "transition_start", "transition_end", "continuation_id",
            "transition_model", "raw_residual_energy", "measurement_energy",
            "shared_component_energy", "seed_specific_energy",
            "measurement_corrected_seed_specific_energy",
            "leave_one_continuation_out_eta_energy", "valid", "notes"]
    with (OUT / "tables" / "residual_decomposition_1024_long.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=lord, extrasaction="ignore")
        w.writeheader(); w.writerows(all_long)
    json.dump({"note": "cell-level energies kept for P6 and P9",
               "cells": {f"{k[0]}|{int(t)}|{m}": v for m in verd for (k, t), v in
                         [((kk, kk[1]), vv) for kk, vv in verd[m].items()]}},
              (OUT / "audit" / "residual_cells.json").open("w"), indent=0, default=float)
    for f in ("tables/residual_decomposition_1024_long.csv",
              "tables/residual_decomposition_1024_summary.csv"):
        print(f"  {f} sha256 {sha(OUT / f)}")


if __name__ == "__main__":
    main()
