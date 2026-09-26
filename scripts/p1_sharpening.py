#!/usr/bin/env python3
"""P1: rebuild the frequency-sharpening null on the CURRENT canonical branch table.

M0:  p_t(b|q)  proportional to  p_{t0}(b|q)^{gamma_t}

The statistic, the fitting grid, the smoothing and the null are imported
verbatim from countdown/src/cd_step100.py, the frozen Round-5 implementation,
so nothing about the test is re-invented here. What changes is the input: the
branch states now come from tables/branch_state_long.csv, i.e. the current
frozen definition (parseable, solver-feasible entrances, Dirichlet alpha=0.5),
and the analysis is run on every track that has a usable reference checkpoint.

Discrepancy        D_t(q) = mean_b | log p_t(b|q) - log p_pred(b|q) |
Null               parametric bootstrap of M0 itself: resample counts at BOTH
                   t0 and t from the fitted null at the real rollout budgets,
                   refit gamma on every draw, recompute D. Finite-rollout noise
                   and gamma-estimation error are therefore inside the null.
Excess             ExcessD_t(q) = D_t(q) - E_0[D_t(q)]
Uncertainty        question-level bootstrap (10,000) for the CI; one-sided
                   parametric p with +1 smoothing.
Second null        question-specific gamma (M0-pq): each question may sharpen
                   at its own rate, which absorbs rate heterogeneity.
"""
import csv, hashlib, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from cd_step100 import (ALPHA, GRID, N_PARAM_BOOT, N_QBOOT, SEED,
                        smooth_np, sharpen_np, nll_grid, fit_gamma)

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


P = ART / "paper_main_figures"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_long():
    """(track, lineage, cid, root_step, ckpt) -> {question: {branch: count}}

    root_step MUST be part of the key: all four Track A roots name their forks
    seed1/seed2/seed3 and evaluate them at the same step 175, so a key without
    the root silently collapses four different continuations into one.
    """
    D = defaultdict(lambda: defaultdict(dict))
    for r in csv.DictReader((P / "tables" / "branch_state_long.csv").open()):
        key = (r["track"], r["lineage_id"], r["continuation_id"], r["root_step"], int(r["checkpoint"]))
        D[key][r["question_id"]][r["branch"]] = int(float(r["raw_branch_count"]))
    return D


def pack(state, qs, sup_of, Bmax):
    C = np.zeros((len(qs), Bmax)); M = np.zeros((len(qs), Bmax), bool)
    for i, q in enumerate(qs):
        for j, b in enumerate(sup_of[q]):
            C[i, j] = state[q].get(b, 0); M[i, j] = True
    return C, M


def fit_gamma_pq(LP0_, C_, M_):
    n = (C_ * M_).sum(-1)
    nl = []
    for g in GRID:
        pr = sharpen_np(LP0_, M_, g)
        nl.append(-(C_ * np.log(np.maximum(pr, 1e-300)) * M_).sum(-1) / np.maximum(n, 1.0))
    return GRID[np.argmin(np.stack(nl), axis=0)]


def analyse(C0, Ct, M, mode, rng, qidx):
    """returns per-question and pair-level results for one (t0, t) pair."""
    P0 = smooth_np(C0, M); LP0 = np.where(M, np.log(np.maximum(P0, 1e-300)), 0.0)
    Pt = smooth_np(Ct, M)
    if mode == "shared_gamma":
        g = fit_gamma(LP0, Ct, M); pred = sharpen_np(LP0, M, g); gam_out = float(g)
    else:
        gq = fit_gamma_pq(LP0, Ct, M); pred = sharpen_np(LP0, M, gq[:, None]); gam_out = gq
    R_ = np.where(M, np.log(np.maximum(Pt, 1e-300)) - np.log(np.maximum(pred, 1e-300)), 0.0)
    Dobs = (np.abs(R_) * M).sum(1) / M.sum(1)
    N0 = (C0 * M).sum(1); Nt = (Ct * M).sum(1); Qn, Bmax = C0.shape
    Cb = np.zeros((N_PARAM_BOOT, Qn, Bmax)); C0b = np.zeros_like(Cb)
    for i in range(Qn):
        k = int(M[i].sum())
        pp = pred[i, :k] / pred[i, :k].sum()
        Cb[:, i, :k] = rng.multinomial(int(Nt[i]), pp, size=N_PARAM_BOOT)
        p0i = P0[i, :k] / P0[i, :k].sum()
        C0b[:, i, :k] = rng.multinomial(int(N0[i]), p0i, size=N_PARAM_BOOT)
    Mb = M[None]
    LP0b = np.where(Mb, np.log(np.maximum(smooth_np(C0b, Mb), 1e-300)), 0.0)
    if mode == "shared_gamma":
        gb = GRID[np.argmin(nll_grid(LP0b, Cb, Mb), axis=0)]
        prb = sharpen_np(LP0b, Mb, gb[:, None, None])
    else:
        gb = fit_gamma_pq(LP0b, Cb, Mb)
        prb = sharpen_np(LP0b, Mb, gb[..., None])
    Rb = np.where(Mb, np.log(np.maximum(smooth_np(Cb, Mb), 1e-300))
                  - np.log(np.maximum(prb, 1e-300)), 0.0)
    Db = (np.abs(Rb) * Mb).sum(2) / M.sum(1)[None]                    # [B, Q]
    E = Dobs - Db.mean(0)
    bs = E[qidx].mean(1)
    p_param = float((Db.mean(1) >= Dobs.mean()).sum() + 1) / (N_PARAM_BOOT + 1)
    return {"Dobs": Dobs, "null_mean": Db.mean(0), "null_sd": Db.std(0),
            "null_lo": np.quantile(Db, .025, axis=0), "null_hi": np.quantile(Db, .975, axis=0),
            "excess": E, "gamma": gam_out, "p_param": p_param,
            "ci": (float(np.quantile(bs, .025)), float(np.quantile(bs, .975))),
            "frac_pos": float((E > 0).mean()), "N0": N0, "Nt": Nt, "K": M.sum(1)}


def main():
    t0 = time.time()
    D = load_long()
    sup_of = defaultdict(list)
    for key, st in D.items():
        for q, d in st.items():
            if not sup_of[q]:
                sup_of[q] = sorted(d)
    Bmax = max(len(v) for v in sup_of.values())

    A, C_, Dt, B_ = ("A", "r11_trunk"), ("C", "r8_7b"), ("D", "public_ckpt100"), ("B", "r11_trunk")
    TR = lambda st: A + ("trunk", "NA", st)                 # trunk arm: its root_step field is NA
    PAIRS = []
    for st in (50, 75, 100, 125, 150, 175, 200):
        PAIRS.append(("A", "r11_trunk", "trunk", TR(25), TR(st), 25, st, "trunk trajectory"))
    for root in (75, 100, 125, 150):
        for s in (1, 2, 3):
            k = A + (f"seed{s}", str(root), 175)
            if k in D:
                PAIRS.append(("A", "r11_trunk", f"root{root}_seed{s}", TR(root), k, root, 175,
                              "stochastic fork, root to endpoint"))
    PAIRS.append(("B", "r11_trunk", "root75_replay", TR(75), B_ + ("replay", "75", 175), 75, 175,
                  "same-seed replay control"))
    for s in (1, 2):
        for st in (10, 15, 20, 25):
            PAIRS.append(("C", "r8_7b", f"seed{s}", C_ + (f"seed{s}", "NA", 5), C_ + (f"seed{s}", "NA", st), 5, st,
                          "7B existence check"))
    for s in range(1, 6):
        for st in (105, 110, 115, 120, 125):
            PAIRS.append(("D", "public_ckpt100", f"seed{s}", Dt + ("base", "100", 100), Dt + (f"seed{s}", "100", st),
                          100, st, "public 5-seed replication"))

    rng = np.random.default_rng(SEED)
    qrng = np.random.default_rng(SEED + 1)
    per_rows, sum_rows = [], []
    for track, lin, cid, k0, k1, c0, c1, note in PAIRS:
        if k0 not in D or k1 not in D:
            print(f"  SKIP {track}/{cid} {c0}->{c1}: missing {k0 if k0 not in D else k1}")
            continue
        qs = sorted(set(D[k0]) & set(D[k1]))
        C0, M = pack(D[k0], qs, sup_of, Bmax)
        Ct, _ = pack(D[k1], qs, sup_of, Bmax)
        qidx = qrng.integers(0, len(qs), size=(N_QBOOT, len(qs)))
        for mode in ("shared_gamma", "question_specific_gamma"):
            r = analyse(C0, Ct, M, mode, rng, qidx)
            gam = r["gamma"]
            for i, q in enumerate(qs):
                per_rows.append({
                    "track": track, "lineage_id": lin, "continuation_id": cid, "question_id": q,
                    "reference_checkpoint": c0, "target_checkpoint": c1, "null_model": mode,
                    "fitted_gamma": round(float(gam if np.isscalar(gam) else gam[i]), 4),
                    "observed_discrepancy": round(float(r["Dobs"][i]), 6),
                    "null_mean": round(float(r["null_mean"][i]), 6),
                    "null_sd": round(float(r["null_sd"][i]), 6),
                    "null_q025": round(float(r["null_lo"][i]), 6),
                    "null_q975": round(float(r["null_hi"][i]), 6),
                    "excess_discrepancy": round(float(r["excess"][i]), 6),
                    "p_value_if_defined": "NA_per_question_p_is_not_defined_in_this_null",
                    "effective_n_reference": int(r["N0"][i]), "effective_n_target": int(r["Nt"][i]),
                    "num_branches": int(r["K"][i])})
            sum_rows.append({
                "track": track, "lineage_id": lin, "continuation_id": cid,
                "reference_checkpoint": c0, "target_checkpoint": c1, "horizon": c1 - c0,
                "null_model": mode, "note": note,
                "fitted_gamma": round(float(gam), 4) if np.isscalar(gam) else "per_question",
                "gamma_mean": round(float(np.mean(gam)), 4), "gamma_sd": round(float(np.std(gam)), 4)
                if not np.isscalar(gam) else 0.0,
                "mean_D_obs": round(float(r["Dobs"].mean()), 6),
                "mean_D_null": round(float(r["null_mean"].mean()), 6),
                "excess_D": round(float(r["excess"].mean()), 6),
                "ci_lo": round(r["ci"][0], 6), "ci_hi": round(r["ci"][1], 6),
                "frac_questions_positive": round(r["frac_pos"], 4),
                "p_parametric_one_sided": r["p_param"], "n_questions": len(qs),
                "n_param_boot": N_PARAM_BOOT, "n_question_boot": N_QBOOT})
            print(f"  {track}/{cid:16} {c0:>3}->{c1:<3} {mode:24} gamma "
                  f"{np.mean(gam):.3f}  ExcessD {r['excess'].mean():+.4f} "
                  f"[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]  frac+ {r['frac_pos']:.3f}  "
                  f"p {r['p_param']:.4f}", flush=True)

    (P / "tables").mkdir(exist_ok=True)
    with (WOUT / "tables" / "sharpening_null.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_rows[0])); w.writeheader(); w.writerows(per_rows)
    with (WOUT / "figure_data" / "fig2_sharpening_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sum_rows[0])); w.writeheader(); w.writerows(sum_rows)
    print(f"  sharpening_null.csv {len(per_rows)} rows sha256 {sha(WOUT/'tables/sharpening_null.csv')}")
    print(f"  fig2_sharpening_summary.csv {len(sum_rows)} rows sha256 "
          f"{sha(WOUT/'figure_data/fig2_sharpening_summary.csv')}")
    print(f"  elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
