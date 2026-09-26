#!/usr/bin/env python3
"""Track-A multi-root replication of the Section 5.5 pre-fork instability result.

CPU only, frozen artifacts only, no new training or generation.

Every definition -- branch probability, smoothing, feature construction, modal
branch, modal-disagreement target, continuous JS target, model family,
regularisation, CV and bootstrap -- is taken verbatim from the Track-D
implementation in sr_p789_susceptibility.py. Nothing is searched over or tuned
on Track A.

Track A design used here:
    root state  = trunk arm evaluated AT the root checkpoint r
    endpoint    = the four balanced arms at step 175
                  (trunk, seed1, seed2, seed3), exactly the arm set that
                  bk4_omnibus.py uses for the K=4 same-state omnibus
    roots       = 75, 100, 125, 150   ->  horizons 100, 75, 50, 25
"""
import csv, hashlib, json, sys, time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


PMF = ART / "paper_main_figures" / "tables"
R9 = ART / "round9_markov_test"
BK4 = ART / "analysis_balanced_k4"
OUT = _mkout(OUTPUTS / "trackA_prefork_instability_v1")
ROOTS = [75, 100, 125, 150]
ARMS = ["trunk", "seed1", "seed2", "seed3"]
ENDPOINT = 175
N_BOOT = 10000
SEED = 20260926                      # same frozen seed as the Track-D analysis
TIE_NEAR = 0.01                      # same near-tie threshold as Track D
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

FEATS = ["normalized_entropy", "top1_mass", "top1_top2_margin",
         "num_feasible_branches", "pass_at_1", "effective_feasible_count"]
CONTROLS = ["num_feasible_branches", "pass_at_1", "effective_feasible_count"]
STATE = ["normalized_entropy", "top1_mass", "top1_top2_margin"]


# ---------------------------------------------------------------- verbatim
def js(pa, pb):
    m = 0.5 * (pa + pb)
    return float(max(0.0, 0.5 * np.sum(pa * np.log(pa / m))
                     + 0.5 * np.sum(pb * np.log(pb / m))))


def feats_of(pv, eff, K, acc):
    p = np.sort(pv)[::-1]
    H = float(-(pv * np.log(pv)).sum() / np.log(len(pv))) if len(pv) > 1 else 0.0
    return {"normalized_entropy": H, "top1_mass": float(p[0]),
            "top1_top2_margin": float(p[0] - p[1]) if len(p) > 1 else 1.0,
            "num_feasible_branches": float(K), "pass_at_1": float(acc),
            "effective_feasible_count": float(eff)}


def cv_predict(X, y, fold_id, kind):
    pred = np.zeros(len(y), float)
    for k in sorted(set(fold_id)):
        tr, te = fold_id != k, fold_id == k
        if tr.sum() == 0 or te.sum() == 0:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        Xt, Xe = (X[tr] - mu) / sd, (X[te] - mu) / sd
        if kind == "clf":
            if len(set(y[tr])) < 2:
                pred[te] = y[tr].mean(); continue
            m = LogisticRegression(C=1.0, max_iter=2000).fit(Xt, y[tr])
            pred[te] = m.predict_proba(Xe)[:, 1]
        else:
            m = Ridge(alpha=1.0).fit(Xt, y[tr])
            pred[te] = m.predict(Xe)
    return pred


def auroc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")


def auprc(y, s):
    return float(average_precision_score(y, s)) if len(set(y)) > 1 else float("nan")


def ci(vals):
    v = np.array([x for x in vals if np.isfinite(x)], float)
    return ((float(np.quantile(v, .025)), float(np.quantile(v, .975)))
            if v.size else (float("nan"), float("nan")))


# ------------------------------------------------------------------- load
def load():
    """track A branch states keyed (continuation_id, root_step, checkpoint, q)."""
    P, MET = {}, {}
    for r in csv.DictReader((PMF / "branch_state_long.csv").open()):
        if r["track"] != "A":
            continue
        k = (r["continuation_id"], r["root_step"], int(r["checkpoint"]),
             r["question_id"])
        P.setdefault(k, {})[r["branch"]] = float(r["smoothed_probability"])
        MET[k] = (int(r["effective_feasible_count"]),
                  int(r["num_feasible_branches"]))
    ACC = {}
    for r in csv.DictReader((PMF / "branch_state_summary.csv").open()):
        if r["track"] == "A":
            ACC[(r["continuation_id"], r["root_step"], int(r["checkpoint"]),
                 r["question_id"])] = float(r["accuracy"])
    return P, MET, ACC


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    P, MET, ACC = load()
    fold = json.load((R9 / "fold_assignment.json").open())["assignment"]

    # optional robustness join: the frozen K=4 within-root BH decisions
    OMNI = {}
    try:
        bo = json.load((BK4 / "balanced_omnibus.json").open())
        for root in ROOTS:
            for q, d in bo["arms"]["4"][str(root)]["blocked"]["per_prompt"].items():
                OMNI[(root, q)] = int(bool(d["reject"]))
    except Exception as e:                       # never fatal: robustness only
        print(f"  omnibus join unavailable: {e}")

    qs = sorted({k[3] for k in P if k[0] == "trunk" and k[2] == ENDPOINT})
    rows, per_root = [], {}
    for root in ROOTS:
        recs = []
        for q in qs:
            kr = ("trunk", "NA", root, q)                 # shared pre-fork state
            if kr not in P:
                continue
            bs = sorted(P[kr])
            pv = np.array([P[kr][b] for b in bs])
            eff, K = MET[kr]
            f = feats_of(pv, eff, K, ACC.get(kr, float("nan")))
            if not np.isfinite(list(f.values())).all():
                continue
            # four endpoint arms on the SAME branch support
            end = {}
            for a in ARMS:
                ka = (a, "NA" if a == "trunk" else str(root), ENDPOINT, q)
                if ka in P and sorted(P[ka]) == bs:
                    end[a] = np.array([P[ka][b] for b in bs])
            if len(end) < 2:
                continue
            w = [bs[int(np.argmax(end[a]))] for a in sorted(end)]
            recs.append({
                "question_id": q, "root": root, "horizon": ENDPOINT - root,
                "n_arms": len(end),
                **{c: f[c] for c in FEATS},
                "modal_disagreement": int(len(set(w)) > 1),
                "endpoint_pairwise_js": float(np.mean(
                    [js(end[a], end[b]) for a, b in combinations(sorted(end), 2)])),
                "omnibus_significant": OMNI.get((root, q), ""),
            })
        per_root[root] = recs
        rows += recs

    # ---- the question set common to all four roots, for the joint bootstrap
    common = sorted(set.intersection(*[{r["question_id"] for r in per_root[k]}
                                       for k in ROOTS]))
    rng = np.random.default_rng(SEED)
    B = rng.integers(0, len(common), (N_BOOT, len(common)))   # ONE resample,
    qpos = {q: i for i, q in enumerate(common)}               # shared by roots

    # ================================================ Part 3: raw margin
    summ = []
    for root in ROOTS:
        R = [r for r in per_root[root] if r["question_id"] in set(common)]
        R.sort(key=lambda r: r["question_id"])
        y = np.array([r["modal_disagreement"] for r in R])
        s = np.array([-r["top1_top2_margin"] for r in R])     # score_margin
        jsv = np.array([r["endpoint_pairwise_js"] for r in R])
        idx = np.array([qpos[r["question_id"]] for r in R])
        order = np.argsort(idx)
        y, s, jsv = y[order], s[order], jsv[order]
        a_b, p_b, r_b = [], [], []
        for b in range(N_BOOT):
            sel = B[b]
            a_b.append(auroc(y[sel], s[sel]))
            p_b.append(auprc(y[sel], s[sel]))
            r_b.append(spearmanr(s[sel], jsv[sel]).statistic)
        alo, ahi = ci(a_b); plo, phi = ci(p_b); rlo, rhi = ci(r_b)
        row = {"root": root, "horizon": ENDPOINT - root, "N": len(y),
               "positives": int(y.sum()), "base_rate": round(float(y.mean()), 4),
               "margin_AUROC": round(auroc(y, s), 4),
               "AUROC_CI_low": round(alo, 4), "AUROC_CI_high": round(ahi, 4),
               "AUPRC": round(auprc(y, s), 4),
               "AUPRC_CI_low": round(plo, 4), "AUPRC_CI_high": round(phi, 4),
               "Spearman_JS": round(float(spearmanr(s, jsv).statistic), 4),
               "Spearman_CI_low": round(rlo, 4), "Spearman_CI_high": round(rhi, 4)}
        summ.append(row)
        np.save(OUT / f"bootstrap_auroc_root{root}.npy", np.array(a_b))

    # ============================ Part 4: state-feature model replication
    Xall = np.array([[r[c] for c in FEATS] for r in rows], float)
    yall = np.array([r["modal_disagreement"] for r in rows])
    fid = np.array([fold[r["question_id"]] for r in rows])    # group by question
    rootcol = np.array([r["root"] for r in rows])
    model_rows = []
    for name, cols in (("state_features", STATE), ("controls_only", CONTROLS),
                       ("state_plus_controls", FEATS)):
        j = [FEATS.index(c) for c in cols]
        pb = cv_predict(Xall[:, j], yall, fid, "clf")
        model_rows.append({"scope": "pooled_4_roots", "feature_set": name,
                           "n": len(yall), "positives": int(yall.sum()),
                           "AUROC": round(auroc(yall, pb), 4),
                           "AUPRC": round(auprc(yall, pb), 4)})
        for root in ROOTS:
            m = rootcol == root
            npos, nneg = int(yall[m].sum()), int((1 - yall[m]).sum())
            model_rows.append({
                "scope": f"root_{root}", "feature_set": name, "n": int(m.sum()),
                "positives": npos,
                "AUROC": round(auroc(yall[m], pb[m]), 4),
                "AUPRC": round(auprc(yall[m], pb[m]), 4),
                "unstable_flag": "UNSTABLE (<10 in a class)"
                                 if min(npos, nneg) < 10 else ""})

    # ==================================================== Part 6: robustness
    rob = []
    for root in ROOTS:
        R = sorted([r for r in per_root[root] if r["question_id"] in set(common)],
                   key=lambda r: r["question_id"])
        base_s = np.array([-r["top1_top2_margin"] for r in R])
        # (1) leave-one-arm-out
        for drop in ARMS:
            keep = [a for a in ARMS if a != drop]
            yy = []
            for r in R:
                q = r["question_id"]
                bs = sorted(P[("trunk", "NA", root, q)])
                w = []
                for a in keep:
                    ka = (a, "NA" if a == "trunk" else str(root), ENDPOINT, q)
                    if ka in P and sorted(P[ka]) == bs:
                        w.append(bs[int(np.argmax(
                            np.array([P[ka][b] for b in bs])))])
                yy.append(int(len(set(w)) > 1) if len(w) >= 2 else -1)
            yy = np.array(yy)
            m = yy >= 0
            rob.append({"root": root, "check": f"leave_out_{drop}",
                        "n": int(m.sum()), "positives": int(yy[m].sum()),
                        "margin_AUROC": round(auroc(yy[m], base_s[m]), 4)})
        # (2) near-tie sensitivity, Track-D threshold
        mg = np.array([r["top1_top2_margin"] for r in R])
        y = np.array([r["modal_disagreement"] for r in R])
        k = mg >= TIE_NEAR
        rob.append({"root": root, "check": f"exclude_margin_lt_{TIE_NEAR}",
                    "n": int(k.sum()), "positives": int(y[k].sum()),
                    "margin_AUROC": round(auroc(y[k], base_s[k]), 4)})
        # (3) omnibus significance as an alternative endpoint target
        oz = [r["omnibus_significant"] for r in R]
        if all(v != "" for v in oz):
            yo = np.array([int(v) for v in oz])
            rob.append({"root": root, "check": "target=omnibus_significant",
                        "n": len(yo), "positives": int(yo.sum()),
                        "margin_AUROC": round(auroc(yo, base_s), 4)})

    # ------------------------------------------------------------- write
    with (OUT / "trackA_prefork_instability_by_question.csv").open("w", newline="") as f:
        cols = ["question_id", "root", "horizon", "n_arms", "top1_mass",
                "top1_top2_margin", "normalized_entropy", "num_feasible_branches",
                "pass_at_1", "effective_feasible_count", "modal_disagreement",
                "endpoint_pairwise_js", "omnibus_significant"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    with (OUT / "trackA_prefork_instability_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)
    with (OUT / "trackA_state_model_metrics.csv").open("w", newline="") as f:
        ks = sorted({k for r in model_rows for k in r})
        w = csv.DictWriter(f, fieldnames=ks); w.writeheader(); w.writerows(model_rows)
    with (OUT / "trackA_robustness.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rob[0])); w.writeheader(); w.writerows(rob)

    # ------------------------------------------------- Part 5: horizon plot
    fig = plt.figure(figsize=(3.3, 2.15))
    ax = fig.add_axes([0.165, 0.205, 0.815, 0.735])
    h = [r["horizon"] for r in summ]
    a = [r["margin_AUROC"] for r in summ]
    lo = [r["margin_AUROC"] - r["AUROC_CI_low"] for r in summ]
    hi = [r["AUROC_CI_high"] - r["margin_AUROC"] for r in summ]
    ax.errorbar(h, a, yerr=[lo, hi], fmt="o-", color="#2b6897", lw=1.5, ms=4.0,
                capsize=2.5, elinewidth=1.0, zorder=4, label="Track A (this work)")
    ax.axhline(0.5, color="#8a8a8a", lw=0.8, ls="--", zorder=2)
    ax.text(103, 0.505, "chance", fontsize=6.5, color="#8a8a8a", ha="right",
            va="bottom")
    ax.plot([25], [0.8044], "s", color="#ac6935", ms=4.5, zorder=5,
            label="Track D reference")
    ax.set_xticks([25, 50, 75, 100])
    ax.set_xlim(12, 112)
    ax.set_xlabel("horizon to endpoint (training updates)", fontsize=8)
    ax.set_ylabel("AUROC, $-$(top1$-$top2) margin", fontsize=8)
    ax.tick_params(labelsize=7)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(color="#e2e2e2", lw=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"trackA_prefork_instability_horizon.{ext}", dpi=200)
    plt.close(fig)

    man = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "seed": SEED, "n_bootstrap": N_BOOT, "arms": ARMS, "roots": ROOTS,
           "endpoint": ENDPOINT, "n_common_questions": len(common),
           "inputs": {str(PMF / "branch_state_long.csv"):
                      sha(PMF / "branch_state_long.csv"),
                      str(PMF / "branch_state_summary.csv"):
                      sha(PMF / "branch_state_summary.csv"),
                      str(R9 / "fold_assignment.json"):
                      sha(R9 / "fold_assignment.json"),
                      str(BK4 / "balanced_omnibus.json"):
                      sha(BK4 / "balanced_omnibus.json")
                      if (BK4 / "balanced_omnibus.json").exists() else "MISSING"},
           "code_sha256": {"ta_prefork.py": sha(__file__)}}
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))

    print(f"\n  common questions across all four roots: {len(common)}\n")
    print(f"  {'root':>5} {'horiz':>6} {'N':>4} {'pos':>4} {'base':>6} "
          f"{'AUROC':>7} {'95% CI':>18} {'AUPRC':>7} {'Spearman(JS)':>22}")
    for r in summ:
        print(f"  {r['root']:>5} {r['horizon']:>6} {r['N']:>4} {r['positives']:>4} "
              f"{r['base_rate']:>6.3f} {r['margin_AUROC']:>7.4f} "
              f"[{r['AUROC_CI_low']:>6.4f}, {r['AUROC_CI_high']:>6.4f}] "
              f"{r['AUPRC']:>7.4f} "
              f"{r['Spearman_JS']:>7.4f} [{r['Spearman_CI_low']:>6.3f},"
              f"{r['Spearman_CI_high']:>6.3f}]")
    print(f"\n  wall {time.time() - t0:.0f}s   ->  {OUT}")


if __name__ == "__main__":
    main()
