#!/usr/bin/env python3
"""P7 pre-fork instability prediction, P8 winner lock-in, P9 entropy -> innovation.

These are about state-dependent *training susceptibility*, which is a different
question from the already-established fact that finite-rollout *measurement*
noise depends on branch entropy. The two are never combined here.

Everything uses the frozen question-level folds, standardises features inside
training folds only, and never lets a post-fork quantity into a pre-fork model.
"""
import csv, hashlib, json, sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr

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
ALPHA = 0.5
STEPS = [100, 105, 110, 115, 120, 125]
CONTS = [f"seed{i}" for i in range(1, 6)]
N_BOOT = 2000
SEED = 20260926
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

FEATS = ["normalized_entropy", "top1_mass", "top1_top2_margin",
         "num_feasible_branches", "pass_at_1", "effective_feasible_count"]
CONTROLS = ["num_feasible_branches", "pass_at_1", "effective_feasible_count"]
STATE = ["normalized_entropy", "top1_mass", "top1_top2_margin"]


def js(pa, pb):
    m = 0.5 * (pa + pb)
    return float(max(0.0, 0.5 * np.sum(pa * np.log(pa / m)) + 0.5 * np.sum(pb * np.log(pb / m))))


def load_states():
    P, MET = {}, {}
    for r in csv.DictReader((PMF / "tables" / "branch_state_long.csv").open()):
        if r["track"] != "D":
            continue
        k = (r["continuation_id"], int(r["checkpoint"]), r["question_id"])
        P.setdefault(k, {})[r["branch"]] = float(r["smoothed_probability"])
        MET[k] = (int(r["effective_feasible_count"]), int(r["num_feasible_branches"]))
    ACC = {}
    for r in csv.DictReader((PMF / "tables" / "branch_state_summary.csv").open()):
        if r["track"] == "D":
            ACC[(r["continuation_id"], int(r["checkpoint"]), r["question_id"])] = float(r["accuracy"])
    return P, MET, ACC


def feats_of(pv, eff, K, acc):
    p = np.sort(pv)[::-1]
    H = float(-(pv * np.log(pv)).sum() / np.log(len(pv))) if len(pv) > 1 else 0.0
    return {"normalized_entropy": H, "top1_mass": float(p[0]),
            "top1_top2_margin": float(p[0] - p[1]) if len(p) > 1 else 1.0,
            "num_feasible_branches": float(K), "pass_at_1": float(acc),
            "effective_feasible_count": float(eff)}


def cv_predict(X, y, fold_id, kind):
    """out-of-fold predictions; standardisation fitted on the training folds only."""
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


def boot_ci(fn, n, rng, reps=N_BOOT):
    v = []
    for _ in range(reps):
        b = rng.integers(0, n, n)
        try:
            v.append(fn(b))
        except Exception:
            pass
    return (float(np.quantile(v, .025)), float(np.quantile(v, .975))) if v else (float("nan"),) * 2


def main():
    P, MET, ACC = load_states()
    fold = json.load((R9 / "fold_assignment.json").open())["assignment"]
    rng = np.random.default_rng(SEED)
    qs = sorted({k[2] for k in P if k[0] == "base"})
    rows_pre, rows_post, lock, ent_inn = [], [], [], []

    # ---------------- P7 pre-fork -----------------------------------------
    def build(at_step, arm):
        X, yb, yc, keep = [], [], [], []
        for q in qs:
            k = (arm if at_step != STEPS[0] else "base", at_step, q)
            if k not in P:
                continue
            bs = sorted(P[k]); pv = np.array([P[k][b] for b in bs])
            eff, K = MET[k]
            f = feats_of(pv, eff, K, ACC.get(k, float("nan")))
            if not np.isfinite(list(f.values())).all():
                continue
            end = {s: np.array([P[(s, STEPS[-1], q)][b] for b in bs])
                   for s in CONTS if (s, STEPS[-1], q) in P}
            if len(end) < 2:
                continue
            w = [bs[int(np.argmax(end[s]))] for s in sorted(end)]
            X.append([f[c] for c in FEATS])
            yb.append(int(len(set(w)) > 1))
            yc.append(float(np.mean([js(end[a], end[b]) for a, b in
                                     combinations(sorted(end), 2)])))
            keep.append(q)
        return np.array(X), np.array(yb), np.array(yc), keep

    for tag, at_step, arm, sink in (("prefork", STEPS[0], "base", rows_pre),
                                    ("postfork_first_checkpoint", STEPS[1], "seed1", rows_post)):
        X, yb, yc, keep = build(at_step, arm)
        fid = np.array([fold[q] for q in keep])
        for name, cols in (("state_features", STATE), ("controls_only", CONTROLS),
                           ("state_plus_controls", FEATS)):
            idx = [FEATS.index(c) for c in cols]
            pb = cv_predict(X[:, idx], yb, fid, "clf")
            pc = cv_predict(X[:, idx], yc, fid, "reg")
            auroc = float(roc_auc_score(yb, pb)) if len(set(yb)) > 1 else float("nan")
            auprc = float(average_precision_score(yb, pb)) if len(set(yb)) > 1 else float("nan")
            rho = float(spearmanr(pc, yc).statistic)
            r2 = float(1 - ((yc - pc) ** 2).sum() / ((yc - yc.mean()) ** 2).sum())
            lo_a, hi_a = boot_ci(lambda b: roc_auc_score(yb[b], pb[b])
                                 if len(set(yb[b])) > 1 else np.nan, len(yb), rng)
            lo_r, hi_r = boot_ci(lambda b: spearmanr(pc[b], yc[b]).statistic, len(yc), rng)
            sink.append({"analysis": tag, "feature_set": name,
                         "features": "|".join(cols), "n_questions": len(keep),
                         "positive_class_rate": round(float(yb.mean()), 4),
                         "AUROC": round(auroc, 4), "AUROC_ci_low": round(lo_a, 4),
                         "AUROC_ci_high": round(hi_a, 4), "AUPRC": round(auprc, 4),
                         "spearman_rho_future_js": round(rho, 4),
                         "spearman_ci_low": round(lo_r, 4), "spearman_ci_high": round(hi_r, 4),
                         "cv_r2_future_js": round(r2, 4),
                         "folds": "frozen question-level 5-fold",
                         "leakage_check": "features taken only at step %d; targets only at step %d"
                                          % (at_step, STEPS[-1])})
            print(f"  P7 {tag:26} {name:20} AUROC {auroc:.3f} [{lo_a:.3f},{hi_a:.3f}]  "
                  f"rho {rho:+.3f} [{lo_r:+.3f},{hi_r:+.3f}]  R2 {r2:+.3f}  "
                  f"(pos rate {yb.mean():.3f}, n {len(keep)})")
    for nm, dat in (("instability_prediction_prefork.csv", rows_pre),
                    ("instability_prediction_postfork.csv", rows_post)):
        with (OUT / "tables" / nm).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dat[0])); w.writeheader(); w.writerows(dat)
    with (OUT / "figure_data" / "instability_prediction.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_pre[0]))
        w.writeheader(); w.writerows(rows_pre + rows_post)

    # ---------------- P8 winner lock-in ------------------------------------
    Xs, ys, gid, marg = [], [], [], []
    for q in qs:
        for t in STEPS[:-1]:
            for s in CONTS:
                k = (s if t != STEPS[0] else "base", t, q)
                ke = (s, STEPS[-1], q)
                if k not in P or ke not in P:
                    continue
                bs = sorted(P[k]); pv = np.array([P[k][b] for b in bs])
                eff, K = MET[k]
                f = feats_of(pv, eff, K, ACC.get(k, float("nan")))
                win_now = bs[int(np.argmax(pv))]
                win_end = bs[int(np.argmax([P[ke][b] for b in bs]))]
                Xs.append([f["top1_top2_margin"], f["normalized_entropy"]])
                ys.append(int(win_now == win_end)); gid.append(q)
                marg.append((f["top1_top2_margin"], int(win_now == win_end), t))
    Xs = np.array(Xs); ys = np.array(ys); fid = np.array([fold[q] for q in gid])
    for name, cols in (("margin_only", [0]), ("entropy_only", [1]), ("margin_plus_entropy", [0, 1])):
        pr = cv_predict(Xs[:, cols], ys, fid, "clf")
        au = float(roc_auc_score(ys, pr))
        lo, hi = boot_ci(lambda b: roc_auc_score(ys[b], pr[b])
                         if len(set(ys[b])) > 1 else np.nan, len(ys), rng)
        mu, sd = Xs[:, cols].mean(0), np.where(Xs[:, cols].std(0) < 1e-12, 1, Xs[:, cols].std(0))
        m = LogisticRegression(C=1.0, max_iter=2000).fit((Xs[:, cols] - mu) / sd, ys)
        cf = m.coef_[0]
        cb = []
        for _ in range(400):
            b = rng.integers(0, len(ys), len(ys))
            if len(set(ys[b])) < 2:
                continue
            mb = LogisticRegression(C=1.0, max_iter=2000).fit((Xs[b][:, cols] - mu) / sd, ys[b])
            cb.append(mb.coef_[0][0])
        lock.append({"model": name, "features": "|".join(["top1_top2_margin", "normalized_entropy"][c]
                                                         for c in cols),
                     "n_state_checkpoints": len(ys),
                     "winner_survival_rate": round(float(ys.mean()), 4),
                     "AUROC": round(au, 4), "AUROC_ci_low": round(lo, 4),
                     "AUROC_ci_high": round(hi, 4),
                     "logistic_coefficient_first_feature": round(float(cf[0]), 4),
                     "coefficient_ci_low": round(float(np.quantile(cb, .025)), 4) if cb else "NA",
                     "coefficient_ci_high": round(float(np.quantile(cb, .975)), 4) if cb else "NA",
                     "standardisation": "z-scored; CV predictions standardised within training folds",
                     "folds": "frozen question-level 5-fold"})
        print(f"  P8 {name:20} AUROC {au:.3f} [{lo:.3f},{hi:.3f}]  coef "
              f"{float(cf[0]):+.3f}  survival {ys.mean():.3f}  n {len(ys)}")
    mv = np.array([m[0] for m in marg]); sv = np.array([m[1] for m in marg])
    qb = np.quantile(mv, np.linspace(0, 1, 7))
    bins = []
    for i in range(6):
        sel = (mv >= qb[i]) & (mv <= qb[i + 1] if i == 5 else mv < qb[i + 1])
        if sel.sum():
            bins.append({"margin_bin": f"[{qb[i]:.3f},{qb[i+1]:.3f})", "bin_index": i,
                         "n": int(sel.sum()), "mean_margin": round(float(mv[sel].mean()), 4),
                         "winner_survival_rate": round(float(sv[sel].mean()), 4)})
    with (OUT / "tables" / "winner_lockin.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lock[0])); w.writeheader(); w.writerows(lock)
    with (OUT / "figure_data" / "winner_survival_by_margin.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(bins[0])); w.writeheader(); w.writerows(bins)
    print("  P8 survival by margin sextile: " +
          " ".join(f"{b['mean_margin']:.2f}->{b['winner_survival_rate']:.2f}" for b in bins))

    # ---------------- P9 entropy -> corrected innovation --------------------
    cells = defaultdict(dict)
    for r in csv.DictReader((ART / "section4_stakeholder_resolution" / "tables" /
                             "residual_decomposition_1024_long.csv").open()):
        if r["transition_model"] != "R1":
            continue
        cells[(r["question_id"], int(r["transition_start"]))] = {
            "eta_corr": float(r["measurement_corrected_seed_specific_energy"])}
    X9, y9, t9, q9 = [], [], [], []
    for (q, t), d in cells.items():
        k = ("base" if t == STEPS[0] else CONTS[0], t, q)
        if k not in P:
            continue
        bs = sorted(P[k]); pv = np.array([P[k][b] for b in bs])
        eff, K = MET[k]
        f = feats_of(pv, eff, K, ACC.get(k, float("nan")))
        X9.append([f["normalized_entropy"], float(K)]); y9.append(d["eta_corr"])
        t9.append(t); q9.append(q)
    X9 = np.array(X9); y9 = np.array(y9); t9 = np.array(t9)
    z = (X9[:, 0] - X9[:, 0].mean()) / X9[:, 0].std()
    A = np.c_[np.ones(len(z)), z, (X9[:, 1] - X9[:, 1].mean()) / X9[:, 1].std()]
    beta = np.linalg.lstsq(A, y9, rcond=None)[0]
    qlist = sorted(set(q9)); qi = defaultdict(list)
    for i, q in enumerate(q9):
        qi[q].append(i)
    bb = []
    for _ in range(N_BOOT):
        b = rng.integers(0, len(qlist), len(qlist))
        sel = [j for i in b for j in qi[qlist[i]]]
        try:
            bb.append(np.linalg.lstsq(A[sel], y9[sel], rcond=None)[0][1])
        except Exception:
            pass
    per_t = {}
    for t in sorted(set(t9)):
        m = t9 == t
        if m.sum() > 10:
            per_t[int(t)] = float(np.linalg.lstsq(A[m], y9[m], rcond=None)[0][1])
    qb9 = np.quantile(X9[:, 0], np.linspace(0, 1, 6))
    mb = []
    for i in range(5):
        s_ = (X9[:, 0] >= qb9[i]) & (X9[:, 0] <= qb9[i + 1] if i == 4 else X9[:, 0] < qb9[i + 1])
        if s_.sum():
            mb.append(f"H~{X9[s_, 0].mean():.2f}: eta {y9[s_].mean():.3f}")
    ent_inn.append({"target": "measurement_corrected_continuation_specific_energy",
                    "transition_model": "R1 (state only)", "n_cells": len(y9),
                    "coefficient_per_sd_entropy": round(float(beta[1]), 6),
                    "ci_low": round(float(np.quantile(bb, .025)), 6),
                    "ci_high": round(float(np.quantile(bb, .975)), 6),
                    "control": "number of feasible branches (z-scored)",
                    "per_transition_coefficients": json.dumps({k: round(v, 4)
                                                               for k, v in per_t.items()}),
                    "per_transition_signs": "".join("+" if v > 0 else "-" for v in per_t.values()),
                    "marginal_bins": " | ".join(mb),
                    "mean_eta_corrected": round(float(y9.mean()), 6)})
    with (OUT / "tables" / "entropy_innovation.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ent_inn[0])); w.writeheader(); w.writerows(ent_inn)
    print(f"  P9 coefficient per SD entropy {beta[1]:+.4f} "
          f"[{np.quantile(bb, .025):+.4f}, {np.quantile(bb, .975):+.4f}]; "
          f"per-transition signs {ent_inn[0]['per_transition_signs']}")
    print("  P9 bins: " + " | ".join(mb))
    for f in ("tables/instability_prediction_prefork.csv",
              "tables/instability_prediction_postfork.csv", "tables/winner_lockin.csv",
              "tables/entropy_innovation.csv", "figure_data/instability_prediction.csv",
              "figure_data/winner_survival_by_margin.csv"):
        print(f"  {f} sha256 {sha(OUT / f)}")


if __name__ == "__main__":
    main()
