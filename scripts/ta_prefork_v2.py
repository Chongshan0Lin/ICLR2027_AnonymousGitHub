#!/usr/bin/env python3
"""Track-A pre-fork instability, v2: dump every bootstrap draw + paper tables.

This changes NOTHING about the analysis. Estimator, target, tie rule, feature
definitions, arm set, seed and draw count are byte-identical to
`<scripts>/ta_prefork.py`; the RNG is consumed in the same order (one
`default_rng(SEED)` followed by exactly one `integers(...)` call), so the
resampled question indices are the same matrix v1 used.

What is new:
  * the AUPRC and Spearman bootstrap draws are persisted, not just AUROC
  * the question-index draw matrix itself is persisted, so every CI can be
    rebuilt without re-running any RNG
  * v1's point estimates and AUROC CIs are re-derived and asserted to match
  * a paper-ready CSV and LaTeX fragment are emitted

No training, generation or model inference. v1 is not touched.
"""
import csv, hashlib, json, time
from itertools import combinations
from pathlib import Path
import numpy as np
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


PMF = ART / "paper_main_figures" / "tables"
BK4 = ART / "analysis_balanced_k4"
V1 = ART / "trackA_prefork_instability_v1"
OUT = _mkout(OUTPUTS / "trackA_prefork_instability_v2")
ROOTS = [75, 100, 125, 150]
ARMS = ["trunk", "seed1", "seed2", "seed3"]
ENDPOINT = 175
N_BOOT = 10000
SEED = 20260926
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
FEATS = ["normalized_entropy", "top1_mass", "top1_top2_margin",
         "num_feasible_branches", "pass_at_1", "effective_feasible_count"]


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


def auroc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")


def auprc(y, s):
    return float(average_precision_score(y, s)) if len(set(y)) > 1 else float("nan")


def ci(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    return ((float(np.quantile(v, .025)), float(np.quantile(v, .975)))
            if v.size else (float("nan"), float("nan")))


def load():
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
    OMNI = {}
    try:
        bo = json.load((BK4 / "balanced_omnibus.json").open())
        for root in ROOTS:
            for q, d in bo["arms"]["4"][str(root)]["blocked"]["per_prompt"].items():
                OMNI[(root, q)] = int(bool(d["reject"]))
    except Exception as e:
        print(f"  omnibus join unavailable: {e}")

    # ---- identical construction to v1 -----------------------------------
    qs = sorted({k[3] for k in P if k[0] == "trunk" and k[2] == ENDPOINT})
    per_root = {}
    for root in ROOTS:
        recs = []
        for q in qs:
            kr = ("trunk", "NA", root, q)
            if kr not in P:
                continue
            bs = sorted(P[kr])
            pv = np.array([P[kr][b] for b in bs])
            eff, K = MET[kr]
            f = feats_of(pv, eff, K, ACC.get(kr, float("nan")))
            if not np.isfinite(list(f.values())).all():
                continue
            end = {}
            for a in ARMS:
                ka = (a, "NA" if a == "trunk" else str(root), ENDPOINT, q)
                if ka in P and sorted(P[ka]) == bs:
                    end[a] = np.array([P[ka][b] for b in bs])
            if len(end) < 2:
                continue
            w = [bs[int(np.argmax(end[a]))] for a in sorted(end)]
            recs.append({"question_id": q, "root": root,
                         "horizon": ENDPOINT - root, **{c: f[c] for c in FEATS},
                         "modal_disagreement": int(len(set(w)) > 1),
                         "endpoint_pairwise_js": float(np.mean(
                             [js(end[a], end[b])
                              for a, b in combinations(sorted(end), 2)])),
                         "omnibus_significant": OMNI.get((root, q), "")})
        per_root[root] = recs

    common = sorted(set.intersection(*[{r["question_id"] for r in per_root[k]}
                                       for k in ROOTS]))
    rng = np.random.default_rng(SEED)                      # same order as v1
    B = rng.integers(0, len(common), (N_BOOT, len(common)))
    qpos = {q: i for i, q in enumerate(common)}
    np.save(OUT / "bootstrap_question_index_draws.npy", B)
    (OUT / "bootstrap_question_order.json").write_text(json.dumps(
        {"note": "row i of bootstrap_question_index_draws.npy indexes into this "
                 "list; the SAME matrix is applied to every root, which is what "
                 "preserves cross-root dependence",
         "seed": SEED, "n_boot": N_BOOT, "questions_in_index_order": common},
        indent=1))

    rows, summ = [], []
    for root in ROOTS:
        R = sorted([r for r in per_root[root] if r["question_id"] in set(common)],
                   key=lambda r: r["question_id"])
        idx = np.array([qpos[r["question_id"]] for r in R])
        order = np.argsort(idx)
        R = [R[i] for i in order]
        rows += R
        y = np.array([r["modal_disagreement"] for r in R])
        s = np.array([-r["top1_top2_margin"] for r in R])
        jsv = np.array([r["endpoint_pairwise_js"] for r in R])
        a_b = np.empty(N_BOOT); p_b = np.empty(N_BOOT); r_b = np.empty(N_BOOT)
        for b in range(N_BOOT):
            sel = B[b]
            a_b[b] = auroc(y[sel], s[sel])
            p_b[b] = auprc(y[sel], s[sel])
            r_b[b] = spearmanr(s[sel], jsv[sel]).statistic
        np.save(OUT / f"bootstrap_auroc_root{root}.npy", a_b)
        np.save(OUT / f"bootstrap_auprc_root{root}.npy", p_b)
        np.save(OUT / f"bootstrap_spearman_root{root}.npy", r_b)
        alo, ahi = ci(a_b); plo, phi = ci(p_b); rlo, rhi = ci(r_b)
        summ.append({"root": root, "horizon": ENDPOINT - root, "N": len(y),
                     "positives": int(y.sum()),
                     "base_rate": round(float(y.mean()), 4),
                     "margin_AUROC": round(auroc(y, s), 4),
                     "margin_AUROC_CI_low": round(alo, 4),
                     "margin_AUROC_CI_high": round(ahi, 4),
                     "margin_AUPRC": round(auprc(y, s), 4),
                     "margin_AUPRC_CI_low": round(plo, 4),
                     "margin_AUPRC_CI_high": round(phi, 4),
                     "spearman_endpoint_JS": round(
                         float(spearmanr(s, jsv).statistic), 4),
                     "spearman_CI_low": round(rlo, 4),
                     "spearman_CI_high": round(rhi, 4)})

    # ---- Task 1 verification against v1 ---------------------------------
    v1 = {int(r["root"]): r for r in
          csv.DictReader((V1 / "trackA_prefork_instability_summary.csv").open())}
    checks, bad = [], []
    for r in summ:
        o = v1[r["root"]]
        for new, old in (("margin_AUROC", "margin_AUROC"),
                         ("margin_AUROC_CI_low", "AUROC_CI_low"),
                         ("margin_AUROC_CI_high", "AUROC_CI_high"),
                         ("margin_AUPRC", "AUPRC"),
                         ("spearman_endpoint_JS", "Spearman_JS"),
                         ("spearman_CI_low", "Spearman_CI_low"),
                         ("spearman_CI_high", "Spearman_CI_high"),
                         ("positives", "positives"), ("N", "N")):
            a, b_ = str(r[new]), str(o[old])
            ok = abs(float(a) - float(b_)) < 5e-5
            checks.append({"root": r["root"], "quantity": new, "v1": b_,
                           "v2": a, "match": "yes" if ok else "NO"})
            if not ok:
                bad.append((r["root"], new, b_, a))
    # v1 also saved AUROC draws: compare the arrays element-wise
    for root in ROOTS:
        f1 = V1 / f"bootstrap_auroc_root{root}.npy"
        if f1.exists():
            same = bool(np.array_equal(np.load(f1),
                                       np.load(OUT / f"bootstrap_auroc_root{root}.npy")))
            checks.append({"root": root, "quantity": "auroc_draws_array",
                           "v1": "10000 draws", "v2": "10000 draws",
                           "match": "yes (element-wise identical)" if same else "NO"})
            if not same:
                bad.append((root, "auroc_draws_array", "", ""))
    with (OUT / "v1_v2_verification.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(checks[0]))
        w.writeheader(); w.writerows(checks)
    if bad:
        print("  !! MISMATCH vs v1 -- STOPPING")
        for b_ in bad:
            print("    ", b_)
        raise SystemExit(1)
    print(f"  v1 vs v2: {len(checks)}/{len(checks)} checks match")

    # ---- Task 2 paper table ---------------------------------------------
    cols = ["root", "horizon", "N", "positives", "base_rate", "margin_AUROC",
            "margin_AUROC_CI_low", "margin_AUROC_CI_high", "margin_AUPRC",
            "margin_AUPRC_CI_low", "margin_AUPRC_CI_high",
            "spearman_endpoint_JS", "spearman_CI_low", "spearman_CI_high"]
    with (OUT / "trackA_prefork_instability_paper_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(summ)

    tex = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\caption{\textbf{Pre-fork branch margin predicts endpoint modal "
           r"disagreement across Track-A fork roots.} Each root is an independent "
           r"full-state fork of the same trunk, evaluated with four balanced arms "
           r"at step 175. The score is the raw negative top-1 minus top-2 margin "
           r"at the shared pre-fork state; no threshold is fitted. Intervals are "
           r"question-level bootstrap percentiles ($10{,}000$ draws, one resample "
           r"shared across roots).}",
           r"\label{tab:trackA-instability-horizons}",
           r"\begin{tabular}{rccccc}", r"\toprule",
           r"Root & Horizon & Disagree & AUROC & 95\% CI & Spearman JS \\",
           r"\midrule"]
    for r in summ:
        tex.append(
            f"{r['root']} & {r['horizon']} & {r['positives']}/{r['N']} & "
            f"{r['margin_AUROC']:.3f} & "
            f"[{r['margin_AUROC_CI_low']:.3f}, {r['margin_AUROC_CI_high']:.3f}] & "
            f"{r['spearman_endpoint_JS']:.3f} "
            f"[{r['spearman_CI_low']:.2f}, {r['spearman_CI_high']:.2f}] \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "trackA_prefork_instability_paper_table.tex").write_text("\n".join(tex) + "\n")

    man = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "supersedes": "none -- v1 is retained unmodified",
           "identical_to_v1": "estimator, target, tie rule, features, arm set, "
                              "seed, draw count and RNG consumption order",
           "new_in_v2": "AUPRC and Spearman bootstrap draws persisted; question "
                        "index draw matrix persisted; v1 verification; paper tables",
           "seed": SEED, "n_bootstrap": N_BOOT,
           "n_common_questions": len(common),
           "verification": f"{len(checks)} checks, all match",
           "inputs": {str(PMF / 'branch_state_long.csv'):
                      sha(PMF / 'branch_state_long.csv'),
                      str(PMF / 'branch_state_summary.csv'):
                      sha(PMF / 'branch_state_summary.csv'),
                      str(BK4 / 'balanced_omnibus.json'):
                      sha(BK4 / 'balanced_omnibus.json')},
           "code_sha256": {"ta_prefork_v2.py": sha(__file__),
                           "ta_prefork.py (v1)": sha("<scripts>/ta_prefork.py")}}
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))

    print(f"\n  {'root':>5} {'hor':>4} {'N':>4} {'pos':>4} {'AUROC':>7} "
          f"{'AUROC 95% CI':>18} {'AUPRC':>7} {'AUPRC 95% CI':>18} "
          f"{'rho':>7} {'rho 95% CI':>16}")
    for r in summ:
        print(f"  {r['root']:>5} {r['horizon']:>4} {r['N']:>4} {r['positives']:>4} "
              f"{r['margin_AUROC']:>7.4f} "
              f"[{r['margin_AUROC_CI_low']:>6.4f},{r['margin_AUROC_CI_high']:>6.4f}] "
              f"{r['margin_AUPRC']:>7.4f} "
              f"[{r['margin_AUPRC_CI_low']:>6.4f},{r['margin_AUPRC_CI_high']:>6.4f}] "
              f"{r['spearman_endpoint_JS']:>7.4f} "
              f"[{r['spearman_CI_low']:>5.3f},{r['spearman_CI_high']:>5.3f}]")
    print(f"\n  wall {time.time() - t0:.0f}s  ->  {OUT}")


if __name__ == "__main__":
    main()
