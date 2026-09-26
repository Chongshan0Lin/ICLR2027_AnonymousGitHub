#!/usr/bin/env python3
"""P0: measurement-confound control for the Track-A modal-disagreement result.

Could "pre-fork margin predicts endpoint modal disagreement" be partly an
artifact of finite-rollout uncertainty in the ESTIMATED endpoint modal branch?

Matched budget throughout. The frozen Track-A endpoint evaluation is 256
rollouts per problem per arm; the branch distribution is conditional on the
parseable, solver-feasible subset, whose size is `effective_feasible_count`
(median 249 of 256). Each arm's counts are split into two disjoint halves by a
multivariate hypergeometric draw WITHOUT replacement -- the same estimator the
frozen split-half floor uses in meta_build.m2_data, not a bootstrap with
replacement.

  within-arm  : do the two halves of ONE arm disagree on the modal branch?
  cross-arm   : do the four arms' first halves all share one modal branch?

Both are evaluated at the same half budget, so they are directly comparable.

Two exclusion rules are declared here, BEFORE any AUROC is computed, and both
are reported. Neither threshold is tuned:
  ANY       measurement-unstable if some arm's halves disagree in ANY replicate
  MAJORITY  measurement-unstable if some arm's halves disagree in > half of them

Tie handling is identical to the frozen Track-A analysis: modal branch is
`bs[argmax(p)]` over the lexicographically sorted branch list, so an exact tie
resolves to the first branch; ties are never dropped.

No training, generation or inference. Writes only to its own new directory.
"""
import csv, hashlib, json, time
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
V1 = ART / "trackA_prefork_instability_v1"
V2 = ART / "trackA_prefork_instability_v2"
OUT = _mkout(OUTPUTS / "trackA_measurement_control_v1")
ROOTS = [75, 100, 125, 150]
ARMS = ["trunk", "seed1", "seed2", "seed3"]
ENDPOINT, ALPHA = 175, 0.5
N_SPLIT, SPLIT_SEED = 100, 20260927
N_BOOT, BOOT_SEED = 10000, 20260926
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
auroc = lambda y, s: float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")
auprc = lambda y, s: float(average_precision_score(y, s)) if len(set(y)) > 1 else float("nan")


def ci(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    return ((float(np.quantile(v, .025)), float(np.quantile(v, .975)))
            if v.size else (float("nan"), float("nan")))


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    C, MET = {}, {}
    for r in csv.DictReader((PMF / "branch_state_long.csv").open()):
        if r["track"] != "A" or int(r["checkpoint"]) != ENDPOINT:
            continue
        k = (r["continuation_id"], r["root_step"], r["question_id"])
        C.setdefault(k, {})[r["branch"]] = int(float(r["raw_branch_count"]))
        MET[k] = int(r["total_rollout_budget"])
    frozen = {}
    for r in csv.DictReader(
            (V1 / "trackA_prefork_instability_by_question.csv").open()):
        frozen[(int(r["root"]), r["question_id"])] = (
            float(r["top1_top2_margin"]), int(r["modal_disagreement"]),
            float(r["endpoint_pairwise_js"]))

    rng = np.random.default_rng(SPLIT_SEED)
    per_q, matched, skipped = [], [], []
    for root in ROOTS:
        recs = []
        for (rt, q) in sorted([k for k in frozen if k[0] == root], key=str):
            keys = {a: (a, "NA" if a == "trunk" else str(root), q) for a in ARMS}
            if not all(k in C for k in keys.values()):
                skipped.append({"root": root, "question_id": q, "reason": "arm missing"})
                continue
            bs = sorted(C[keys["trunk"]])
            if any(sorted(C[k]) != bs for k in keys.values()):
                skipped.append({"root": root, "question_id": q, "reason": "support mismatch"})
                continue
            cnt = {a: np.array([C[keys[a]][b] for b in bs], dtype=np.int64) for a in ARMS}
            if any(int(cnt[a].sum()) < 4 for a in ARMS):
                skipped.append({"root": root, "question_id": q,
                                "reason": "effective_feasible_count < 4 in some arm"})
                continue
            within = {a: 0 for a in ARMS}
            cross = 0
            for _ in range(N_SPLIT):
                first = {}
                for a in ARMS:
                    n = int(cnt[a].sum())
                    h1 = rng.multivariate_hypergeometric(cnt[a], n // 2)
                    h2 = cnt[a] - h1
                    p1 = (h1 + ALPHA) / (h1.sum() + ALPHA * len(bs))
                    p2 = (h2 + ALPHA) / (h2.sum() + ALPHA * len(bs))
                    if bs[int(np.argmax(p1))] != bs[int(np.argmax(p2))]:
                        within[a] += 1
                    first[a] = bs[int(np.argmax(p1))]
                if len(set(first.values())) > 1:
                    cross += 1
            wmax = max(within.values())
            mg, y, jsv = frozen[(root, q)]
            recs.append({
                "question_id": q, "root": root, "horizon": ENDPOINT - root,
                "nominal_budget_per_arm": MET[keys["trunk"]],
                "effective_n_per_arm": "|".join(str(int(cnt[a].sum())) for a in ARMS),
                "half_budget_per_arm": "|".join(str(int(cnt[a].sum()) // 2) for a in ARMS),
                **{f"within_flips_{a}": within[a] for a in ARMS},
                "within_flip_rate_max_arm": round(wmax / N_SPLIT, 4),
                "within_flip_rate_mean_arm": round(
                    float(np.mean(list(within.values()))) / N_SPLIT, 4),
                "cross_arm_disagree_rate_half_budget": round(cross / N_SPLIT, 4),
                "measurement_unstable_ANY": int(wmax > 0),
                "measurement_unstable_MAJORITY": int(wmax > N_SPLIT // 2),
                "top1_top2_margin": mg, "modal_disagreement_full_budget": y,
                "endpoint_pairwise_js": jsv})
        per_q += recs
        matched.append({
            "root": root, "horizon": ENDPOINT - root, "n_questions": len(recs),
            "mean_within_arm_halfsplit_disagreement_rate": round(float(np.mean(
                [r["within_flip_rate_mean_arm"] for r in recs])), 4),
            "mean_cross_arm_4way_disagreement_rate_half_budget": round(float(np.mean(
                [r["cross_arm_disagree_rate_half_budget"] for r in recs])), 4),
            "full_budget_cross_arm_disagreement_rate": round(float(np.mean(
                [r["modal_disagreement_full_budget"] for r in recs])), 4)})

    summ = []
    brng = np.random.default_rng(BOOT_SEED)
    for rule in ("ANY", "MAJORITY"):
        for root in ROOTS:
            R = sorted([r for r in per_q if r["root"] == root],
                       key=lambda r: r["question_id"])
            keep = [r for r in R if not r[f"measurement_unstable_{rule}"]]
            y0 = np.array([r["modal_disagreement_full_budget"] for r in R])
            s0 = np.array([-r["top1_top2_margin"] for r in R])
            y = np.array([r["modal_disagreement_full_budget"] for r in keep])
            s = np.array([-r["top1_top2_margin"] for r in keep])
            jv = np.array([r["endpoint_pairwise_js"] for r in keep])
            row = {"rule": rule, "root": root, "horizon": ENDPOINT - root,
                   "original_N": len(y0), "original_positives": int(y0.sum()),
                   "original_AUROC": round(auroc(y0, s0), 4),
                   "n_excluded": len(R) - len(keep),
                   "remaining_N": len(keep), "remaining_positives": int(y.sum()),
                   "post_exclusion_AUROC": "", "AUROC_CI_low": "",
                   "AUROC_CI_high": "", "post_exclusion_AUPRC": "",
                   "spearman_endpoint_JS": "", "delta_AUROC": "", "note": ""}
            if len(keep) >= 10 and len(set(y.tolist())) > 1:
                Bi = brng.integers(0, len(keep), (N_BOOT, len(keep)))
                ab = np.array([auroc(y[b], s[b]) for b in Bi])
                lo, hi = ci(ab)
                np.save(OUT / f"bootstrap_auroc_{rule}_root{root}.npy", ab)
                row.update({"post_exclusion_AUROC": round(auroc(y, s), 4),
                            "AUROC_CI_low": round(lo, 4), "AUROC_CI_high": round(hi, 4),
                            "post_exclusion_AUPRC": round(auprc(y, s), 4),
                            "spearman_endpoint_JS": round(
                                float(spearmanr(s, jv).statistic), 4),
                            "delta_AUROC": round(auroc(y, s) - auroc(y0, s0), 4)})
            else:
                row["note"] = ("fewer than 10 problems remain, or one class is empty "
                               "after exclusion -- not computable")
            summ.append(row)

    for nm, rows_ in (("trackA_measurement_control_by_question.csv", per_q),
                      ("trackA_measurement_control_summary.csv", summ),
                      ("trackA_matched_budget_comparison.csv", matched),
                      ("trackA_measurement_control_skipped.csv", skipped or
                       [{"root": "", "question_id": "", "reason": "none"}])):
        with (OUT / nm).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_[0]))
            w.writeheader(); w.writerows(rows_)

    man = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "split_seed": SPLIT_SEED, "n_split_replicates": N_SPLIT,
           "bootstrap_seed": BOOT_SEED, "n_bootstrap": N_BOOT, "alpha": ALPHA,
           "roots": ROOTS, "arms": ARMS, "endpoint": ENDPOINT,
           "split_estimator": "numpy multivariate_hypergeometric without "
                              "replacement; same estimator as the frozen split-half "
                              "floor in meta_build.m2_data",
           "budget_note": "nominal endpoint budget 256 rollouts per problem per arm; "
                          "the branch distribution is conditional on the parseable "
                          "solver-feasible subset (median 249), so the split is of "
                          "that effective count. Within-arm and cross-arm quantities "
                          "use the SAME half budget.",
           "tie_handling": "identical to the frozen Track-A analysis: modal branch = "
                           "bs[argmax(p)] over the lexicographically sorted branch "
                           "list; ties resolve to the first branch, never dropped",
           "measurement_unstable_rules": {
               "ANY": "some arm's halves disagree in >=1 of 100 replicates",
               "MAJORITY": "some arm's halves disagree in >50 of 100 replicates"},
           "rules_declared_before_any_auroc_was_computed": True,
           "n_skipped": len(skipped),
           "inputs": {str(PMF / "branch_state_long.csv"):
                      sha(PMF / "branch_state_long.csv"),
                      str(V1 / "trackA_prefork_instability_by_question.csv"):
                      sha(V1 / "trackA_prefork_instability_by_question.csv"),
                      str(V2 / "trackA_prefork_instability_paper_table.csv"):
                      sha(V2 / "trackA_prefork_instability_paper_table.csv")},
           "code_sha256": {"ta_p0_measctl.py": sha(__file__)}}
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))

    print("\n  matched-budget comparison (both at the same half budget)")
    print(f"  {'root':>5} {'hor':>4} {'n':>4} {'within-arm':>11} {'cross-arm':>10} "
          f"{'full-budget cross':>18}")
    for m in matched:
        print(f"  {m['root']:>5} {m['horizon']:>4} {m['n_questions']:>4} "
              f"{m['mean_within_arm_halfsplit_disagreement_rate']:>11.4f} "
              f"{m['mean_cross_arm_4way_disagreement_rate_half_budget']:>10.4f} "
              f"{m['full_budget_cross_arm_disagreement_rate']:>18.4f}")
    for rule in ("ANY", "MAJORITY"):
        print(f"\n  exclusion rule {rule}")
        print(f"  {'root':>5} {'N0':>4} {'pos0':>5} {'AUROC0':>7} {'excl':>5} "
              f"{'N':>4} {'pos':>4} {'AUROC':>8} {'95% CI':>18} {'dAUROC':>8}")
        for r in [x for x in summ if x["rule"] == rule]:
            pa = r["post_exclusion_AUROC"]
            cis = (f"[{r['AUROC_CI_low']:.4f},{r['AUROC_CI_high']:.4f}]"
                   if pa != "" else "")
            print(f"  {r['root']:>5} {r['original_N']:>4} {r['original_positives']:>5} "
                  f"{r['original_AUROC']:>7.4f} {r['n_excluded']:>5} "
                  f"{r['remaining_N']:>4} {r['remaining_positives']:>4} "
                  f"{str(pa):>8} {cis:>18} {str(r['delta_AUROC']):>8}")
    print(f"\n  skipped states: {len(skipped)}   wall {time.time() - t0:.0f}s  ->  {OUT}")


if __name__ == "__main__":
    main()
