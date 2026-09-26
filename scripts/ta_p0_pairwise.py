#!/usr/bin/env python3
"""P0 addendum: matched PAIRWISE within-arm vs cross-arm modal disagreement.

The v1 control compared a 2-way event (do an arm's two halves disagree?) with a
4-way event (do all four arms share one modal branch?). Those are not the same
event: the 4-way version has six chances to disagree, so it is mechanically
larger even with no real difference between arms. The ratio reported in v1 is
inflated by that asymmetry and must not be used as the primary comparison.

This recomputes the comparison so both sides are the SAME event:

    within-arm pairwise : modal(half1 of arm a) != modal(half2 of arm a)
    cross-arm  pairwise : modal(half1 of arm a) != modal(half1 of arm b),
                          averaged over the 6 unordered pairs

Both sides compare two modal-branch estimates, each from the same ~n/2
effective rollouts, using the identical modal-branch definition. The only
difference is whether the two estimates come from one arm or from two.

The splits are the SAME ones v1 drew: identical seed, identical RNG consumption
order (one multivariate_hypergeometric per arm per replicate, arms in the same
order), with only extra bookkeeping added. This is asserted against v1's
recorded within-arm flip counts.

Note on conservativeness: the two within-arm halves are disjoint draws from one
finite pool, so they are slightly anti-correlated and therefore disagree a
little MORE often than two independent samples of the same size would. That
biases the within-arm side upward, which makes the contrast below conservative.

No training, generation or inference. v1 is not modified.
"""
import csv, hashlib, json, time
from itertools import combinations
from pathlib import Path
import numpy as np

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
V1 = ART / "trackA_measurement_control_v1"
OUT = _mkout(OUTPUTS / "trackA_measurement_control_v2")
ROOTS = [75, 100, 125, 150]
ARMS = ["trunk", "seed1", "seed2", "seed3"]
ENDPOINT, ALPHA = 175, 0.5
N_SPLIT, SPLIT_SEED = 100, 20260927        # identical to v1
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    C = {}
    for r in csv.DictReader((PMF / "branch_state_long.csv").open()):
        if r["track"] != "A" or int(r["checkpoint"]) != ENDPOINT:
            continue
        C.setdefault((r["continuation_id"], r["root_step"], r["question_id"]),
                     {})[r["branch"]] = int(float(r["raw_branch_count"]))
    frozen = {}
    for r in csv.DictReader(
            (Path("<artifacts>/results/"
                  "trackA_prefork_instability_v1")
             / "trackA_prefork_instability_by_question.csv").open()):
        frozen[(int(r["root"]), r["question_id"])] = float(r["top1_top2_margin"])
    v1q = {(int(r["root"]), r["question_id"]): r
           for r in csv.DictReader(
               (V1 / "trackA_measurement_control_by_question.csv").open())}

    rng = np.random.default_rng(SPLIT_SEED)
    per_q, summ = [], []
    mismatch = []
    for root in ROOTS:
        recs = []
        for (rt, q) in sorted([k for k in frozen if k[0] == root], key=str):
            keys = {a: (a, "NA" if a == "trunk" else str(root), q) for a in ARMS}
            if not all(k in C for k in keys.values()):
                continue
            bs = sorted(C[keys["trunk"]])
            if any(sorted(C[k]) != bs for k in keys.values()):
                continue
            cnt = {a: np.array([C[keys[a]][b] for b in bs], dtype=np.int64)
                   for a in ARMS}
            if any(int(cnt[a].sum()) < 4 for a in ARMS):
                continue
            within = {a: 0 for a in ARMS}
            pairw = {p: 0 for p in combinations(ARMS, 2)}
            fourway = 0
            for _ in range(N_SPLIT):
                first = {}
                for a in ARMS:                       # same order, same draws as v1
                    n = int(cnt[a].sum())
                    h1 = rng.multivariate_hypergeometric(cnt[a], n // 2)
                    h2 = cnt[a] - h1
                    p1 = (h1 + ALPHA) / (h1.sum() + ALPHA * len(bs))
                    p2 = (h2 + ALPHA) / (h2.sum() + ALPHA * len(bs))
                    m1, m2 = bs[int(np.argmax(p1))], bs[int(np.argmax(p2))]
                    if m1 != m2:
                        within[a] += 1
                    first[a] = m1
                for (x, y) in pairw:                 # NEW: the 6 cross-arm pairs
                    if first[x] != first[y]:
                        pairw[(x, y)] += 1
                if len(set(first.values())) > 1:
                    fourway += 1
            # the splits must be the ones v1 drew
            v = v1q.get((root, q))
            if v is not None:
                for a in ARMS:
                    if int(v[f"within_flips_{a}"]) != within[a]:
                        mismatch.append((root, q, a, v[f"within_flips_{a}"], within[a]))
            wr = float(np.mean(list(within.values()))) / N_SPLIT
            cr = float(np.mean(list(pairw.values()))) / N_SPLIT
            recs.append({
                "question_id": q, "root": root, "horizon": ENDPOINT - root,
                "effective_n_per_arm": "|".join(str(int(cnt[a].sum())) for a in ARMS),
                "within_arm_pairwise_rate": round(wr, 4),
                "cross_arm_pairwise_rate": round(cr, 4),
                "cross_arm_fourway_rate": round(fourway / N_SPLIT, 4),
                "top1_top2_margin": frozen[(root, q)]})
        per_q += recs
        W = float(np.mean([r["within_arm_pairwise_rate"] for r in recs]))
        X = float(np.mean([r["cross_arm_pairwise_rate"] for r in recs]))
        F = float(np.mean([r["cross_arm_fourway_rate"] for r in recs]))
        summ.append({"root": root, "horizon": ENDPOINT - root,
                     "n_questions": len(recs),
                     "within_arm_pairwise_disagreement": round(W, 4),
                     "cross_arm_pairwise_disagreement": round(X, 4),
                     "difference": round(X - W, 4),
                     "ratio": round(X / W, 2) if W > 0 else "inf",
                     "cross_arm_fourway_disagreement_reference": round(F, 4)})

    if mismatch:
        print(f"  !! {len(mismatch)} within-arm flip counts differ from v1 -- the "
              f"splits are NOT the same draws; STOPPING")
        for m in mismatch[:5]:
            print("    ", m)
        raise SystemExit(1)
    print(f"  split reproduction: within-arm flip counts identical to v1 for all "
          f"{len(per_q)} root-question cells")

    for nm, rows_ in (("trackA_pairwise_measurement_control_by_question.csv", per_q),
                      ("trackA_pairwise_measurement_control_summary.csv", summ)):
        with (OUT / nm).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_[0]))
            w.writeheader(); w.writerows(rows_)

    man = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "supersedes": "the within-arm vs FOUR-WAY comparison in "
                         "trackA_measurement_control_v1, which compared unequal "
                         "event definitions and therefore overstated the contrast",
           "retains": "the four-way half-budget vs full-budget comparison in v1 is "
                      "unaffected: both of those quantities use the same event "
                      "definition and differ only in budget",
           "split_seed": SPLIT_SEED, "n_split_replicates": N_SPLIT, "alpha": ALPHA,
           "event_definition": "both sides are 'two modal-branch estimates, each from "
                               "the same ~n/2 effective rollouts, disagree'; the only "
                               "difference is whether they come from one arm or two",
           "cross_arm_pairs": 6,
           "conservativeness_note": "the two within-arm halves are disjoint draws from "
                                    "one finite pool and are therefore slightly "
                                    "anti-correlated, so they disagree a little more "
                                    "often than two independent equal-size samples "
                                    "would; this biases the within-arm side upward and "
                                    "makes the contrast conservative",
           "split_reproduction_check": "within-arm flip counts identical to v1 for "
                                       f"all {len(per_q)} cells",
           "inputs": {str(PMF / "branch_state_long.csv"):
                      sha(PMF / "branch_state_long.csv"),
                      str(V1 / "trackA_measurement_control_by_question.csv"):
                      sha(V1 / "trackA_measurement_control_by_question.csv")},
           "code_sha256": {"ta_p0_pairwise.py": sha(__file__)}}
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))

    print(f"\n  {'root':>5} {'hor':>4} {'n':>4} {'within-arm':>11} {'cross-arm':>10} "
          f"{'diff':>8} {'ratio':>7} {'(4-way ref)':>12}")
    for s in summ:
        print(f"  {s['root']:>5} {s['horizon']:>4} {s['n_questions']:>4} "
              f"{s['within_arm_pairwise_disagreement']:>11.4f} "
              f"{s['cross_arm_pairwise_disagreement']:>10.4f} "
              f"{s['difference']:>8.4f} {str(s['ratio']):>7} "
              f"{s['cross_arm_fourway_disagreement_reference']:>12.4f}")
    W = [s["within_arm_pairwise_disagreement"] for s in summ]
    X = [s["cross_arm_pairwise_disagreement"] for s in summ]
    R = [s["ratio"] for s in summ if isinstance(s["ratio"], float)]
    print(f"\n  within-arm pairwise {min(W):.4f}-{max(W):.4f} | "
          f"cross-arm pairwise {min(X):.4f}-{max(X):.4f} | "
          f"ratio {min(R):.2f}-{max(R):.2f}x")
    print(f"  wall {time.time() - t0:.0f}s  ->  {OUT}")


if __name__ == "__main__":
    main()
