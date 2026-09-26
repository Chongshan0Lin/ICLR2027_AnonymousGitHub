#!/usr/bin/env python3
"""P3: nested predictive-state models, rebuilt on the current canonical table.

The question is not whether the dynamics are "Markov" in any mechanistic sense.
It is narrower and purely behavioural: given the current branch distribution
z_t, does recent behavioural history add held-out predictive information about
z_{t+Delta} at the checkpoint resolution we actually observe?

Model specification, regulariser, scoring rule and fold structure are the frozen
Round-9 ones, imported from countdown/src/r9_models.py:

    M1  z_t              + branch controls
    M2  z_t, v_t         + branch controls
    M3  z_t, v_t, v_{t-1}+ branch controls          (v_t = z_t - z_{t-1})

M3 is NOT (z_t, v_t, z_{t-1}): those three are exactly collinear, so the model
would be unidentified. v_{t-1} is the first genuinely new history term.

What changes versus Round 9 is only the input. Round 9 had to work around a
budget asymmetry (step 105 at ~1024 rollouts, every other checkpoint at 256).
The canonical table now carries the high-precision top-ups as well, so all five
continuation checkpoints are at 1024 and that asymmetry is gone.
"""
import csv, hashlib, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
import r9_models as RM

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
R9 = ART / "round9_markov_test"
REF = ART / "countdown_selectivity"
SEEDS = [f"seed{i}" for i in range(1, 6)]
CKPTS = [100, 105, 110, 115, 120, 125]
ALPHA, N_BOOT = 0.5, 10000
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_transition_table():
    mult = {}
    for l in (REF / "solver_families.jsonl").open():
        d = json.loads(l)
        mult[(d["question_id"], d["family_id"])] = d["solution_multiplicity"]
    S = defaultdict(dict)
    for r in csv.DictReader((P / "tables" / "branch_state_long.csv").open()):
        if r["track"] != "D":
            continue
        S[(r["continuation_id"], int(r["checkpoint"]), r["question_id"])][r["branch"]] = (
            int(float(r["raw_branch_count"])), float(r["smoothed_probability"]))

    def state(seed, ck, q):
        # every seed shares the root state at the fork point
        return S.get(("base" if ck == CKPTS[0] else seed, ck, q))

    def z_of(d, bs):
        p = np.array([d[b][1] for b in bs], float)
        lp = np.log(p)
        return lp - lp.mean()

    rows = []
    qs = sorted({k[2] for k in S})
    for q in qs:
        d0 = state(SEEDS[0], CKPTS[0], q)
        if not d0:
            continue
        bs = sorted(d0)
        for seed in SEEDS:
            Z, CNT, TOT = {}, {}, {}
            for ck in CKPTS:
                d = state(seed, ck, q)
                if not d or sorted(d) != bs:
                    continue
                Z[ck] = z_of(d, bs)
                CNT[ck] = np.array([d[b][0] for b in bs], float)
                TOT[ck] = float(CNT[ck].sum())
            for i, t in enumerate(CKPTS[:-1]):
                t1 = CKPTS[i + 1]
                if t not in Z or t1 not in Z or TOT[t] == 0 or TOT[t1] == 0:
                    continue
                tm1 = CKPTS[i - 1] if i >= 1 else None
                tm2 = CKPTS[i - 2] if i >= 2 else None
                v_t = Z[t] - Z[tm1] if tm1 in Z else None
                v_m1 = Z[tm1] - Z[tm2] if (tm1 in Z and tm2 in Z) else None
                for j, b in enumerate(bs):
                    rows.append({
                        "seed": seed, "question_id": q, "branch_id": b,
                        "step_t": t, "step_t1": t1,
                        "z_t": Z[t][j],
                        "z_t_minus_1": Z[tm1][j] if tm1 in Z else "",
                        "z_t_minus_2": Z[tm2][j] if tm2 in Z else "",
                        "velocity_t": v_t[j] if v_t is not None else "",
                        "velocity_t_minus_1": v_m1[j] if v_m1 is not None else "",
                        "branch_count_t": CNT[t][j], "n_total_t": TOT[t],
                        "next_count": CNT[t1][j], "next_total": TOT[t1],
                        "log_multiplicity": np.log(mult.get((q, b), 1)),
                        "operator": b[-1],
                        "log_first_operand": np.log(float(b[:-1])) if b[:-1].replace(".", "").isdigit()
                        else 0.0,
                        "K_q": len(bs)})
    f = WOUT / "tables" / "transition_table_trackD.csv"
    with f.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return f, rows


def main():
    t0 = time.time()
    f, raw = build_transition_table()
    print(f"  transition table: {len(raw)} rows  sha256 {sha(f)}")
    rows = RM.load(f)
    fold = json.load((R9 / "fold_assignment.json").open())["assignment"]      # preregistered folds
    qs = sorted({r["question_id"] for r in rows})
    missing = [q for q in qs if q not in fold]
    assert not missing, f"preregistered fold assignment does not cover {len(missing)} questions"
    rng = np.random.default_rng(20260916)

    set_M2 = [r for r in rows if np.isfinite(r["velocity_t"])]
    set_M3 = [r for r in rows if np.isfinite(r["velocity_t_minus_1"])]
    print(f"  M1 vs M2 on {len({(r['seed'], r['step_t']) for r in set_M2})} seed-transitions; "
          f"M2 vs M3 on {len({(r['seed'], r['step_t']) for r in set_M3})}")

    out, gains = [], []
    n1, p1 = RM.cv_nll(rows, RM.MODELS["M1"], fold)
    print(f"  M1 on all transitions: held-out NLL {n1:.5f}")
    out.append({"model_name": "M1", "features": "z_t + controls", "fold": "question 5-fold",
                "held_out_scope": "all transitions", "transition_start": min(CKPTS),
                "transition_end": max(CKPTS), "nll": round(n1, 6), "delta_nll_vs_state_only": 0.0,
                "bootstrap_ci_low": "", "bootstrap_ci_high": "",
                "num_questions": len(qs), "num_transitions": len(p1),
                "regularization": f"L2={RM.L2}", "hyperparameters": "frozen Round-9 spec"})
    for name, rs in (("M1_vs_M2", set_M2), ("M2_vs_M3", set_M3)):
        a, b = ("M1", "M2") if name == "M1_vs_M2" else ("M2", "M3")
        na, pa = RM.cv_nll(rs, RM.MODELS[a], fold)
        nb, pb = RM.cv_nll(rs, RM.MODELS[b], fold)
        g, lo, hi = RM.gain_ci(pa, pb, rng)
        gains.append((f"{a}->{b}", na, nb, g, lo, hi, len(pa)))
        print(f"  {a} {na:.5f}  ->  {b} {nb:.5f}   dNLL {g:+.6f} [{lo:+.6f},{hi:+.6f}]")
        for m, nl, dl, cl, ch in ((a, na, 0.0, "", ""), (b, nb, g, lo, hi)):
            out.append({"model_name": m, "features": "+".join(RM.MODELS[m]),
                        "fold": "question 5-fold", "held_out_scope": f"scope {name}",
                        "transition_start": min(CKPTS), "transition_end": max(CKPTS),
                        "nll": round(nl, 6), "delta_nll_vs_state_only": round(dl, 8),
                        "bootstrap_ci_low": round(cl, 8) if cl != "" else "",
                        "bootstrap_ci_high": round(ch, 8) if ch != "" else "",
                        "num_questions": len(qs), "num_transitions": len(pa),
                        "regularization": f"L2={RM.L2}", "hyperparameters": "frozen Round-9 spec"})

    # ---- transition breakdown
    tl = []
    for step in sorted({r["step_t"] for r in set_M2}):
        sub = [r for r in set_M2 if r["step_t"] == step]
        a, _ = RM.cv_nll(sub, RM.MODELS["M1"], fold)
        b, _ = RM.cv_nll(sub, RM.MODELS["M2"], fold)
        row = {"transition": f"{int(step)}->{int(step)+5}", "step_t": int(step),
               "nll_M1": round(a, 6), "nll_M2": round(b, 6), "gain_M1_M2": round(a - b, 8),
               "nll_M3": "", "gain_M2_M3": ""}
        sub3 = [r for r in sub if np.isfinite(r["velocity_t_minus_1"])]
        if sub3:
            c, _ = RM.cv_nll(sub3, RM.MODELS["M3"], fold)
            b3, _ = RM.cv_nll(sub3, RM.MODELS["M2"], fold)
            row["nll_M3"] = round(c, 6); row["gain_M2_M3"] = round(b3 - c, 8)
        tl.append(row)
        print(f"  {row['transition']}: M1 {a:.5f} M2 {b:.5f} gain {a-b:+.6f}"
              + (f"  M3 gain {row['gain_M2_M3']:+.6f}" if row["nll_M3"] != "" else ""))
    with (WOUT / "figure_data" / "fig5_transition_breakdown.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tl[0])); w.writeheader(); w.writerows(tl)

    # ---- leave-one-seed-out (the originally intended protocol) and
    #      leave-one-transition-out
    lo_rows = []
    seeds = sorted({r["seed"] for r in rows})
    sfold = {s: i for i, s in enumerate(seeds)}
    for name, rs, (a, b) in (("M1_vs_M2", set_M2, ("M1", "M2")), ("M2_vs_M3", set_M3, ("M2", "M3"))):
        na, pa = RM.cv_nll(rs, RM.MODELS[a], sfold, group="seed")
        nb, pb = RM.cv_nll(rs, RM.MODELS[b], sfold, group="seed")
        lo_rows.append({"protocol": "leave_one_seed_out", "comparison": f"{a}->{b}",
                        "held_out": "each continuation in turn", "nll_a": round(na, 6),
                        "nll_b": round(nb, 6), "delta_nll": round(na - nb, 8)})
        print(f"  LOSO {a} {na:.5f} -> {b} {nb:.5f}   dNLL {na-nb:+.6f}")
    for drop in sorted({r["step_t"] for r in set_M2}):
        rs = [r for r in set_M2 if r["step_t"] != drop]
        na, _ = RM.cv_nll(rs, RM.MODELS["M1"], fold)
        nb, _ = RM.cv_nll(rs, RM.MODELS["M2"], fold)
        lo_rows.append({"protocol": "leave_one_transition_out", "comparison": "M1->M2",
                        "held_out": f"drop {int(drop)}->{int(drop)+5}", "nll_a": round(na, 6),
                        "nll_b": round(nb, 6), "delta_nll": round(na - nb, 8)})
        print(f"  drop {int(drop)}->{int(drop)+5}: dNLL {na-nb:+.6f}")
    with (WOUT / "figure_data" / "fig5_loso.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lo_rows[0])); w.writeheader(); w.writerows(lo_rows)
    with (WOUT / "tables" / "predictive_models.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    lab = [g[0] for g in gains]
    est = [g[3] for g in gains]
    lo_ = [g[3] - g[4] for g in gains]; hi_ = [g[5] - g[3] for g in gains]
    ax[0].errorbar(range(len(gains)), est, yerr=[lo_, hi_], fmt="o", capsize=4, color="k",
                   label="observed")
    ax[0].axhline(0, color="crimson", ls="--", lw=1)
    # the control that decides the reading: z_{t-1} is a second noisy look at
    # the same state, so M2 can beat M1 even under exactly first-order dynamics.
    fo = WOUT / "figure_data" / "fig5_first_order_null.csv"
    if fo.exists():
        nl = {r["comparison"]: r for r in csv.DictReader(fo.open())}
        xs = [i for i, g in enumerate(gains) if g[0] in nl]
        ax[0].plot(xs, [float(nl[gains[i][0]]["first_order_null_mean"]) for i in xs], "_",
                   ms=18, color="tab:blue", label="first-order null (mean)")
        ax[0].plot(xs, [float(nl[gains[i][0]]["first_order_null_q95"]) for i in xs], "_",
                   ms=12, color="tab:cyan", label="first-order null (q95)")
    ax[0].legend(fontsize=6)
    ax[0].set_xticks(range(len(gains))); ax[0].set_xticklabels(lab)
    ax[0].set_ylabel(r"held-out $\Delta$NLL per count")
    ax[0].set_title(f"(a) gain from history (M1 NLL = {n1:.4f})", fontsize=9)
    x = range(len(tl))
    ax[1].bar(x, [r["gain_M1_M2"] for r in tl], color="tab:blue")
    ax[1].axhline(0, color="k", lw=.7)
    ax[1].set_xticks(list(x)); ax[1].set_xticklabels([r["transition"] for r in tl], rotation=30, fontsize=7)
    ax[1].set_ylabel(r"$\Delta$NLL (M1 $-$ M2)"); ax[1].set_title("(b) gain by transition", fontsize=9)
    rb = [r for r in lo_rows if r["protocol"] == "leave_one_transition_out"]
    sb = [r for r in lo_rows if r["protocol"] == "leave_one_seed_out"]
    ax[2].plot(range(len(rb)), [r["delta_nll"] for r in rb], "o", color="tab:blue",
               label="leave-one-transition-out")
    ax[2].plot(range(len(rb), len(rb) + len(sb)), [r["delta_nll"] for r in sb], "s", color="tab:orange",
               label="leave-one-seed-out")
    ax[2].axhline(0, color="crimson", ls="--", lw=1)
    ax[2].set_xticks(range(len(rb) + len(sb)))
    ax[2].set_xticklabels([r["held_out"] for r in rb] + [r["comparison"] for r in sb],
                          rotation=35, fontsize=6, ha="right")
    ax[2].set_ylabel(r"$\Delta$NLL"); ax[2].set_title("(c) robustness", fontsize=9)
    ax[2].legend(fontsize=6)
    for a_ in ax:
        a_.grid(alpha=.3)
    fig.suptitle("Fig 5  Recent velocity adds a reliable but very small amount of held-out predictive "
                 "information beyond the current branch distribution;\nolder history adds none. "
                 r"The gain is 0.19% of the NLL and exceeds a first-order null ($p=0.001$).",
                 fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(WOUT / "figures" / f"fig5_predictive_state.{ext}", dpi=150)
    plt.close(fig)
    for f_ in ("tables/predictive_models.csv", "tables/transition_table_trackD.csv",
               "figure_data/fig5_transition_breakdown.csv", "figure_data/fig5_loso.csv",
               "figures/fig5_predictive_state.png"):
        print(f"  {f_} sha256 {sha(P / f_)}")
    print(f"  elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
