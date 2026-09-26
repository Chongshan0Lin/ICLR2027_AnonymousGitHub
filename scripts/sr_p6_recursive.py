#!/usr/bin/env python3
"""P6: recursive simulation with the re-audited 1024-rollout decomposition.

The reference model to beat is S1: "all one-step residual variation is fresh
independent process noise". If that were right, recursively injecting the
empirical residual would reproduce the observed rate of cross-continuation
divergence.

    S0  deterministic drift            z_{t+d,s} = F(.)
    S1  naive iid full residual        + r sampled independently per continuation
    S2  shared + continuation-specific + mu_{q,t} shared across the simulated
                                          continuations of one question/transition,
                                          eta sampled independently

Measurement noise is never injected as process noise. It is added only when a
simulated latent state is turned into an observed branch count, exactly as the
real evaluation does: a multinomial draw of the real rollout budget.

Every simulated trajectory starts from the real shared pre-fork state at step
100 and is advanced with the same out-of-fold predictor used for the residuals.
"""
import csv, hashlib, json, sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
ALPHA = 0.5
STEPS = [100, 105, 110, 115, 120, 125]
CONTS = [f"seed{i}" for i in range(1, 6)]
N_REP = 200
SEED = 20260925
PRIMARY_MODEL = "R1"          # state-only; R2 is reported as a sensitivity
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def clr(p):
    lp = np.log(np.maximum(p, 1e-300))
    return lp - lp.mean()


def js(pa, pb):
    m = 0.5 * (pa + pb)
    return float(max(0.0, 0.5 * np.sum(pa * np.log(pa / m)) + 0.5 * np.sum(pb * np.log(pb / m))))


def main():
    rows = RM.load(PMF / "tables" / "transition_table_trackD.csv")
    fold = json.load((R9 / "fold_assignment.json").open())["assignment"]
    feats = RM.MODELS["M1"]
    rng = np.random.default_rng(SEED)

    # ---- static per (question, branch) controls, and the real states -------
    ctrl, branches, budget = {}, {}, {}
    obs_p, obs_cnt = {}, {}
    for r in rows:
        q, b = r["question_id"], r["branch_id"]
        branches.setdefault(q, [])
        if b not in branches[q]:
            branches[q].append(b)
        ctrl[(q, b)] = [r[f] for f in feats if f != "z_t"]
        budget[(q, int(r["step_t1"]))] = float(r["next_total"])
        obs_cnt[(q, int(r["step_t1"]), r["seed"])] = None
    for q in branches:
        branches[q] = sorted(branches[q])

    Z, P, CNT = {}, {}, {}
    for r in rows:
        q, b, t, s = r["question_id"], r["branch_id"], int(r["step_t"]), r["seed"]
        CNT.setdefault((q, t, s), {})[b] = float(r["branch_count_t"])
        CNT.setdefault((q, int(r["step_t1"]), s), {})[b] = float(r["next_count"])
    for (q, t, s), d in CNT.items():
        bs = branches[q]
        c = np.array([d.get(b, 0.0) for b in bs])
        n = c.sum()
        p = (c + ALPHA) / (n + ALPHA * len(bs))
        P[(q, t, s)] = p; Z[(q, t, s)] = clr(p)

    # ---- one out-of-fold predictor per transition, fitted once -------------
    beta_by = {}
    for t in STEPS[:-1]:
        sub = [r for r in rows if int(r["step_t"]) == t]
        if not sub:
            continue
        units = RM.units_for(sub, feats)
        for k in sorted(set(fold.values())):
            tr = [u for u in units if fold[u["question_id"]] != k]
            if tr:
                beta_by[(t, k)] = fit(tr, feats, RM.L2)

    def advance(q, t, z):
        """apply the out-of-fold predictor for question q at transition t."""
        bs = branches[q]
        X = np.array([[z[j]] + ctrl[(q, b)] for j, b in enumerate(bs)], float)
        X = X - X.mean(0, keepdims=True)
        beta, sd = beta_by[(t, fold[q])]
        return predict({"X": X}, beta, sd)

    # measurement energy of the target state, per (question, checkpoint)
    MEAS = {}
    mrng = np.random.default_rng(SEED + 1)
    for (q, t, s), p in list(P.items()):
        if s != CONTS[0]:
            continue
        n = int(budget.get((q, t), 1024))
        K = len(branches[q])
        v = []
        for _ in range(24):
            a = mrng.multinomial(n, p); b = mrng.multinomial(n, p)
            za = clr((a + ALPHA) / (a.sum() + ALPHA * K))
            zb = clr((b + ALPHA) / (b.sum() + ALPHA * K))
            v.append(float(np.sum((za - zb) ** 2) / 2.0))
        MEAS[(q, t)] = float(np.mean(v))

    # ---- empirical residual pools, out of fold ----------------------------
    RES = {}
    for t in STEPS[:-1]:
        for q in branches:
            if (q, t, CONTS[0]) not in Z:
                continue
            for s in CONTS:
                if (q, t, s) not in Z or (q, t + 5, s) not in Z:
                    continue
                pr = advance(q, t, Z[(q, t, s)])
                RES[(q, t, s)] = Z[(q, t + 5, s)] - clr(pr)
    # Frozen semantics, from countdown/src/r11_forecast.py:
    #   mu[(q,t)]  = the across-continuation mean residual for that cell
    #   eta_pool[(t,K)] = every (r - mu), POOLED ACROSS QUESTIONS with the same
    #                     support size at that transition
    # The pooling matters. Drawing S1's residual from a question's own five
    # residuals would leave mu identical in every draw, i.e. it would smuggle
    # S2's shared component into S1 and make the contrast meaningless. That is
    # the mistake this rewrite fixes.
    MU, eta_pool = {}, defaultdict(list)
    for t in STEPS[:-1]:
        for q in branches:
            rr = {s: RES[(q, t, s)] for s in CONTS if (q, t, s) in RES}
            if len(rr) < len(CONTS):
                continue
            M = np.mean([rr[s] for s in CONTS], axis=0)
            MU[(q, t)] = M
            for s in CONTS:
                eta_pool[(t, len(M))].append(rr[s] - M)
    # S1 injects the WHOLE residual energy as fresh independent noise:
    # measurement-corrected eta energy plus the shared energy of that cell.
    S_ = len(CONTS)
    cellE = {}
    for (q, t), M in MU.items():
        rr = np.array([RES[(q, t, s)] for s in CONTS])
        m = rr.mean(0)
        e_within = float(np.mean(((rr - m) ** 2).sum(1)))
        e_eta = e_within * S_ / (S_ - 1)
        meas = MEAS.get((q, t + 5), 0.0)
        cellE[(q, t)] = {"eta_corr": max(1e-9, e_eta - meas), "shared": float((M ** 2).sum())}

    qs = sorted({q for (q, t, s) in RES if t == STEPS[0]})
    print(f"  {len(qs)} questions carried through the full recursion, "
          f"{len(STEPS) - 1} transitions, {N_REP} replicates")

    # ---- observed targets --------------------------------------------------
    obs = {}
    for t in STEPS[1:]:
        pj, wd = [], []
        for q in qs:
            ps = {s: P[(q, t, s)] for s in CONTS if (q, t, s) in P}
            if len(ps) < 2:
                continue
            pj.append(np.mean([js(ps[a], ps[b]) for a, b in combinations(sorted(ps), 2)]))
            w = [branches[q][int(np.argmax(ps[s]))] for s in sorted(ps)]
            wd.append(np.mean([w[i] != w[j] for i, j in combinations(range(len(w)), 2)]))
        obs[t] = (float(np.mean(pj)), float(np.mean(wd)))

    # ---- simulators --------------------------------------------------------
    def simulate(kind, rep):
        r = np.random.default_rng([SEED, rep, {"S0": 0, "S1": 1, "S2": 2, "S2g": 3}[kind]])
        state = {(q, s): Z[(q, STEPS[0], s)] for q in qs for s in CONTS}
        out = {}
        for i, t in enumerate(STEPS[:-1]):
            t1 = t + 5
            for q in qs:
                shared = MU.get((q, t)) if kind in ("S2", "S2g") else None
                for s in CONTS:
                    z = state[(q, s)]
                    pr = clr(advance(q, t, z))
                    K = len(branches[q])
                    ce = cellE.get((q, t))
                    if kind == "S0" or ce is None:
                        znew = pr
                    elif kind == "S1":
                        sc = np.sqrt(max(1e-9, (ce["eta_corr"] + ce["shared"]) / max(1, K - 1)))
                        znew = pr + r.normal(0, sc, K)
                    elif kind == "S2":
                        pool = eta_pool.get((t, K), [])
                        znew = pr + (shared if shared is not None else 0.0) \
                            + (pool[r.integers(len(pool))] if pool else 0.0)
                    else:   # S2g: shared mu + Gaussian corrected eta, like-for-like with S1
                        sc = np.sqrt(max(1e-9, ce["eta_corr"] / max(1, K - 1)))
                        znew = pr + (shared if shared is not None else 0.0) + r.normal(0, sc, K)
                    state[(q, s)] = znew
            # turn the latent states into observed counts at the real budget
            for q in qs:
                n = int(budget.get((q, t1), 1024))
                ps = {}
                for s in CONTS:
                    e = np.exp(state[(q, s)] - state[(q, s)].max())
                    p = e / e.sum()
                    c = r.multinomial(n, p)
                    ps[s] = (c + ALPHA) / (c.sum() + ALPHA * len(p))
                out.setdefault(t1, []).append(ps)
        agg = {}
        for t1, lst in out.items():
            pj, wd = [], []
            for ps in lst:
                ks = sorted(ps)
                pj.append(np.mean([js(ps[a], ps[b]) for a, b in combinations(ks, 2)]))
                w = [int(np.argmax(ps[s])) for s in ks]
                wd.append(np.mean([w[i] != w[j] for i, j in combinations(range(len(w)), 2)]))
            agg[t1] = (float(np.mean(pj)), float(np.mean(wd)))
        return agg

    recs = []
    for kind in ("S0", "S1", "S2", "S2g"):
        for rep in range(N_REP):
            a = simulate(kind, rep)
            for t1 in sorted(a):
                pj, wd = a[t1]
                recs.append({"simulator": kind, "transition_model": PRIMARY_MODEL,
                             "start_checkpoint": STEPS[0], "target_checkpoint": t1,
                             "horizon": t1 - STEPS[0], "simulation_replicate": rep,
                             "predicted_pairwise_js": round(pj, 8),
                             "observed_pairwise_js": round(obs[t1][0], 8),
                             "predicted_winner_disagreement": round(wd, 8),
                             "observed_winner_disagreement": round(obs[t1][1], 8),
                             "absolute_js_error": round(abs(pj - obs[t1][0]), 8),
                             "absolute_winner_error": round(abs(wd - obs[t1][1]), 8)})
        m = [x for x in recs if x["simulator"] == kind and x["horizon"] == 25]
        print(f"  {kind}: endpoint predicted JS "
              f"{np.mean([x['predicted_pairwise_js'] for x in m]):.5f} vs observed "
              f"{obs[125][0]:.5f} | winner disagreement "
              f"{np.mean([x['predicted_winner_disagreement'] for x in m]):.4f} vs "
              f"{obs[125][1]:.4f}")
    with (OUT / "tables" / "recursive_forecast_1024.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)

    fd_js, fd_w = [], []
    for kind in ("S0", "S1", "S2", "S2g"):
        for h in sorted({x["horizon"] for x in recs}):
            m = [x for x in recs if x["simulator"] == kind and x["horizon"] == h]
            pj = np.array([x["predicted_pairwise_js"] for x in m])
            wd = np.array([x["predicted_winner_disagreement"] for x in m])
            fd_js.append({"simulator": kind, "horizon": h,
                          "predicted_js_mean": round(float(pj.mean()), 8),
                          "predicted_js_lo": round(float(np.quantile(pj, .025)), 8),
                          "predicted_js_hi": round(float(np.quantile(pj, .975)), 8),
                          "observed_js": round(obs[100 + h][0], 8),
                          "abs_error": round(abs(float(pj.mean()) - obs[100 + h][0]), 8)})
            fd_w.append({"simulator": kind, "horizon": h,
                         "predicted_winner_disagreement_mean": round(float(wd.mean()), 8),
                         "predicted_lo": round(float(np.quantile(wd, .025)), 8),
                         "predicted_hi": round(float(np.quantile(wd, .975)), 8),
                         "observed_winner_disagreement": round(obs[100 + h][1], 8),
                         "abs_error": round(abs(float(wd.mean()) - obs[100 + h][1]), 8)})
    for nm, dat in (("recursive_js_forecast.csv", fd_js), ("recursive_winner_forecast.csv", fd_w)):
        with (OUT / "figure_data" / nm).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dat[0])); w.writeheader(); w.writerows(dat)

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    cols = {"S0": "grey", "S1": "tab:orange", "S2": "tab:blue", "S2g": "tab:green"}
    lbl = {"S0": "S0 drift only", "S1": "S1 iid full residual",
           "S2": "S2 shared + empirical eta", "S2g": "S2g shared + Gaussian eta"}
    for kind in ("S0", "S1", "S2", "S2g"):
        d = [x for x in fd_js if x["simulator"] == kind]
        ax[0].plot([x["horizon"] for x in d], [x["predicted_js_mean"] for x in d], "o-",
                   color=cols[kind], label=lbl[kind])
        ax[0].fill_between([x["horizon"] for x in d], [x["predicted_js_lo"] for x in d],
                           [x["predicted_js_hi"] for x in d], color=cols[kind], alpha=.15)
        d = [x for x in fd_w if x["simulator"] == kind]
        ax[1].plot([x["horizon"] for x in d],
                   [x["predicted_winner_disagreement_mean"] for x in d], "o-",
                   color=cols[kind], label=lbl[kind])
    d = [x for x in fd_js if x["simulator"] == "S2"]
    ax[0].plot([x["horizon"] for x in d], [x["observed_js"] for x in d], "k^--", label="observed")
    d = [x for x in fd_w if x["simulator"] == "S2"]
    ax[1].plot([x["horizon"] for x in d], [x["observed_winner_disagreement"] for x in d],
               "k^--", label="observed")
    ax[0].set_xlabel("horizon after the fork (steps)"); ax[0].set_ylabel("mean pairwise JS")
    ax[0].set_title("(a) cross-continuation divergence", fontsize=9)
    ax[1].set_xlabel("horizon after the fork (steps)"); ax[1].set_ylabel("winner disagreement")
    ax[1].set_title("(b) modal-branch disagreement", fontsize=9)
    for a in ax:
        a.grid(alpha=.3); a.legend(fontsize=7)
    fig.suptitle("Recursive simulation from the shared pre-fork state, 1024-rollout "
                 "decomposition", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for e in ("pdf", "png"):
        fig.savefig(OUT / "figures" / f"recursive_forecast_final.{e}", dpi=150)
    plt.close(fig)
    # ---- leave-one-continuation-out robustness -----------------------------
    # The whole contrast is about structure ACROSS continuations, so dropping one
    # and redoing the comparison is the robustness check that matters.
    lo_rows = []
    for drop in CONTS:
        keep = [c for c in CONTS if c != drop]
        Sk = len(keep)
        MU2, pool2, cell2 = {}, defaultdict(list), {}
        for t in STEPS[:-1]:
            for q in branches:
                rr = {s: RES[(q, t, s)] for s in keep if (q, t, s) in RES}
                if len(rr) < Sk:
                    continue
                M = np.mean([rr[s] for s in keep], axis=0)
                MU2[(q, t)] = M
                arr = np.array([rr[s] for s in keep])
                e_w = float(np.mean(((arr - arr.mean(0)) ** 2).sum(1)))
                e_eta = e_w * Sk / (Sk - 1)
                cell2[(q, t)] = {"eta_corr": max(1e-9, e_eta - MEAS.get((q, t + 5), 0.0)),
                                 "shared": float((M ** 2).sum())}
                for s in keep:
                    pool2[(t, len(M))].append(rr[s] - M)
        obs2 = {}
        for t in STEPS[1:]:
            pj, wd = [], []
            for q in qs:
                ps = {s: P[(q, t, s)] for s in keep if (q, t, s) in P}
                if len(ps) < 2:
                    continue
                ks = sorted(ps)
                pj.append(np.mean([js(ps[a], ps[b]) for a, b in combinations(ks, 2)]))
                w = [int(np.argmax(ps[s])) for s in ks]
                wd.append(np.mean([w[i] != w[j] for i, j in combinations(range(len(w)), 2)]))
            obs2[t] = (float(np.mean(pj)), float(np.mean(wd)))
        for kind in ("S1", "S2"):
            acc = defaultdict(list)
            for rep in range(60):
                r = np.random.default_rng([SEED, rep, 7, CONTS.index(drop),
                                           0 if kind == "S1" else 1])
                state = {(q, s): Z[(q, STEPS[0], s)] for q in qs for s in keep}
                for t in STEPS[:-1]:
                    t1 = t + 5
                    for q in qs:
                        ce = cell2.get((q, t)); K = len(branches[q])
                        sh = MU2.get((q, t)) if kind == "S2" else None
                        for s in keep:
                            pr = clr(advance(q, t, state[(q, s)]))
                            if ce is None:
                                state[(q, s)] = pr
                            elif kind == "S1":
                                sc = np.sqrt(max(1e-9, (ce["eta_corr"] + ce["shared"]) / max(1, K - 1)))
                                state[(q, s)] = pr + r.normal(0, sc, K)
                            else:
                                pl = pool2.get((t, K), [])
                                state[(q, s)] = pr + (sh if sh is not None else 0.0) \
                                    + (pl[r.integers(len(pl))] if pl else 0.0)
                    for q in qs:
                        n = int(budget.get((q, t1), 1024)); ps = {}
                        for s in keep:
                            e = np.exp(state[(q, s)] - state[(q, s)].max()); pp = e / e.sum()
                            c = r.multinomial(n, pp)
                            ps[s] = (c + ALPHA) / (c.sum() + ALPHA * len(pp))
                        ks = sorted(ps)
                        acc[(t1, "js")].append(np.mean([js(ps[a], ps[b])
                                                        for a, b in combinations(ks, 2)]))
                        w = [int(np.argmax(ps[s])) for s in ks]
                        acc[(t1, "w")].append(np.mean([w[i] != w[j]
                                                       for i, j in combinations(range(len(w)), 2)]))
            for t1 in STEPS[1:]:
                lo_rows.append({"dropped_continuation": drop, "simulator": kind,
                                "horizon": t1 - STEPS[0],
                                "predicted_js": round(float(np.mean(acc[(t1, "js")])), 8),
                                "observed_js": round(obs2[t1][0], 8),
                                "abs_js_error": round(abs(float(np.mean(acc[(t1, "js")])) - obs2[t1][0]), 8),
                                "predicted_winner_disagreement":
                                    round(float(np.mean(acc[(t1, "w")])), 8),
                                "observed_winner_disagreement": round(obs2[t1][1], 8),
                                "abs_winner_error":
                                    round(abs(float(np.mean(acc[(t1, "w")])) - obs2[t1][1]), 8)})
        e = {k: [r for r in lo_rows if r["dropped_continuation"] == drop
                 and r["simulator"] == k and r["horizon"] == 25][0] for k in ("S1", "S2")}
        print(f"  LOO drop {drop}: endpoint JS error S1 {e['S1']['abs_js_error']:.5f} -> "
              f"S2 {e['S2']['abs_js_error']:.5f} "
              f"({1 - e['S2']['abs_js_error'] / e['S1']['abs_js_error']:+.1%}); winner "
              f"{e['S1']['abs_winner_error']:.4f} -> {e['S2']['abs_winner_error']:.4f} "
              f"({1 - e['S2']['abs_winner_error'] / e['S1']['abs_winner_error']:+.1%})")
    with (OUT / "tables" / "recursive_forecast_loso.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lo_rows[0])); w.writeheader(); w.writerows(lo_rows)

    for f in ("tables/recursive_forecast_1024.csv", "tables/recursive_forecast_loso.csv",
              "figure_data/recursive_js_forecast.csv",
              "figure_data/recursive_winner_forecast.csv", "figures/recursive_forecast_final.png"):
        print(f"  {f} sha256 {sha(OUT / f)}")


if __name__ == "__main__":
    main()
