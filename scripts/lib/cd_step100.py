#!/usr/bin/env python3
"""Step-100 focused follow-up: is the step-100 excess residual real?

The earlier 128-rollout pass found one checkpoint (step 100) whose mean |residual|
from the frequency-sharpening null M0 sat above a split-half sampling floor, while
every later checkpoint sat at or below it. That floor was a crude instrument. Here
the noise control is a proper parametric bootstrap of M0 itself: simulate entrance
counts (at t0 AND at t) from the fitted null, refit gamma on each simulated draw,
and read the residual magnitude the null alone produces. The signal is the excess
of the observed residual over that null expectation, per question.
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2]  # artifact repo root
OLD = R / "results/countdown_selectivity"
OUT = R / "results/countdown_selectivity_step100"
MOD, FIG = OUT / "models", OUT / "figures"
STEPS = [25, 75, 100, 125, 150]
T0, PRIMARY = 25, 100
ALPHA, N_FOLD = 0.5, 5
SEED = 20260915
N_PARAM_BOOT, N_QBOOT = 2000, 10000
GRID = np.arange(0.20, 4.0001, 0.05)


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ loading
def load_fresh(step):
    rows = []
    for p in sorted(OUT.glob(f"generations/ckpt{step:04d}_s*.jsonl")):
        rows += [json.loads(l) for l in p.open()]
    return {r["question_id"]: r for r in rows}


def load_old(step):
    p = OLD / f"generations/ckpt{step:04d}.jsonl"
    if not p.exists():
        return None
    return {r["question_id"]: r for r in (json.loads(l) for l in p.open())}


def audit_rows(data, tag):
    out = []
    for s in sorted(data):
        rows = list(data[s].values())
        n = sum(r["n"] for r in rows)
        f = lambda k: sum(1 for r in rows for x in r["rollouts"] if x[k])
        ent, entf = f("entrance_family"), f("entrance_is_solver_feasible")
        out.append({"source": tag, "training_step": s, "n_questions": len(rows),
                    "n_rollouts": n,
                    "parse_success_rate": f("parse_success") / n,
                    "truncation_rate": f("truncated") / n,
                    "entrance_found_rate": ent / n,
                    "entrance_feasible_rate": entf / n,
                    "entrance_feasible_given_found": entf / max(1, ent),
                    "pass@1": f("reaches_target") / n})
    return out


# ------------------------------------------------------- packed arrays / M0
def pack(dist, qs, Bmax):
    """counts [Q,Bmax] and mask [Q,Bmax] for a fixed question order."""
    C = np.zeros((len(qs), Bmax))
    M = np.zeros((len(qs), Bmax), bool)
    for i, q in enumerate(qs):
        cnt, sup = dist[q]
        for j, b in enumerate(sup):
            C[i, j] = cnt.get(b, 0)
            M[i, j] = True
    return C, M


def smooth_np(C, M):
    # axis -1 is the family axis for both the [Q,K] observed arrays and the
    # [B,Q,K] bootstrap arrays; axis 1 would be the question axis for the latter
    K = M.sum(-1, keepdims=True)
    tot = (C * M).sum(-1, keepdims=True) + ALPHA * K
    P = np.where(M, (C + ALPHA) / tot, 0.0)
    return P


def sharpen_np(LP0, M, g):
    """softmax(g * log p0) over the support; g may be scalar or [...,1,1]."""
    z = np.where(M, g * LP0, -np.inf)
    z = z - z.max(-1, keepdims=True)
    e = np.where(M, np.exp(z), 0.0)
    return e / e.sum(-1, keepdims=True)


def nll_grid(LP0, C, M):
    """mean NLL per grid gamma. LP0/C/M broadcast over leading dims."""
    out = []
    n = (C * M).sum((-1, -2))
    for g in GRID:
        pr = sharpen_np(LP0, M, g)
        out.append(-(C * np.log(np.maximum(pr, 1e-300)) * M).sum((-1, -2)) / n)
    return np.stack(out)                                    # [G, ...]


def fit_gamma(LP0, C, M):
    return float(GRID[int(np.argmin(nll_grid(LP0, C, M)))])


def main() -> int:
    for d in (MOD, FIG):
        d.mkdir(parents=True, exist_ok=True)
    feas, mult = defaultdict(set), {}
    for l in (OLD / "solver_families.jsonl").open():
        d = json.loads(l)
        feas[d["question_id"]].add(d["family_id"])
        mult[(d["question_id"], d["family_id"])] = d["solution_multiplicity"]

    fresh = {s: load_fresh(s) for s in STEPS}
    old = {s: o for s in STEPS if (o := load_old(s)) is not None}

    # ------------------------------------------------- 1. parser / audit
    A = audit_rows(fresh, "fresh_512") + audit_rows(old, "old_128")
    with (OUT / "parser_audit_512.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(A[0]))
        w.writeheader()
        w.writerows(A)
    log(f"{'src':>10}{'step':>6}{'nq':>5}{'nroll':>8}{'parse':>8}{'trunc':>8}"
        f"{'entfnd':>8}{'entfeas|f':>11}{'pass@1':>9}")
    for a in A:
        log(f"{a['source']:>10}{a['training_step']:>6}{a['n_questions']:>5}"
            f"{a['n_rollouts']:>8}{a['parse_success_rate']:>8.4f}"
            f"{a['truncation_rate']:>8.4f}{a['entrance_found_rate']:>8.4f}"
            f"{a['entrance_feasible_given_found']:>11.4f}{a['pass@1']:>9.4f}")

    # ------------------------------- 2. frozen eligible set from the old run
    def dists(data, s):
        out = {}
        for q, r in data[s].items():
            c = Counter(x["entrance_family"] for x in r["rollouts"]
                        if x["entrance_family"] in feas[q])
            out[q] = (c, sorted(feas[q]))
        return out
    d_old_t0 = dists(old, T0)
    elig = sorted(q for q in d_old_t0
                  if sum(d_old_t0[q][0].values()) >= 20
                  and len(d_old_t0[q][1]) >= 2)
    log(f"\nfrozen analysis set: {len(elig)} questions "
        f"(rule re-applied to the old step-{T0} run)")
    (OUT / "eligible_questions.json").write_text(json.dumps(
        {"n": len(elig), "rule": "old-run step-25: >=20 feasible-entrance "
         "rollouts and >=2 solver-feasible families", "questions": elig},
        indent=2))

    # ------------------------------------- 3. fresh entrance distributions
    D = {s: dists(fresh, s) for s in STEPS}
    rows = []
    for s in STEPS:
        for q in elig:
            cnt, sup = D[s][q]
            tot = sum(cnt.values())
            for b in sup:
                rows.append({"question_id": q, "training_step": s,
                             "checkpoint": f"checkpoint-{s}", "family_id": b,
                             "count": cnt.get(b, 0),
                             "probability": cnt.get(b, 0) / tot if tot else 0.0,
                             "operator": b[-1],
                             "solution_multiplicity": mult[(q, b)]})
    with (OUT / "entrance_distribution_512.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    log(f"entrance_distribution_512.csv: {len(rows)} rows")

    Bmax = max(len(D[T0][q][1]) for q in elig)
    C0, M = pack(D[T0], elig, Bmax)
    P0 = smooth_np(C0, M)
    LP0 = np.where(M, np.log(np.maximum(P0, 1e-300)), 0.0)
    N0 = (C0 * M).sum(1)

    # --------------------------- 4. M0 with one frozen fold assignment
    rng = random.Random(SEED)
    ql = list(range(len(elig)))
    rng.shuffle(ql)
    folds = [np.array(sorted(ql[i::N_FOLD])) for i in range(N_FOLD)]
    (MOD / "fold_assignment.json").write_text(json.dumps(
        {"seed": SEED, "n_fold": N_FOLD,
         "folds": [[elig[i] for i in f] for f in folds]}, indent=2))

    later = [s for s in STEPS if s != T0]
    Ct, gam, null_rows = {}, {}, []
    for s in later:
        C, _ = pack(D[s], elig, Bmax)
        Ct[s] = C
        g = fit_gamma(LP0, C, M)
        gam[s] = g
        nl, gs = [], []
        for fi in range(N_FOLD):
            te = folds[fi]
            tr = np.array([i for i in range(len(elig)) if i not in set(te)])
            gtr = fit_gamma(LP0[tr], C[tr], M[tr])
            gs.append(gtr)
            nl.append(float(nll_grid(LP0[te], C[te], M[te])[
                int(np.argmin(np.abs(GRID - gtr)))]))
        null_rows.append({"training_step": s, "gamma_all": g,
                          "gamma_cv_mean": sum(gs) / len(gs),
                          "heldout_nll": sum(nl) / len(nl),
                          "n_questions": len(elig),
                          "n_rollouts_used": float((C * M).sum())})
        log(f"  step {s:>3}: gamma {g:.3f}  cv-gamma {sum(gs)/len(gs):.3f}  "
            f"held-out NLL {sum(nl)/len(nl):.4f}")
    with (MOD / "frequency_null_512.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(null_rows[0]))
        w.writeheader()
        w.writerows(null_rows)

    # --------------------- 5. observed residuals and per-question D_t(q)
    res_rows, Dobs = [], {}
    for s in later:
        Pt = smooth_np(Ct[s], M)
        pred = sharpen_np(LP0, M, gam[s])
        Rr = np.where(M, np.log(np.maximum(Pt, 1e-300))
                      - np.log(np.maximum(pred, 1e-300)), 0.0)
        Dobs[s] = (np.abs(Rr) * M).sum(1) / M.sum(1)
        for i, q in enumerate(elig):
            for j, b in enumerate(D[s][q][1]):
                res_rows.append({"question_id": q, "training_step": s,
                                 "family_id": b, "count": Ct[s][i, j],
                                 "p_t": Pt[i, j], "p_t0": P0[i, j],
                                 "p_pred": pred[i, j], "residual": Rr[i, j],
                                 "solution_multiplicity": mult[(q, b)],
                                 "operator": b[-1]})
    with (OUT / "residual_trajectories_512.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res_rows[0]))
        w.writeheader()
        w.writerows(res_rows)

    # ------------------------------- 6. parametric bootstrap of M0 itself
    log(f"\n=== parametric bootstrap of M0  ({N_PARAM_BOOT} draws)")
    nprng = np.random.default_rng(SEED)
    boot = {}
    for s in later:
        Nt = (Ct[s] * M).sum(1)
        pred = sharpen_np(LP0, M, gam[s])
        Cb = np.zeros((N_PARAM_BOOT, len(elig), Bmax))
        C0b = np.zeros_like(Cb)
        for i in range(len(elig)):
            k = int(M[i].sum())
            p = pred[i, :k] / pred[i, :k].sum()
            Cb[:, i, :k] = nprng.multinomial(int(Nt[i]), p, size=N_PARAM_BOOT)
            p0i = P0[i, :k] / P0[i, :k].sum()
            C0b[:, i, :k] = nprng.multinomial(int(N0[i]), p0i,
                                              size=N_PARAM_BOOT)
        Mb = M[None]
        P0b = smooth_np(C0b, Mb)
        LP0b = np.where(Mb, np.log(np.maximum(P0b, 1e-300)), 0.0)
        gb = GRID[np.argmin(nll_grid(LP0b, Cb, Mb), axis=0)]   # [B]
        prb = sharpen_np(LP0b, Mb, gb[:, None, None])
        Ptb = smooth_np(Cb, Mb)
        Rb = np.where(Mb, np.log(np.maximum(Ptb, 1e-300))
                      - np.log(np.maximum(prb, 1e-300)), 0.0)
        Db = (np.abs(Rb) * Mb).sum(2) / M.sum(1)[None]          # [B, Q]
        boot[s] = {"E0": Db.mean(0), "meanD": Db.mean(1), "gamma": gb}
        log(f"  step {s:>3}: obs meanD {Dobs[s].mean():.4f}   null meanD "
            f"{Db.mean():.4f} [{np.quantile(Db.mean(1), .025):.4f},"
            f"{np.quantile(Db.mean(1), .975):.4f}]   null gamma "
            f"{gb.mean():.3f}")

    # ---------- 7. ExcessD with a 10k question bootstrap; step 100 primary
    log("\n=== ExcessD_t = D_t(q) - E_0[D_t(q)]   (question bootstrap)")
    qrng = np.random.default_rng(SEED + 1)
    idx = qrng.integers(0, len(elig), size=(N_QBOOT, len(elig)))
    ex_rows = []
    for s in later:
        E = Dobs[s] - boot[s]["E0"]
        bs = E[idx].mean(1)
        lo, hi = np.quantile(bs, .025), np.quantile(bs, .975)
        # +1 smoothing so a bootstrap that never exceeds the observed value
        # reports 1/(B+1) rather than an impossible zero
        p_param = float((boot[s]["meanD"] >= Dobs[s].mean()).sum() + 1) \
            / (N_PARAM_BOOT + 1)
        ex_rows.append({
            "training_step": s, "mean_D_obs": float(Dobs[s].mean()),
            "mean_D_null": float(boot[s]["E0"].mean()),
            "excess_D": float(E.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "frac_questions_positive": float((E > 0).mean()),
            "median_excess": float(np.median(E)),
            "p_parametric_one_sided": p_param,
            "n_questions": len(elig)})
        log(f"  step {s:>3}: ExcessD {E.mean():+.4f} [{lo:+.4f},{hi:+.4f}]  "
            f"frac q>0 {(E > 0).mean():.3f}  p_param {p_param:.4f}")
    with (OUT / "excess_D.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ex_rows[0]))
        w.writeheader()
        w.writerows(ex_rows)
    per_q = []
    for s in later:
        E = Dobs[s] - boot[s]["E0"]
        for i, q in enumerate(elig):
            per_q.append({"question_id": q, "training_step": s,
                          "D_obs": Dobs[s][i], "D_null": boot[s]["E0"][i],
                          "excess_D": E[i]})
    with (OUT / "excess_D_per_question.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_q[0]))
        w.writeheader()
        w.writerows(per_q)

    # --------------------- 8. old-128 -> fresh-512 residual replication
    log("\n=== replication of the residual ordering, old 128 vs fresh 512")
    def spearman(x, y):
        x, y = np.asarray(x), np.asarray(y)
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        return float(np.corrcoef(rx, ry)[0, 1])
    d_old = {s: dists(old, s) for s in old}
    C0o, Mo = pack(d_old[T0], elig, Bmax)
    P0o = smooth_np(C0o, Mo)
    LP0o = np.where(Mo, np.log(np.maximum(P0o, 1e-300)), 0.0)
    rep_rows = []
    for s in later:
        if s not in d_old:
            continue
        Co, _ = pack(d_old[s], elig, Bmax)
        go = fit_gamma(LP0o, Co, Mo)
        Po = smooth_np(Co, Mo)
        pro = sharpen_np(LP0o, Mo, go)
        Ro = np.log(np.maximum(Po, 1e-300)) - np.log(np.maximum(pro, 1e-300))
        Pt = smooth_np(Ct[s], M)
        prn = sharpen_np(LP0, M, gam[s])
        Rn = np.log(np.maximum(Pt, 1e-300)) - np.log(np.maximum(prn, 1e-300))
        a, b = Ro[M], Rn[M]
        rep_rows.append({"training_step": s, "n_families": int(M.sum()),
                         "gamma_old": go, "gamma_new": gam[s],
                         "spearman_old_new": spearman(a, b),
                         "pearson_old_new": float(np.corrcoef(a, b)[0, 1]),
                         "sign_agreement": float(((a > 0) == (b > 0)).mean())})
        log(f"  step {s:>3}: rho {rep_rows[-1]['spearman_old_new']:+.4f}  "
            f"sign {rep_rows[-1]['sign_agreement']:.3f}  "
            f"gamma {go:.2f}->{gam[s]:.2f}")
    with (OUT / "replication_old_vs_new.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rep_rows[0]))
        w.writeheader()
        w.writerows(rep_rows)

    # ------------------------------ 9. 256/256 split-half reliability
    log("\n=== 256/256 split-half reliability of the residual")
    rel_rows = []
    for s in later:
        hA, hB = {}, {}
        for q in elig:
            r = fresh[s][q]
            ca, cb = Counter(), Counter()
            for x in r["rollouts"]:
                b = x["entrance_family"]
                if b in feas[q]:
                    (ca if x["rollout_id"] % 2 == 0 else cb)[b] += 1
            hA[q], hB[q] = (ca, sorted(feas[q])), (cb, sorted(feas[q]))
        CA, _ = pack(hA, elig, Bmax)
        CB, _ = pack(hB, elig, Bmax)
        pr = sharpen_np(LP0, M, gam[s])
        lp = np.log(np.maximum(pr, 1e-300))
        RA = np.log(np.maximum(smooth_np(CA, M), 1e-300)) - lp
        RB = np.log(np.maximum(smooth_np(CB, M), 1e-300)) - lp
        a, b = RA[M], RB[M]
        rel_rows.append({"training_step": s, "n_pairs": int(M.sum()),
                         "spearman": spearman(a, b),
                         "sign_agreement": float(((a > 0) == (b > 0)).mean())})
        log(f"  step {s:>3}: rho {rel_rows[-1]['spearman']:+.4f}  "
            f"sign {rel_rows[-1]['sign_agreement']:.3f}")
    with (OUT / "split_half_reliability_512.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rel_rows[0]))
        w.writeheader()
        w.writerows(rel_rows)


    # ------------- 10. a harder null: per-question gamma (M0-pq)
    # A single global gamma forces every question to sharpen at the same rate.
    # If the excess is just heterogeneity in how fast different questions
    # collapse, letting gamma vary by question should absorb it.
    log("\n=== M0-pq: per-question gamma null")
    def fit_gamma_pq(LP0_, C_, M_):
        """gamma per question; leading bootstrap dims are preserved."""
        n = (C_ * M_).sum(-1)
        nl = []
        for g in GRID:
            pr = sharpen_np(LP0_, M_, g)
            nl.append(-(C_ * np.log(np.maximum(pr, 1e-300)) * M_).sum(-1)
                      / np.maximum(n, 1.0))
        return GRID[np.argmin(np.stack(nl), axis=0)]

    pq_rows = []
    for s in later:
        gq = fit_gamma_pq(LP0, Ct[s], M)                       # [Q]
        prq = sharpen_np(LP0, M, gq[:, None])
        Rq = np.where(M, np.log(np.maximum(smooth_np(Ct[s], M), 1e-300))
                      - np.log(np.maximum(prq, 1e-300)), 0.0)
        Dq = (np.abs(Rq) * M).sum(1) / M.sum(1)
        Nt = (Ct[s] * M).sum(1)
        Cb = np.zeros((N_PARAM_BOOT, len(elig), Bmax))
        C0b = np.zeros_like(Cb)
        for i in range(len(elig)):
            k = int(M[i].sum())
            pp = prq[i, :k] / prq[i, :k].sum()
            Cb[:, i, :k] = nprng.multinomial(int(Nt[i]), pp, size=N_PARAM_BOOT)
            p0i = P0[i, :k] / P0[i, :k].sum()
            C0b[:, i, :k] = nprng.multinomial(int(N0[i]), p0i,
                                              size=N_PARAM_BOOT)
        Mb = M[None]
        LP0b = np.where(Mb, np.log(np.maximum(smooth_np(C0b, Mb), 1e-300)), 0.)
        gqb = fit_gamma_pq(LP0b, Cb, Mb)                       # [B, Q]
        prb = sharpen_np(LP0b, Mb, gqb[..., None])
        Rb = np.where(Mb, np.log(np.maximum(smooth_np(Cb, Mb), 1e-300))
                      - np.log(np.maximum(prb, 1e-300)), 0.0)
        Db = (np.abs(Rb) * Mb).sum(2) / M.sum(1)[None]
        E = Dq - Db.mean(0)
        bs = E[idx].mean(1)
        pp_ = float((Db.mean(1) >= Dq.mean()).sum() + 1) / (N_PARAM_BOOT + 1)
        pq_rows.append({"training_step": s, "mean_D_obs": float(Dq.mean()),
                        "mean_D_null": float(Db.mean()),
                        "excess_D": float(E.mean()),
                        "ci_lo": float(np.quantile(bs, .025)),
                        "ci_hi": float(np.quantile(bs, .975)),
                        "frac_questions_positive": float((E > 0).mean()),
                        "gamma_pq_mean": float(gq.mean()),
                        "gamma_pq_sd": float(gq.std()),
                        "p_parametric_one_sided": pp_})
        log(f"  step {s:>3}: ExcessD(pq) {E.mean():+.4f} "
            f"[{np.quantile(bs, .025):+.4f},{np.quantile(bs, .975):+.4f}]  "
            f"frac q>0 {(E > 0).mean():.3f}  gamma_q {gq.mean():.2f}"
            f"+-{gq.std():.2f}  p {pp_:.4f}")
    with (OUT / "excess_D_per_question_gamma.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pq_rows[0]))
        w.writeheader()
        w.writerows(pq_rows)

    # ------- 11. direction: do residuals track downstream solution multiplicity?
    # Excess residual only says M0 is the wrong shape. Selective reweighting
    # predicts a direction: branches with more downstream solutions should be
    # up-weighted relative to what frequency sharpening alone predicts.
    log("\n=== direction: residual vs log solution multiplicity")
    MU = np.zeros((len(elig), Bmax))
    for i, q in enumerate(elig):
        for j, b in enumerate(D[T0][q][1]):
            MU[i, j] = math.log(max(1, mult[(q, b)]))

    def rank1(v):
        o = np.argsort(v, kind="stable")
        r = np.empty(len(v), float)
        r[o] = np.arange(len(v), dtype=float)
        return r

    def mean_within_q_spearman(Rmat):
        vals = []
        for i in range(len(elig)):
            k = int(M[i].sum())
            if k < 3 or len(set(MU[i, :k])) < 2 or len(set(Rmat[i, :k])) < 2:
                continue
            a, b_ = rank1(Rmat[i, :k]), rank1(MU[i, :k])
            c = np.corrcoef(a, b_)[0, 1]
            if np.isfinite(c):
                vals.append(c)
        return np.array(vals)

    dir_rows = []
    for s in later:
        pred = sharpen_np(LP0, M, gam[s])
        Rr = np.where(M, np.log(np.maximum(smooth_np(Ct[s], M), 1e-300))
                      - np.log(np.maximum(pred, 1e-300)), 0.0)
        v = mean_within_q_spearman(Rr)
        vb = v[np.random.default_rng(SEED + 2).integers(0, len(v),
                                                        (N_QBOOT, len(v)))]
        bs = vb.mean(1)
        dir_rows.append({"training_step": s, "n_questions_used": len(v),
                         "mean_within_question_spearman": float(v.mean()),
                         "ci_lo": float(np.quantile(bs, .025)),
                         "ci_hi": float(np.quantile(bs, .975)),
                         "frac_questions_positive": float((v > 0).mean())})
        log(f"  step {s:>3}: rho(resid, log mult) {v.mean():+.4f} "
            f"[{np.quantile(bs, .025):+.4f},{np.quantile(bs, .975):+.4f}]  "
            f"n_q {len(v)}  frac>0 {(v > 0).mean():.3f}")
    with (OUT / "residual_vs_multiplicity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dir_rows[0]))
        w.writeheader()
        w.writerows(dir_rows)

    json.dump({"excess": ex_rows, "replication": rep_rows,
               "reliability": rel_rows, "audit": A, "null": null_rows,
               "per_question_gamma": pq_rows, "direction": dir_rows,
               "n_eligible": len(elig)},
              (OUT / "summary_stats.json").open("w"), indent=2)

    # ------------------------------------------------------------ figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st = [r["training_step"] for r in ex_rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(st, [r["excess_D"] for r in ex_rows],
                yerr=[[r["excess_D"] - r["ci_lo"] for r in ex_rows],
                      [r["ci_hi"] - r["excess_D"] for r in ex_rows]],
                marker="o", capsize=4, color="#c0392b")
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("training step")
    ax.set_ylabel(r"ExcessD  =  $D_t(q) - E_0[D_t(q)]$")
    ax.set_title("Residual magnitude above the M0 parametric-bootstrap null")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_excessD_trajectory.png", dpi=160)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(boot[PRIMARY]["meanD"], bins=40, color="#95a5a6",
            label="M0 null (2000 draws)")
    ax.axvline(Dobs[PRIMARY].mean(), color="#c0392b", lw=2,
               label=f"observed (step {PRIMARY})")
    ax.set_xlabel("mean |residual| over questions")
    ax.set_ylabel("draws")
    ax.set_title(f"Step {PRIMARY}: observed vs the M0 null")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig2_step100_null_vs_observed.png", dpi=160)

    if rep_rows:
        fig, axes = plt.subplots(1, len(rep_rows), figsize=(4 * len(rep_rows), 4),
                                 squeeze=False)
        for k, r in enumerate(rep_rows):
            s = r["training_step"]
            Co, _ = pack(d_old[s], elig, Bmax)
            go = r["gamma_old"]
            Ro = (np.log(np.maximum(smooth_np(Co, Mo), 1e-300))
                  - np.log(np.maximum(sharpen_np(LP0o, Mo, go), 1e-300)))
            Rn = (np.log(np.maximum(smooth_np(Ct[s], M), 1e-300))
                  - np.log(np.maximum(sharpen_np(LP0, M, gam[s]), 1e-300)))
            ax = axes[0][k]
            ax.scatter(Ro[M], Rn[M], s=6, alpha=.4, color="#2c3e50")
            ax.axhline(0, color="k", lw=.5)
            ax.axvline(0, color="k", lw=.5)
            ax.set_xlabel("residual, old 128")
            ax.set_ylabel("residual, fresh 512")
            ax.set_title(f"step {s}  rho={r['spearman_old_new']:+.3f}")
        fig.tight_layout()
        fig.savefig(FIG / "fig3_replication_old_vs_new.png", dpi=160)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["training_step"] for r in rel_rows],
            [r["spearman"] for r in rel_rows], marker="s", color="#2980b9",
            label="split-half Spearman (256/256)")
    ax.plot([r["training_step"] for r in rel_rows],
            [r["sign_agreement"] for r in rel_rows], marker="^",
            color="#27ae60", label="sign agreement")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("training step")
    ax.set_title("Within-checkpoint reliability of the residual")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig4_split_half_reliability.png", dpi=160)
    log(f"\nfigures -> {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
