#!/usr/bin/env python3
"""Shared machinery for the Round 6 predictive-state models.

Every model predicts the next checkpoint's entrance counts for one question as a
softmax over that question's solver-feasible branches. Because the softmax is
taken within a question, any question-level covariate (arity, target magnitude)
cancels exactly; only branch-varying features can enter, which is why the
control set is restricted to per-branch quantities.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

R = Path(__file__).resolve().parents[2]  # artifact repo root
OUT = R / "results/round6_predictive_state"

CTRL = ["log_multiplicity", "op_plus", "op_minus", "op_mult", "op_div",
        "log_first_operand"]
MODELS = {
    "M0": ["z_0"],
    "M1": ["z_t", "z_0"] + CTRL,
    "M2": ["z_t", "z_0", "s_t_centered"] + CTRL,
    "M3": ["z_t", "z_0", "s_t_centered", "v_t"] + CTRL,
    "M4": ["z_t", "z_0", "s_t_centered", "v_t", "v_t_minus_1",
           "cum_visit", "cum_success"] + CTRL,
}
# z_t_minus_1 is exactly z_t - v_t, so it cannot be added on top of M3; the
# long-memory model instead adds the previous velocity (an acceleration term,
# which needs a second lag) and cumulative visitation/success.


def load_transitions(path=None):
    path = path or (OUT / "transition_table.csv")
    rows = []
    with open(path) as f:
        for d in csv.DictReader(f):
            for k, v in list(d.items()):
                if k not in ("question_id", "branch_id", "operator"):
                    d[k] = float(v)
            op = d["operator"]
            d["op_plus"] = float(op == "+")
            d["op_minus"] = float(op == "-")
            d["op_mult"] = float(op == "*")
            d["op_div"] = float(op == "/")
            d["log_first_operand"] = math.log(max(1.0, abs(d["first_operand"])))
            rows.append(d)
    return rows


def units_for(rows, feats):
    """One softmax unit per (transition, question): centered X, counts, N."""
    by = defaultdict(list)
    for r in rows:
        by[(r["step_t"], r["question_id"])].append(r)
    out = []
    for k, rs in by.items():
        X = np.array([[r[f] for f in feats] for r in rs], float)
        X = X - X.mean(0, keepdims=True)          # identified up to a constant
        y = np.array([r["count_next"] for r in rs], float)
        out.append({"key": k, "step": k[0], "question_id": k[1], "X": X,
                    "y": y, "N": y.sum(),
                    "branches": [r["branch_id"] for r in rs]})
    return out


def scale_of(units):
    X = np.vstack([u["X"] for u in units])
    sd = X.std(0)
    sd[sd < 1e-9] = 1.0
    return sd


def nll_and_grad(beta, units, sd, l2):
    tot = 0.0
    g = np.zeros_like(beta)
    n = 0.0
    for u in units:
        eta = (u["X"] / sd) @ beta
        eta -= eta.max()
        e = np.exp(eta)
        p = e / e.sum()
        tot -= float(u["y"] @ np.log(np.maximum(p, 1e-300)))
        g += (u["X"] / sd).T @ (u["N"] * p - u["y"])
        n += u["N"]
    return (tot / n + l2 * float(beta @ beta),
            g / n + 2 * l2 * beta)


def fit(units, feats, l2=1e-3, sd=None):
    sd = scale_of(units) if sd is None else sd
    b0 = np.zeros(len(feats))
    r = minimize(nll_and_grad, b0, args=(units, sd, l2), jac=True,
                 method="L-BFGS-B", options={"maxiter": 500})
    return r.x, sd


def predict(u, beta, sd):
    eta = (u["X"] / sd) @ beta
    eta -= eta.max()
    e = np.exp(eta)
    return e / e.sum()


def eval_units(units, beta, sd):
    """held-out NLL per count plus distributional and ranking metrics."""
    tot = n = 0.0
    js, sp, top1 = [], [], []
    for u in units:
        p = predict(u, beta, sd)
        tot -= float(u["y"] @ np.log(np.maximum(p, 1e-300)))
        n += u["N"]
        q = (u["y"] + 0.5) / (u["N"] + 0.5 * len(u["y"]))
        m = 0.5 * (p + q)
        js.append(float(0.5 * (p * np.log(np.maximum(p / m, 1e-300))).sum()
                        + 0.5 * (q * np.log(np.maximum(q / m, 1e-300))).sum()))
        if len(u["y"]) > 1 and len(set(u["y"])) > 1:
            rp = np.argsort(np.argsort(p)).astype(float)
            rq = np.argsort(np.argsort(u["y"])).astype(float)
            if rp.std() > 0 and rq.std() > 0:
                sp.append(float(np.corrcoef(rp, rq)[0, 1]))
        top1.append(float(np.argmax(p) == np.argmax(u["y"])))
    return {"nll": tot / n, "js": float(np.mean(js)),
            "rank_spearman": float(np.mean(sp)) if sp else float("nan"),
            "top1_acc": float(np.mean(top1)), "n_counts": n,
            "n_units": len(units)}


def folds_of(questions, n_fold=5, seed=20260915):
    qs = sorted(set(questions))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(qs))
    return {qs[perm[i]]: i % n_fold for i in range(len(qs))}


# --------------------------------------------------------- transition builder
ALPHA, BETA_A, BETA_B = 0.5, 0.5, 0.5


def make_transitions(steps, elig, sup_of, cnt, cor, ntot, mult, qmeta):
    """Branch-transition rows from raw counts.

    Used for both the observed data and every simulated bootstrap trajectory, so
    that the null is processed through exactly the same estimator as the data.
    """
    P, S, Z = {}, {}, {}
    for s in steps:
        for q in elig:
            sup = sup_of[q]
            K = len(sup)
            tot = ntot[(s, q)]
            lp = {}
            for b in sup:
                p = (cnt[(s, q, b)] + ALPHA) / (tot + ALPHA * K)
                P[(s, q, b)] = p
                lp[b] = math.log(p)
                nb, cb = cnt[(s, q, b)], cor[(s, q, b)]
                S[(s, q, b)] = (cb + BETA_A) / (nb + BETA_A + BETA_B)
            m = sum(lp.values()) / K
            for b in sup:
                Z[(s, q, b)] = lp[b] - m
    tr = []
    for i in range(2, len(steps) - 1):
        tm2, tm1, t, tp1 = steps[i - 2], steps[i - 1], steps[i], steps[i + 1]
        for q in elig:
            sup = sup_of[q]
            den = max(1, sum(cnt[(t, q, b)] for b in sup))
            sbar = sum(S[(t, q, b)] * cnt[(t, q, b)] for b in sup) / den
            cv = {b: math.log1p(sum(cnt[(u, q, b)] for u in steps if u <= t))
                  for b in sup}
            cs = {b: math.log1p(sum(cor[(u, q, b)] for u in steps if u <= t))
                  for b in sup}
            for b in sup:
                tr.append({
                    "question_id": q, "branch_id": b,
                    "step_t": float(t), "step_next": float(tp1),
                    "step_prev": float(tm1), "step_prev2": float(tm2),
                    "delta_t": float(tp1 - t),
                    "z_t": Z[(t, q, b)], "z_t_minus_1": Z[(tm1, q, b)],
                    "z_t_minus_2": Z[(tm2, q, b)],
                    "v_t": Z[(t, q, b)] - Z[(tm1, q, b)],
                    "v_t_minus_1": Z[(tm1, q, b)] - Z[(tm2, q, b)],
                    "cum_visit": cv[b], "cum_success": cs[b],
                    "z_0": Z[(steps[0], q, b)],
                    "p_t": P[(t, q, b)], "p_0": P[(steps[0], q, b)],
                    "s_t": S[(t, q, b)],
                    "s_t_centered": S[(t, q, b)] - sbar,
                    "branch_rollout_count_t": float(cnt[(t, q, b)]),
                    "count_next": float(cnt[(tp1, q, b)]),
                    "n_next_total": float(ntot[(tp1, q)]),
                    "z_next": Z[(tp1, q, b)],
                    "delta_z": Z[(tp1, q, b)] - Z[(t, q, b)],
                    "log_multiplicity": math.log(max(1, mult[(q, b)])),
                    "operator": b[-1],
                    "first_operand": float(int(b[:-1])),
                    "log_first_operand": math.log(max(1.0, abs(int(b[:-1])))),
                    "op_plus": float(b[-1] == "+"), "op_minus": float(b[-1] == "-"),
                    "op_mult": float(b[-1] == "*"), "op_div": float(b[-1] == "/"),
                    "arity": float(len(qmeta[q]["nums"])),
                    "target": float(qmeta[q]["target"]),
                    "K_q": float(len(sup))})
    return tr
