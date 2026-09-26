#!/usr/bin/env python3
"""Round 9 §5-§8, §12, §13, §15: nested predictive models of the next checkpoint.

Each model predicts the next checkpoint's entrance counts as a softmax over a
question's branches, so question-level covariates cancel and only branch-varying
terms enter. Scoring is held-out multinomial NLL under a frozen 5-fold split by
question, with every seed and checkpoint of a question kept in one fold.

M3 cannot be (z_t, v_t, z_{t-1}) as the plan writes it: v_t = z_t - z_{t-1}
makes the three exactly collinear. It adds the previous velocity instead, which
is the first genuinely new history term (equivalently z_{t-2}).
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2]  # artifact repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from r6_lib import fit, predict  # noqa: E402

import os as _os
OUT = Path(_os.environ.get("ARTIFACT_OUTPUTS",
          R / "outputs")) / "round9_markov_test"
OUT.mkdir(parents=True, exist_ok=True)
CTRL = ["log_multiplicity", "op_plus", "op_minus", "op_mult", "op_div",
        "log_first_operand"]
MODELS = {"M1": ["z_t"] + CTRL,
          "M2": ["z_t", "velocity_t"] + CTRL,
          "M3": ["z_t", "velocity_t", "velocity_t_minus_1"] + CTRL}
N_BOOT = 10000
L2 = 1e-3


def load(path=None):
    rows = []
    with open(path or (OUT / "transition_table.csv")) as f:
        for d in csv.DictReader(f):
            op = d["operator"]
            d["op_plus"] = float(op == "+")
            d["op_minus"] = float(op == "-")
            d["op_mult"] = float(op == "*")
            d["op_div"] = float(op == "/")
            for k in ("z_t", "velocity_t", "velocity_t_minus_1", "next_count",
                      "log_multiplicity", "log_first_operand", "step_t",
                      "next_total", "z_t_minus_1", "n_total_t",
                      "branch_count_t"):
                d[k] = float(d[k]) if d[k] != "" else float("nan")
            rows.append(d)
    return rows


def units_for(rows, feats):
    by = defaultdict(list)
    for r in rows:
        by[(r["seed"], r["step_t"], r["question_id"])].append(r)
    out = []
    for k, rs in by.items():
        X = np.array([[r[f] for f in feats] for r in rs], float)
        if not np.isfinite(X).all():
            continue
        X = X - X.mean(0, keepdims=True)
        y = np.array([r["next_count"] for r in rs], float)
        out.append({"key": k, "seed": k[0], "step": k[1], "question_id": k[2],
                    "X": X, "y": y, "N": y.sum(),
                    "branches": [r["branch_id"] for r in rs]})
    return [u for u in out if u["N"] > 0]


def cv_nll(rows, feats, fold, group="question_id"):
    """held-out NLL per count, plus per-unit detail for localisation."""
    units = units_for(rows, feats)
    per = {}
    for k in sorted(set(fold.values())):
        tr = [u for u in units if fold[u[group]] != k]
        te = [u for u in units if fold[u[group]] == k]
        if not tr or not te:
            continue
        beta, sd = fit(tr, feats, L2)
        for u in te:
            p = predict(u, beta, sd)
            per[u["key"]] = {"nll": -float(u["y"] @ np.log(np.maximum(p, 1e-300))),
                             "N": u["N"], "step": u["step"], "seed": u["seed"],
                             "question_id": u["question_id"], "p": p, "y": u["y"]}
    tot = sum(v["nll"] for v in per.values())
    n = sum(v["N"] for v in per.values())
    return tot / n, per


def gain_ci(perA, perB, rng):
    """question-clustered bootstrap of the NLL gain A -> B."""
    qs = sorted({v["question_id"] for v in perA.values()})
    a, b, c = defaultdict(float), defaultdict(float), defaultdict(float)
    for k in perA:
        q = perA[k]["question_id"]
        a[q] += perA[k]["nll"]
        b[q] += perB[k]["nll"]
        c[q] += perA[k]["N"]
    A = np.array([a[q] for q in qs])
    B = np.array([b[q] for q in qs])
    C = np.array([c[q] for q in qs])
    idx = rng.integers(0, len(qs), (N_BOOT, len(qs)))
    d = (A[idx].sum(1) - B[idx].sum(1)) / C[idx].sum(1)
    return ((A.sum() - B.sum()) / C.sum(),
            float(np.quantile(d, .025)), float(np.quantile(d, .975)))


def main() -> int:
    rows = load()
    qs = sorted({r["question_id"] for r in rows})
    rng = np.random.default_rng(20260916)
    perm = rng.permutation(len(qs))
    fold = {qs[perm[i]]: i % 5 for i in range(len(qs))}
    (OUT / "fold_assignment.json").write_text(json.dumps(
        {"n_fold": 5, "seed": 20260916, "assignment": fold}, indent=2))

    # each comparison is scored on the rows where both models are defined
    set_M2 = [r for r in rows if np.isfinite(r["velocity_t"])]
    set_M3 = [r for r in rows if np.isfinite(r["velocity_t_minus_1"])]
    print(f"M1 vs M2 on {len({(r['seed'], r['step_t']) for r in set_M2})} "
          f"seed-transitions; M2 vs M3 on "
          f"{len({(r['seed'], r['step_t']) for r in set_M3})}")

    res, gains = [], []
    n1, p1 = cv_nll(rows, MODELS["M1"], fold)
    print(f"\nM1 on all transitions: held-out NLL {n1:.5f}")
    for name, rs in (("M1_vs_M2", set_M2), ("M2_vs_M3", set_M3)):
        a, b = ("M1", "M2") if name == "M1_vs_M2" else ("M2", "M3")
        na, pa = cv_nll(rs, MODELS[a], fold)
        nb, pb = cv_nll(rs, MODELS[b], fold)
        g, lo, hi = gain_ci(pa, pb, rng)
        gains.append({"comparison": f"{a}->{b}", "nll_a": na, "nll_b": nb,
                      "delta_nll": g, "ci_lo": lo, "ci_hi": hi,
                      "n_units": len(pa)})
        print(f"  {a} {na:.5f}  ->  {b} {nb:.5f}   dNLL {g:+.6f} "
              f"[{lo:+.6f},{hi:+.6f}]")
        res += [{"model": a, "scope": name, "heldout_nll": na},
                {"model": b, "scope": name, "heldout_nll": nb}]

    # ---- §12 temporal localisation
    print("\n=== gain by transition")
    tl = []
    for step in sorted({r["step_t"] for r in set_M2}):
        sub = [r for r in set_M2 if r["step_t"] == step]
        a, _ = cv_nll(sub, MODELS["M1"], fold)
        b, _ = cv_nll(sub, MODELS["M2"], fold)
        row = {"step_t": int(step), "nll_M1": a, "nll_M2": b, "gain_M1_M2": a - b}
        if any(np.isfinite(r["velocity_t_minus_1"]) for r in sub):
            c, _ = cv_nll(sub, MODELS["M3"], fold)
            row.update({"nll_M3": c, "gain_M2_M3": b - c})
        tl.append(row)
        print(f"  {int(step)}->{int(step)+5}: M1 {a:.5f}  M2 {b:.5f}  "
              f"gain {a-b:+.6f}"
              + (f"   M3 {row['nll_M3']:.5f}  gain {row['gain_M2_M3']:+.6f}"
                 if "nll_M3" in row else ""))
    with (OUT / "models/temporal_localization.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step_t", "nll_M1", "nll_M2",
                                          "gain_M1_M2", "nll_M3", "gain_M2_M3"])
        w.writeheader()
        w.writerows(tl)

    # ---- §15 leave-one-seed-out
    print("\n=== leave-one-seed-out")
    lo_rows = []
    seeds = sorted({r["seed"] for r in rows})
    sfold = {s: i for i, s in enumerate(seeds)}
    for name, rs, (a, b) in (("M1_vs_M2", set_M2, ("M1", "M2")),
                             ("M2_vs_M3", set_M3, ("M2", "M3"))):
        na, pa = cv_nll(rs, MODELS[a], sfold, group="seed")
        nb, pb = cv_nll(rs, MODELS[b], sfold, group="seed")
        lo_rows.append({"comparison": f"{a}->{b}", "nll_a": na, "nll_b": nb,
                        "delta_nll": na - nb})
        print(f"  {a} {na:.5f} -> {b} {nb:.5f}   dNLL {na-nb:+.6f}")
    with (OUT / "models/seed_cv_predictions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lo_rows[0]))
        w.writeheader()
        w.writerows(lo_rows)

    # ---- §13 direct velocity coefficient on delta z
    print("\n=== velocity coefficient on next-step change")
    vt = np.array([r["velocity_t"] for r in set_M2])
    zt = np.array([r["z_t"] for r in set_M2])
    dz = np.array([float(r["next_z"]) - r["z_t"] for r in set_M2])
    qid = np.array([r["question_id"] for r in set_M2])
    X = np.column_stack([np.ones(len(zt)), zt, vt])
    beta = np.linalg.lstsq(X, dz, rcond=None)[0]
    uq = np.array(sorted(set(qid)))
    idxq = {q: np.where(qid == q)[0] for q in uq}
    draws = []
    for _ in range(2000):
        sel = np.concatenate([idxq[q] for q in rng.choice(uq, len(uq))])
        draws.append(np.linalg.lstsq(X[sel], dz[sel], rcond=None)[0])
    draws = np.array(draws)
    vrow = {"beta_z_t": float(beta[1]), "beta_velocity": float(beta[2]),
            "vel_ci_lo": float(np.quantile(draws[:, 2], .025)),
            "vel_ci_hi": float(np.quantile(draws[:, 2], .975)),
            "z_ci_lo": float(np.quantile(draws[:, 1], .025)),
            "z_ci_hi": float(np.quantile(draws[:, 1], .975))}
    print(f"  beta_z_t      {beta[1]:+.4f} "
          f"[{vrow['z_ci_lo']:+.4f},{vrow['z_ci_hi']:+.4f}]")
    print(f"  beta_velocity {beta[2]:+.4f} "
          f"[{vrow['vel_ci_lo']:+.4f},{vrow['vel_ci_hi']:+.4f}]")
    with (OUT / "models/velocity_coefficient.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(vrow))
        w.writeheader()
        w.writerow(vrow)

    for m in ("M1", "M2", "M3"):
        rs = set_M3 if m == "M3" else set_M2
        u = units_for(rs, MODELS[m])
        beta, sd = fit(u, MODELS[m], L2)
        with (OUT / f"models/{m}_coefficients.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["feature", "coefficient"])
            for k, v in zip(MODELS[m], (beta / sd).tolist()):
                w.writerow([k, v])
    json.dump({"gains": gains, "temporal": tl, "leave_one_seed_out": lo_rows,
               "velocity": vrow, "M1_all_transitions_nll": n1},
              (OUT / "models/summary.json").open("w"), indent=2)
    with (OUT / "models/nll_gains.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(gains[0]))
        w.writeheader()
        w.writerows(gains)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
