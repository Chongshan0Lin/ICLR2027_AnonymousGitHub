#!/usr/bin/env python3
"""Balanced Track A analysis: the same-state omnibus at K=4 arms, every root.

Two frozen tests, generalised from 3 arms to any number of arms and run at both
K=3 and K=4. The K=3 pass is a reproduction check: if it does not return the
frozen 67 / 56 / 93 / 84, the generalisation is wrong and the K=4 numbers are
not to be trusted.

Test 1, blocked permutation (primary). Statistic = mean pairwise JS over all
arm pairs, on p_hat smoothed over the frozen feasible support. All arms are
evaluated on the same prompts at eval seed 909 and vLLM derives the per-slot
sampler seed as `eval_seed + rollout_index`, so slot i is shared across arms.
The design-matched null permutes the arm labels **within each slot**. Labels
outside the feasible support go to a sink category that permutes with the rest
but never enters p_hat, so the denominator moves with the permutation exactly as
the frozen definition requires.

Test 2, marginal omnibus (secondary, from the v4 addendum). Ratio-estimator
influence functions, stacked contrast against the trunk arm, paired Rademacher
wild bootstrap, Moore-Penrose pseudo-inverse with rank as the degrees of
freedom.

Nothing in analysis_v3/, analysis_v4_cpu/ or analysis_v4_cpu_addendum/ is
touched; this writes to its own namespace.
"""
import csv, hashlib, json, time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
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


R11 = ART / "round11_3b_trunk"
REF = ART / "countdown_selectivity"
OUT = _mkout(OUTPUTS / "analysis_balanced_k4")
ALPHA, B_PERM, B_WILD, Q_BH, WORKERS = 0.5, 20000, 20000, 0.05, 32
ROOTS = [150, 125, 100, 75]
SEED = 20260928
RARE = 5
OTHER = "OTHER"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

FEAS = defaultdict(set)
for l in (REF / "solver_families.jsonl").open():
    d = json.loads(l)
    FEAS[d["question_id"]].add(d["family_id"])


def load(tag):
    out = {}
    for l in (R11 / "eval" / f"{tag}.jsonl").open():
        r = json.loads(l)
        out[r["question_id"]] = {int(x["rollout_id"]):
                                 (str(x["entrance_family"]) if x["entrance_family"] is not None
                                  else None) for x in r["rollouts"]}
    return out


def phat(C, K):
    F = C[..., :K]
    return (F + ALPHA) / (F.sum(-1, keepdims=True) + ALPHA * K)


def mean_pair_js(P):
    """P: (..., A, K) -> mean JS over all arm pairs."""
    A = P.shape[-2]
    tot = 0.0
    for a, b in combinations(range(A), 2):
        pa, pb = P[..., a, :], P[..., b, :]
        m = 0.5 * (pa + pb)
        tot = tot + np.maximum(0.0, 0.5 * (pa * np.log(pa / m)).sum(-1)
                               + 0.5 * (pb * np.log(pb / m)).sum(-1))
    return tot / (A * (A - 1) / 2)


def blocked(args):
    """blocked within-slot permutation for one prompt, any number of arms."""
    root, q, labs, seed = args
    sup = sorted(FEAS[q]); K = len(sup); SINK = K
    idx = {b: i for i, b in enumerate(sup)}
    slots = sorted(set.intersection(*[set(d) for d in labs]))
    A = len(labs)
    L = np.empty((len(slots), A), np.int64)
    for i, s in enumerate(slots):
        for a in range(A):
            L[i, a] = idx.get(labs[a][s], SINK)
    obs_c = np.zeros((A, K + 1))
    for a in range(A):
        obs_c[a] = np.bincount(L[:, a], minlength=K + 1)
    if (obs_c[:, :K].sum(1) == 0).all():
        return root, q, {"computable": False, "reason": "no arm has a feasible slot"}
    obs = float(mean_pair_js(phat(obs_c, K)))
    # only slots whose labels are not all identical can change anything
    var = ~(L == L[:, :1]).all(1)
    nv = int(var.sum())
    if nv == 0:
        return root, q, {"computable": True, "p": 1.0, "statistic": obs, "n_slots": len(slots),
                         "n_variable_slots": 0, "K": K, "A": A,
                         "slot_disagreement": 0.0}
    rng = np.random.default_rng([seed, root, abs(hash(q)) % (2 ** 31), A])
    Lv = L[var]
    B = B_PERM
    perm = np.argsort(rng.random((B, nv, A)), axis=2)
    g = np.take_along_axis(np.broadcast_to(Lv, (B, nv, A)), perm, axis=2)
    base = np.bincount(L[~var, 0], minlength=K + 1) if (~var).sum() else np.zeros(K + 1)
    flat = (g + (K + 1) * np.arange(A)[None, None, :]
            + (A * (K + 1)) * np.arange(B)[:, None, None]).ravel()
    cnt = np.bincount(flat, minlength=B * A * (K + 1)).reshape(B, A, K + 1).astype(float)
    cnt += base[None, None, :]
    nd = mean_pair_js(phat(cnt, K))
    p = float((1 + int(np.sum(nd >= obs))) / (B + 1))
    dis = float(np.mean([len({labs[a][s] for a in range(A)}) > 1 for s in slots]))
    return root, q, {"computable": True, "p": p, "statistic": obs, "n_slots": len(slots),
                     "n_variable_slots": nv, "K": K, "A": A, "slot_disagreement": dis}


def marginal(args):
    """K-arm marginal omnibus: ratio-estimator influence functions + wild bootstrap."""
    root, q, labs, seed = args
    sup_all = sorted(FEAS[q]); A = len(labs)
    slots = sorted(set.intersection(*[set(d) for d in labs]))
    n = len(slots)
    pooled = Counter()
    for d in labs:
        for s in slots:
            pooled[d[s]] += 1
    keep = [b for b in sup_all if pooled.get(b, 0) >= RARE]
    cats = keep + [OTHER]; K = len(cats)
    ix = {b: i for i, b in enumerate(cats)}

    def cat(v):
        return ix.get(v, ix[OTHER]) if v in sup_all else None

    X = np.zeros((A, n, K)); F = np.zeros((A, n))
    for a in range(A):
        for i, s in enumerate(slots):
            c = cat(labs[a][s])
            if c is None:
                continue
            X[a, i, c] = 1.0; F[a, i] = 1.0
    muF = F.mean(1)
    if np.any(muF <= 0):
        return root, q, {"computable": False, "reason": "an arm has zero feasible slots"}
    ph = X.sum(1) / F.sum(1)[:, None]
    PHI = np.empty((n, A * K))
    for a in range(A):
        PHI[:, a * K:(a + 1) * K] = (X[a] - ph[a][None, :] * F[a][:, None]) / muF[a]
    C = np.zeros(((A - 1) * K, A * K))
    for j in range(A - 1):
        for k in range(K):
            C[j * K + k, (j + 1) * K + k] = 1.0
            C[j * K + k, k] = -1.0
    delta = C @ np.concatenate(ph)
    G = PHI - PHI.mean(0)
    S = C @ ((G.T @ G) / n) @ C.T
    Sp = np.linalg.pinv(S, rcond=1e-10)
    rank = int(np.linalg.matrix_rank(S, tol=1e-10))
    if rank == 0:
        return root, q, {"computable": False, "reason": "zero-rank covariance"}
    stat = float(n * delta @ Sp @ delta)
    rng = np.random.default_rng([seed, root, abs(hash(q)) % (2 ** 31), A, 1])
    W = rng.choice([-1.0, 1.0], size=(B_WILD, n))
    DB = ((W @ G) / n) @ C.T
    null = np.einsum("bi,ij,bj->b", DB, Sp, DB) * n
    return root, q, {"computable": True, "p": float((1 + int(np.sum(null >= stat))) / (B_WILD + 1)),
                     "statistic": stat, "rank": rank, "K": K, "A": A, "n_slots": n}


def bh(p, q=Q_BH):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); prev = 1.0
    for r in range(m - 1, -1, -1):
        i = o[r]; prev = adj[i] = min(prev, p[i] * m / (r + 1))
    return adj <= q, adj


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tags = {"trunk": "trunk_s0_step175"}
    L = {"trunk": load(tags["trunk"])}
    for root in ROOTS:
        for s in (1, 2, 3):
            L[(root, s)] = load(f"fork_s{s}_r{root:03d}_step175")
    qs = sorted(set.intersection(*[set(v) for v in L.values()]))
    print(f"  arms loaded, common prompts {len(qs)}")

    frozen = json.load((R11 / "analysis_v4_cpu" / "blocked_permutation_final.json").open())["roots"]
    res = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "design": "Track A multi-root full-state forks, now balanced: trunk + seed1 + seed2 "
                     "+ seed3 at every root",
           "tests": {"blocked": f"within-slot permutation, B={B_PERM}, mean pairwise JS, "
                                f"BH q={Q_BH} within each root",
                     "marginal": f"ratio-estimator influence functions, paired Rademacher wild "
                                 f"bootstrap B={B_WILD}, pinv, rank as dof, BH q={Q_BH}"},
           "frozen_K3_reference": {r: frozen[str(r)]["significant"] for r in ROOTS},
           "arms": {}}
    rows = [("arms", "root", "test", "significant", "computable", "uncomputable",
             "mean_statistic", "mean_slot_disagreement")]
    for A, armsel in ((3, [1, 2]), (4, [1, 2, 3])):
        res["arms"][A] = {}
        for root in ROOTS:
            labs_of = lambda q: [L["trunk"][q]] + [L[(root, s)][q] for s in armsel]
            jobs = [(root, q, labs_of(q), SEED) for q in qs]
            with ProcessPoolExecutor(max_workers=WORKERS) as ex:
                bl = list(ex.map(blocked, jobs, chunksize=1))
                mg = list(ex.map(marginal, jobs, chunksize=2))
            out = {}
            for nm, dat in (("blocked", bl), ("marginal", mg)):
                ok = [(q_, d) for _, q_, d in dat if d["computable"]]
                bad = [(q_, d) for _, q_, d in dat if not d["computable"]]
                rej, adj = bh([d["p"] for _, d in ok]) if ok else (np.array([]), np.array([]))
                ms = float(np.mean([d["statistic"] for _, d in ok])) if ok else float("nan")
                ds = float(np.mean([d.get("slot_disagreement", np.nan) for _, d in ok])) \
                    if nm == "blocked" else float("nan")
                out[nm] = {"significant": int(rej.sum()), "computable": len(ok),
                           "uncomputable": len(bad),
                           "uncomputable_reasons": dict(Counter(d["reason"] for _, d in bad)),
                           "mean_statistic": ms, "mean_slot_disagreement": ds,
                           "per_prompt": {q_: {**d, "q_adj": float(adj[i]), "reject": bool(rej[i])}
                                          for i, (q_, d) in enumerate(ok)}}
                rows.append((A, root, nm, int(rej.sum()), len(ok), len(bad),
                             round(ms, 8), round(ds, 6) if ds == ds else "NA"))
                tag = ""
                if A == 3 and nm == "blocked":
                    f = frozen[str(root)]["significant"]
                    tag = f"   [frozen K=3 reference {f}: " \
                          f"{'REPRODUCED' if int(rej.sum()) == f else 'DIFFERS'}]"
                print(f"  K={A} root {root:>3} {nm:9} significant {int(rej.sum()):>3}/"
                      f"{len(ok)}  stat {ms:.5f}{tag}", flush=True)
            res["arms"][A][root] = out
    json.dump(res, (OUT / "balanced_omnibus.json").open("w"), indent=1, default=str)
    with (OUT / "balanced_omnibus_summary.csv").open("w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    print(f"  elapsed {time.time() - t0:.0f}s")
    for f in ("balanced_omnibus.json", "balanced_omnibus_summary.csv"):
        print(f"  {f} sha256 {sha(OUT / f)}")


if __name__ == "__main__":
    main()
