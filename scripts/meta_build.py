#!/usr/bin/env python3
"""Three meta-figures in the Pythia (ICML 2023) small-multiple grammar.

One scientific quantity per meta-figure, one panel per trajectory, identical
axes, one global legend, no summary panel.

Trajectory labels are the ones the restructuring brief uses (a)-(e); the
registry track id is carried alongside because they differ.

    (a) Qwen2.5-3B-Instruct / Countdown, self-trained     registry track A
    (b) Qwen2.5-7B / Countdown                            registry track C
    (c) public Qwen2.5-3B / Countdown, dense continuation registry track D
    (d) Qwen-derived 1.5B RLVR-Code / HumanEval           registry track E
    (e) OLMo-3.1-7B-RL-Zero-Code / HumanEval              registry track F

(d) and (e) carry no branch-level rows at all: the branch labeller failed its
preregistered blind audit, so every branch-space quantity is quarantined. Their
panels are drawn empty on the shared axes with a subdued note. The explicit
MISSING_* codes live in the audit report, not in the publication figure.
"""
import csv, hashlib, json, platform, subprocess, time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
D3 = ART
FIG, REP = OUTPUTS / "figures", OUTPUTS / "reports"
DAT = FIG / "meta_data"
DATR = ART / "figures" / "meta_data"   # frozen reads
ALPHA, SEED = 0.5, 20260930
NPERM, NBOOT = 200, 2000
NFLOOR = 25
NULL_PRIMARY = "question_specific_gamma"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

# label, registry track, model/task, design, checkpoint span for the subtitle
# Colour carries meaning, not identity: OBS is "observed" in every panel of every
# meta-figure and NULLC is "null" in every panel, so the one global legend is
# literal. Panel identity is carried by position and title, as in Pythia Fig. 4/7.
OBS, NULLC = "#1f4e9c", "0.55"
PANELS = [
    ("a", "A", "Qwen2.5-3B / Countdown", "full-state fork", OBS),
    ("b", "C", "Qwen2.5-7B / Countdown", "independent seeds", OBS),
    ("c", "D", "Qwen2.5-3B public / Countdown", "dense continuation", OBS),
    ("d", "E", "Qwen-1.5B Code / HumanEval", "single lineage", OBS),
    ("e", "F", "OLMo-3.1-7B Code / HumanEval", "single lineage", OBS),
]
# The Code branch representation is now the deterministic AST label frozen in
# code_branch_codebook_v2.json, so panels (d)/(e) carry real curves wherever the
# quantity only needs one lineage. Meta-figure 2 still cannot be populated: it
# compares two training realizations and each Code lineage is a single public run.
CODE_TRACKS = {"E": "qwen_code", "F": "olmo_code"}
CODE_SPAN = {"E": (100, 800), "F": (800, 2000)}
CODE_NOTE_SINGLE = "single public lineage:\nno second training\nrealization to compare"
QUARANTINED = set()
QUARANTINE_NOTE = "branch labelling\nnot validated\n(blind audit macro-F1\n0.479 / 0.724)"
QUARANTINE_PLUS_RUNS = ("branch labelling\nnot validated;\nalso no second\ntraining realization")

# sharpening reference checkpoint and the trajectory span used to normalise x
SHARP = {"A": ("25", ["trunk"], 25, 200), "C": ("5", ["seed1", "seed2"], 5, 25),
         "D": ("100", [f"seed{i}" for i in range(1, 6)], 100, 125)}
# divergence families: registry track, root, checkpoints, arms, origin
DIV_FAMS = [("A", "75", [175], ["seed1", "seed2", "seed3"], 75),
            ("A", "100", [175], ["seed1", "seed2", "seed3"], 100),
            ("A", "125", [175], ["seed1", "seed2", "seed3"], 125),
            ("A", "150", [175], ["seed1", "seed2", "seed3"], 150),
            ("C", "NA", [5, 10, 15, 20, 25], ["seed1", "seed2"], 0),
            ("D", "100", [105, 110, 115, 120, 125],
             [f"seed{i}" for i in range(1, 6)], 100)]
# lineages carrying an ordered checkpoint sequence, for increment memory
MEM_LINEAGES = {"A": [("NA", "trunk")], "C": [("NA", "seed1"), ("NA", "seed2")],
                "D": [("100", f"seed{i}") for i in range(1, 6)]}
# a continuation that inherits a shared root state: that state is prepended, so
# the first increment is the one that carries the fork perturbation
MEM_PREFIX = {"D": ("100", "base", 100)}
OBS_INTERVAL = {"A": 25, "C": 5, "D": 5}

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7.5, "axes.linewidth": 0.8,
                     "figure.dpi": 150})


def qboot(by_q, n=NBOOT, seed=7):
    """question-cluster bootstrap of the mean of pooled per-question values."""
    qs = sorted(by_q)
    if not qs:
        return float("nan"), float("nan"), float("nan")
    flat = np.concatenate([np.asarray(by_q[q], float) for q in qs])
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arrs = [np.asarray(by_q[q], float) for q in qs]
    idx = rng.integers(0, len(qs), size=(n, len(qs)))
    b = np.empty(n)
    for i in range(n):
        v = np.concatenate([arrs[j] for j in idx[i]])
        v = v[np.isfinite(v)]
        b[i] = v.mean() if v.size else np.nan
    b = b[np.isfinite(b)]
    return float(flat.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))


def js(pa, pb):
    m = 0.5 * (pa + pb)
    return float(max(0.0, 0.5 * np.sum(pa * np.log(pa / m))
                     + 0.5 * np.sum(pb * np.log(pb / m))))


def load_long():
    S = defaultdict(dict)
    for r in csv.DictReader((PMF / "branch_state_long.csv").open()):
        S[(r["track"], r["continuation_id"], r["root_step"], int(r["checkpoint"]),
           r["question_id"])][r["branch"]] = (int(float(r["raw_branch_count"])),
                                              float(r["smoothed_probability"]),
                                              float(r["clr_value"]))
    return S


# ============================================================ meta figure 1
def m1_data(null):
    out = {}
    for tr, (ref, conts, lo, hi) in SHARP.items():
        rows = [r for r in null if r["track"] == tr
                and r["null_model"] == NULL_PRIMARY
                and r["reference_checkpoint"] == ref
                and r["continuation_id"] in conts]
        series = []
        for tg in sorted({int(r["target_checkpoint"]) for r in rows}):
            g = [r for r in rows if int(r["target_checkpoint"]) == tg]
            by_q = defaultdict(list)
            for r in g:
                sd = float(r["null_sd"])
                if sd > 0:
                    by_q[r["question_id"]].append(
                        (float(r["observed_discrepancy"]) - float(r["null_mean"])) / sd)
            m, l, h = qboot(by_q, seed=101)
            eby = defaultdict(list)
            for r in g:
                eby[r["question_id"]].append(float(r["excess_discrepancy"]))
            em, el, eh = qboot(eby, seed=101)
            series.append({"target_checkpoint": tg,
                           "x_normalised": (tg - lo) / (hi - lo),
                           "n_rows": len(g), "n_questions": len(by_q),
                           "z_reweight_mean": m, "z_lo": l, "z_hi": h,
                           "excess_mean": em, "excess_lo": el, "excess_hi": eh})
        series.insert(0, {"target_checkpoint": int(ref), "x_normalised": 0.0,
                          "n_rows": 0, "n_questions": 0,
                          "z_reweight_mean": 0.0, "z_lo": 0.0, "z_hi": 0.0,
                          "excess_mean": 0.0, "excess_lo": 0.0, "excess_hi": 0.0,
                          "is_reference_anchor": True})
        out[tr] = {"reference_checkpoint": int(ref), "span": [lo, hi], "series": series}
    cp = DATR / "meta1_code_panels.csv"
    if cp.exists():
        for tr, fam in CODE_TRACKS.items():
            rs = [r for r in csv.DictReader(cp.open()) if r["model"] == fam]
            if not rs:
                continue
            lo, hi = CODE_SPAN[tr]
            out[tr] = {"reference_checkpoint": lo, "span": [lo, hi],
                       "series": [{"target_checkpoint": lo, "x_normalised": 0.0,
                                   "n_rows": 0, "n_questions": 0,
                                   "z_reweight_mean": 0.0, "z_lo": 0.0, "z_hi": 0.0,
                                   "excess_mean": 0.0, "excess_lo": 0.0,
                                   "excess_hi": 0.0, "is_reference_anchor": True}]
                                 + [{"target_checkpoint": int(r["target_checkpoint"]),
                                   "x_normalised": float(r["x_normalised"]),
                                   "n_rows": int(r["n_rows"]),
                                   "n_questions": int(r["n_questions"]),
                                   "z_reweight_mean": float(r["z_reweight_mean"]),
                                   "z_lo": float(r["z_lo"]), "z_hi": float(r["z_hi"]),
                                   "excess_mean": float(r["excess_mean"]),
                                   "excess_lo": float(r["excess_lo"]),
                                   "excess_hi": float(r["excess_hi"])} for r in rs]}
    return out


# ============================================================ meta figure 2
def m2_data(S):
    rng = np.random.default_rng(SEED)
    pairs = []
    for track, root, cks, conts, origin in DIV_FAMS:
        for ck in cks:
            qs = sorted({k[4] for k in S if k[0] == track and k[2] == root
                         and k[3] == ck and k[1] in conts})
            for q in qs:
                have = [c for c in conts if (track, c, root, ck, q) in S]
                if len(have) < 2:
                    continue
                bs = sorted(S[(track, have[0], root, ck, q)])
                pv, cv = {}, {}
                for c in have:
                    d = S[(track, c, root, ck, q)]
                    if sorted(d) != bs:
                        continue
                    pv[c] = np.array([d[b][1] for b in bs], float)
                    cv[c] = np.array([d[b][0] for b in bs], float)
                # The split-half floor and the matched half-budget JS are both
                # random draws. A single draw leaves enough Monte-Carlo noise to
                # flip the sign of a near-zero excess, so both are averaged over
                # NFLOOR independent splits. This changes no estimand; it only
                # removes the estimator's own sampling noise.
                fl, halfjs = defaultdict(list), defaultdict(list)
                usable = {c: cc for c, cc in cv.items() if int(cc.sum()) >= 4}
                for _ in range(NFLOOR):
                    hh = {}
                    for c, cc in usable.items():
                        n = int(cc.sum())
                        a_ = rng.multivariate_hypergeometric(cc.astype(int), n // 2)
                        b_ = cc.astype(int) - a_
                        hh[c] = a_
                        fl[c].append(js((a_ + ALPHA) / (a_.sum() + ALPHA * len(bs)),
                                        (b_ + ALPHA) / (b_.sum() + ALPHA * len(bs))))
                    for i, j in combinations(sorted(hh), 2):
                        halfjs[(i, j)].append(
                            js((hh[i] + ALPHA) / (hh[i].sum() + ALPHA * len(bs)),
                               (hh[j] + ALPHA) / (hh[j].sum() + ALPHA * len(bs))))
                for i, j in combinations(sorted(pv), 2):
                    if (i, j) not in halfjs:
                        continue
                    vm = float(np.mean(halfjs[(i, j)]))
                    fm = 0.5 * (float(np.mean(fl[i])) + float(np.mean(fl[j])))
                    pairs.append({"track": track, "root_step": root, "checkpoint": ck,
                                  "horizon": ck - origin, "question_id": q,
                                  "continuation_i": i, "continuation_j": j,
                                  "n_floor_draws": NFLOOR,
                                  "js_matched": vm, "floor_matched": fm,
                                  "excess_js": vm - fm,
                                  "modal_agreement": int(np.argmax(pv[i]) == np.argmax(pv[j]))})
    out = {}
    for tr in SHARP:
        g0 = [r for r in pairs if r["track"] == tr]
        if not g0:
            continue
        hmax = max(r["horizon"] for r in g0)
        series = []
        for h in sorted({r["horizon"] for r in g0}):
            g = [r for r in g0 if r["horizon"] == h]
            by_q = defaultdict(list)
            for r in g:
                by_q[r["question_id"]].append(r["excess_js"])
            m, l, hh = qboot(by_q, seed=202)
            am = defaultdict(list)
            for r in g:
                am[r["question_id"]].append(r["modal_agreement"])
            a_m, a_l, a_h = qboot(am, seed=202)
            series.append({"horizon": h, "x_normalised": h / hmax, "n_pairs": len(g),
                           "n_questions": len(by_q),
                           "excess_js_mean": m, "excess_js_lo": l, "excess_js_hi": hh,
                           "floor_mean": float(np.mean([r["floor_matched"] for r in g])),
                           "modal_agreement_mean": a_m, "modal_agreement_lo": a_l,
                           "modal_agreement_hi": a_h})
        out[tr] = {"max_horizon": hmax, "series": series}
    return pairs, out


# ============================================================ meta figure 3
def m3_data(S):
    """CLR-increment directional memory with a within-trajectory shuffled-time null.

    Available for a single lineage, which is the point: it is the only one of the
    three quantities that does not need a second training realization.
    """
    rng = np.random.default_rng(SEED + 1)
    out, rawrows = {}, []
    for tr, lins in MEM_LINEAGES.items():
        units = defaultdict(list)          # (question) -> list of z-sequences
        for root, cont in lins:
            cks = sorted({k[3] for k in S if k[0] == tr and k[1] == cont and k[2] == root})
            qs = sorted({k[4] for k in S if k[0] == tr and k[1] == cont and k[2] == root})
            pre = MEM_PREFIX.get(tr)
            for q in qs:
                seq, steps = [], []
                if pre is not None:
                    d0 = S.get((tr, pre[1], pre[0], pre[2], q))
                    if d0 is None:
                        continue
                    seq.append(np.array([d0[b][2] for b in sorted(d0)], float))
                    steps.append(pre[2])
                ok = True
                for ck in cks:
                    d = S.get((tr, cont, root, ck, q))
                    if d is None:
                        ok = False
                        break
                    seq.append(np.array([d[b][2] for b in sorted(d)], float))
                    steps.append(ck)
                if ok and len(seq) >= 3:
                    units[q].append((cont, steps, seq))
        T = min(len(s) for v in units.values() for _, _, s in v)
        max_lag = T - 2
        if max_lag < 1:
            continue

        def memory(seqs):
            by_lag = {L: defaultdict(list) for L in range(1, max_lag + 1)}
            for q, lst in seqs.items():
                for _, _, seq in lst:
                    v = [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]
                    for L in range(1, max_lag + 1):
                        for i in range(len(v) - L):
                            a, b = v[i], v[i + L]
                            na, nb = np.linalg.norm(a), np.linalg.norm(b)
                            if na > 1e-12 and nb > 1e-12:
                                by_lag[L][q].append(float(a @ b / (na * nb)))
            return by_lag

        obs = memory(units)
        # shuffled-time null: permute the checkpoint ORDER within each unit, which
        # preserves branch dimension, question identity, the multiset of states
        # (hence checkpoint marginals) and the measurement budget
        null_means = {L: [] for L in range(1, max_lag + 1)}
        for _ in range(NPERM):
            sh = {}
            for q, lst in units.items():
                sh[q] = [(c, ck, [s[i] for i in rng.permutation(len(s))])
                         for c, ck, s in lst]
            bl = memory(sh)
            for L in range(1, max_lag + 1):
                v = np.concatenate([np.asarray(bl[L][q], float) for q in bl[L]]) \
                    if bl[L] else np.array([])
                null_means[L].append(float(v.mean()) if v.size else np.nan)
        series = []
        for L in range(1, max_lag + 1):
            m, lo, hi = qboot(obs[L], seed=303 + L)
            nm = np.array(null_means[L], float)
            nm = nm[np.isfinite(nm)]
            series.append({"lag_intervals": L, "n_questions": len(obs[L]),
                           "n_pairs": int(sum(len(v) for v in obs[L].values())),
                           "memory_mean": m, "memory_lo": lo, "memory_hi": hi,
                           "null_mean": float(nm.mean()) if nm.size else float("nan"),
                           "null_lo": float(np.quantile(nm, .025)) if nm.size else float("nan"),
                           "null_hi": float(np.quantile(nm, .975)) if nm.size else float("nan")})
            for q, vals in obs[L].items():
                rawrows.append({"track": tr, "lag_intervals": L, "question_id": q,
                                "n_cos": len(vals), "mean_cos": float(np.mean(vals))})
        out[tr] = {"n_checkpoints": T, "max_lag": max_lag,
                   "observation_interval_steps": OBS_INTERVAL.get(tr),
                   "starts_at_shared_root": tr in MEM_PREFIX, "series": series}
    cp = DATR / "meta3_code_panels.csv"
    if cp.exists():
        for tr, fam in CODE_TRACKS.items():
            rs = [r for r in csv.DictReader(cp.open()) if r["model"] == fam]
            if not rs:
                continue
            out[tr] = {"n_checkpoints": int(rs[0]["n_checkpoints"]),
                       "max_lag": int(rs[0]["max_lag"]),
                       "observation_interval_steps": rs[0]["observation_interval_steps"],
                       "starts_at_shared_root": False,
                       "series": [{"lag_intervals": int(r["lag_intervals"]),
                                   "n_questions": int(r["n_questions"]),
                                   "n_pairs": int(r["n_pairs"]),
                                   "memory_mean": float(r["memory_mean"]),
                                   "memory_lo": float(r["memory_lo"]),
                                   "memory_hi": float(r["memory_hi"]),
                                   "null_mean": float(r["null_mean"]),
                                   "null_lo": float(r["null_lo"]),
                                   "null_hi": float(r["null_hi"])} for r in rs]}
    return out, rawrows


# ============================================================ meta figure 4
SEQ = {"A": [[("trunk", "NA", c) for c in (25, 50, 75, 100, 125, 150, 175, 200)]],
       "C": [[(f"seed{i}", "NA", c) for c in (5, 10, 15, 20, 25)] for i in (1, 2)],
       "D": [[("base", "100", 100)] + [(f"seed{i}", "100", c)
                                       for c in (105, 110, 115, 120, 125)]
             for i in range(1, 6)]}
CODE_SEQ = {"E": ("qwen_code", ["step-100", "step-200", "step-300", "step-400",
                                "step-500", "step-600", "step-700", "step-800"],
                  {f"step-{i}00": i * 100 for i in range(1, 9)}),
            "F": ("olmo_code", ["step_0800", "step_1000", "step_1250", "step_1500",
                                "step_1750", "FINAL-main"],
                  {"step_0800": 800, "step_1000": 1000, "step_1250": 1250,
                   "step_1500": 1500, "step_1750": 1750, "FINAL-main": 2000})}
NFLOOR_MOVE = 25


def _counts_countdown(S):
    """(track, cont, root, ckpt, q) -> (ordered branches, count vector)"""
    out = {}
    for k, d in S.items():
        bs = sorted(d)
        out[k] = (bs, np.array([d[b][0] for b in bs], float))
    return out


def _counts_code():
    """Code counts over the frozen viable support, evaluation split only."""
    base = ART / "code_branch_v2"
    cbp = base / "code_branch_codebook_v2.json"
    ap = base / "code_branch_assignments_eval.jsonl"
    if not (cbp.exists() and ap.exists()):
        return None, None
    cb = json.load(cbp.open())
    SUP = {q: cb["prompts"][q]["viable_branches"] for q in cb["prompts"]}
    acc = defaultdict(Counter)
    for l in ap.open(encoding="utf-8"):
        r = json.loads(l)
        if r["branch"] and r["branch"] in SUP.get(r["prompt_id"], ()):
            acc[(r["family"], r["state"], r["prompt_id"])][r["branch"]] += 1
    out = {}
    for (fam, st, q), c in acc.items():
        bs = SUP[q]
        out[(fam, st, q)] = (bs, np.array([c.get(b, 0) for b in bs], float))
    return out, SUP


def _move_cell(rng, cref, ct, K):
    """matched half-budget JS to the reference, and the within-checkpoint floor."""
    if cref.sum() < 4 or ct.sum() < 4:
        return None
    obs, fl = [], []
    for _ in range(NFLOOR_MOVE):
        a0 = rng.multivariate_hypergeometric(cref.astype(int), int(cref.sum()) // 2)
        b0 = cref.astype(int) - a0
        a1 = rng.multivariate_hypergeometric(ct.astype(int), int(ct.sum()) // 2)
        b1 = ct.astype(int) - a1
        sm = lambda c: (c + ALPHA) / (c.sum() + ALPHA * K)
        obs.append(js(sm(a0), sm(a1)))
        fl.append(0.5 * (js(sm(a0), sm(b0)) + js(sm(a1), sm(b1))))
    return float(np.mean(obs)), float(np.mean(fl))


def m4_data(S):
    rng = np.random.default_rng(SEED + 3)
    CD = _counts_countdown(S)
    CODE, _ = _counts_code()
    per_cell, out = [], {}
    for tr in ("A", "C", "D", "E", "F"):
        cells = defaultdict(lambda: defaultdict(list))        # step -> q -> values
        if tr in SEQ:
            for seq in SEQ[tr]:
                ref = seq[0]
                steps = [c for _, _, c in seq]
                for cont, root, ck in seq[1:]:
                    for q in sorted({k[4] for k in S if k[0] == tr}):
                        kr = (tr, ref[0], ref[1], ref[2], q)
                        kt = (tr, cont, root, ck, q)
                        if kr not in CD or kt not in CD:
                            continue
                        bs_r, cr = CD[kr]
                        bs_t, ct = CD[kt]
                        if bs_r != bs_t:
                            continue
                        K = len(bs_r)
                        sm = lambda c: (c + ALPHA) / (c.sum() + ALPHA * K)
                        full = js(sm(cr), sm(ct))
                        m = _move_cell(rng, cr, ct, K)
                        cells[ck][q].append((full, m))
                        per_cell.append({"track": tr, "continuation": cont, "root": root,
                                         "checkpoint": ck, "question_id": q,
                                         "d_move_full_budget": round(full, 8),
                                         "d_move_matched": round(m[0], 8) if m else None,
                                         "floor_matched": round(m[1], 8) if m else None,
                                         "excess_move": round(m[0] - m[1], 8) if m else None})
            lo, hi = steps[0], steps[-1]
        elif CODE is not None:
            fam, states, num = CODE_SEQ[tr]
            ref = states[0]
            for st in states[1:]:
                for q in sorted({k[2] for k in CODE if k[0] == fam}):
                    kr, kt = (fam, ref, q), (fam, st, q)
                    if kr not in CODE or kt not in CODE:
                        continue
                    bs_r, cr = CODE[kr]
                    bs_t, ct = CODE[kt]
                    if bs_r != bs_t or len(bs_r) < 2:
                        continue
                    K = len(bs_r)
                    sm = lambda c: (c + ALPHA) / (c.sum() + ALPHA * K)
                    full = js(sm(cr), sm(ct))
                    m = _move_cell(rng, cr, ct, K)
                    cells[num[st]][q].append((full, m))
                    per_cell.append({"track": tr, "continuation": fam, "root": ref,
                                     "checkpoint": num[st], "question_id": q,
                                     "d_move_full_budget": round(full, 8),
                                     "d_move_matched": round(m[0], 8) if m else None,
                                     "floor_matched": round(m[1], 8) if m else None,
                                     "excess_move": round(m[0] - m[1], 8) if m else None})
            lo, hi = num[ref], num[states[-1]]
        else:
            continue
        if not cells:
            continue
        series = [{"checkpoint": lo, "x_normalised": 0.0, "n_questions": 0,
                   "d_move_mean": 0.0, "d_move_lo": 0.0, "d_move_hi": 0.0,
                   "excess_move_mean": 0.0, "excess_move_lo": 0.0, "excess_move_hi": 0.0,
                   "is_reference_anchor": True}]
        for ck in sorted(cells):
            fq = {q: [v[0] for v in lst] for q, lst in cells[ck].items()}
            eq = {q: [v[1][0] - v[1][1] for v in lst if v[1]] for q, lst in cells[ck].items()}
            eq = {q: v for q, v in eq.items() if v}
            dm, dl, dh = qboot(fq, seed=404)
            em, el, eh = qboot(eq, seed=404) if eq else (float("nan"),) * 3
            series.append({"checkpoint": ck, "x_normalised": (ck - lo) / (hi - lo),
                           "n_questions": len(fq), "d_move_mean": dm,
                           "d_move_lo": dl, "d_move_hi": dh,
                           "excess_move_mean": em, "excess_move_lo": el,
                           "excess_move_hi": eh, "is_reference_anchor": False})
        out[tr] = {"span": [lo, hi], "series": series}
    return out, per_cell


# ============================================================ plotting
def panel_grid(ylabel, ylim, legend_handles, fname, xlabel, note_map, draw, subtitle,
               xsetup=None, panels=None, width=13.2):
    panels = panels or PANELS
    fig, axes = plt.subplots(1, len(panels), figsize=(width, 2.55), sharey=True)
    axes = np.atleast_1d(axes)
    for i, (lab, tr, model, design, col) in enumerate(panels):
        a = axes[i]
        a.axhline(0, color="0.35", lw=0.8, zorder=1)
        drew = draw(a, tr, col)
        a.set_ylim(*ylim)
        if drew is False and xsetup is None:
            pass
        a.set_title(f"({lab}) {model}\n{subtitle(tr, design)}", fontsize=7.1, pad=4,
                    linespacing=1.35)
        a.set_xlabel(xlabel, fontsize=7.5)
        a.grid(alpha=.25, lw=.5)
        if i == 0:
            a.set_ylabel(ylabel, fontsize=8)
        if not drew:
            if xsetup is not None:
                xsetup(a)          # identical axes even where there is no data
            a.set_facecolor("0.965")
            a.text(0.5, 0.5, note_map(tr), transform=a.transAxes, ha="center",
                   va="center", fontsize=6.6, color="0.45", linespacing=1.5)
    anchor = 0.905 if len(panels) >= 5 else 0.845
    fig.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(anchor, 0.5),
               frameon=True, fontsize=7.5, borderpad=0.6)
    fig.tight_layout(rect=(0, 0, anchor - 0.005, 1))
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"{fname}.{e}", bbox_inches="tight")
    plt.close(fig)


def dump(name, rows, fields=None):
    p = DAT / name
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return p


def main():
    t0 = time.time()
    for d in (FIG, REP, DAT):
        d.mkdir(parents=True, exist_ok=True)
    null = list(csv.DictReader((PMF / "sharpening_null.csv").open()))
    summ = list(csv.DictReader((PMF / "branch_state_summary.csv").open()))
    S = load_long()

    # ---------------- meta figure 1 -------------------------------------
    M1 = m1_data(null)
    rows1 = [dict(panel=p, track=t, model=m, **s) for p, t, m, _, _ in PANELS
             if t in M1 for s in M1[t]["series"]]
    dump("meta1_reweighting.csv", rows1)
    ylim1 = (min(0, min(s["z_lo"] for v in M1.values() for s in v["series"]) - 1),
             max(s["z_hi"] for v in M1.values() for s in v["series"]) * 1.12)

    def draw1(a, tr, col):
        if tr not in M1:
            return False
        s = M1[tr]["series"]
        est = [r for r in s if not r.get("is_reference_anchor")]
        x = [r["x_normalised"] for r in s]
        a.plot(x, [r["z_reweight_mean"] for r in s], "-", color=col, lw=1.8, zorder=3)
        a.plot([r["x_normalised"] for r in est], [r["z_reweight_mean"] for r in est],
               "o", color=col, ms=3.2, zorder=4)
        # the reference checkpoint is Z = 0 BY DEFINITION, not by estimation: it is
        # the state the null is fitted from. Open marker, no band.
        a.plot([0.0], [0.0], "o", mfc="white", mec=col, mew=1.2, ms=4.4, zorder=5)
        a.fill_between([r["x_normalised"] for r in est], [r["z_lo"] for r in est],
                       [r["z_hi"] for r in est], color=col, alpha=.22, lw=0, zorder=2)
        a.set_xlim(-0.03, 1.03)
        return True

    panel_grid(
        "standardised excess\nover sharpening null  $Z$", ylim1,
        [Line2D([], [], color=OBS, lw=1.8, marker="o", ms=3.2, label="observed"),
         Patch(facecolor=OBS, alpha=.22, label="95% question bootstrap"),
         Line2D([], [], color=OBS, lw=0, marker="o", mfc="white", mec=OBS, mew=1.2,
                ms=4.4, label="reference checkpoint ($Z=0$ by definition)"),
         Line2D([], [], color="0.35", lw=0.8, label="scalar-sharpening null ($Z=0$)")],
        "meta_reweighting_5way", "normalised progress",
        lambda tr: QUARANTINE_NOTE if tr in QUARANTINED else "no data",
        draw1,
        lambda tr, d: (f"steps {M1[tr]['span'][0]}–{M1[tr]['span'][1]}, ref "
                       f"{M1[tr]['reference_checkpoint']}" if tr in M1 else d),
        xsetup=lambda a: a.set_xlim(-0.03, 1.03))

    # ---------------- meta figure 2 -------------------------------------
    pairs, M2 = m2_data(S)
    dump("meta2_pairs_raw.csv", pairs)
    rows2 = [dict(panel=p, track=t, model=m, **s) for p, t, m, _, _ in PANELS
             if t in M2 for s in M2[t]["series"]]
    dump("meta2_stochastic_futures.csv", rows2)
    lo2 = min(s["excess_js_lo"] for v in M2.values() for s in v["series"])
    hi2 = max(s["excess_js_hi"] for v in M2.values() for s in v["series"])
    ylim2 = (min(0, lo2) - 0.002, hi2 * 1.18)

    def draw2(a, tr, col):
        if tr not in M2:
            return False
        s = M2[tr]["series"]
        x = [r["x_normalised"] for r in s]
        a.plot(x, [r["excess_js_mean"] for r in s], "-o", color=col, lw=1.8, ms=3.2, zorder=3)
        a.fill_between(x, [r["excess_js_lo"] for r in s], [r["excess_js_hi"] for r in s],
                       color=col, alpha=.22, lw=0, zorder=2)
        a.text(0.97, 0.94, f"modal agr. {s[-1]['modal_agreement_mean']:.2f}",
               transform=a.transAxes, ha="right", va="top", fontsize=6.3, color="0.35")
        a.set_xlim(-0.03, 1.03)
        return True

    panel_grid(
        "excess pairwise JS\nover measurement floor", ylim2,
        [Line2D([], [], color=OBS, lw=1.8, marker="o", ms=3.2, label="observed"),
         Patch(facecolor=OBS, alpha=.22, label="95% question bootstrap"),
         Line2D([], [], color="0.35", lw=0.8, label="measurement floor (0)")],
        "meta_controlled_stochastic_3way", "normalised horizon",
        lambda tr: CODE_NOTE_SINGLE if tr in CODE_TRACKS else "no second\ntraining realization",
        draw2,
        lambda tr, d: ((f"{d} (4 roots)" if tr == "A" else d)
                       + f", {M2[tr]['series'][0]['n_pairs']}–"
                       f"{M2[tr]['series'][-1]['n_pairs']} pairs" if tr in M2 else d),
        xsetup=lambda a: a.set_xlim(-0.03, 1.03),
        panels=[p for p in PANELS if p[1] in M2], width=8.9)

    # ---------------- meta figure 3 -------------------------------------
    M3, raw3 = m3_data(S)
    dump("meta3_memory_per_question.csv", raw3)
    rows3 = [dict(panel=p, track=t, model=m, **s) for p, t, m, _, _ in PANELS
             if t in M3 for s in M3[t]["series"]]
    dump("meta3_temporal_memory.csv", rows3)
    # Requested range is approximately [-0.5, 0.35]. The shuffled-null band reaches
    # -0.54, so the floor is set to -0.55: a clipped null band would read as a
    # tighter null than it is. Nothing is cut at either end.
    lo3 = min(min(s["memory_lo"], s["null_lo"]) for v in M3.values() for s in v["series"])
    hi3 = max(max(s["memory_hi"], s["null_hi"]) for v in M3.values() for s in v["series"])
    ylim3 = (-0.55, 0.35)
    assert lo3 >= ylim3[0] and hi3 <= ylim3[1], f"y-range clips data: {lo3:.4f} {hi3:.4f}"

    LAGMAX = max(v["max_lag"] for v in M3.values())

    def draw3(a, tr, col):
        if tr not in M3:
            return False
        s = M3[tr]["series"]
        x = [r["lag_intervals"] for r in s]
        a.fill_between(x, [r["null_lo"] for r in s], [r["null_hi"] for r in s],
                       color=NULLC, alpha=.5, lw=0, zorder=2)
        a.plot(x, [r["memory_mean"] for r in s], "-o", color=col, lw=1.8, ms=3.2, zorder=3)
        a.fill_between(x, [r["memory_lo"] for r in s], [r["memory_hi"] for r in s],
                       color=col, alpha=.22, lw=0, zorder=2)
        a.set_xticks(range(1, LAGMAX + 1))
        a.set_xlim(0.6, LAGMAX + 0.4)
        return True

    panel_grid(
        "CLR-increment directional\nmemory  $M(\\mathrm{lag})$", ylim3,
        [Line2D([], [], color=OBS, lw=1.8, marker="o", ms=3.2, label="observed"),
         Patch(facecolor=OBS, alpha=.22, label="95% question bootstrap"),
         Patch(facecolor=NULLC, alpha=.5, label="shuffled-time null (95%)")],
        "meta_temporal_memory_5way", "lag (observation intervals)",
        lambda tr: QUARANTINE_NOTE if tr in QUARANTINED else "no data",
        draw3,
        lambda tr, d: (f"{M3[tr]['n_checkpoints']} checkpoints, interval "
                       f"{M3[tr]['observation_interval_steps']} steps"
                       if tr in M3 else d),
        xsetup=lambda a: (a.set_xticks(range(1, LAGMAX + 1)),
                          a.set_xlim(0.6, LAGMAX + 0.4)))

    # ---------------- meta figure 4: behavioural displacement ---------------
    M4, cells4 = m4_data(S)
    dump("meta4_displacement_per_cell.csv", cells4)
    rows4 = [dict(panel=p, track=t, model=m, **s) for p, t, m, _, _ in PANELS
             if t in M4 for s in M4[t]["series"]]
    dump("meta4_behavioral_displacement.csv", rows4)
    hi4 = max(s["d_move_hi"] for v in M4.values() for s in v["series"])
    ylim4 = (0, hi4 * 1.14)

    def draw4(a, tr, col):
        if tr not in M4:
            return False
        s = M4[tr]["series"]
        est = [r for r in s if not r["is_reference_anchor"]]
        a.plot([r["x_normalised"] for r in s], [r["d_move_mean"] for r in s], "-",
               color=col, lw=1.8, zorder=3)
        a.plot([r["x_normalised"] for r in est], [r["d_move_mean"] for r in est], "o",
               color=col, ms=3.2, zorder=4)
        a.fill_between([r["x_normalised"] for r in est], [r["d_move_lo"] for r in est],
                       [r["d_move_hi"] for r in est], color=col, alpha=.22, lw=0, zorder=2)
        a.plot([r["x_normalised"] for r in est],
               [r["excess_move_mean"] for r in est], "--", color=NULLC, lw=2.0, zorder=3)
        a.plot([0.0], [0.0], "o", mfc="white", mec=col, mew=1.2, ms=4.4, zorder=5)
        a.set_xlim(-0.03, 1.03)
        return True

    panel_grid(
        "displacement from earliest\ncheckpoint  $D_{\\mathrm{move}}$ (JS)", ylim4,
        [Line2D([], [], color=OBS, lw=1.8, marker="o", ms=3.2, label="$D_{move}$ observed"),
         Patch(facecolor=OBS, alpha=.22, label="95% question bootstrap"),
         Line2D([], [], color=NULLC, lw=2.0, ls="--",
                label="excess over matched\nsplit-half floor"),
         Line2D([], [], color=OBS, lw=0, marker="o", mfc="white", mec=OBS, mew=1.2,
                ms=4.4, label="reference checkpoint")],
        "meta_behavioral_displacement_5way", "normalised progress",
        lambda tr: "no branch-state\nsequence", draw4,
        lambda tr, d: (f"ref {M4[tr]['span'][0]}, to {M4[tr]['span'][1]}"
                       if tr in M4 else d),
        xsetup=lambda a: a.set_xlim(-0.03, 1.03))

    # ---------------- appendix: entropy and top-1 mass, exported separately
    app = []
    for p, tr, model, design, _ in PANELS:
        if tr not in SHARP:
            continue
        conts = SHARP[tr][1]
        rows = [r for r in summ if r["track"] == tr and r["continuation_id"] in conts]
        for ck in sorted({int(r["checkpoint"]) for r in rows}):
            g = [r for r in rows if int(r["checkpoint"]) == ck]
            ent = defaultdict(list)
            top = defaultdict(list)
            for r in g:
                ent[r["question_id"]].append(float(r["normalized_entropy"]))
                top[r["question_id"]].append(float(r["top1_prob"]))
            em, el, eh = qboot(ent, seed=404)
            tm, tl, th = qboot(top, seed=404)
            app.append({"panel": p, "track": tr, "model": model, "checkpoint": ck,
                        "n_questions": len(ent),
                        "normalized_entropy_mean": em, "normalized_entropy_lo": el,
                        "normalized_entropy_hi": eh,
                        "top1_mass_mean": tm, "top1_mass_lo": tl, "top1_mass_hi": th})
    dump("appendix_entropy_top1.csv", app)

    meta = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "panels": [{"panel": p, "registry_track": t, "model_task": m, "design": d}
                       for p, t, m, d, _ in PANELS],
            "inputs_sha256": {str(PMF / n): sha(PMF / n) for n in
                              ("branch_state_long.csv", "branch_state_summary.csv",
                               "sharpening_null.csv", "trajectory_registry.csv")},
            "code_sha256": {str(Path(__file__).name): sha(Path(__file__))},
            "git_commit": subprocess.run(["git", "-C", "<artifacts>", "rev-parse",
                                          "HEAD"], capture_output=True, text=True,
                                         timeout=20).stdout.strip(),
            "seeds": {"split_half_floor": SEED, "shuffled_time_null": SEED + 1},
            "n_permutations_shuffled_null": NPERM, "n_bootstrap": NBOOT,
            "null_model": NULL_PRIMARY,
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "matplotlib": matplotlib.__version__, "host": platform.node()},
            "wall_sec": round(time.time() - t0, 1)}
    with (DAT / "meta_build_manifest.json").open("w") as f:
        json.dump(meta, f, indent=1)

    print("  meta 1  Z over sharpening null")
    for t in ("A", "C", "D"):
        s = M1[t]["series"]
        print(f"    {t}: Z {s[0]['z_reweight_mean']:+.2f} → {s[-1]['z_reweight_mean']:+.2f} "
              f"[{s[-1]['z_lo']:+.2f},{s[-1]['z_hi']:+.2f}] over targets "
              f"{s[0]['target_checkpoint']}–{s[-1]['target_checkpoint']}")
    print("  meta 2  excess JS")
    for t in ("A", "C", "D"):
        s = M2[t]["series"]
        print(f"    {t}: {s[0]['excess_js_mean']:+.5f} → {s[-1]['excess_js_mean']:+.5f} "
              f"[{s[-1]['excess_js_lo']:+.5f},{s[-1]['excess_js_hi']:+.5f}] "
              f"modal agr {s[-1]['modal_agreement_mean']:.3f}")
    print("  meta 3  CLR-increment memory")
    for t in ("A", "C", "D"):
        s = M3[t]["series"]
        print(f"    {t}: {M3[t]['n_checkpoints']} ckpts, lags 1–{M3[t]['max_lag']}; "
              + ", ".join(f"L{r['lag_intervals']} {r['memory_mean']:+.3f} "
                          f"(null {r['null_mean']:+.3f})" for r in s))
    print("  meta 4  displacement from the earliest checkpoint")
    for t in ("A", "C", "D", "E", "F"):
        if t in M4:
            e = [r for r in M4[t]["series"] if not r["is_reference_anchor"]]
            print(f"    {t}: D_move {e[0]['d_move_mean']:.5f} -> {e[-1]['d_move_mean']:.5f} "
                  f"[{e[-1]['d_move_lo']:.5f},{e[-1]['d_move_hi']:.5f}]  "
                  f"excess {e[-1]['excess_move_mean']:+.5f}")
    for n in ("meta_reweighting_5way", "meta_controlled_stochastic_3way",
              "meta_temporal_memory_5way", "meta_behavioral_displacement_5way"):
        print(f"  {n}.pdf sha256 {sha(FIG / (n + '.pdf'))}")
    print(f"  wall {meta['wall_sec']} s")


if __name__ == "__main__":
    main()
