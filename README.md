# Behavioural branch dynamics under RLVR — code and frozen artifacts

Anonymous artifact for an ICLR 2027 submission. Everything here is code and
frozen analysis output; there are no model weights and no author information.

## What the paper studies

We treat RLVR as a stochastic dynamical process over a distribution on
*behavioural branches* — reproducibly identified entrances into a solution
(the first arithmetic entrance for Countdown, the first substantive AST
decision for HumanEval). We track how that distribution evolves along training
trajectories, and we resume training repeatedly from the *same complete
training state* to ask how reproducible its future is.

## Layout

```
artifacts/   frozen analysis outputs, and the frozen inputs they are computed from
scripts/     the executed analysis path for each claim
scripts/lib/ the branch-construction, parsing, solver and model primitives
config/      the effective training and DeepSpeed configurations
outputs/     created at runtime; nothing here is tracked
```

`artifacts/` mirrors the directory layout the analysis scripts expect, so each
script needed only a one-line change per root constant to run from this
repository. Scripts **read** from `artifacts/` and **write** to `outputs/`, so
rerunning never overwrites a frozen artifact in place — you can always diff
your regenerated output against the shipped one.

## Claim → artifact → script

| # | headline claim | paper | frozen artifact | executed analysis |
|---|---|---|---|---|
| 1 | RLVR selectively reweights viable branches rather than uniformly sharpening | Fig. 2 | `artifacts/paper_main_figures/tables/sharpening_null.csv` | `scripts/p1_sharpening.py` |
| 2 | One complete training state admits multiple behavioural futures | Fig. 3, Tab. 4 | `artifacts/analysis_balanced_k4/balanced_omnibus.json` | `scripts/bk4_omnibus.py` |
| 3 | Directional memory in branch space is short-ranged | Fig. 4 | `artifacts/figures/meta_data/meta3_temporal_memory.csv` | `scripts/meta_build.py` |
| 4 | The current branch state is a near-first-order predictive summary | Tab. 1 | `artifacts/predictive_history.csv` | `scripts/p3_predictive.py` |
| 5 | Residual dependence shapes long-horizon divergence | Fig. 5 | `artifacts/section4_stakeholder_resolution/tables/recursive_forecast_1024.csv` | `scripts/sr_p5_residual.py, scripts/sr_p6_recursive.py` |
| 6 | The current state predicts future instability (Track D) | Fig. 6 | `artifacts/section4_stakeholder_resolution/tables/instability_prediction_prefork.csv` | `scripts/sr_p789_susceptibility.py, scripts/tc_item4_raw.py` |
| 7 | The same holds on the independently trained Track-A lineage, at horizons up to 100 updates | Tab. 22, Fig. 8 | `artifacts/trackA_prefork_instability_v2/trackA_prefork_instability_paper_table.csv` | `scripts/ta_prefork.py, scripts/ta_prefork_v2.py` |
| 8 | The Track-A signal is not an artifact of finite-rollout modal uncertainty | appendix robustness table | `artifacts/trackA_measurement_control_v2/trackA_pairwise_measurement_control_summary.csv` | `scripts/ta_p0_measctl.py, scripts/ta_p0_pairwise.py` |

## Environment

Python 3.11 with `numpy`, `scipy`, `scikit-learn`, `matplotlib`. No GPU and no
network access are needed for anything in this repository.

```
pip install -r requirements.txt
```

## Reproducing results from the frozen artifacts (no retraining)

```
python scripts/p1_sharpening.py            # claim 1
python scripts/bk4_omnibus.py              # claim 2  (~1 min, 32 workers)
python scripts/meta_build.py               # claim 3  + figures 2-5
python scripts/p3_predictive.py            # claim 4
python scripts/sr_p5_residual.py           # claim 5
python scripts/sr_p6_recursive.py          # claim 5
python scripts/sr_p789_susceptibility.py   # claim 6
python scripts/tc_item4_raw.py             # claim 6, raw-margin ranking
python scripts/ta_prefork.py               # claim 7
python scripts/ta_prefork_v2.py            # claim 7, all bootstrap draws
python scripts/ta_p0_measctl.py            # claim 8
python scripts/ta_p0_pairwise.py           # claim 8, matched pairwise
python scripts/fig_final.py                # renders the paper figures
```

Set `ARTIFACT_ROOT` to point at a different export, or `ARTIFACT_OUTPUTS` to
redirect where results are written. Both default to this repository.

Two notes on exact reproduction:

* `bk4_omnibus.py` seeds its per-prompt permutation stream with
  `abs(hash(question_id))`, and Python randomises string hashing per process.
  Set `PYTHONHASHSEED` to a fixed value for bitwise-identical significance
  counts; without it, prompts whose BH-adjusted *q* sits at the 0.05 boundary
  can move by one. The observed test statistics are unaffected and reproduce to
  eight decimal places.
* `ta_prefork_v2.py` re-derives the `ta_prefork.py` bootstrap exactly and
  asserts the match; `v1_v2_verification.csv` records the comparison.

## What is provided as frozen output rather than as a rerun path

Training and generation are not reproducible from this repository, by design:

* **Model checkpoints and training runs.** The primary lineage is a 3B model
  trained with GRPO, forked from four complete resumable states. The
  checkpoints are hundreds of gigabytes and are not redistributable here. The
  effective configuration is in `config/` and in the paper appendix.
* **Raw generation corpora.** The endpoint evaluation is 106 problems x 256
  rollouts x 13 arms, about 362 MB of full model generations. We ship instead
  the exact three-field projection (`question_id`, `rollout_id`,
  `entrance_family`) that the omnibus analysis consumes, under
  `artifacts/round11_3b_trunk/eval/`. This is sufficient to rerun that analysis
  unchanged and reproduces every observed test statistic exactly.
* **HumanEval rollouts.** Provided as the frozen branch codebook and the
  per-state trajectory summaries, not as raw completions.

Everything downstream of those — every statistic, interval, table and figure in
the paper — reruns from what is here.

## Provenance

Exact training and evaluation configurations, branch-construction procedures,
controlled-continuation protocols, statistical estimators, robustness analyses
and artifact provenance are documented in the paper appendix. Each script
writes a manifest recording its inputs, their SHA256 hashes, its random seeds
and its own hash.
