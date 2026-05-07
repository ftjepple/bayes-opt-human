# Bayes-Opt-Human: Design Notes

This document explains the model, diagnostics, and workflow that BayesOpt-Human
implements. It is aimed at readers who want to understand *why* the library
makes the choices it does; for usage examples see [README.md](README.md) and
[notebooks/demo.ipynb](notebooks/demo.ipynb).

## 1. Problem and guiding principles

Bayes-Opt-Human targets expensive black-box optimization problems where each
function evaluation is costly — in time, money, or physical resources — and the
user wants to retain control over every decision. Unlike fully automated
Bayesian optimization libraries, Bayes-Opt-Human operates **one step at a time**:

- **Human-in-the-loop.** The library recommends; the user decides. Evaluation
  happens outside the library.
- **Horizon-aware.** The total evaluation budget is known and finite. The
  recommendation engine uses budget pressure to shift its ranking toward
  exploitation as the budget is consumed.
- **Diagnostic-rich.** Every recommendation is backed by quantitative
  diagnostics (LOO, coverage, modality, SVM sense-check).
- **Inspectable.** No hidden automated loop. Each round is an explicit step
  that the user can interrogate.
- **Prior-aware.** The user can encode beliefs about modality and output
  transform, and the system accounts for them.

## 2. Workflow

Each round follows a fixed three-phase sequence. Each phase has a
`get → report → choose` triple:

```
1. DATA      get_data()       → report_data()       → choose_data()
2. MODEL     get_models()     → report_models()     → choose_model()
3. CANDIDATE get_candidates() → report_candidates() → choose_candidate()
```

The `get_*` call does the expensive work (loading, fitting, acquisition
optimization). `report_*` produces tables and plots but does not mutate state.
`choose_*` commits the user's choice and returns the payload needed by the
next phase. Within one round the user can call `report_*` freely before
committing.

`choose_candidate` returns the next evaluation point in raw (user-facing)
coordinates. The user evaluates it externally, appends the result to the CSV,
and runs another round.

## 3. Data

**Inputs.** Observations are stored as CSV files with one row per
evaluation: `N_DIM` parameter columns plus one scalar objective column. The
user supplies the raw values and the parameter bounds; the library owns
normalization.

**Normalization.** All internal operations — GP fitting, acquisition
optimization, diagnostics — run on inputs normalized to `[0, 1)` using the
supplied bounds:

```
x_norm_i = (x_raw_i - lower_i) / (upper_i - lower_i)
```

Outputs shown to the user are denormalized back to raw coordinates.

**Validation.** On load, the loader rejects missing values, non-numeric
columns, and datasets with fewer than two observations. The step API
additionally checks that parameter names match, that the objective column
exists, and that the observation count does not exceed the configured horizon.

**Progress diagnostics.** The data phase reports stagnation length (consecutive
observations without improvement in best-so-far) and a progress chart showing
the objective trace and its running best.

## 4. Configuration and priors

Configuration is centralized in a single `OptimizationConfig` dataclass. The
user sets the following before model fitting; they persist across rounds
unless explicitly changed:

| Parameter   | Options                                 | Effect                                      |
| ----------- | --------------------------------------- | ------------------------------------------- |
| `direction` | `"minimize"` / `"maximize"`             | Orients acquisition functions               |
| `modality`  | `"unimodal"` / `"multimodal"` / `"unknown"` | Influences recommendations + diagnostics |
| `horizon`   | positive int                            | Total evaluation budget                     |
| `warmstart` | int ≥ 0, < horizon                      | Observations that existed before the run   |
| `bounds`    | per-parameter `(lo, hi)` or tuple       | Search space                                |

Tunables for acquisition and GP fitting (κ values, priors, KNN `k`, jitter
bounds) have sensible defaults on `OptimizationConfig`; see the class for
the full list.

## 5. Gaussian process surrogate

**Base configuration.** ARD Matérn-5/2 kernel (GPyTorch `MaternKernel` with
`nu=2.5`, `ard_num_dims=N_DIM`), fixed-noise observation model with adaptive
jitter, constant mean. The model assumes noiseless observations.

**Adaptive jitter.** Jitter scales with the variance of the transformed,
normalized targets: initial jitter `max(1e-6, 1e-4 · var(y))`, multiplied by 10
on Cholesky failure, hard ceiling at `1e-2 · var(y)`. Hitting the ceiling
raises a conditioning warning. The jitter used on each fit is logged.

**Output transforms.** Four transforms are evaluated, applied **before** IQR
normalization:

| Transform      | Formula                                        | Use case                                                                        |
| -------------- | ---------------------------------------------- | ------------------------------------------------------------------------------- |
| `none`         | `y' = y`                                       | Default; well-behaved objectives                                                |
| `log`          | `y' = log(y)`, or `-log(-y)` for negative data | Strictly positive **or** strictly negative heavy-tailed objectives. Sign mode is detected on first fit and then locked; mixed-sign data is rejected. |
| `signed_log1p` | `y' = sign(y) · log(1 + \|y\|)`                | Mixed-sign objectives with wide dynamic range; compresses both tails symmetrically and is continuous through zero |
| `clipped_log`  | `y' = log(max(y, ε))`                          | Positive objectives with occasional near-zero or non-positive values; censoring is only enabled when the optimum is not among the censored values |


After the transform, targets are IQR-normalized:
`y_norm = (y' - median(y')) / IQR(y')`. IQR is robust to outliers in the
transformed targets.

**Input warping.** Optional Beta-CDF warping per dimension (via BoTorch's
`Warp` input transform), with two learnable parameters per dimension. Warping
is only **considered** when `N_OBS ≥ 10 · N_DIM`; below that threshold the
additional parameters cannot be identified and LOO comparisons are
misleading. Warping is disabled by default.

**Surrogate space.** Surrogate space is the coordinate system in which the GP
has isotropic unit correlation decay:

```
x_sur_i = w_i(x_flat_i) / l_i
```

where `w_i` is the learned warp (identity if warping is disabled) and `l_i` is
the learned ARD length scale. Euclidean distance in surrogate space is
directly comparable to correlation drop. Model-aware diagnostics (coverage,
modality, space-filling) operate here. Surrogate space is only defined after
fitting and changes between rounds, so cross-round comparisons should use flat
space instead.

**Model selection.** The library fits each output transform (no warping), and
optionally re-fits each with warping when the data-to-dimension ratio supports
it. Candidates are ranked by analytical LOO log predictive density (LPD),
highest first. LOO-MAE and calibration variance are computed and shown in
the table as diagnostics, but they do not influence the sort order. The
user sees the ranked table plus per-candidate diagnostics (length scales,
warp parameters, residuals, transform comparison) and makes the call. The
library recommends; it does not enforce. From round two onward the engine
adds inertia: it keeps the previous model unless a paired LOO-LPD t-test
on the best alternative clears a significance threshold — and that
threshold is stricter when the alternative is a more complex kernel
(ISO < ARD < WARP) than the incumbent.

**Model selection as meta-exploration.** Choosing the surrogate model
(and output transform) each round is itself an explore-vs-exploit
decision, one layer above the candidate selection. *Exploitation* means
sticking with the current spec: the trajectory of past observations was
chosen against that model, so its posterior is "tuned" to the data, and
switching can disrupt convergence. *Exploration* means trying a
different spec on the chance that it fits better, accepting that the
data path may no longer be optimal for the new model. The inertia rules
above implement this trade-off explicitly: they bias toward exploitation
unless the LPD evidence is strong enough — and stricter when the
proposed switch is to a more complex kernel, because the cost of being
wrong scales with model complexity.

**Demotion and warnings.** Each candidate is health-checked before ranking.
If any check hits a "red" threshold the candidate is **demoted**: it still
appears in the ranked table but below a separator, and the recommendation
engine will not pick it unless no non-demoted candidate is available.
A candidate is demoted when any of:

- **Severe miscalibration** — LOO calibration variance outside `[0.1, 5.0]`
  (extremely over- or under-confident).
- **Length scales pinned at bounds** — at least half the learned length
  scales sit within 5% of the log-range from either configured boundary,
  so the kernel is either collapsing those dimensions or ignoring them.
- **Extreme warp parameters** — when warping is enabled, at least half the
  Beta concentration parameters fall outside `[0.1, 20]`.

Milder versions of the same three signals surface as **advisory warnings**
(amber in the UI). They show up in the diagnostics table but do not affect
the ranking or the demotion line.

Two transforms are **filtered out before ranking** and never enter the
candidate table at all: `clipped_log` fails its eligibility check (needs at
least five positive observations plus a direction-dependent extremity test)
and `log` fails on mixed-sign data (it requires strictly positive or strictly
negative observations).

## 6. Candidate generation

After model selection, the GP is refit on all observations and one candidate
is produced per strategy. No batching — the user picks one from the set.

Acquisition functions are optimized with BoTorch's `optimize_acqf` (multi-start
L-BFGS-B).

| # | Strategy                         | Category      |
| - | -------------------------------- | ------------- |
| 1 | Max-min distance (flat)          | Space-filling |
| 2 | Max-min distance (surrogate)     | Space-filling |
| 3 | Max posterior variance (MPV)     | Exploration   |
| 4 | Expected improvement (EI)        | Balanced      |
| 5 | UCB/LCB, low κ (0.5)             | Exploitation  |
| 6 | UCB/LCB, medium κ (1.5)          | Balanced      |
| 7 | UCB/LCB, high κ (2.5)            | Exploration   |
| 8 | Max GP mean                      | Exploitation  |

**Direction handling.** Minimization uses LCB (`μ - κσ`); maximization uses
UCB (`μ + κσ`). EI is oriented the same way.

**Per-candidate diagnostics.** Each candidate carries its posterior mean and
std, expected improvement, coverage gain (flat and surrogate), and the SVM
sense-check prediction.

## 7. Diagnostics

### 7.1 GP fit

- **LOO-MAE / LPD / calibration variance** — analytical leave-one-out from the
  closed-form formulas. Primary measures of GP predictive quality and
  calibration. LOO-MAE and calibration variance are both reported on a
  natural scale so absolute values can be read without further context:
  LOO-MAE is in units of one IQR of the (transformed) targets, so a value of
  `0.1` means the typical leave-one-out error is one tenth of the observed
  inter-quartile range; calibration variance is the variance of standardized
  residuals `(y - μ_LOO) / σ_LOO` and equals `1.0` for a perfectly calibrated
  GP. LPD is reported in nats and is only meaningful when comparing models
  on the same data, which is how the ranking uses it.
- **Conditioning** — condition number / log-determinant. Paired with the
  jitter used; large jitter is a warning.
- **Length scales** — per-dimension, plotted on a log scale. Values near the
  permitted bounds are flagged (near lower: overfitting; near upper: dimension
  being ignored).
- **Warp parameters** — Beta-CDF parameters per dimension, when warping is
  enabled.
- **Feature importance** — `1 / l_i`, higher is more informative.

### 7.2 Coverage

Average and minimum KNN distance over the observation set, computed in both
flat and surrogate space and compared to the expected value for an optimal
space-filling design of the same size. Default `k = min(5, N_OBS - 1)`.

### 7.3 Modality

KNN-based local optimum detection: a point is a candidate local optimum if it
beats all `k=5` nearest neighbors. Clusters of such points estimate the number
of distinct basins. Computed in both flat and surrogate space. Gated by a
reliability check: if `N_OBS / N_DIM < 5` a warning is emitted, since the
heuristic needs enough density to be meaningful.

### 7.4 SVM sense-check

An independent classifier that serves as a sanity check on the GP:

- Soft-margin RBF SVM (`sklearn.SVC`) fit on the observations.
- Binary target: top-quartile-vs-rest.
- Reliability assessed with leave-one-out balanced accuracy plus a binomial CI.
- For each candidate, the SVM's predicted class is reported alongside the GP's
  recommendation.

If the LOO balanced accuracy is not significantly better than chance, the
sense-check is flagged as unreliable for the current dataset and should be
disregarded.

## 8. Candidate Recommendations

The recommendation engine synthesizes the diagnostics into a ranked list of
candidates plus plain-language rationales. Each recommendation is a structured
object carrying the candidate coordinates, source strategy, rationale,
confidence level, and priority.

The engine allocates weight across four strategy arms and scores each
candidate as `w_arm · within-arm percentile`.

**Signals.** Four scalar signals drive the allocation:

- **u (urgency)** = `1 − (remaining − 1) / (budget − 1)`. Rises from 0 at the
  start to 1 on the final evaluation.
- **q (GP quality)** = `exp(−|log(calibration_var)|)`. Peaks at 1 when the
  LOO standardized-residual variance is 1.0, decays symmetrically in log
  space when σ is too small or too large.
- **s (stagnation)** = `max(0, (s_raw − 0.3) / 0.7)`, where
  `s_raw = stag_length / n_opt_obs`. A 30% deadband suppresses small-sample
  noise early in the run.
- **c (coverage need)** = `1 − 1/ratio_to_optimal`. Rises when observed
  points leave large gaps in the search space.

From these, two pulls:

    exploit_pull = u
    explore_pull = clip(1, (1 − u) + s + c)

Additive aggregation of the three exploration drivers lets "stagnating
*and* undersampled" be a stronger case for exploring than either alone.

**Arm weights.** Four arms — matching the four candidate categories —
receive weight:

| Arm | Candidates | Weight (unnormalized) |
|---|---|---|
| **exploit** | Max GP Mean, low-κ UCB/LCB | `exploit_pull · max(q, 0.2)` |
| **balanced** | EI, medium-κ UCB/LCB | `0.5 · (exploit_pull + explore_pull) · √q` |
| **model-explore** | Max-PV, high-κ UCB/LCB, Max-min distance (surrogate) | `explore_pull · q` |
| **geometric-explore** | Max-min distance (flat) | `explore_pull · (1 − q) + 0.05` |

Weights are normalized to sum to 1. Three design points:

- The `max(q, 0.2)` floor on the exploit arm preserves a minimum exploit
  option when the GP is badly miscalibrated — even a suspect µ carries
  some ranking information.
- The `√q` on the balanced arm captures EI's graceful degradation under
  poor calibration; it is less q-sensitive than the model-explore arm.
- The `+ 0.05` on the geometric arm keeps it alive as a safety option
  even when the GP is perfectly calibrated.

The key coupling: **calibration decides whether exploration is
model-based or geometric**. When `q → 1`, exploration weight flows to
the Max-PV / high-κ UCB arm; when `q → 0`, it flows to flat max-min
distance, which does not depend on the GP at all.

**Per-candidate score.** Each candidate is ranked against peers in the
*same arm* on the arm's natural metric (μ, EI, σ, flat coverage gain),
yielding a within-arm percentile `p_within ∈ [0, 1]`. The global score
is then

    score_i = w_arm(i) · p_within(i).

The top candidate in each arm is flagged `is_arm_winner`, and surfaced
with a ★ in the candidate table so the user can always see "the best
geometric option", "the best exploit option", etc. regardless of overall
rank. A separate diversify step demotes near-duplicates (two candidates
within half the average KNN distance are not both surfaced at the top).

Qualitative horizon behaviour:

- **Early stage** (remaining > 50% of horizon): explore-pull is high;
  allocation is dominated by model-explore or geometric-explore depending
  on GP calibration.
- **Mid stage** (20–50% remaining): the balanced arm (EI, medium-κ
  UCB/LCB) carries a growing share. GP-vs-SVM disagreement downgrades
  confidence.
- **Late stage** (< 20% remaining): exploit-pull dominates; the exploit
  arm usually wins.
- **Final evaluation** (remaining = 1): if the GP is well-calibrated the
  exploit arm dominates. If the GP is badly miscalibrated, the geometric
  arm keeps most of the weight — the design choice being that trusting a
  broken model's mean on the last round is worse than one last geometric
  probe.

Confidence is derived from LOO-MAE (relative to target IQR), conditioning
warnings, and length-scale stability.

## 9. Visualization

Reports (a mix of Matplotlib figures and formatted tables) are surfaced at
every phase of the workflow via the same `get → report → choose` pattern
described in §2. Each phase exposes two complementary registries of
reports that serve different questions:

- **Summary reports** compare *across* the items at that phase. They sit
  side-by-side for every loaded dataset (data phase), every fitted model
  candidate (model phase), or every generated candidate point (candidate
  phase), and answer "which option stands out?" — e.g. a grid of progress
  charts across datasets, a table of per-model calibration diagnostics, or
  a projection showing all candidates overlaid on the observations.
- **Individual reports** drill down into one selected item. They answer
  "is this specific choice a good one?" — e.g. the objective trace and
  coverage gaps for a single dataset, the length scales and LOO residuals
  for a single fitted GP, or the 2D projection and closest-neighbour table
  for a single candidate point.

Both types are embedded inline in the Jupyter notebook or in the Panel UI,
and both are registered declaratively so the set of available reports
grows over time without widening the public API. See §11 for the dispatch
conventions and
`bayesopt_human/reporting/{data,models,candidates}.py` for the current
catalog.

## 10. Module layout

```
bayesopt_human/
├── config.py              # OptimizationConfig dataclass
├── utils.py               # Shared utilities (stagnation length, formatters)
├── data/                  # CSV loading + bounds-based normalization
├── transforms/            # Output transforms, IQR normalization, Beta-CDF warping
├── gp/                    # Model, LOO, adaptive jitter, model selection, surrogate space
├── acquisition/           # Acquisition functions, space-filling, candidate orchestration
├── diagnostics/           # GP fit, coverage, modality, SVM sense-check
├── recommendations/       # Rule-based engine + recommendation schema
├── visualization/         # Progress, length scales, projections, candidate table
├── optimizer/             # OptimizationStep (get/report/choose orchestration), state persistence
└── ui/                    # Panel web app (optional, `pip install -e '.[ui]'`)
```

## 11. Reporting API

`report_data`, `report_models`, and `report_candidates` share a common shape:

```python
step.report_data(result, report=None, choice=None)
```

- `report=None` lists the available reports (as strings) for the current
  `choice` context and prints their one-line descriptions. The listing is
  generated from the registries in `bayesopt_human/reporting/` so it stays
  in sync with the actual set of reports.
- `report="all"` renders every report in the current context.
- `report="<name>"` renders one specific report.
- `choice=None` selects the **summary** registry — multi-config aggregate
  reports.
- `choice=<int>` selects the **individual** registry — reports for one
  specific config (data entry index, model candidate index, or
  recommendation rank for candidates).

Summary and individual reports may share a keyword (e.g. `progress`,
`histogram`, `coverage_metrics`, `modality`) — the `choice` argument
disambiguates.

The full catalog is declared in
`bayesopt_human/reporting/{data,models,candidates}.py` as two lists per
phase (`SUMMARY_REPORTS`, `INDIVIDUAL_REPORTS`) of `Report` entries with a
`key`, a one-line `description`, and the function that renders the report.
Adding a new report is a single-entry change in the appropriate list; the
help listing, the UI dropdown, and the `report="all"` fan-out all pick it up
automatically. The UI step files (`bayesopt_human/ui/steps/`) import the
same registries so the Panel web app stays in sync with the notebook API.

## 12. Implementation notes

- **State persistence.** Each `choose_candidate` call appends a `RoundRecord`
  to an `OptimizationState` JSON file alongside the data CSV. This records the
  model choice, the selected candidate, and the diagnostics that informed the
  decision, so later rounds (and the UI) can show the full history.
- **Surrogate space is ephemeral.** It is recomputed on each model fit. Do not
  cache surrogate-space coordinates across rounds.
- **No automated loop.** The library deliberately does not provide a
  `run_for_n_iterations()` entry point. If you want one, you can write it in
  ten lines on top of `OptimizationStep` — but that loses the human-in-the-loop
  contract.
- **Constraints and failed evaluations.** Not currently handled. The library
  assumes every observation is a valid (coordinate, objective) pair inside the
  declared bounds.
