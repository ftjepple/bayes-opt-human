# Datasheet: BBO Capstone Query–Response Dataset

This datasheet documents the dataset accumulated during the Imperial
College ML & AI **Bayesian Black-Box Optimisation (BBO)** capstone,
as tracked in this repository. 

## Motivation

- **Purpose.** The dataset supports **sequential optimisation** of
  eight synthetic black-box objective functions under a strict
  evaluation budget. Each function starts from a course-provided
  warmstart; the participant then submits one query per function
  per round over **thirteen rounds** (one per week), and the platform returns the function
  evaluation. The dataset captures every submitted point query and
  every returned value, which is what the author's optimisation
  library ([`bayesopt_human`](bayesopt_human/)) fits GPs on and
  proposes the next query from.
- **Task supported.** Empirical study of **Bayesian
  optimisation-style** workflows (Gaussian processes, acquisition
  functions, human-in-the-loop decisions) on problems of dimension
  2 through 8 inside the unit hypercube.
- **Creator and context.** The **initial warmstart designs** and the
  **hidden function evaluations** were provided by the BBO capstone course. All
  **per-round queries** were chosen by the author using the
  Bayes-Opt-Human library's `OptimizationStep` API (or, equivalently,
  its Panel web UI) on top of the accumulated history. The recorded
  `(x, y)` pairs are the join of those two sources.
- **Funding.** Academic coursework only; no external funding.

## Composition

- **Instances.** One instance is **one evaluation of one function**:
  an input vector `x` in `[0, 1]^D` (where `D` depends on the
  function) paired with a scalar output `y` returned by the
  platform.

- **Structure.** The dataset lives under `data/` in this repository
  and contains **8 accumulated per-function CSVs** (tracked in git and
  bundled with the library for reproducible demos):
  `data/fn_1.csv` … `data/fn_8.csv`. Each file has one row per
  evaluation, one column per parameter dimension, and one
  `objective` column for the function response. Rows are ordered
  chronologically (warmstart first, then one row per round).
  After all thirteen rounds the sizes are:

  | Function | Dims | Warmstart `n` | Opt. Rounds | Total `n` |
  | --- | ---: | ---: | ---: | ---: |
  | `fn_1` | 2 | 10 | 13 | 23 |
  | `fn_2` | 2 | 10 | 13 | 23 |
  | `fn_3` | 3 | 15 | 13 | 28 |
  | `fn_4` | 4 | 30 | 13 | 43 |
  | `fn_5` | 4 | 20 | 13 | 33 |
  | `fn_6` | 5 | 20 | 13 | 33 |
  | `fn_7` | 6 | 30 | 13 | 43 |
  | `fn_8` | 8 | 40 | 13 | 53 |

- **Format.**
  Standard CSV with a header row.
  Parameter columns are named `x1`, `x2`, … `x<D>` (all in
  `[0, 1]`) and the objective column is named `objective`.

- **Size.** After all thirteen rounds, the total number of `(x, y)`
  pairs across all eight functions is **279** (23 + 23 + 28 + 43 + 33 + 33 + 43 + 53). The on-disk footprint of the CSVs is roughly 27 KB. 

- **Completeness and gaps.** No missing values: every submitted
  point has a recorded function response. The dataset is complete in
  that sense. What it is *not* is a uniform map of `[0, 1]^D` —
  see the bias note in **Uses** below.

- **Sensitive or confidential content.** **None.** The objective functions
  are synthetic, there is no personal data.

## Collection process

- **Initial data.** The course supplies the initial inputs and outputs (warmstart)
  for each function. These are fixed over the entire run and form
  the first `n_warmstart` rows of each function's accumulated CSV.

- **Per-round data.** For every round and for every function the author runs the workflow cycle from `bayesopt_human` to generate the coordinates for the next round's point queries. These are submitted to the course platform, which provides a scalar objective function value for the submitted queries. Inputs and outputs are appended to the per-function csv files before the next round starts. See `model_card.md` §3 for the detailed decision logic applied at each step.

- **Sampling strategy.** **Adaptive sequential design**, not i.i.d. sampling. Every round's candidate is chosen based on the GP posterior fit to all points observed so far, so later points cluster near historically strong regions (exploitation) or test explicit hypotheses (space-filling, boundary probes, high-κ UCB corners). This induces a **strong spatial and temporal bias** relative to uniform exploration; see **Uses** for the consequences.

- **Configuration fixed up front.** Every function has an `OptimizationConfig` defined in `data/config.py` with the horizon, warmstart count, bounds, objective-column name, direction, and modality prior. These were chosen before the first round and were **not** adjusted between rounds.

- **Time frame.** Thirteen rounds of submissions, one round per week, over the capstone period.

## Preprocessing / cleaning / labelling

- **Preprocessing of raw data** The course provides warmstart data in the form of two .npy files per function (one for inputs and one for outputs). After every submission, the platform returns two .txt files containing the inputs and outputs for all functions and all post-warmstart optimisation rounds. These datasources were processed with a python helper script and converted to per-function csv files.
- **Processing of stored data: none.** The `fn_<k>.csv` files contain the raw `(x, y)` pairs returned by the platform. No scaling, transformation, or filtering is applied before writing to disk.
- **Model-side processing (not written back).** Inside the library the outputs go through an output transform (`none`, `log`, `clipped_log`, or `signed_log1p`) followed by IQR normalisation before the GP fit. These transforms are ephemeral model-space operations and do not overwrite anything in the CSV.

## Uses

- **Appropriate uses.**
  - Reproducing or critiquing a documented Bayesian optimisation workflow under a tiny evaluation budget.
  - Illustration of a detailed step-by-step Bayesian optimisation workflow with rich diagnostics.
  - Studying common failure modes of small-data BO: over-exploitation, narrow-peak capture, boundary optima.
  - Academic learning: the eight functions span dimensionalities 2 through 8 and include a visibly degenerate case (`fn_1`) and a boundary-solution case (`fn_5`), which makes them a useful set for demonstrating a wide range of black box function shapes.

- **Inappropriate uses.**
  - **Any real-world decision.** The objectives are synthetic course fixtures with no confirmed grounding in physical or social processes. 
  - **Uniform exploration / distributional assumptions.** The point cloud is the output of an adaptive sequential policy, so it is **not** an unbiased sample of the hypercube. The density of the point cloud is a result of the library's exploitation policy, and the author's ad-hoc decisions, not the underlying domain.
  - **Cross-function comparison of raw `y`.** The objective scales differ by many orders of magnitude — compare `fn_1` near 0 with `fn_5` at ~8600 — so any score that compares raw values across functions is meaningless without per-function normalisation.
  - **Inference of the hidden oracle's analytical form.** The objective function definitions are course IP and are *not* in this repository. The observed `(x, y)` pairs are enough to fit a surrogate but not enough to reverse-engineer the generating functions with any confidence at this sample size.

## Distribution

- **Availability.** The dataset is stored in this GitHub
  repository. Specifically:

  - **Tracked in git** (shipped with the library): the eight
    `data/fn_<k>.csv` files, the eight `data/fn_<k>_state.json`
    files (persisted optimiser state, including the round history
    of past model and candidate choices), and the `data/config.py`
    manifest. A fresh clone can run
    `python -m bayesopt_human.ui --config data/config.py --data-dir data/`
    immediately on all eight problems and pick up where the last
    committed round left off.
  - **Not tracked in git** (local only): the course-provided
    warmstart `.npy` files under `data/warmstart/function_*/`, the
    unprocessed `.txt` files produced by the platform every round,
    and any scratch utility scripts (e.g. for data pre-processing).

- **Terms of use.** MIT licence — see `LICENSE` — for everything
  tracked in git. The hidden objective function definitions remain course
  intellectual property and are not redistributed here; this
  datasheet describes only the *observed inputs and outputs* and
  the methodology used to collect them.

## Maintenance

- **Maintainer.** The student author for the duration of the
  capstone and any agreed retention period afterwards.

