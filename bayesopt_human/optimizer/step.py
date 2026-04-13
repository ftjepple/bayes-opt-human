"""Orchestrates one round: get -> report -> choose for data, models, candidates."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bayesopt_human.acquisition.candidates import Candidate, CandidateGenerator
from bayesopt_human.config import OptimizationConfig
from bayesopt_human.data.loader import load_observations
from bayesopt_human.data.normalizer import BoundsNormalizer
from bayesopt_human.diagnostics.coverage import (
    CoverageReport,
    coverage_gain,
    coverage_metrics,
)
from bayesopt_human.diagnostics.gp_diagnostics import GPFitReport, gp_fit_summary
from bayesopt_human.diagnostics.modality import ModalityReport, detect_local_optima
from bayesopt_human.diagnostics.svm import SVMReport, svm_predict_candidate, svm_sense_check
from bayesopt_human.gp.model import FittedGP, GPModel
from bayesopt_human.gp.selection import (
    ModelCandidate,
    ModelSelector,
    length_scale_flags,
    model_n_params,
    warp_flags,
)
from bayesopt_human.gp.model_recommendation import (
    ModelRecommendation,
    recommend_model,
)
from bayesopt_human.gp.surrogate_space import SurrogateSpaceTransform
from bayesopt_human.transforms.warping import InputWarping
from bayesopt_human.optimizer.state import (
    ModelChoiceRecord,
    OptimizationState,
    load_state,
    record_round,
    save_state,
)
from bayesopt_human.recommendations.engine import RecommendationEngine
from bayesopt_human.recommendations.schema import Recommendation
from bayesopt_human.utils import apply_seed
from bayesopt_human.visualization.candidates import display_candidate_table

logger = logging.getLogger("bayesopt_human")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class DataEntry:
    """Summary of one loaded data file."""
    filepath: str
    df: pd.DataFrame
    objective_column: str
    param_names: list[str]
    n_obs: int
    n_dim: int


@dataclass
class DataResult:
    """Container returned by get_data()."""
    entries: list[DataEntry]


@dataclass
class DataPayload:
    """Container for loaded and normalized data."""

    df: pd.DataFrame
    X_raw: np.ndarray
    X_norm: np.ndarray
    y: np.ndarray
    normalizer: BoundsNormalizer
    param_names: list[str]


# ---------------------------------------------------------------------------
# Model containers
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    """Container returned by get_models()."""
    candidates: list[ModelCandidate]
    data: DataPayload
    recommendation: ModelRecommendation | None = None


# ---------------------------------------------------------------------------
# Candidate containers
# ---------------------------------------------------------------------------

@dataclass
class CandidateResult:
    """Container returned by get_candidates()."""
    candidates: list[Candidate]
    recommendations: list[Recommendation]
    gp_report: GPFitReport
    coverage: CoverageReport
    modality: ModalityReport
    svm_report: SVMReport
    candidate_table: pd.DataFrame
    data: DataPayload
    fitted_gp: FittedGP


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt(v: float, width: int = 8) -> str:
    """Format a metric value, switching to scientific notation when large."""
    if abs(v) >= 1000:
        return f"{v:{width}.2e}"
    return f"{v:{width}.4f}"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class OptimizationStep:
    """Orchestrates a single round of the optimization workflow.

    This is the primary entry point for the Jupyter notebook.
    Uses a consistent get -> report -> choose pattern:

    1. Data:       get_data -> report_data -> choose_data
    2. Models:     get_models -> report_models -> choose_model
    3. Candidates: get_candidates -> report_candidates -> choose_candidate
    """

    def __init__(self, configs: list[OptimizationConfig]) -> None:
        self.configs = configs


    # -----------------------------------------------------------------------
    # Phase 1: Data
    # -----------------------------------------------------------------------

    def get_data(
        self,
        data_dir: str,
    ) -> DataResult:
        """Load and validate data files for each config in self.configs. Does NOT normalize.

        Args:
            data_dir: Directory containing CSV files.

        Returns:
            DataResult with entries for each valid file found.
        """
        path = Path(data_dir)
        if not path.is_dir():
            raise ValueError(f"Not a directory: {data_dir}")
        self._data_dir = str(path)

        entries: list[DataEntry] = []
        for config in self.configs:
            csv_path = path / f"{config.name}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"CSV file not found for config '{config.name}': {csv_path}"
                )
            # load_observations already verifies that ``objective_column``
            # exists in the CSV and raises ``ValueError`` otherwise.
            df = load_observations(str(csv_path), config.objective_column)
            param_names = [c for c in df.columns if c != config.objective_column]

            if len(df) > config.horizon:
                raise ValueError(
                    f"CSV for '{config.name}' contains {len(df)} observations, "
                    f"which exceeds horizon {config.horizon}"
                )

            # ``update_from_data`` is the single source of truth for expanding
            # tuple bounds into a per-parameter dict.
            config.update_from_data(len(df), param_names)

            if set(config.bounds.keys()) != set(param_names):
                raise ValueError(
                    f"Parameter names in bounds {list(config.bounds.keys())} "
                    f"do not match CSV columns {param_names} for '{config.name}'"
                )

            entries.append(DataEntry(
                filepath=str(csv_path),
                df=df,
                objective_column=config.objective_column,
                param_names=param_names,
                n_obs=len(df),
                n_dim=len(param_names),
            ))

        # Print summary table
        n_files = len(entries)
        print(f"Found {n_files} data file{'s' if n_files > 1 else ''}:")
        for idx, entry in enumerate(entries):
            fname = Path(entry.filepath).name
            params = ", ".join(entry.param_names)
            print(
                f"  [{idx}] {fname:30s} | {entry.n_obs:3d} obs, "
                f"{entry.n_dim} dims  | params: {params}"
            )
        return DataResult(entries=entries)

    def report_data(
        self,
        result: DataResult,
        report: str | None = None,
        choice: int | None = None,
    ) -> list[str]:
        """Show data reports (summary across configs, or individual per config)."""
        from bayesopt_human.reporting.base import run_reports
        from bayesopt_human.reporting.data import (
            INDIVIDUAL_REPORTS,
            SUMMARY_REPORTS,
            DataIndividualContext,
            DataSummaryContext,
        )

        if choice is None:
            ctx = DataSummaryContext(result=result, configs=self.configs)
            return run_reports(
                SUMMARY_REPORTS,
                report,
                context=ctx,
                header="Available summary reports for data:",
                kind="summary",
            )

        if not isinstance(choice, int) or choice < 0 or choice >= len(result.entries):
            raise ValueError(
                f"Choice index {choice} out of range (0-{len(result.entries) - 1})."
            )

        entry = result.entries[choice]
        entry_name = Path(entry.filepath).stem
        config = next((c for c in self.configs if c.name == entry_name), None)
        if config is None:
            raise ValueError(f"No config found for entry {entry_name}")

        ctx = DataIndividualContext(result=result, entry=entry, config=config)
        return run_reports(
            INDIVIDUAL_REPORTS,
            report,
            context=ctx,
            header=f"Available individual reports for function [{choice}]:",
            kind="individual",
        )

    def choose_data(
        self,
        result: DataResult,
        choice: int = 0,
        bounds: dict[str, tuple[float, float]] | None = None,
    ) -> DataPayload:
        """Select a data file and normalize.

        Args:
            result: DataResult from get_data().
            choice: Index into entries list.
            bounds: Override bounds. Falls back to config bounds,
                then auto-detects from data range with a warning.

        Returns:
            DataPayload with normalized data.
        """
        if choice < 0 or choice >= len(result.entries):
            raise ValueError(
                f"Choice index {choice} out of range "
                f"(0-{len(result.entries) - 1})."
            )

        entry = result.entries[choice]
        obj_col = entry.objective_column
        p_cols = entry.param_names
        df = entry.df

        # Find the config for this entry by matching name to filename (without .csv)
        entry_name = Path(entry.filepath).stem
        config = next((c for c in self.configs if c.name == entry_name), None)
        if config is None:
            raise ValueError(f"No config found for entry {entry_name}")

        # Determine bounds
        bnds = bounds or config.bounds
        if bnds is None:
            # Auto-detect from data range
            logger.warning(
                "No bounds provided. Auto-detecting from data range. "
                "This may not reflect the true search space."
            )
            bnds = {}
            for col in p_cols:
                lo, hi = float(df[col].min()), float(df[col].max())
                # Add small margin so max value normalizes to < 1.0
                margin = max((hi - lo) * 0.01, 1e-6)
                bnds[col] = (lo, hi + margin)
        else:
            # Validate bounds match dimensions
            for col in p_cols:
                if col not in bnds:
                    raise ValueError(
                        f"No bounds provided for parameter {col!r}. "
                        f"Available bounds: {list(bnds.keys())}"
                    )

        normalizer = BoundsNormalizer({name: bnds[name] for name in p_cols})

        X_raw = df[p_cols].values
        X_norm = normalizer.normalize(X_raw)
        y = df[obj_col].values

        # Update config
        config.update_from_data(len(y), p_cols)

        fname = Path(entry.filepath).name
        print(f"Selected [{choice}]: {fname} ({entry.n_obs} obs, {entry.n_dim} dims)")

        # Store the locked config for downstream steps
        self._locked_config = config

        return DataPayload(
            df=df,
            X_raw=X_raw,
            X_norm=X_norm,
            y=y,
            normalizer=normalizer,
            param_names=p_cols,
        )

    # -----------------------------------------------------------------------
    # Phase 2: Models
    # -----------------------------------------------------------------------

    def get_models(
        self,
        data: DataPayload,
    ) -> ModelResult:
        """Evaluate output transforms, kernel types, and optional warping.

        Displays ranked table with LOO-MAE and LOO-LPD.
        Review the table, then call report_models() or choose_model().

        Args:
            data: DataPayload from choose_data.

        Returns:
            ModelResult with ranked candidates.
        """
        # Use the locked config from choose_data
        config = getattr(self, '_locked_config', None)
        if config is None:
            raise RuntimeError("No config locked. Call choose_data first.")
        apply_seed(config.random_seed)
        selector = ModelSelector(config)

        # Step 1: Evaluate transforms (ARD + non-ARD)
        print("Evaluating output transforms (ARD + isotropic)...")
        transform_candidates = selector.evaluate_transforms(
            data.X_norm, data.y,
        )

        if not transform_candidates:
            raise RuntimeError("All output transform evaluations failed.")

        # Step 2: Optionally evaluate warping (ARD only)
        warping_candidates: list[ModelCandidate] = []

        if InputWarping.sufficient_data(config.n_obs, config.n_dim, config.warping_min_ratio):
            print("Evaluating input warping...")
            ard_candidates = [mc for mc in transform_candidates if mc.ard]
            for mc in ard_candidates[:3]:
                warp_mc = selector.evaluate_warping(
                    data.X_norm, data.y, mc.output_transform, ard=True
                )
                if warp_mc is not None:
                    warping_candidates.append(warp_mc)

        # Combine all candidates
        all_candidates = transform_candidates + warping_candidates

        # Compute LS flags (for display and demotion)
        for mc in all_candidates:
            flags, ls_rag = length_scale_flags(
                mc,
                ls_lower_bound=config.length_scale_lower_bound,
                ls_upper_bound=config.length_scale_upper_bound,
            )
            mc.length_scale_flags = flags
            mc.ls_rag = ls_rag

        # Compute WRP flags (for display and demotion)
        for mc in all_candidates:
            wrp_fl, wrp_rag = warp_flags(mc)
            mc.warp_flags = wrp_fl
            mc.wrp_rag = wrp_rag

        # Compute FIT RAG and demotion
        def _kernel_label(mc):
            return "WARP" if mc.warping else ("ARD" if mc.ard else "ISO")

        for mc in all_candidates:
            # FIT RAG: red = severe miscalibration (demoted), amber = advisory
            if mc.calibration_var > 5.0:
                mc.fit_rag = 'red'
                mc.fit_note = "FIT: severely overconfident"
            elif mc.calibration_var < 0.1:
                mc.fit_rag = 'red'
                mc.fit_note = "FIT: severely underconfident"
            elif mc.calibration_var > 2.0:
                mc.fit_rag = 'amber'
                mc.fit_note = "FIT: overconfident"
            elif mc.calibration_var < 0.5:
                mc.fit_rag = 'amber'
                mc.fit_note = "FIT: underconfident"
            else:
                mc.fit_rag = 'green'
                mc.fit_note = ""

            # Demotion: red LS, red FIT, or red WRP
            mc.demoted = (mc.ls_rag == 'red') or (mc.fit_rag == 'red') or (mc.wrp_rag == 'red')

        # Sort: demoted models below the line, rest ranked by LPD
        all_candidates.sort(key=lambda c: (c.demoted, -c.loo_lpd))

        # Load previous model choice from state file
        data_dir = getattr(self, '_data_dir', None)
        prev_state = load_state(data_dir, config.name) if data_dir else None
        self._loaded_state = prev_state

        # Identify previous model choice index
        prev_choice = prev_state.model_choice if prev_state else None
        prev_choice_idx = None
        if prev_choice is not None:
            for ci, mc in enumerate(all_candidates):
                if (mc.output_transform == prev_choice.output_transform
                        and mc.ard == prev_choice.ard
                        and mc.warping == prev_choice.warping):
                    prev_choice_idx = ci
                    break

        # Model recommendation with inertia
        recommendation = recommend_model(all_candidates, prev_choice_idx)
        rec_idx = recommendation.recommended_idx

        # Re-sort: move recommended model to index 0
        if rec_idx != 0:
            rec_candidate = all_candidates.pop(rec_idx)
            all_candidates.insert(0, rec_candidate)
            # Update prev_choice_idx for the shift
            if prev_choice_idx is not None:
                if prev_choice_idx == rec_idx:
                    prev_choice_idx = 0
                elif prev_choice_idx < rec_idx:
                    prev_choice_idx += 1
            recommendation = ModelRecommendation(
                action=recommendation.action,
                recommended_idx=0,
                rationale=recommendation.rationale,
                t_stat=recommendation.t_stat,
                p_value=recommendation.p_value,
            )

        # Display table
        from tabulate import tabulate
        import sys

        def rag_circle(rag):
            in_jupyter = 'ipykernel' in sys.modules
            if in_jupyter:
                color = {'red': '#d62728', 'amber': '#ffbf00', 'green': '#2ca02c'}[rag]
                return f'<span style="font-size:2em; color:{color}; vertical-align:middle;">&#9679;</span>'
            else:
                return {'red': '●', 'amber': '◍', 'green': '•'}[rag]

        table_rows = []
        demoted_seen = False
        n_cols = 12
        for idx, mc in enumerate(all_candidates):
            kernel = _kernel_label(mc)
            ls_flags = getattr(mc, 'length_scale_flags', [])
            ls_note = " | ".join(ls_flags)
            wrp_flags_list = getattr(mc, 'warp_flags', [])
            wrp_note = " | ".join(wrp_flags_list)
            censored_note = ""
            if mc.censored_fraction > 0:
                n_total = len(data.y)
                n_censored = int(round(mc.censored_fraction * n_total))
                censored_note = f"({n_censored}/{n_total} censored)"
            mean_r = getattr(mc, 'mean_r', float('nan'))
            # Annotation tags
            tags = []
            if idx == recommendation.recommended_idx:
                tags.append("[rec]")
            if idx == prev_choice_idx and idx != recommendation.recommended_idx:
                tags.append("[prev]")
            elif idx == prev_choice_idx:
                tags.append("[prev]")
            tag_str = " " + ", ".join(tags) if tags else ""
            notes = "; ".join(filter(None, [mc.fit_note, censored_note, ls_note, wrp_note])) + tag_str
            wrp_cell = rag_circle(mc.wrp_rag) if mc.warping else "\u2014"
            # Insert separator above first demoted model
            if not demoted_seen and mc.demoted:
                table_rows.append(['--------'] * n_cols)
                demoted_seen = True
            table_rows.append([
                idx,
                mc.output_transform,
                kernel,
                model_n_params(mc),
                f"{_fmt(mc.loo_mae)} \u00b1 {_fmt(mc.loo_std)}",
                _fmt(mc.loo_lpd, width=10),
                _fmt(mc.calibration_var, width=10),
                _fmt(mean_r, width=10),
                rag_circle(mc.fit_rag),
                rag_circle(mc.ls_rag),
                wrp_cell,
                notes
            ])

        headers = [
            "Idx", "Transform", "Kern", "#Params", "LOO-MAE", "LPD",
            "Var(r)", "Mean(r)", "FIT", "LS", "WRP", "Notes"
        ]
        align = ["left"] * n_cols
        in_jupyter = 'ipykernel' in sys.modules
        if prev_choice is not None:
            n_rounds = len(prev_state.rounds)
            prev_kernel = "WARP" if prev_choice.warping else ("ARD" if prev_choice.ard else "ISO")
            if prev_choice_idx is not None:
                print(f"\nPrevious choice (round {n_rounds}, {prev_choice.n_obs_when_chosen} obs): "
                      f"[{prev_choice_idx}] {prev_choice.output_transform} ({prev_kernel})")
            else:
                print(f"\nPrevious choice (round {n_rounds}, {prev_choice.n_obs_when_chosen} obs): "
                      f"{prev_choice.output_transform} ({prev_kernel}) "
                      f"\u2014 not evaluated this round.")
        print("\nModel Selection Results:")
        if in_jupyter:
            from IPython.display import display, HTML
            import pandas as pd
            # Filter out separator rows
            display_rows = []
            rec_display_row = None
            prev_display_row = None
            for r in table_rows:
                if r[0] == '--------':
                    continue
                if r[0] == recommendation.recommended_idx:
                    rec_display_row = len(display_rows)
                if r[0] == prev_choice_idx and prev_choice_idx != recommendation.recommended_idx:
                    prev_display_row = len(display_rows)
                display_rows.append(r)
            df = pd.DataFrame(display_rows, columns=headers)
            styles = [dict(selector="th", props=[("text-align", "left")]),
                      dict(selector="td", props=[("text-align", "left")]),
                      dict(selector="tbody tr", props=[("background-color", "transparent")])]
            extra_styles = []
            # Blue outline for recommended model
            if rec_display_row is not None:
                row_sel = f"tbody tr:nth-child({rec_display_row + 1})"
                extra_styles.append(
                    {"selector": row_sel,
                     "props": [("outline", "3px solid #4a90d9"),
                               ("outline-offset", "-3px")]})
            # Amber outline for previous choice (if different from recommended)
            if prev_display_row is not None:
                row_sel = f"tbody tr:nth-child({prev_display_row + 1})"
                extra_styles.append(
                    {"selector": row_sel,
                     "props": [("outline", "3px solid #e6a817"),
                               ("outline-offset", "-3px")]})
            styler = df.style.hide(axis='index').set_table_styles([*styles, *extra_styles])
            display(HTML(styler.to_html(escape=False)))
        else:
            print(tabulate(table_rows, headers=headers, tablefmt="pipe", floatfmt=".4f", stralign=align, colalign=align))
        print("\nRanked by recommendation (LOO-LPD within sections). Models below the line are demoted.")
        print("RAG columns:")
        print("  FIT: calibration of LOO residuals (Var(r)).")
        print("       Red = Var(r) > 5.0 or < 0.1 (demoted). Amber = > 2.0 or < 0.5. Green = well-calibrated.")
        print("  LS:  length-scale diagnostics.")
        print("       Red = >= half at bounds (demoted). Amber = advisory. Green = no issues.")
        print("  WRP: warp parameter diagnostics (\u2014 = non-warp model).")
        print("       Red = >= half extreme (demoted). Amber = advisory. Green = no issues.")

        # Recommendation summary
        rec_mc = all_candidates[recommendation.recommended_idx]
        rec_kernel = _kernel_label(rec_mc)
        action_label = recommendation.action.value.upper()
        print(f"\nRecommendation ({action_label}): [{recommendation.recommended_idx}] "
              f"{rec_mc.output_transform} ({rec_kernel}, "
              f"LPD={_fmt(rec_mc.loo_lpd).strip()}, "
              f"LOO-MAE={_fmt(rec_mc.loo_mae).strip()} \u00b1 {_fmt(rec_mc.loo_std).strip()}, "
              f"Var(r)={_fmt(rec_mc.calibration_var).strip()})")
        print(f"  {recommendation.rationale}")

        return ModelResult(candidates=all_candidates, data=data, recommendation=recommendation)

    def report_models(
        self,
        result: ModelResult,
        report: str | None = None,
        choice: int | None = None,
    ) -> list[str]:
        """Show model diagnostic reports (summary or individual)."""
        from bayesopt_human.reporting.base import run_reports
        from bayesopt_human.reporting.models import (
            INDIVIDUAL_REPORTS,
            SUMMARY_REPORTS,
            ModelIndividualContext,
            ModelSummaryContext,
        )

        if choice is None:
            ctx = ModelSummaryContext(result=result)
            return run_reports(
                SUMMARY_REPORTS,
                report,
                context=ctx,
                header="Available summary reports for models:",
                kind="summary",
            )

        if not isinstance(choice, int) or choice < 0 or choice >= len(result.candidates):
            raise ValueError(
                f"Choice index {choice} out of range (0-{len(result.candidates) - 1})."
            )

        selected = result.candidates[choice]
        kernel = "WARP" if selected.warping else ("ARD" if selected.ard else "ISO")
        header = (
            f"Available individual reports for model [{choice}]: "
            f"{selected.output_transform} ({kernel})"
        )
        if report is not None:
            print(f"Diagnostics for [{choice}]: {selected.output_transform} ({kernel})")

        ctx = ModelIndividualContext(
            result=result,
            candidate=selected,
            config=self._locked_config,
            choice=choice,
        )
        return run_reports(
            INDIVIDUAL_REPORTS,
            report,
            context=ctx,
            header=header,
            kind="individual",
        )

    def choose_model(
        self,
        result: ModelResult,
        choice: int = 0,
        show_stability: bool = False,
    ) -> FittedGP:
        """Select a model configuration by index.

        Args:
            result: ModelResult from get_models().
            choice: Index into the ranked candidates list (0 = best).
            show_stability: If True, run expensive LOO parameter stability.

        Returns:
            FittedGP for the chosen model configuration.
        """
        if choice < 0 or choice >= len(result.candidates):
            raise ValueError(
                f"Choice index {choice} out of range "
                f"(0-{len(result.candidates) - 1})."
            )

        # Use the locked config from choose_data
        config = getattr(self, '_locked_config', None)
        if config is None:
            raise RuntimeError("No config locked. Call choose_data first.")

        selected = result.candidates[choice]
        kernel = "WARP" if selected.warping else ("ARD" if selected.ard else "ISO")
        print(f"Selected model [{choice}]: {selected.output_transform} "
              f"({kernel}, LPD={_fmt(selected.loo_lpd).strip()}, "
              f"LOO-MAE={_fmt(selected.loo_mae).strip()} \u00b1 {_fmt(selected.loo_std).strip()}, "
              f"Var(r)={_fmt(selected.calibration_var).strip()})")
        if result.recommendation is not None:
            action_label = result.recommendation.action.value.upper()
            if choice == result.recommendation.recommended_idx:
                print(f"  (Following {action_label} recommendation)")
            else:
                print(f"  (Overriding {action_label} recommendation "
                      f"for [{result.recommendation.recommended_idx}])")
        if selected.calibration_var > 2.0:
            print("  WARNING: Model is overconfident (Var(r) > 2.0). "
                  "Predicted uncertainties may be too narrow.")
        elif selected.calibration_var < 0.5:
            print("  WARNING: Model is underconfident (Var(r) < 0.5). "
                  "Predicted uncertainties may be too wide.")

        # LOO parameter stability (expensive)
        if show_stability:
            from bayesopt_human.gp.loo import loo_parameter_stability

            data = result.data
            print("\nRunning LOO parameter stability analysis...")
            gp_model = GPModel(config)
            stability = loo_parameter_stability(
                lambda X, y: gp_model.fit(
                    X, y,
                    output_transform=selected.output_transform,
                    warping=selected.warping,
                    ard=selected.ard,
                ),
                data.X_norm, data.y,
            )
            n_folds = stability.get("n_successful_folds", 0)
            print(f"  ({n_folds}/{len(data.X_norm)} folds succeeded)")
            for line in stability.get("length_scales_summary", []):
                print(f"  {line}")

        self._chosen_model_record = ModelChoiceRecord(
            output_transform=selected.output_transform,
            ard=selected.ard,
            warping=selected.warping,
            loo_mae=selected.loo_mae,
            loo_lpd=selected.loo_lpd,
            calibration_var=selected.calibration_var,
            n_obs_when_chosen=config.n_obs,
        )

        return selected.fitted_gp

    # -----------------------------------------------------------------------
    # Phase 3: Candidates
    # -----------------------------------------------------------------------

    def get_candidates(
        self,
        data: DataPayload,
        fitted_gp: FittedGP,
    ) -> CandidateResult:
        """Generate candidates, compute diagnostics, make recommendations.

        Prints candidate table and top recommendation only.
        Call report_candidates() for detailed diagnostics or projections.

        Args:
            data: DataPayload from choose_data.
            fitted_gp: Fitted GP from choose_model.

        Returns:
            CandidateResult containing candidates, diagnostics, recommendations.
        """
        # Use the locked config from choose_data
        config = getattr(self, '_locked_config', None)
        if config is None:
            raise RuntimeError("No config locked. Call choose_data first.")
        apply_seed(config.random_seed)
        sur_transform = SurrogateSpaceTransform(fitted_gp)

        # Generate candidates
        print("Generating candidates...")
        gen = CandidateGenerator(config)
        candidates = gen.generate_all(
            fitted_gp, data.X_norm, data.y, data.normalizer
        )
        print(f"  Generated {len(candidates)} candidates")

        # Diagnostics
        print("Computing diagnostics...")

        gp_report = gp_fit_summary(fitted_gp, data.X_norm, data.y)

        cov_report = coverage_metrics(
            data.X_norm, k=config.knn_k,
            surrogate_transform=sur_transform,
        )

        mod_report = detect_local_optima(
            data.X_norm, data.y, config.direction,
            k=config.modality_k,
            surrogate_transform=sur_transform,
        )

        svm_report = svm_sense_check(data.X_norm, data.y, config.direction)

        # Fill in coverage gain and SVM predictions for each candidate
        for cand in candidates:
            cg_flat, cg_sur = coverage_gain(
                data.X_norm, cand.coordinates_normalized,
                k=config.knn_k,
                surrogate_transform=sur_transform,
            )
            cand.coverage_gain_flat = cg_flat
            cand.coverage_gain_surrogate = cg_sur
            cand.svm_prediction = svm_predict_candidate(
                svm_report, cand.coordinates_normalized
            )

        # Recommendations
        print("Generating recommendations...")
        rec_engine = RecommendationEngine(config)
        recommendations = rec_engine.generate(
            candidates, gp_report, cov_report,
            svm_report, data.y,
        )

        # Candidate table
        cand_table = display_candidate_table(candidates, recommendations)
        print("\nCandidate Comparison:")
        print(cand_table.to_string(index=False))
        print("  Sur. Mean, Sur. Std, and EI are in surrogate space (output-transformed + IQR-normalized).")

        # Top recommendation
        if recommendations:
            top = recommendations[0]
            print(f"\nTop recommendation: {top.source}")
            print(f"  Coordinates: {top.candidate}")
            print(f"  Rationale: {top.rationale}")
            print(f"  Confidence: {top.confidence}")

        return CandidateResult(
            candidates=candidates,
            recommendations=recommendations,
            gp_report=gp_report,
            coverage=cov_report,
            modality=mod_report,
            svm_report=svm_report,
            candidate_table=cand_table,
            data=data,
            fitted_gp=fitted_gp,
        )

    def report_candidates(
        self,
        result: CandidateResult,
        report: str | None = None,
        choice: int | None = None,
    ) -> list[str]:
        """Show candidate reports (summary or individual, with 2D projections)."""
        from bayesopt_human.reporting.base import run_reports
        from bayesopt_human.reporting.candidates import (
            CandidateContext,
            INDIVIDUAL_REPORTS,
            SUMMARY_REPORTS,
        )

        if choice is None:
            ctx = CandidateContext(result=result, config=self._locked_config)
            return run_reports(
                SUMMARY_REPORTS,
                report,
                context=ctx,
                header="Available summary reports for candidates:",
                kind="summary",
            )

        # Individual reports: resolve rank to a single candidate
        rank_map: dict[int, Candidate] = {}
        for rec in result.recommendations:
            for cand in result.candidates:
                if cand.source == rec.source:
                    rank_map[rec.priority] = cand
                    break
        if choice not in rank_map:
            valid_ranks = sorted(rank_map.keys())
            raise ValueError(
                f"Rank {choice} not found. Valid ranks: {valid_ranks}"
            )
        single_cand = rank_map[choice]
        ctx = CandidateContext(
            result=result,
            config=self._locked_config,
            candidate=single_cand,
            ranks={single_cand.source: choice},
        )
        return run_reports(
            INDIVIDUAL_REPORTS,
            report,
            context=ctx,
            header="Available individual reports for candidates:",
            kind="individual",
        )

    def choose_candidate(
        self,
        result: CandidateResult,
        choice: int | np.ndarray = 1,
        save_to: str | None = None,
    ) -> np.ndarray:
        """Finalize the user's choice by recommendation rank.

        Args:
            result: CandidateResult from get_candidates().
            choice: Recommendation rank (1 = top recommendation) or custom
                coordinates in raw space as an array.
            save_to: Optional filepath to save the selected point as CSV.

        Returns:
            Selected point coordinates in raw space.
        """
        if isinstance(choice, (int, np.integer)):
            # Build rank -> candidate mapping
            rank_map: dict[int, Candidate] = {}
            for rec in result.recommendations:
                for cand in result.candidates:
                    if cand.source == rec.source:
                        rank_map[rec.priority] = cand
                        break

            if choice not in rank_map:
                valid_ranks = sorted(rank_map.keys())
                raise ValueError(
                    f"Rank {choice} not found. Valid ranks: {valid_ranks}"
                )
            cand = rank_map[choice]
            coords = cand.coordinates_raw
            print(f"Selected rank #{choice}: {cand.source}")
            print(f"  Raw coordinates: {coords}")
        else:
            coords = np.asarray(choice, dtype=np.float64)
            print(f"Selected custom coordinates: {coords}")

        if save_to is not None:
            row = pd.DataFrame([coords], columns=result.data.param_names)
            row.to_csv(save_to, index=False)
            print(f"  Saved to {save_to}")

        # Persist optimization state
        config = getattr(self, '_locked_config', None)
        data_dir = getattr(self, '_data_dir', None)
        model_record = getattr(self, '_chosen_model_record', None)

        if config is not None and data_dir is not None and model_record is not None:
            state = getattr(self, '_loaded_state', None) or OptimizationState(
                schema_version=1,
                config_name=config.name,
                last_updated="",
                model_choice=None,
                rounds=[],
            )

            if isinstance(choice, (int, np.integer)):
                chosen_cand = rank_map[choice]
                rec_for_cand = next(
                    (r for r in result.recommendations if r.source == chosen_cand.source),
                    None,
                )
                record_round(
                    state=state,
                    n_obs=config.n_obs,
                    model_choice=model_record,
                    candidate_source=chosen_cand.source,
                    candidate_category=chosen_cand.category,
                    candidate_confidence=rec_for_cand.confidence if rec_for_cand else "unknown",
                    candidate_posterior_mean=chosen_cand.posterior_mean,
                    candidate_posterior_std=chosen_cand.posterior_std,
                    candidate_expected_improvement=chosen_cand.expected_improvement,
                    was_custom=False,
                )
            else:
                record_round(
                    state=state,
                    n_obs=config.n_obs,
                    model_choice=model_record,
                    candidate_source="custom",
                    candidate_category="custom",
                    candidate_confidence="unknown",
                    candidate_posterior_mean=float('nan'),
                    candidate_posterior_std=float('nan'),
                    candidate_expected_improvement=float('nan'),
                    was_custom=True,
                )

            save_state(state, data_dir)

        return coords

