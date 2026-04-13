"""Optimization state persistence across rounds."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelChoiceRecord:
    """Persisted identity and metrics of a chosen model."""

    output_transform: str
    ard: bool
    warping: bool
    loo_mae: float
    loo_lpd: float
    calibration_var: float
    n_obs_when_chosen: int


@dataclass
class RoundRecord:
    """One completed optimization round."""

    round_number: int
    n_obs: int
    timestamp: str
    # Model fields
    model_output_transform: str
    model_ard: bool
    model_warping: bool
    model_loo_mae: float
    model_loo_lpd: float
    model_calibration_var: float
    # Candidate fields
    candidate_source: str
    candidate_category: str
    candidate_confidence: str
    candidate_posterior_mean: float
    candidate_posterior_std: float
    candidate_expected_improvement: float
    candidate_was_custom: bool


@dataclass
class OptimizationState:
    """Full persisted state for one optimization problem."""

    schema_version: int
    config_name: str
    last_updated: str
    model_choice: ModelChoiceRecord | None
    rounds: list[RoundRecord]


def state_filepath(data_dir: str, config_name: str) -> Path:
    """Return path: data_dir / {config_name}_state.json."""
    return Path(data_dir) / f"{config_name}_state.json"


def load_state(data_dir: str, config_name: str) -> OptimizationState | None:
    """Load state from JSON.

    Returns None if file is missing, corrupt, or has wrong schema version.
    """
    fp = state_filepath(data_dir, config_name)
    if not fp.exists():
        return None
    try:
        raw = json.loads(fp.read_text(encoding="utf-8"))
        if raw.get("schema_version") != 1:
            logger.warning("State file %s has unsupported schema version, ignoring.", fp)
            return None

        mc_raw = raw.get("model_choice")
        model_choice = ModelChoiceRecord(**mc_raw) if mc_raw else None

        rounds = []
        for r in raw.get("rounds", []):
            rounds.append(RoundRecord(**r))

        return OptimizationState(
            schema_version=raw["schema_version"],
            config_name=raw["config_name"],
            last_updated=raw.get("last_updated", ""),
            model_choice=model_choice,
            rounds=rounds,
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        # OSError: read failures. JSONDecodeError: malformed file.
        # KeyError / TypeError: raw dict shape doesn't match the current
        # RoundRecord / ModelChoiceRecord fields (e.g. older schema variant).
        logger.warning("Failed to parse state file %s, ignoring.", fp, exc_info=True)
        return None


def save_state(state: OptimizationState, data_dir: str) -> None:
    """Serialize state to JSON."""
    fp = state_filepath(data_dir, state.config_name)
    fp.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "schema_version": state.schema_version,
        "config_name": state.config_name,
        "last_updated": state.last_updated,
        "model_choice": asdict(state.model_choice) if state.model_choice else None,
        "rounds": [asdict(r) for r in state.rounds],
    }
    fp.write_text(json.dumps(data, indent=2), encoding="utf-8")


def record_round(
    state: OptimizationState,
    n_obs: int,
    model_choice: ModelChoiceRecord,
    candidate_source: str,
    candidate_category: str,
    candidate_confidence: str,
    candidate_posterior_mean: float,
    candidate_posterior_std: float,
    candidate_expected_improvement: float,
    was_custom: bool,
) -> None:
    """Append a new round and update model_choice + last_updated."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    round_number = len(state.rounds) + 1

    state.rounds.append(RoundRecord(
        round_number=round_number,
        n_obs=n_obs,
        timestamp=now,
        model_output_transform=model_choice.output_transform,
        model_ard=model_choice.ard,
        model_warping=model_choice.warping,
        model_loo_mae=model_choice.loo_mae,
        model_loo_lpd=model_choice.loo_lpd,
        model_calibration_var=model_choice.calibration_var,
        candidate_source=candidate_source,
        candidate_category=candidate_category,
        candidate_confidence=candidate_confidence,
        candidate_posterior_mean=candidate_posterior_mean,
        candidate_posterior_std=candidate_posterior_std,
        candidate_expected_improvement=candidate_expected_improvement,
        candidate_was_custom=was_custom,
    ))

    state.model_choice = model_choice
    state.last_updated = now
