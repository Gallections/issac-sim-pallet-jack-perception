"""Canonical repository paths for experiment-scoped datasets."""

from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_ID = "experiment-01"
_EXPERIMENT_ID_PATTERN = re.compile(r"experiment-\d{2,}\Z")


def _experiment_root(data_kind: str, experiment_id: str) -> Path:
    if not isinstance(experiment_id, str) or not _EXPERIMENT_ID_PATTERN.fullmatch(experiment_id):
        raise ValueError("experiment_id must look like 'experiment-01'")
    return PROJECT_ROOT / "data" / data_kind / experiment_id


def raw_experiment_root(experiment_id: str = DEFAULT_EXPERIMENT_ID) -> Path:
    """Return the raw-data directory for one experiment."""
    return _experiment_root("raw", experiment_id)


def processed_experiment_root(experiment_id: str = DEFAULT_EXPERIMENT_ID) -> Path:
    """Return the processed-data directory for one experiment."""
    return _experiment_root("processed", experiment_id)


DEFAULT_RAW_ROOT = raw_experiment_root()
DEFAULT_PROCESSED_ROOT = processed_experiment_root()
DEFAULT_SPLITS_ROOT = DEFAULT_PROCESSED_ROOT / "splits"
DEFAULT_REAL_RAW_ROOT = DEFAULT_RAW_ROOT / "real-pallet-jack-data"
DEFAULT_REAL_PROCESSED_ROOT = DEFAULT_PROCESSED_ROOT / "real-pallet-jack-data"
