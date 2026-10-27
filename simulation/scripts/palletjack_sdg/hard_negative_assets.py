"""Asset selection helpers for manual-pallet-jack hard-negative generation."""

from pathlib import Path
from typing import Sequence, TypeVar


HARD_NEGATIVE_LOCAL_ASSET_FILENAMES = {
    "pallet_stacker": "pallet_jack_advanced.usdz",
}
LOCAL_SCENE_DISTRACTOR_USDZ_FILENAMES = {
    "forklift": "forklift.usdz",
    "ladder": "ladder.usdz",
    "shopping_cart": "shopping_cart.usdz",
    "warehouse_rack": "warehouse_rack.usdz",
}

T = TypeVar("T")


def resolve_local_hard_negative_usdz_paths(
    script_path: str | Path,
) -> dict[str, Path]:
    """Resolve the local stacker asset used as a target-free hard negative."""
    simulation_directory = Path(script_path).resolve().parents[2]
    asset_directory = simulation_directory / "assets"
    paths = {
        equipment_type: asset_directory / filename
        for equipment_type, filename in HARD_NEGATIVE_LOCAL_ASSET_FILENAMES.items()
    }
    missing_paths = [path for path in paths.values() if not path.is_file()]
    if missing_paths:
        missing_names = ", ".join(path.name for path in missing_paths)
        raise FileNotFoundError(
            "Missing local hard-negative asset(s): "
            f"{missing_names}. Place them under {asset_directory}."
        )
    return paths


def resolve_local_scene_distractor_usdz_paths(
    script_path: str | Path,
) -> dict[str, Path]:
    """Resolve all local USDZ props shared by normal and negative scene pools."""
    simulation_directory = Path(script_path).resolve().parents[2]
    asset_directory = simulation_directory / "assets"
    paths = {
        asset_name: asset_directory / filename
        for asset_name, filename in LOCAL_SCENE_DISTRACTOR_USDZ_FILENAMES.items()
    }
    missing_paths = [path for path in paths.values() if not path.is_file()]
    if missing_paths:
        missing_names = ", ".join(path.name for path in missing_paths)
        raise FileNotFoundError(
            "Missing local scene distractor asset(s): "
            f"{missing_names}. Place them under {asset_directory}."
        )
    return paths


def combine_regular_distractor_asset_pool(
    existing_paths: Sequence[T],
    local_distractor_paths: Sequence[T],
    local_stacker_paths: Sequence[T],
) -> list[T]:
    """Add local distractor props and stackers to a regular scene category."""
    return [*existing_paths, *local_distractor_paths, *local_stacker_paths]


def combine_hard_negative_asset_pool(
    warehouse_paths: Sequence[T],
    additional_paths: Sequence[T],
    hard_negative_paths: Sequence[T],
) -> list[T]:
    """Combine all existing distractor groups and target-confusable equipment."""
    return [*warehouse_paths, *additional_paths, *hard_negative_paths]


def should_add_palletjack_targets(distractors_only: bool) -> bool:
    """Return whether a generation run should include labeled pallet-truck targets."""
    return not distractors_only
