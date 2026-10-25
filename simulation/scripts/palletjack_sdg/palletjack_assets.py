"""Resolve local pallet-jack USDZ assets used by the SDG script."""

from pathlib import Path


def resolve_local_palletjack_usdz_paths(script_path: str | Path) -> list[Path]:
    """Return the required Experiment 2 USDZ files relative to the simulation folder."""
    simulation_directory = Path(script_path).resolve().parents[2]
    asset_directory = simulation_directory / "assets" / "pallet_jacks" / "experiment-02"
    asset_paths = [
        asset_directory / "electric_pallet_jack.usdz",
        asset_directory / "pallet_jack_advanced.usdz",
    ]
    missing_asset_paths = [path for path in asset_paths if not path.is_file()]
    if missing_asset_paths:
        missing_names = ", ".join(path.name for path in missing_asset_paths)
        raise FileNotFoundError(
            "Missing local pallet-jack asset(s): "
            f"{missing_names}. Place them under {asset_directory}."
        )
    return asset_paths
