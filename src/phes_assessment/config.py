"""Configuration centralisée des chemins de données."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    root: Path
    data_dir: Path
    chirps_dir: Path
    era5_dir: Path
    fabdem_dir: Path
    n10_dir: Path

    @classmethod
    def from_repo_root(cls, root: Path | None = None) -> "DataPaths":
        base = root or Path(__file__).resolve().parents[2]
        data_dir = base / "data"
        return cls(
            root=base,
            data_dir=data_dir,
            chirps_dir=data_dir / "chirps",
            era5_dir=data_dir / "era5",
            fabdem_dir=data_dir / "N10E000-N20E010_FABDEM_V1-2",
            n10_dir=base / "n10_e001",
        )


def get_paths(root: Path | None = None) -> DataPaths:
    """Helper for callers to fetch the repository data paths."""

    return DataPaths.from_repo_root(root)
