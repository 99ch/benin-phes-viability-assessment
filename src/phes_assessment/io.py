"""Fonctions d'entrée/sortie pour les différents ensembles de données."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import rasterio


def list_rasters(directory: Path, suffix: str = ".tif") -> list[Path]:
    """Retourne la liste triée des rasters disponibles dans un dossier."""

    if not directory.exists():
        return []
    return sorted(p for p in directory.glob(f"*{suffix}") if p.is_file())


def read_raster_metadata(raster_path: Path) -> dict:
    """Extrait les métadonnées principales d'un raster."""

    with rasterio.open(raster_path) as dataset:
        return {
            "path": str(raster_path),
            "bounds": dataset.bounds,
            "crs": dataset.crs.to_string() if dataset.crs else None,
            "width": dataset.width,
            "height": dataset.height,
            "transform": tuple(dataset.transform),
            "count": dataset.count,
            "dtype": dataset.dtypes[0],
        }


def batch_metadata(paths: Iterable[Path]) -> list[dict]:
    """Collecte les métadonnées d'une série de rasters."""

    return [read_raster_metadata(path) for path in paths]
