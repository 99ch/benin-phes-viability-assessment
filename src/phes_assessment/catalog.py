"""Utilitaires pour résumer les jeux de données disponibles."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

from .io import list_rasters, read_raster_metadata

DATE_PATTERNS = [
    re.compile(r".*?(\d{4})\.(\d{2})\.tif$"),  # CHIRPS mensuel
    re.compile(r".*?(\d{4})\.tif$"),  # ERA5 annuel
]


@dataclass
class DatasetSummary:
    name: str
    file_count: int
    start_date: Optional[date]
    end_date: Optional[date]
    sample_path: Optional[Path]
    crs: Optional[str]
    resolution: Optional[tuple[float, float]]


def _parse_date(filename: str) -> Optional[date]:
    for pattern in DATE_PATTERNS:
        match = pattern.match(filename)
        if match:
            year = int(match.group(1))
            month = int(match.group(2)) if len(match.groups()) > 1 else 1
            return date(year, month, 1)
    return None


def summarize_directory(name: str, directory: Path, files: Iterable[Path] | None = None) -> DatasetSummary:
    """Construit un résumé rapide (temporalité + projection) pour un dossier donné."""

    rasters = list(files) if files is not None else list_rasters(directory)
    rasters.sort()
    if not rasters:
        return DatasetSummary(name, 0, None, None, None, None, None)

    dates = [_parse_date(path.name) for path in rasters]
    dates = [d for d in dates if d is not None]
    metadata = read_raster_metadata(rasters[0])
    res_x = abs(metadata["transform"][0])
    res_y = abs(metadata["transform"][4])

    return DatasetSummary(
        name=name,
        file_count=len(rasters),
        start_date=min(dates) if dates else None,
        end_date=max(dates) if dates else None,
        sample_path=rasters[0],
        crs=metadata["crs"],
        resolution=(res_x, res_y),
    )
