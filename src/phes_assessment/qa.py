"""Contrôles qualité basiques sur les jeux de données climatiques et topographiques."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import rasterio

from .config import DataPaths, get_paths
from .fabdem import DEFAULT_SITES_FILE
from .io import list_rasters, read_raster_metadata
from .sites import load_sites

MONTHLY_PATTERN = re.compile(r".*?(\d{4})[._-](\d{2})")
YEAR_PATTERN = re.compile(r".*?(\d{4})")


@dataclass
class DatasetQAReport:
    dataset: str
    file_count: int
    coverage_start: date | None
    coverage_end: date | None
    expected_start: date | None = None
    expected_end: date | None = None
    missing_periods: list[str] = field(default_factory=list)
    duplicate_periods: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def missing_label(self) -> str:
        if not self.missing_periods:
            return "-"
        if len(self.missing_periods) > 6:
            head = ", ".join(self.missing_periods[:5])
            return f"{head} … ({len(self.missing_periods)} manquants)"
        return ", ".join(self.missing_periods)

    def duplicate_label(self) -> str:
        if not self.duplicate_periods:
            return "-"
        return ", ".join(self.duplicate_periods)

    def notes_label(self) -> str:
        if not self.notes:
            return "-"
        return " | ".join(self.notes)


def _generate_months(start: date, end: date) -> list[date]:
    months: list[date] = []
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        months.append(cursor)
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return months


def _month_label(dt: date) -> str:
    return dt.strftime("%Y-%m")


def _find_duplicates(dates: Sequence[date]) -> list[str]:
    seen: dict[date, int] = {}
    duplicates: list[str] = []
    for dt in dates:
        seen[dt] = seen.get(dt, 0) + 1
    for dt, count in seen.items():
        if count > 1:
            duplicates.append(f"{_month_label(dt)} (x{count})")
    return sorted(duplicates)


def _parse_month_from_name(name: str) -> date | None:
    match = MONTHLY_PATTERN.match(name)
    if not match:
        return None
    year = int(match.group(1))
    month = int(match.group(2))
    if 1 <= month <= 12:
        return date(year, month, 1)
    return None


def _parse_year_from_name(name: str) -> int | None:
    match = YEAR_PATTERN.match(name)
    if not match:
        return None
    return int(match.group(1))


def _find_compressed_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.glob("*.tif.gz*") if path.is_file())


def check_chirps(
    paths: DataPaths,
    expected_start_year: int,
    expected_end_year: int,
) -> DatasetQAReport:
    rasters = list_rasters(paths.chirps_dir)
    compressed = _find_compressed_files(paths.chirps_dir)
    if not rasters:
        return DatasetQAReport(
            dataset="CHIRPS",
            file_count=0,
            coverage_start=None,
            coverage_end=None,
            expected_start=date(expected_start_year, 1, 1),
            expected_end=date(expected_end_year, 12, 1),
            notes=["Aucun fichier .tif détecté"]
            + ([f"{len(compressed)} fichiers compressés (.tif.gz) présents"] if compressed else []),
        )

    parsed_dates: list[date] = []
    unparsed: list[str] = []
    for raster in rasters:
        dt = _parse_month_from_name(raster.name)
        if dt is None:
            unparsed.append(raster.name)
            continue
        parsed_dates.append(dt)

    expected_months = _generate_months(date(expected_start_year, 1, 1), date(expected_end_year, 12, 1))
    parsed_set = set(parsed_dates)
    missing = [_month_label(dt) for dt in expected_months if dt not in parsed_set]
    duplicates = _find_duplicates(parsed_dates)

    notes: list[str] = []
    if unparsed:
        notes.append(f"{len(unparsed)} fichiers ignorés (pattern inconnu)")
    if compressed:
        rel_dir = paths.chirps_dir.relative_to(paths.root) if paths.chirps_dir.is_relative_to(paths.root) else paths.chirps_dir
        cmd = f"for f in {rel_dir}/*.tif.gz*; do gunzip -f \"$f\"; done"
        notes.append(f"{len(compressed)} fichiers compressés détectés → décompresser via `{cmd}`")

    return DatasetQAReport(
        dataset="CHIRPS",
        file_count=len(rasters),
        coverage_start=min(parsed_dates) if parsed_dates else None,
        coverage_end=max(parsed_dates) if parsed_dates else None,
        expected_start=expected_months[0],
        expected_end=expected_months[-1],
        missing_periods=missing,
        duplicate_periods=duplicates,
        notes=notes,
    )


def check_era5(
    paths: DataPaths,
    expected_start_year: int,
    expected_end_year: int,
) -> DatasetQAReport:
    rasters = list_rasters(paths.era5_dir)
    compressed = _find_compressed_files(paths.era5_dir)
    if not rasters:
        return DatasetQAReport(
            dataset="ERA5",
            file_count=0,
            coverage_start=None,
            coverage_end=None,
            expected_start=date(expected_start_year, 1, 1),
            expected_end=date(expected_end_year, 12, 1),
            notes=["Aucun fichier .tif détecté"]
            + ([f"{len(compressed)} fichiers compressés (.tif.gz) présents"] if compressed else []),
        )

    years_present: list[int] = []
    band_notes: list[str] = []
    for raster in rasters:
        year = _parse_year_from_name(raster.name)
        if year is None:
            band_notes.append(f"Nom ignoré: {raster.name}")
            continue
        years_present.append(year)
        try:
            with rasterio.open(raster) as dataset:
                if dataset.count < 12:
                    band_notes.append(f"{raster.name}: {dataset.count} bandes (<12)")
        except rasterio.errors.RasterioIOError as exc:
            band_notes.append(f"Impossible d'ouvrir {raster.name}: {exc}")

    if years_present:
        coverage_start = date(min(years_present), 1, 1)
        coverage_end = date(max(years_present), 12, 1)
    else:
        coverage_start = coverage_end = None

    expected_years = list(range(expected_start_year, expected_end_year + 1))
    missing_years = [str(year) for year in expected_years if year not in years_present]

    notes = band_notes
    if compressed:
        rel_dir = paths.era5_dir.relative_to(paths.root) if paths.era5_dir.is_relative_to(paths.root) else paths.era5_dir
        cmd = f"for f in {rel_dir}/*.tif.gz*; do gunzip -f \"$f\"; done"
        notes.append(f"{len(compressed)} fichiers compressés détectés → décompresser via `{cmd}`")

    return DatasetQAReport(
        dataset="ERA5",
        file_count=len(rasters),
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        expected_start=date(expected_start_year, 1, 1),
        expected_end=date(expected_end_year, 12, 1),
        missing_periods=missing_years,
        duplicate_periods=[],
        notes=notes,
    )


def check_fabdem(
    paths: DataPaths,
    sites_csv: Path | None = None,
) -> DatasetQAReport:
    rasters = list_rasters(paths.fabdem_dir)
    if not rasters:
        return DatasetQAReport(
            dataset="FABDEM",
            file_count=0,
            coverage_start=None,
            coverage_end=None,
            notes=["Aucune tuile FABDEM détectée"],
        )

    metadata = [read_raster_metadata(path) for path in rasters]
    bounds_array = np.array([[meta["bounds"].left, meta["bounds"].bottom, meta["bounds"].right, meta["bounds"].top] for meta in metadata])
    global_bounds = (
        float(bounds_array[:, 0].min()),
        float(bounds_array[:, 1].min()),
        float(bounds_array[:, 2].max()),
        float(bounds_array[:, 3].max()),
    )

    res_x = float(metadata[0]["transform"][0])
    res_y = float(metadata[0]["transform"][4])
    notes = [
        f"Résolution approx. {abs(res_x):.1f} m",
        f"Emprise globale lon[{global_bounds[0]:.2f},{global_bounds[2]:.2f}] / lat[{global_bounds[1]:.2f},{global_bounds[3]:.2f}]",
    ]

    csv_path = sites_csv or (paths.data_dir / DEFAULT_SITES_FILE)
    if csv_path.exists():
        df_sites = load_sites(csv_path)
        outside: list[str] = []
        upper_lat = pd.to_numeric(df_sites["Upper latitude"], errors="coerce")
        upper_lon = pd.to_numeric(df_sites["Upper longitude"], errors="coerce")
        lower_lat = pd.to_numeric(df_sites["Lower latitude"], errors="coerce")
        lower_lon = pd.to_numeric(df_sites["Lower longitude"], errors="coerce")

        for idx, row in df_sites.iterrows():
            pid = str(row["Pair Identifier"])
            ulat = upper_lat.iloc[idx]
            ulon = upper_lon.iloc[idx]
            llat = lower_lat.iloc[idx]
            llon = lower_lon.iloc[idx]
            if not np.isnan(ulat) and not np.isnan(ulon):
                if not (global_bounds[0] <= ulon <= global_bounds[2] and global_bounds[1] <= ulat <= global_bounds[3]):
                    outside.append(f"{pid} (upper)")
            if not np.isnan(llat) and not np.isnan(llon):
                if not (global_bounds[0] <= llon <= global_bounds[2] and global_bounds[1] <= llat <= global_bounds[3]):
                    outside.append(f"{pid} (lower)")
        if outside:
            notes.append(f"Sites hors emprise : {', '.join(outside)}")
    else:
        notes.append(f"CSV sites introuvable: {csv_path}")

    return DatasetQAReport(
        dataset="FABDEM",
        file_count=len(rasters),
        coverage_start=None,
        coverage_end=None,
        notes=notes,
    )


def run_quality_checks(
    paths: DataPaths | None = None,
    start_year: int = 2002,
    end_year: int = 2023,
    sites_csv: Path | None = None,
) -> list[DatasetQAReport]:
    paths = paths or get_paths()
    reports = [
        check_chirps(paths, start_year, end_year),
        check_era5(paths, start_year, end_year),
        check_fabdem(paths, sites_csv),
    ]
    return reports