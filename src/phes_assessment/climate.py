"""Agrégation des séries climatiques CHIRPS et ERA5 par site."""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rasterio_mask

from .basins import load_site_basins
from .config import DataPaths, get_paths
from .sites import build_site_masks, load_sites

MONTHLY_PATTERNS = [
    re.compile(r".*?(\d{4})\.(\d{2})\.tif$"),
    re.compile(r".*?(\d{4})_(\d{2})\.tif$"),
]
ANNUAL_PATTERN = re.compile(r".*?(\d{4})\.tif$")


def _within_range(dt: date, start: date | None, end: date | None) -> bool:
    if start and dt < start:
        return False
    if end and dt > end:
        return False
    return True

def _parse_raster_date(path: Path) -> date:
    name = path.name
    for pattern in MONTHLY_PATTERNS:
        match = pattern.match(name)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            return date(year, month, 1)
    match = ANNUAL_PATTERN.match(name)
    if match:
        year = int(match.group(1))
        return date(year, 1, 1)
    raise ValueError(f"Impossible d'extraire une date depuis {name}")


def _mean_valid(values: Iterable[float]) -> float:
    arr = [val for val in values if np.isfinite(val)]
    if not arr:
        return float("nan")
    return float(np.mean(arr))


def _infer_month(description: str | None, band_index: int) -> int:
    if description:
        match = re.search(r"(\d{1,2})", description)
        if match:
            month = int(match.group(1))
            if 1 <= month <= 12:
                return month
    return band_index


def _sample_dataset(
    dataset: rasterio.io.DatasetReader,
    geometries: Dict[str, Dict],
    band_index: int,
) -> Dict[str, float]:
    results: Dict[str, float] = {}
    for pair, geometry in geometries.items():
        try:
            data, _ = rasterio_mask(
                dataset,
                [geometry],
                crop=False,
                filled=False,
                indexes=band_index,
                all_touched=True,
            )
        except ValueError:
            results[pair] = float("nan")
            continue
        arr = data[0] if data.ndim == 3 else data
        arr = np.ma.array(arr)
        results[pair] = _mean_valid(arr.compressed())
    return results


def aggregate_series(
    paths: DataPaths | None = None,
    sites_csv: Path | None = None,
    dataset: str = "both",
    buffer_meters: float = 500.0,
    basins_geojson: Path | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    on_raster_processed: Callable[[str, Path], None] | None = None,
) -> pd.DataFrame:
    """Construit un tableau (site, date) avec les valeurs CHIRPS/ERA5 moyennées."""

    paths = paths or get_paths()
    csv_path = sites_csv or (paths.data_dir / "n10_e001_12_sites_complete.csv")
    df_sites = load_sites(csv_path)
    if basins_geojson:
        basin_map = load_site_basins(basins_geojson)
        site_pairs = [str(pair) for pair in df_sites["Pair Identifier"].tolist()]
        missing = [pair for pair in site_pairs if pair not in basin_map]
        if missing:
            raise ValueError(
                "Basins manquants pour les sites suivants : " + ", ".join(missing)
            )
        geometries = {pair: basin_map[pair].geometry.__geo_interface__ for pair in site_pairs}
    else:
        site_masks = build_site_masks(df_sites, buffer_meters=buffer_meters)
        geometries = {mask.pair_identifier: mask.geometry.__geo_interface__ for mask in site_masks}

    records: Dict[Tuple[str, date], Dict[str, float]] = defaultdict(dict)

    def _process_directory(
        directory: Path,
        value_key: str,
        multiband: bool = False,
    ) -> None:
        for raster_path in sorted(directory.glob("*.tif")):
            if multiband:
                dt = _parse_raster_date(raster_path)
                year = dt.year
                year_start = date(year, 1, 1)
                year_end = date(year, 12, 1)
                if start_date and year_end < start_date:
                    continue
                if end_date and year_start > end_date:
                    continue
                processed_file = False
                with rasterio.open(raster_path) as dataset:
                    descriptions = dataset.descriptions or [None] * dataset.count
                    for band_index in range(1, dataset.count + 1):
                        month = _infer_month(descriptions[band_index - 1], band_index)
                        try:
                            dt_band = date(year, month, 1)
                        except ValueError:
                            continue
                        if not _within_range(dt_band, start_date, end_date):
                            continue
                        processed_file = True
                        samples = _sample_dataset(dataset, geometries, band_index=band_index)
                        for pair, value in samples.items():
                            records[(pair, dt_band)][value_key] = value
                if processed_file and on_raster_processed:
                    on_raster_processed(value_key, raster_path)
            else:
                dt = _parse_raster_date(raster_path)
                if not _within_range(dt, start_date, end_date):
                    continue
                with rasterio.open(raster_path) as dataset:
                    samples = _sample_dataset(dataset, geometries, band_index=1)
                for pair, value in samples.items():
                    records[(pair, dt)][value_key] = value
                if on_raster_processed:
                    on_raster_processed(value_key, raster_path)

    if dataset in ("both", "chirps"):
        _process_directory(paths.chirps_dir, "precip_mm")
    if dataset in ("both", "era5"):
        _process_directory(paths.era5_dir, "etp_mm", multiband=True)

    flat_rows = [
        {
            "pair_identifier": pair,
            "date": dt,
            **values,
        }
        for (pair, dt), values in sorted(records.items(), key=lambda item: (item[0][0], item[0][1]))
    ]

    df = pd.DataFrame(flat_rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=False)
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]
    return df


def export_series(
    output: Path,
    paths: DataPaths | None = None,
    sites_csv: Path | None = None,
    dataset: str = "both",
    buffer_meters: float = 500.0,
    basins_geojson: Path | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    dataframe: pd.DataFrame | None = None,
) -> Path:
    df = dataframe
    if df is None:
        df = aggregate_series(
            paths,
            sites_csv,
            dataset,
            buffer_meters=buffer_meters,
            basins_geojson=basins_geojson,
            start_date=start_date,
            end_date=end_date,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if output.suffix == ".parquet":
            df.to_parquet(output)
        else:
            df.to_csv(output, index=False)
        return output
    except ImportError:
        fallback = output.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        return fallback
