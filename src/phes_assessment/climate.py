"""Agrégation des séries climatiques CHIRPS et ERA5 par site."""
from __future__ import annotations

import gzip
import re
import shutil
import tempfile
import warnings
from collections import defaultdict
from contextlib import contextmanager
import calendar
from datetime import date
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.mask import mask as rasterio_mask

from .basins import load_site_basins
from .config import DataPaths, get_paths
from .sites import build_site_masks, load_sites

MONTHLY_PATTERNS = [
    re.compile(r".*?(\d{4})\.(\d{2})\.tif$"),
    re.compile(r".*?(\d{4})_(\d{2})\.tif$"),
]
ANNUAL_PATTERN = re.compile(r".*?(\d{4})\.tif$")
VALID_GEOMETRY_MODES = {"auto", "buffers", "basins"}


def _list_raster_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    candidates = [path for path in directory.iterdir() if path.is_file() and ".tif" in path.name.lower()]
    return sorted(candidates)


def _is_gzipped(path: Path) -> bool:
    return ".tif.gz" in path.name.lower()


@contextmanager
def _prepared_raster(path: Path) -> Iterator[Path]:
    if not _is_gzipped(path):
        yield path
        return

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        with gzip.open(path, "rb") as src:
            shutil.copyfileobj(src, tmp)
    try:
        yield tmp_path
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def _validate_dataset(dataset: rasterio.io.DatasetReader, source: Path) -> None:
    crs = dataset.crs
    if crs is None or not crs.is_geographic:
        raise ValueError(
            f"Raster {source} doit être en coordonnées géographiques (EPSG:4326)."
        )
    if dataset.res[0] == 0 or dataset.res[1] == 0:
        raise ValueError(f"Raster {source} possède une résolution invalide ({dataset.res}).")


def _resolve_geometries(
    df_sites: pd.DataFrame,
    *,
    buffer_meters: float,
    basins_geojson: Path | None,
    geometry_mode: str,
) -> Dict[str, Dict]:
    mode = geometry_mode.lower()
    if mode not in VALID_GEOMETRY_MODES:
        raise ValueError(f"geometry_mode doit être l'un de: {', '.join(sorted(VALID_GEOMETRY_MODES))}")

    use_basins = False
    if mode == "basins":
        if basins_geojson is None:
            raise ValueError("geometry_mode='basins' nécessite --basins")
        use_basins = True
    elif mode == "auto":
        use_basins = basins_geojson is not None

    if use_basins:
        basin_map = load_site_basins(basins_geojson)  # type: ignore[arg-type]
        site_pairs = [str(pair) for pair in df_sites["Pair Identifier"].tolist()]
        missing = [pair for pair in site_pairs if pair not in basin_map]
        if missing:
            raise ValueError("Basins manquants pour les sites suivants : " + ", ".join(missing))
        return {pair: basin_map[pair].geometry.__geo_interface__ for pair in site_pairs}

    site_masks = build_site_masks(df_sites, buffer_meters=buffer_meters)
    return {mask.pair_identifier: mask.geometry.__geo_interface__ for mask in site_masks}


def _within_range(dt: date, start: date | None, end: date | None) -> bool:
    if start and dt < start:
        return False
    if end and dt > end:
        return False
    return True

def _parse_raster_date(path: Path) -> date:
    name = path.name
    name = re.sub(r"\.gz(\.\d+)?$", "", name, flags=re.IGNORECASE)
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


YEAR_MONTH_PATTERN = re.compile(r"(?:19|20)\d{2}[^0-9]?([01]\d)")
SINGLE_MONTH_PATTERN = re.compile(r"\b(0?[1-9]|1[0-2])\b")


def _infer_month(description: str | None, band_index: int) -> int:
    if description:
        match = YEAR_MONTH_PATTERN.search(description)
        if match:
            month = int(match.group(1))
            if 1 <= month <= 12:
                return month
        match = SINGLE_MONTH_PATTERN.search(description)
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
    geometry_mode: str = "auto",
) -> pd.DataFrame:
    """Construit un tableau (site, date) avec les valeurs CHIRPS/ERA5 moyennées."""

    paths = paths or get_paths()
    csv_path = sites_csv or (paths.data_dir / "n10_e001_12_sites_complete.csv")
    df_sites = load_sites(csv_path)
    basins_resolved = None
    if basins_geojson:
        if not basins_geojson.exists():
            raise FileNotFoundError(f"GeoJSON des bassins introuvable: {basins_geojson}")
        basins_resolved = basins_geojson
    geometries = _resolve_geometries(
        df_sites,
        buffer_meters=buffer_meters,
        basins_geojson=basins_resolved,
        geometry_mode=geometry_mode,
    )

    records: Dict[Tuple[str, date], Dict[str, float]] = defaultdict(dict)

    def _convert_etp_value(raw_value: float, dt_band: date) -> float:
        if not np.isfinite(raw_value):
            return float("nan")
        value = abs(float(raw_value))
        days = calendar.monthrange(dt_band.year, dt_band.month)[1]
        if value > 50:
            return value  # déjà en mm/mois
        if value > 1:
            return value * days  # mm/jour → mm/mois
        return value * 1000.0 * days  # m/jour → mm/mois

    def _process_directory(
        directory: Path,
        value_key: str,
        multiband: bool = False,
    ) -> None:
        raster_files = _list_raster_files(directory)
        if not raster_files:
            warnings.warn(f"Aucun raster .tif trouvé dans {directory} pour {value_key.upper()}.")
            return
        for raster_path in raster_files:
            if multiband:
                try:
                    dt = _parse_raster_date(raster_path)
                except ValueError:
                    warnings.warn(f"Nom de fichier ignoré (date introuvable): {raster_path.name}")
                    continue
                year = dt.year
                processed_file = False
                try:
                    with _prepared_raster(raster_path) as prepared:
                        with rasterio.open(prepared) as dataset:
                            _validate_dataset(dataset, raster_path)
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
                                if value_key == "etp_mm":
                                    samples = {
                                        pair: _convert_etp_value(val, dt_band)
                                        for pair, val in samples.items()
                                    }
                                for pair, value in samples.items():
                                    records[(pair, dt_band)][value_key] = value
                except (EOFError, OSError, gzip.BadGzipFile) as exc:
                    warnings.warn(f"Raster corrompu ignoré ({raster_path.name}): {exc}")
                    continue
                if processed_file and on_raster_processed:
                    on_raster_processed(value_key, raster_path)
            else:
                try:
                    dt = _parse_raster_date(raster_path)
                except ValueError:
                    warnings.warn(f"Nom de fichier ignoré (date introuvable): {raster_path.name}")
                    continue
                if not _within_range(dt, start_date, end_date):
                    continue
                try:
                    with _prepared_raster(raster_path) as prepared:
                        with rasterio.open(prepared) as dataset:
                            _validate_dataset(dataset, raster_path)
                            samples = _sample_dataset(dataset, geometries, band_index=1)
                except (EOFError, OSError, gzip.BadGzipFile) as exc:
                    warnings.warn(f"Raster corrompu ignoré ({raster_path.name}): {exc}")
                    continue
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
    geometry_mode: str = "auto",
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
            geometry_mode=geometry_mode,
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
