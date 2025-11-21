#!/usr/bin/env python3
"""Télécharge et convertit les rasters ERA5 (évapotranspiration potentielle).

Le script s'appuie sur l'API Copernicus (cdsapi) pour récupérer la variable
`potential_evaporation` du jeu de données
`reanalysis-era5-single-levels-monthly-means`, puis génère des GeoTIFF
annuels (12 bandes) compatibles avec `phes_assessment.cli climate-series`.

Utilisation type :

```
python process_era5_standard_evapotranspiration.py \
    --start-year 2002 --end-year 2023 \
    --output-dir data/era5 \
    --cache-dir data/era5_raw \
    --north 12.8 --south 6.0 --west -1.5 --east 3.5
```

Assurez-vous d'avoir créé `~/.cdsapirc` avec vos identifiants Copernicus.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import typer
import xarray as xr
from rich.console import Console
from rich.progress import track
from rich.table import Table

from phes_assessment.config import get_paths

try:  # pragma: no cover - dépendance optionnelle
    import cdsapi
except ImportError:  # pragma: no cover
    cdsapi = None  # type: ignore[assignment]


app = typer.Typer(help="Télécharge et convertit les séries ERA5 (évapotranspiration)")
console = Console()

DATASET = "reanalysis-era5-single-levels-monthly-means"
VARIABLE = "potential_evaporation"


@app.command()
def main(
    root: Optional[Path] = typer.Option(None, "--root", help="Chemin vers la racine du dépôt"),
    output_dir: Optional[Path] = typer.Option(None, help="Dossier cible des GeoTIFF annuels"),
    cache_dir: Optional[Path] = typer.Option(Path("data/era5_raw"), help="Dossier cache des NetCDF téléchargés"),
    start_year: int = typer.Option(2002, help="Première année à traiter", show_default=True),
    end_year: int = typer.Option(2023, help="Dernière année à traiter", show_default=True),
    north: float = typer.Option(12.8, help="Latitude nord de l'emprise (degrés)"),
    south: float = typer.Option(6.0, help="Latitude sud de l'emprise (degrés)"),
    west: float = typer.Option(-1.5, help="Longitude ouest de l'emprise (degrés)"),
    east: float = typer.Option(3.5, help="Longitude est de l'emprise (degrés)"),
    grid: float = typer.Option(0.25, help="Pas de grille (°)", show_default=True),
    skip_download: bool = typer.Option(False, help="Ne pas télécharger si les NetCDF existent"),
    force: bool = typer.Option(False, help="Forcer le re-téléchargement même si le NetCDF est présent"),
) -> None:
    """Télécharge les NetCDF ERA5 puis génère des GeoTIFF par année."""

    if start_year > end_year:
        raise typer.BadParameter("start-year doit être ≤ end-year.")
    if north <= south:
        raise typer.BadParameter("north doit être strictement supérieur à south.")
    if east <= west:
        raise typer.BadParameter("east doit être strictement supérieur à west.")
    if grid <= 0:
        raise typer.BadParameter("grid doit être strictement positif.")

    paths = get_paths(root)
    era5_dir = (paths.data_dir / "era5") if output_dir is None else _resolve(paths.root, output_dir)
    cache_path = _resolve(paths.root, cache_dir) if cache_dir else (paths.root / "data/era5_raw")
    era5_dir.mkdir(parents=True, exist_ok=True)
    cache_path.mkdir(parents=True, exist_ok=True)

    download_client = None
    if not skip_download:
        if cdsapi is None:
            raise typer.BadParameter(
                "cdsapi non installé. Exécutez `pip install cdsapi` ou fournissez --skip-download si les NetCDF existent."
            )
        download_client = cdsapi.Client()

    summary: Dict[int, Dict[str, float]] = {}
    area = [north, west, south, east]
    years = range(start_year, end_year + 1)

    for year in track(years, description="Traitement ERA5"):
        nc_path = cache_path / f"era5_potential_evaporation_{year}.nc"
        tif_path = era5_dir / f"era5_{year}.tif"

        if not skip_download and (force or not nc_path.exists()):
            console.print(f"Téléchargement ERA5 {year}…")
            _download_year(download_client, year, area, grid, nc_path)
        elif not nc_path.exists():
            raise typer.BadParameter(
                f"NetCDF manquant pour {year}: {nc_path}. Désactivez --skip-download ou placez le fichier manuellement."
            )

        stats = _netcdf_to_geotiff(nc_path, tif_path)
        summary[year] = stats

    _print_summary(summary, era5_dir)


def _resolve(root: Path, candidate: Path) -> Path:
    return candidate if candidate.is_absolute() else (root / candidate)


def _download_year(client: cdsapi.Client, year: int, area: list[float], grid: float, target: Path) -> None:  # type: ignore[name-defined]
    request = {
        "product_type": "monthly_averaged_reanalysis",
        "variable": VARIABLE,
        "year": str(year),
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": "00:00",
        "format": "netcdf",
        "area": area,
        "grid": [grid, grid],
    }
    client.retrieve(DATASET, request, str(target))


def _netcdf_to_geotiff(netcdf_path: Path, output_path: Path) -> Dict[str, float]:
    ds = xr.open_dataset(netcdf_path)
    try:
        data = _extract_variable(ds)
        times = pd.to_datetime(data["time"].values)
        arr = np.asarray(data, dtype=np.float32)
        lat = data["latitude"].values
        lon = data["longitude"].values

        arr, lat = _ensure_descending_lat(arr, lat)
        arr, lon = _ensure_ascending_lon(arr, lon)

        # ERA5 « monthly averaged reanalysis » fournit des moyennes journalières (m/jour)
        # déjà intégrées sur 24 h. On convertit donc en mm/mois via : |arr| × nb_jours × 1000.
        days_in_month = np.array([ts.days_in_month for ts in times], dtype=np.float32)
        arr_mm = np.abs(arr) * days_in_month[:, None, None] * 1000.0

        lat_res = abs(lat[1] - lat[0]) if len(lat) > 1 else 0.25
        lon_res = abs(lon[1] - lon[0]) if len(lon) > 1 else 0.25
        north = lat[0] + lat_res / 2
        west = lon[0] - lon_res / 2
        transform = from_origin(west, north, lon_res, lat_res)

        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": arr_mm.shape[0],
            "height": arr_mm.shape[1],
            "width": arr_mm.shape[2],
            "transform": transform,
            "crs": "EPSG:4326",
            "compress": "deflate",
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            for idx in range(arr_mm.shape[0]):
                dst.write(arr_mm[idx], idx + 1)
                dst.set_band_description(idx + 1, f"{times[idx]:%Y-%m}")
                dst.update_tags(
                    idx + 1,
                    units="mm/month",
                    era5_variable=VARIABLE,
                )

        return {
            "months": float(arr_mm.shape[0]),
            "mean_mm": float(np.nanmean(arr_mm)),
            "time_start": float(times[0].value),
            "time_end": float(times[-1].value),
        }
    finally:
        ds.close()


def _extract_variable(ds: xr.Dataset) -> xr.DataArray:
    for candidate in ("pev", "potential_evaporation"):
        if candidate in ds:
            return ds[candidate]
    raise ValueError("La variable potential_evaporation n'a pas été trouvée dans le NetCDF ERA5.")


def _ensure_descending_lat(arr: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if lat[0] < lat[-1]:
        lat = lat[::-1]
        arr = arr[:, ::-1, :]
    return arr, lat


def _ensure_ascending_lon(arr: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if lon[0] > lon[-1]:
        lon = lon[::-1]
        arr = arr[:, :, ::-1]
    return arr, lon


def _print_summary(summary: Dict[int, Dict[str, float]], output_dir: Path) -> None:
    table = Table(title="GeoTIFF ERA5 générés", header_style="bold green")
    table.add_column("Année", justify="right")
    table.add_column("Bandes", justify="right")
    table.add_column("ETP moyenne (mm/mois)", justify="right")
    table.add_column("Fichier")

    for year in sorted(summary):
        stats = summary[year]
        table.add_row(
            str(year),
            f"{int(stats['months'])}",
            f"{stats['mean_mm']:.1f}",
            str(output_dir / f"era5_{year}.tif"),
        )

    console.print(table)

if __name__ == "__main__":
    app()