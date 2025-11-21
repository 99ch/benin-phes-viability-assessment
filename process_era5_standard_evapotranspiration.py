#!/usr/bin/env python3#!/usr/bin/env python3

"""Pipeline de traitement ERA5 → GeoTIFF multi-bandes (ETP standardisée)."""Traitement ERA5/CHIRPS standard (version harmonisée).



Ce script fournit deux commandes complémentaires :Ce script conserve le point d'entrée historique mais délègue désormais

entièrement au module `phes_assessment.climate` afin d'éviter tout

- ``download`` : télécharge les champs horaires ERA5 (single levels) via l'APIéchantillonnage « pixel unique » et garantir une cohérence parfaite avec

  Copernicus CDS pour la zone d'étude et la période choisie ;les commandes CLI (`phes-data climate-series`).

- ``convert`` : convertit un NetCDF ERA5 (ou un dossier de NetCDF) en GeoTIFF"""

  annuel compatible avec ``phes_assessment.climate`` (12 bandes mensuelles,from __future__ import annotations

  unités en millimètres).

from pathlib import Path

L'objectif est de garantir une reproductibilité parfaite des rastersfrom typing import Optional

``data/era5/era5_YYYY.tif`` utilisés par ``phes-data climate-series``.

import pandas as pd

Exemples d'usage :import typer

from rich.console import Console

.. code-block:: bashfrom rich.table import Table



    # 1) Télécharger les fichiers NetCDF horaires pour 2020-2023from phes_assessment.climate import aggregate_series, export_series

    python process_era5_standard_evapotranspiration.py download \from phes_assessment.config import get_paths

        --start-year 2020 --end-year 2023 \

        --north 12.5 --west 0.0 --south 9.0 --east 3.0 \DEFAULT_SITES = "n10_e001_12_sites_complete.csv"

        --output-dir data/era5/raw

app = typer.Typer(help="Génère les séries climatiques ERA5/CHIRPS via les utilitaires officiels")

    # 2) Convertir chaque NetCDF en GeoTIFF multi-bandes (mm/mois)console = Console()

    python process_era5_standard_evapotranspiration.py convert \

        --source data/era5/raw/era5_2020.nc \

        --output data/era5/era5_2020.tif@app.command()

def main(

    # 3) Conversion en lot d'un dossier complet    root: Optional[Path] = typer.Option(None, "--root", help="Chemin vers la racine du dépôt"),

    python process_era5_standard_evapotranspiration.py convert \    sites_csv: Optional[Path] = typer.Option(None, "--sites", help="CSV des sites (défaut: data/n10_e001_12_sites_complete.csv)"),

        --source data/era5/raw \    basins: Optional[Path] = typer.Option(None, "--basins", help="GeoJSON des bassins versants"),

        --output-dir data/era5    output: Path = typer.Option(Path("results/site_stats_era5_standard.parquet"), help="Fichier de sortie (csv/parquet)"),

    start_year: Optional[int] = typer.Option(2002, help="Année de début (None = tout l'historique)", show_default=True),

Les valeurs ERA5 de ``potential_evaporation`` sont fournies en mètres et    end_year: Optional[int] = typer.Option(2023, help="Année de fin (None = tout l'historique)", show_default=True),

négatives (flux sortant). Nous appliquons donc par défaut un facteur ``-1000``    buffer_meters: float = typer.Option(500.0, help="Rayon des buffers lorsqu'aucun bassin n'est fourni", show_default=True),

(pour obtenir des millimètres positifs) puis un agrégat ``sum`` par mois.    geometry_mode: str = typer.Option(

"""        "auto",

from __future__ import annotations        help="auto = basins si disponible, sinon buffers ; on peut forcer 'basins' ou 'buffers'",

        show_default=True,

import sys    ),

from dataclasses import dataclass) -> None:

from pathlib import Path    """Produit des séries climatiques cohérentes avec le pipeline principal."""

from typing import Iterable, Optional

    paths = get_paths(root)

import numpy as np    csv_path = _resolve_path(paths.data_dir / DEFAULT_SITES, sites_csv, paths.root)

import pandas as pd    basins_path = _resolve_path(None, basins, paths.root) if basins else None

import rasterio

import typer    start_date = pd.Timestamp(start_year, 1, 1) if start_year else None

import xarray as xr    end_date = pd.Timestamp(end_year, 12, 31) if end_year else None

from rasterio.transform import from_origin

from rich.console import Console    df = aggregate_series(

from rich.table import Table        paths,

        csv_path,

try:  # pragma: no cover - dépendance optionnelle        dataset="both",

    import cdsapi        buffer_meters=buffer_meters,

except ImportError:  # pragma: no cover        basins_geojson=basins_path,

    cdsapi = None  # type: ignore[assignment]        start_date=start_date.date() if start_date is not None else None,

        end_date=end_date.date() if end_date is not None else None,

console = Console()        geometry_mode=geometry_mode,

app = typer.Typer(help="Téléchargement et conversion ERA5 (ETP) pour le workflow PHES")    )



    export_path = export_series(

def _ensure_directory(path: Path) -> None:        _resolve_path(Path("results/site_stats_era5_standard.parquet"), output, paths.root),

    path.mkdir(parents=True, exist_ok=True)        paths,

        csv_path,

        dataset="both",

def _hours() -> list[str]:        buffer_meters=buffer_meters,

    return [f"{hour:02d}:00" for hour in range(24)]        basins_geojson=basins_path,

        geometry_mode=geometry_mode,

        dataframe=df,

def _months() -> list[str]:    )

    return [f"{month:02d}" for month in range(1, 13)]

    console.print(f"Séries sauvegardées dans {export_path}")

    _print_summary(df)

@dataclass

class ConversionResult:

    source: Pathdef _resolve_path(default: Path | None, candidate: Optional[Path], root: Path) -> Path:

    output: Path    if candidate is None:

    year: int        if default is None:

    band_count: int            raise ValueError("Impossible de résoudre le chemin demandé")

    precip_mm_mean: float        return default

    etp_mm_mean: float    return candidate if candidate.is_absolute() else (root / candidate)





@app.command("download")def _print_summary(df: pd.DataFrame) -> None:

def download_era5(    if df.empty:

    start_year: int = typer.Option(..., help="Première année incluse"),        console.print("[yellow]Aucune donnée générée : vérifier les rasters disponibles.[/yellow]")

    end_year: int = typer.Option(..., help="Dernière année incluse"),        return

    north: float = typer.Option(..., help="Latitude max (Nord)"),

    west: float = typer.Option(..., help="Longitude Ouest"),    table = Table(title="Séries ERA5 + CHIRPS", header_style="bold cyan")

    south: float = typer.Option(..., help="Latitude min (Sud)"),    table.add_column("Sites", justify="right")

    east: float = typer.Option(..., help="Longitude Est"),    table.add_column("Période")

    output_dir: Path = typer.Option(Path("data/era5/raw"), help="Répertoire des NetCDF téléchargés"),    table.add_column("Précip. moyenne (mm/mois)", justify="right")

    variable: str = typer.Option("potential_evaporation", help="Variable ERA5 à récupérer"),    table.add_column("ETP moyenne (mm/mois)", justify="right")

) -> None:

    """Télécharge les champs ERA5 horaires via CDS (format NetCDF)."""    sites = df["pair_identifier"].nunique()

    date_min = df["date"].min()

    if cdsapi is None:    date_max = df["date"].max()

        typer.echo("[!] Installez cdsapi (pip install cdsapi) et configurez ~/.cdsapirc avant de lancer le téléchargement.")    precip_mean = df["precip_mm"].mean()

        raise typer.Exit(code=1)    etp_mean = df["etp_mm"].mean()



    if end_year < start_year:    table.add_row(

        raise typer.BadParameter("end-year doit être >= start-year")        str(sites),

        f"{date_min:%Y-%m} → {date_max:%Y-%m}",

    client = cdsapi.Client()  # type: ignore[call-arg]        f"{precip_mean:.1f}" if pd.notna(precip_mean) else "-",

    _ensure_directory(output_dir)        f"{etp_mean:.1f}" if pd.notna(etp_mean) else "-",

    )

    for year in range(start_year, end_year + 1):    console.print(table)

        target = output_dir / f"era5_{year}.nc"

        if target.exists():

            console.print(f"[green]Déjà présent : {target}")if __name__ == "__main__":

            continue    app()
        console.print(f"[cyan]Téléchargement ERA5 {year} → {target}")
        client.retrieve(  # pragma: no cover - appel réseau
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": variable,
                "year": str(year),
                "month": _months(),
                "day": [f"{day:02d}" for day in range(1, 32)],
                "time": _hours(),
                "area": [north, west, south, east],
                "format": "netcdf",
            },
            str(target),
        )


@app.command("convert")
def convert_netcdf(
    source: Path = typer.Option(..., help="Fichier NetCDF ERA5 ou dossier contenant des .nc"),
    output: Optional[Path] = typer.Option(None, help="Chemin du GeoTIFF cible (si source unique)"),
    output_dir: Path = typer.Option(Path("data/era5"), help="Dossier des GeoTIFF générés (si conversion par lot)"),
    variable: str = typer.Option("potential_evaporation", help="Nom de la variable dans le NetCDF"),
    aggregation: str = typer.Option("sum", help="Agrégation mensuelle: sum ou mean"),
    multiplier: float = typer.Option(-1000.0, help="Facteur multiplicatif appliqué après agrégation"),
    offset: float = typer.Option(0.0, help="Décalage ajouté après multiplication"),
) -> None:
    """Convertit un ou plusieurs NetCDF ERA5 en GeoTIFF multi-bandes."""

    sources: Iterable[Path]
    if source.is_dir():
        sources = sorted(p for p in source.iterdir() if p.suffix.lower() in {".nc", ".nc4"})
        if not sources:
            raise typer.BadParameter(f"Aucun fichier NetCDF trouvé dans {source}")
        _ensure_directory(output_dir)
    else:
        sources = [source]
        if output is None:
            _ensure_directory(output_dir)

    summaries: list[ConversionResult] = []
    for netcdf_path in sources:
        result = _convert_single(
            netcdf_path,
            variable=variable,
            aggregation=aggregation,
            multiplier=multiplier,
            offset=offset,
            explicit_output=output if source.is_file() and output else None,
            default_output_dir=output_dir,
        )
        summaries.append(result)
        console.print(
            f"[green]✔ GeoTIFF {result.output.name} – {result.band_count} bandes, année {result.year}"
        )

    _print_summary_table(summaries)


def _convert_single(
    netcdf_path: Path,
    *,
    variable: str,
    aggregation: str,
    multiplier: float,
    offset: float,
    explicit_output: Path | None,
    default_output_dir: Path,
) -> ConversionResult:
    ds = xr.open_dataset(netcdf_path)
    if variable not in ds:
        raise typer.BadParameter(f"La variable '{variable}' est absente de {netcdf_path.name}")
    data = ds[variable]
    if "time" not in data.dims:
        raise typer.BadParameter(f"La variable '{variable}' ne possède pas de dimension temporelle")

    data = data.sortby("time")
    time_index = pd.to_datetime(data["time"].values)
    unique_years = np.unique(time_index.year)
    if unique_years.size != 1:
        raise typer.BadParameter(
            f"Le fichier {netcdf_path.name} couvre plusieurs années ({unique_years}). "
            "Générez un fichier par année avant conversion."
        )
    year = int(unique_years[0])

    if aggregation == "sum":
        monthly = data.resample(time="MS").sum()
    elif aggregation == "mean":
        monthly = data.resample(time="MS").mean()
    else:
        raise typer.BadParameter("aggregation doit valoir 'sum' ou 'mean'")

    monthly = (monthly * multiplier) + offset
    monthly = monthly.transpose("time", "latitude", "longitude")

    latitudes = monthly["latitude"].values
    longitudes = monthly["longitude"].values

    # Assurer un ordre décroissant en latitude pour correspondre à la convention GeoTIFF
    if latitudes[0] < latitudes[-1]:
        monthly = monthly.reindex(latitude=latitudes[::-1])
        latitudes = monthly["latitude"].values

    if latitudes.size < 2 or longitudes.size < 2:
        raise typer.BadParameter("Le NetCDF doit contenir au moins deux pas de latitude et de longitude.")
    y_res = float(abs(latitudes[1] - latitudes[0]))
    x_res = float(abs(longitudes[1] - longitudes[0]))
    transform = from_origin(float(longitudes.min()), float(latitudes.max()), x_res, y_res)

    values = monthly.values.astype(np.float32)
    band_count = values.shape[0]
    if band_count != 12:
        console.print(
            f"[yellow]⚠ {netcdf_path.name} ne contient pas 12 mois (bandes = {band_count})."
        )

    output_path = explicit_output or (default_output_dir / f"era5_{year}.tif")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=values.shape[1],
        width=values.shape[2],
        count=band_count,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan,
        compress="lzw",
    ) as dst:
        timestamps = pd.to_datetime(monthly["time"].values)
        for idx in range(band_count):
            dst.write(values[idx], idx + 1)
            month_label = f"{timestamps[idx].month:02d}"
            dst.set_band_description(idx + 1, month_label)

    precip_mean = float(np.nanmean(values))
    return ConversionResult(
        source=netcdf_path,
        output=output_path,
        year=year,
        band_count=band_count,
        precip_mm_mean=precip_mean,
        etp_mm_mean=precip_mean,
    )


def _print_summary_table(results: list[ConversionResult]) -> None:
    if not results:
        return
    table = Table(title="Conversion ERA5 → GeoTIFF", header_style="bold cyan")
    table.add_column("Fichier NetCDF")
    table.add_column("GeoTIFF")
    table.add_column("Année", justify="right")
    table.add_column("Bandes", justify="right")
    table.add_column("Moyenne (mm/mois)", justify="right")
    for entry in results:
        table.add_row(
            entry.source.name,
            entry.output.name,
            str(entry.year),
            str(entry.band_count),
            f"{entry.precip_mm_mean:.1f}" if np.isfinite(entry.precip_mm_mean) else "-",
        )
    console.print(table)


if __name__ == "__main__":
    try:
        app()
    except typer.Exit:
        raise
    except Exception as exc:  # pragma: no cover - utilisation CLI
        console.print(f"[red]Erreur : {exc}")
        sys.exit(1)
