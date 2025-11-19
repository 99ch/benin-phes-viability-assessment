"""Interface en ligne de commande pour l'inventaire des données."""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .catalog import summarize_directory
from .config import get_paths
from .climate import aggregate_series, export_series
from .fabdem import DEFAULT_SITES_FILE, sample_sites, summarize_head_gaps
from .qa import run_quality_checks
from .sites import build_site_masks, export_masks_geojson, load_sites

app = typer.Typer(help="Outils CLI pour l'étude PHES")
console = Console()

@app.command()
def data_catalog(root: Optional[Path] = typer.Option(None, "--root", help="Chemin vers la racine du dépôt")) -> None:
    """Affiche un résumé des principaux jeux de données disponibles localement."""

    paths = get_paths(root)
    table = Table(title="Inventaire datasets", header_style="bold cyan")
    table.add_column("Dataset")
    table.add_column("Fichiers", justify="right")
    table.add_column("Début")
    table.add_column("Fin")
    table.add_column("CRS")
    table.add_column("Résolution (m)")

    for label, directory in (
        ("CHIRPS", paths.chirps_dir),
        ("ERA5", paths.era5_dir),
        ("FABDEM", paths.fabdem_dir),
    ):
        summary = summarize_directory(label, directory)
        res = "-"
        if summary.resolution:
            res = f"{summary.resolution[0]:.1f} x {summary.resolution[1]:.1f}"
        table.add_row(
            summary.name,
            str(summary.file_count),
            summary.start_date.isoformat() if summary.start_date else "-",
            summary.end_date.isoformat() if summary.end_date else "-",
            summary.crs or "-",
            res,
        )

    console.print(table)
    console.print(f"Date d'exécution : {datetime.now():%Y-%m-%d %H:%M:%S}")


@app.command()
def data_qa(
    root: Optional[Path] = typer.Option(None, "--root", help="Chemin vers la racine du dépôt"),
    sites_csv: Optional[Path] = typer.Option(None, "--sites", help="CSV des sites pour vérifier l'emprise FABDEM"),
    start_year: int = typer.Option(2002, help="Année attendue de début pour les climatologies", show_default=True),
    end_year: int = typer.Option(2023, help="Année attendue de fin pour les climatologies", show_default=True),
) -> None:
    """Exécute des contrôles qualité rapides sur CHIRPS, ERA5 et FABDEM."""

    paths = get_paths(root)
    csv_path = sites_csv or (paths.data_dir / DEFAULT_SITES_FILE)
    reports = run_quality_checks(paths, start_year=start_year, end_year=end_year, sites_csv=csv_path)

    table = Table(title="Contrôles qualité datasets", header_style="bold magenta")
    table.add_column("Dataset")
    table.add_column("Fichiers", justify="right")
    table.add_column("Couverture réelle")
    table.add_column("Couverture attendue")
    table.add_column("Manquants")
    table.add_column("Doublons")
    table.add_column("Notes")

    for report in reports:
        coverage_real = "-"
        if report.coverage_start and report.coverage_end:
            coverage_real = f"{report.coverage_start:%Y-%m} → {report.coverage_end:%Y-%m}"
        coverage_expected = "-"
        if report.expected_start and report.expected_end:
            coverage_expected = f"{report.expected_start:%Y-%m} → {report.expected_end:%Y-%m}"
        table.add_row(
            report.dataset,
            str(report.file_count),
            coverage_real,
            coverage_expected,
            report.missing_label(),
            report.duplicate_label(),
            report.notes_label(),
        )

    console.print(table)


@app.command()
def fabdem_sample(
    root: Optional[Path] = typer.Option(None, "--root", help="Chemin vers la racine du dépôt"),
    sites_csv: Optional[Path] = typer.Option(None, "--sites", help="Chemin du CSV contenant les sites"),
    output: Optional[Path] = typer.Option(None, "--output", help="Fichier CSV pour enregistrer les résultats"),
) -> None:
    """Échantillonne les altitudes FABDEM pour les sites sélectionnés."""

    paths = get_paths(root)
    csv_path = sites_csv or (paths.data_dir / DEFAULT_SITES_FILE)
    df = sample_sites(paths, csv_path)
    summary = summarize_head_gaps(df)

    table = Table(title="Comparaison head catalogue vs FABDEM", header_style="bold green")
    table.add_column("Pair")
    table.add_column("Head catalogue (m)", justify="right")
    table.add_column("Head DEM (m)", justify="right")
    table.add_column("Delta (m)", justify="right")

    for _, row in summary.iterrows():
        table.add_row(
            str(row["Pair Identifier"]),
            f"{row['Catalog head (m)']:.1f}",
            f"{row['DEM head (m)']:.1f}",
            f"{row['Head delta (m)']:.1f}",
        )

    console.print(table)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        console.print(f"Résultats détaillés exportés dans {output}")


@app.command()
def climate_series(
    root: Optional[Path] = typer.Option(None, "--root", help="Chemin vers la racine du dépôt"),
    sites_csv: Optional[Path] = typer.Option(None, "--sites", help="CSV des sites"),
    dataset: str = typer.Option("both", help="Dataset à traiter: both, chirps ou era5"),
    output: Optional[Path] = typer.Option(Path("results/climate_series.csv"), help="Fichier de sortie (csv ou parquet)"),
    start_year: Optional[int] = typer.Option(2002, help="Année de début (None pour tout l'historique)", show_default=True),
    end_year: Optional[int] = typer.Option(2023, help="Année de fin (None pour tout l'historique)", show_default=True),
    buffer_meters: float = typer.Option(500.0, help="Rayon du buffer appliqué autour de chaque réservoir (m)", show_default=True),
) -> None:
    """Calcule les séries climatiques (CHIRPS/ERA5) agrégées par site."""

    paths = get_paths(root)
    start_date = date(start_year, 1, 1) if start_year else None
    end_date = date(end_year, 12, 31) if end_year else None
    progress_callback = None

    def _build_progress_callback(progress: Progress, tasks: dict[str, int]):
        def _callback(metric: str, _path: Path) -> None:
            task_id = tasks.get(metric)
            if task_id is not None:
                progress.advance(task_id)

        return _callback

    df = None
    needs_progress = dataset in ("both", "chirps", "era5")
    if needs_progress:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
            transient=True,
        )
        tasks: dict[str, int] = {}
        with progress:
            if dataset in ("both", "chirps"):
                tasks["precip_mm"] = progress.add_task("CHIRPS", total=None, start=True)
            if dataset in ("both", "era5"):
                tasks["etp_mm"] = progress.add_task("ERA5", total=None, start=True)
            progress_callback = _build_progress_callback(progress, tasks)
            df = aggregate_series(
                paths,
                sites_csv,
                dataset,
                buffer_meters=buffer_meters,
                start_date=start_date,
                end_date=end_date,
                on_raster_processed=progress_callback,
            )
    else:
        df = aggregate_series(
            paths,
            sites_csv,
            dataset,
            buffer_meters=buffer_meters,
            start_date=start_date,
            end_date=end_date,
        )

    console.print(f"{len(df)} enregistrements générés pour {df['pair_identifier'].nunique()} sites")
    console.print(df.head())

    if output:
        actual = export_series(
            output,
            paths,
            sites_csv,
            dataset,
            buffer_meters=buffer_meters,
            start_date=start_date,
            end_date=end_date,
            dataframe=df,
        )
        console.print(f"Séries sauvegardées dans {actual}")


@app.command()
def site_masks(
    root: Optional[Path] = typer.Option(None, "--root", help="Chemin vers la racine du dépôt"),
    sites_csv: Optional[Path] = typer.Option(None, "--sites", help="CSV des sites"),
    output: Path = typer.Option(Path("results/site_masks.geojson"), help="Fichier GeoJSON de sortie"),
    buffer_meters: float = typer.Option(500.0, help="Rayon du buffer pour chaque réservoir (m)", show_default=True),
    utm_epsg: int = typer.Option(32631, help="Code EPSG utilisé pour projeter les points", show_default=True),
) -> None:
    """Exporte les buffers de chaque site PHES sous forme de GeoJSON."""

    paths = get_paths(root)
    csv_path = sites_csv or (paths.data_dir / DEFAULT_SITES_FILE)
    df = load_sites(csv_path)
    masks = build_site_masks(df, buffer_meters=buffer_meters, utm_epsg=utm_epsg)
    metadata = {
        "buffer_meters": buffer_meters,
        "utm_epsg": utm_epsg,
        "site_count": len(masks),
        "source_csv": str(csv_path),
    }
    geojson_path = export_masks_geojson(output, masks, metadata=metadata)
    console.print(f"{len(masks)} masques exportés dans {geojson_path}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
