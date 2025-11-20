"""Interface en ligne de commande pour l'inventaire des données."""
from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .ahp import AHPWeights, compute_ahp_scores, load_hydrology_summary as load_ahp_summary
from .ahp import AHPWeights, compute_ahp_scores, load_hydrology_summary as load_ahp_summary
from .basins import delineate_basins, export_basins_geojson, load_site_basins
from .catalog import summarize_directory
from .config import get_paths
from .climate import aggregate_series, export_series
from .fabdem import DEFAULT_SITES_FILE, sample_sites, summarize_head_gaps
from .hydrology import (
    HydrologyModelConfig,
    load_climate_series,
    load_site_parameters,
    run_hydrology_simulation_from_data,
)
from .qa import run_quality_checks
from .sensitivity import HAS_SALIB, SENSITIVITY_PROBLEM, run_sensitivity_analysis
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
    basins: Optional[Path] = typer.Option(None, "--basins", help="GeoJSON des bassins versants à utiliser à la place des buffers"),
) -> None:
    """Calcule les séries climatiques (CHIRPS/ERA5) agrégées par site."""

    paths = get_paths(root)
    start_date = date(start_year, 1, 1) if start_year else None
    end_date = date(end_year, 12, 31) if end_year else None
    progress_callback = None
    basins_path = basins
    if basins_path and not basins_path.is_absolute():
        basins_path = paths.root / basins_path

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
                basins_geojson=basins_path,
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
            basins_geojson=basins_path,
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
            basins_geojson=basins_path,
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


@app.command("site-basins")
def site_basins(
    root: Optional[Path] = typer.Option(None, "--root", help="Chemin vers la racine du dépôt"),
    sites_csv: Optional[Path] = typer.Option(None, "--sites", help="CSV des sites"),
    output: Path = typer.Option(Path("results/site_basins.geojson"), help="Fichier GeoJSON de sortie"),
    dem_margin_km: float = typer.Option(15.0, help="Demi-largeur (km) du cadre DEM autour de chaque site", show_default=True),
    work_dir: Optional[Path] = typer.Option(None, "--work-dir", help="Dossier pour conserver les rasters intermédiaires"),
) -> None:
    """Dérive les bassins versants en amont de chaque site PHES à partir du FABDEM."""

    paths = get_paths(root)
    csv_path = sites_csv or (paths.data_dir / DEFAULT_SITES_FILE)
    actual_output = output if output.is_absolute() else (paths.root / output)
    actual_work_dir = work_dir
    if actual_work_dir and not actual_work_dir.is_absolute():
        actual_work_dir = paths.root / actual_work_dir

    try:
        basins = delineate_basins(paths, csv_path, margin_km=dem_margin_km, work_dir=actual_work_dir)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    metadata = {
        "sites_csv": str(csv_path),
        "dem_margin_km": dem_margin_km,
    }
    geojson_path = export_basins_geojson(actual_output, basins, metadata=metadata)
    console.print(f"{len(basins)} bassins exportés dans {geojson_path}")


@app.command()
def hydro_sim(
    root: Optional[Path] = typer.Option(None, "--root", help="Chemin vers la racine du dépôt"),
    climate: Path = typer.Option(Path("results/climate_series.csv"), help="Séries climatiques agrégées (CSV/Parquet)"),
    sites_csv: Optional[Path] = typer.Option(None, "--sites", help="CSV des 12 sites"),
    basins: Optional[Path] = typer.Option(None, "--basins", help="GeoJSON des bassins versants pour définir l'aire contributive"),
    iterations: int = typer.Option(10000, help="Nombre de tirages Monte Carlo", show_default=True),
    seed: Optional[int] = typer.Option(42, help="Graine aléatoire"),
    output: Optional[Path] = typer.Option(Path("results/hydrology_summary.parquet"), help="Fichier de sortie (csv/parquet)"),
    sensitivity_method: Optional[str] = typer.Option(None, help="Méthode de sensibilité: sobol, morris ou none"),
    sensitivity_samples: int = typer.Option(64, help="Nombre d'échantillons pour l'analyse de sensibilité", show_default=True),
    sensitivity_metric: str = typer.Option(
        "median_balance",
        help="Métrique analysée: median_balance, prob_positive, dry_median_balance, dry_p90_deficit",
        show_default=True,
    ),
    sensitivity_iterations: Optional[int] = typer.Option(2000, help="Iterations Monte Carlo pour la sensibilité"),
    sensitivity_output: Optional[Path] = typer.Option(None, help="Fichier output des indices de sensibilité"),
) -> None:
    """Lance la simulation hydrologique stochastique pour les 12 sites."""

    paths = get_paths(root)
    climate_path = climate if climate.is_absolute() else (paths.root / climate)
    csv_path = sites_csv or (paths.data_dir / DEFAULT_SITES_FILE)
    climate_df = load_climate_series(climate_path)
    basins_path = basins
    if basins_path and not basins_path.is_absolute():
        basins_path = paths.root / basins_path
    basin_areas = None
    if basins_path:
        basin_map = load_site_basins(basins_path)
        basin_areas = {pair: basin.area_m2 for pair, basin in basin_map.items()}
    params = load_site_parameters(csv_path, basin_areas_m2=basin_areas)
    config = HydrologyModelConfig(iterations=iterations, seed=seed)
    df = run_hydrology_simulation_from_data(climate_df, params, config)

    table = Table(title="Synthèse hydrologique", header_style="bold blue")
    table.add_column("Site")
    table.add_column("Med. bilan (GL)", justify="right")
    table.add_column("Prob. bilan>0", justify="right")
    table.add_column("Prob. stock jamais vide", justify="right")
    table.add_column("Prob. saison sèche>0", justify="right")
    table.add_column("Med. saison sèche (GL)", justify="right")
    table.add_column("P90 déficit saison sèche (GL)", justify="right")

    for _, row in df.iterrows():
        table.add_row(
            row["pair_identifier"],
            f"{row['median_annual_balance_gl']:.1f}",
            f"{row['prob_positive_annual_balance']:.2f}",
            f"{row['prob_storage_never_empty']:.2f}",
            f"{row['dry_season_prob_positive']:.2f}" if pd.notna(row["dry_season_prob_positive"]) else "-",
            f"{row['dry_season_median_balance_gl']:.1f}" if pd.notna(row["dry_season_median_balance_gl"]) else "-",
            f"{row['dry_season_p90_deficit_gl']:.1f}" if pd.notna(row["dry_season_p90_deficit_gl"]) else "-",
        )

    console.print(table)

    if output:
        actual = output if output.is_absolute() else (paths.root / output)
        actual.parent.mkdir(parents=True, exist_ok=True)
        if actual.suffix == ".parquet":
            try:
                df.to_parquet(actual)
                console.print(f"Résumé sauvegardé dans {actual}")
            except ImportError:
                fallback = actual.with_suffix(".csv")
                df.to_csv(fallback, index=False)
                console.print(
                    "[yellow]pyarrow/fastparquet manquant, export CSV de secours dans"
                    f" {fallback}[/yellow]"
                )
        else:
            df.to_csv(actual, index=False)
            console.print(f"Résumé sauvegardé dans {actual}")

    method_normalized = None
    if sensitivity_method:
        lowered = sensitivity_method.lower()
        if lowered not in {"sobol", "morris", "none"}:
            raise typer.BadParameter("La méthode doit être 'sobol', 'morris' ou 'none'.", param_hint="--sensitivity-method")
        if lowered != "none":
            method_normalized = lowered

    if method_normalized:
        if not HAS_SALIB:
            console.print("[yellow]SALib non installé : impossible de calculer les indices de sensibilité.[/yellow]")
        else:
            metric_lower = sensitivity_metric.lower()
            valid_metrics = {"median_balance", "prob_positive", "dry_median_balance", "dry_p90_deficit"}
            if metric_lower not in valid_metrics:
                raise typer.BadParameter(
                    f"Métrique inconnue ({sensitivity_metric}). Choisir parmi {', '.join(sorted(valid_metrics))}.",
                    param_hint="--sensitivity-metric",
                )
            sens_iterations = sensitivity_iterations or iterations
            sensitivity_df = run_sensitivity_analysis(
                climate_df,
                params,
                config,
                method=method_normalized,
                samples=sensitivity_samples,
                metric=metric_lower,
                iterations=sens_iterations,
                seed=seed,
            )

            table = Table(
                title=f"Analyse de sensibilité ({method_normalized.upper()} – {metric_lower})",
                header_style="bold magenta",
            )
            table.add_column("Site")
            table.add_column("Paramètre")
            if method_normalized == "sobol":
                table.add_column("S1", justify="right")
                table.add_column("ST", justify="right")
            else:
                table.add_column("mu*", justify="right")
                table.add_column("sigma", justify="right")

            for _, row in sensitivity_df.iterrows():
                for name in SENSITIVITY_PROBLEM["names"]:
                    if method_normalized == "sobol":
                        table.add_row(
                            row["pair_identifier"],
                            name,
                            f"{row.get(f'{name}_S1', float('nan')):.3f}",
                            f"{row.get(f'{name}_ST', float('nan')):.3f}",
                        )
                    else:
                        table.add_row(
                            row["pair_identifier"],
                            name,
                            f"{row.get(f'{name}_mu_star', float('nan')):.3f}",
                            f"{row.get(f'{name}_sigma', float('nan')):.3f}",
                        )
            console.print(table)

            output_path = sensitivity_output
            if output_path is None and output:
                actual = output if output.is_absolute() else (paths.root / output)
                suffix = actual.suffix or ".csv"
                output_path = actual.with_name(f"{actual.stem}_sensitivity{suffix}")
            elif output_path and not output_path.is_absolute():
                output_path = paths.root / output_path

            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if output_path.suffix == ".parquet":
                    sensitivity_df.to_parquet(output_path)
                else:
                    sensitivity_df.to_csv(output_path, index=False)
                console.print(f"Indices sauvegardés dans {output_path}")


@app.command()
def ahp_rank(
    root: Optional[Path] = typer.Option(None, "--root", help="Chemin vers la racine du dépôt"),
    sites_csv: Optional[Path] = typer.Option(None, "--sites", help="CSV des sites"),
    hydrology_summary: Path = typer.Option(
        Path("results/hydrology_summary.parquet"),
        help="Synthèse hydrologique (csv/parquet) issue de hydro-sim",
        show_default=True,
    ),
    output: Optional[Path] = typer.Option(
        Path("results/ahp_rankings.parquet"),
        help="Fichier d'export (csv/parquet)",
        show_default=True,
    ),
    economic_weight: float = typer.Option(0.4, help="Poids du critère économique", show_default=True),
    hydrology_weight: float = typer.Option(0.4, help="Poids du critère hydrologique", show_default=True),
    infrastructure_weight: float = typer.Option(0.2, help="Poids du critère infrastructure", show_default=True),
    cycles_per_year: int = typer.Option(300, help="Cycles équivalents par an", show_default=True),
    lifetime_years: int = typer.Option(60, help="Durée de vie (années)", show_default=True),
    discount_rate: float = typer.Option(0.05, help="Taux d'actualisation", show_default=True),
    round_trip_efficiency: float = typer.Option(0.81, help="Efficacité cycle", show_default=True),
) -> None:
    """Classe les sites via un AHP simplifié reliant classe économique et hydrologie."""

    paths = get_paths(root)
    sites_path = sites_csv or (paths.data_dir / DEFAULT_SITES_FILE)
    hydro_path = hydrology_summary if hydrology_summary.is_absolute() else (paths.root / hydrology_summary)

    sites_df = load_sites(sites_path)
    hydro_df = load_ahp_summary(hydro_path)
    weights = AHPWeights(
        economic=economic_weight,
        hydrology=hydrology_weight,
        infrastructure=infrastructure_weight,
    )
    scores = compute_ahp_scores(
        sites_df,
        hydro_df,
        weights=weights,
        cycles_per_year=cycles_per_year,
        lifetime_years=lifetime_years,
        discount_rate=discount_rate,
        round_trip_efficiency=round_trip_efficiency,
    )

    table = Table(title="Classement AHP", header_style="bold green")
    table.add_column("Rang", justify="right")
    table.add_column("Site")
    table.add_column("Classe éco", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Score éco", justify="right")
    table.add_column("Score hydro", justify="right")
    table.add_column("LCOS ($/MWh)", justify="right")
    table.add_column("Prob annuelle", justify="right")
    table.add_column("Prob saison sèche", justify="right")

    for _, row in scores.iterrows():
        table.add_row(
            str(row["rank"]),
            row["pair_identifier"],
            row["class"],
            f"{row['final_score']:.3f}",
            f"{row['economic_score']:.3f}" if not math.isnan(row["economic_score"]) else "-",
            f"{row['hydrology_score']:.3f}" if not math.isnan(row["hydrology_score"]) else "-",
            f"{row['lcos_usd_per_mwh']:.1f}" if not math.isnan(row["lcos_usd_per_mwh"]) else "-",
            f"{row['prob_positive_annual_balance']:.2f}" if not math.isnan(row["prob_positive_annual_balance"]) else "-",
            f"{row['dry_season_prob_positive']:.2f}" if not math.isnan(row["dry_season_prob_positive"]) else "-",
        )

    console.print(table)

    if output:
        actual = output if output.is_absolute() else (paths.root / output)
        actual.parent.mkdir(parents=True, exist_ok=True)
        if actual.suffix == ".parquet":
            scores.to_parquet(actual)
        else:
            scores.to_csv(actual, index=False)
        console.print(f"Classement exporté dans {actual}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
