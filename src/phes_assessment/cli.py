"""Interface en ligne de commande pour l'inventaire des données."""
from __future__ import annotations

import json
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
from .figures import generate_all_figures, generate_article_figures
from .qa import run_quality_checks
from .sensitivity import HAS_SALIB, SENSITIVITY_PROBLEM, run_sensitivity_analysis
from .sites import build_site_masks, export_masks_geojson, load_sites

app = typer.Typer(help="Outils CLI pour l'étude PHES")
console = Console()


def _resolve_path(base: Path, candidate: Optional[Path]) -> Optional[Path]:
    if candidate is None:
        return None
    if candidate.is_absolute():
        return candidate
    return (base / candidate).resolve()


def _ensure_range(label: str, minimum: float, maximum: float) -> tuple[float, float]:
    if minimum < 0 or maximum < 0:
        raise typer.BadParameter(f"{label} doit être positif.")
    if minimum >= maximum:
        raise typer.BadParameter(f"{label} min ({minimum}) doit être strictement inférieur au max ({maximum}).")
    return minimum, maximum


def _load_weights_config(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    data: dict[str, float]
    if suffix in {".yaml", ".yml"}:
        try:  # pragma: no cover - dépendance optionnelle
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise typer.BadParameter(
                "pyyaml n'est pas installé : utilisez un fichier JSON ou installez PyYAML."
            ) from exc
        parsed = yaml.safe_load(text)
    else:
        parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise typer.BadParameter("Le fichier de configuration doit contenir un objet clé/valeur.")
    data = {}
    for key, value in parsed.items():
        try:
            data[key] = float(value)
        except (TypeError, ValueError):
            raise typer.BadParameter(
                f"Valeur non numérique pour '{key}' dans {path}"
            )
    return data


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
    csv_path = _resolve_path(paths.root, sites_csv) if sites_csv else (paths.data_dir / DEFAULT_SITES_FILE)
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
    csv_path = _resolve_path(paths.root, sites_csv) if sites_csv else (paths.data_dir / DEFAULT_SITES_FILE)
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
        output_path = _resolve_path(paths.root, output) or output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        console.print(f"Résultats détaillés exportés dans {output_path}")


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
    geometry_mode: str = typer.Option(
        "auto",
        help="Mode d'échantillonnage spatial: auto (basins si dispo), buffers ou basins",
        show_default=True,
    ),
) -> None:
    """Calcule les séries climatiques (CHIRPS/ERA5) agrégées par site."""

    paths = get_paths(root)
    start_date = date(start_year, 1, 1) if start_year else None
    end_date = date(end_year, 12, 31) if end_year else None
    progress_callback = None
    basins_path = _resolve_path(paths.root, basins)
    if basins_path is None:
        raise typer.BadParameter(
            "climate-series requiert désormais un GeoJSON de bassins. Fournissez --basins." ,
            param_hint="--basins",
        )
    csv_path = _resolve_path(paths.root, sites_csv) if sites_csv else None

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
                csv_path,
                dataset,
                buffer_meters=buffer_meters,
                basins_geojson=basins_path,
                start_date=start_date,
                end_date=end_date,
                on_raster_processed=progress_callback,
                geometry_mode=geometry_mode,
            )
    else:
        df = aggregate_series(
            paths,
            csv_path,
            dataset,
            buffer_meters=buffer_meters,
            basins_geojson=basins_path,
            start_date=start_date,
            end_date=end_date,
            geometry_mode=geometry_mode,
        )

    console.print(f"{len(df)} enregistrements générés pour {df['pair_identifier'].nunique()} sites")
    console.print(df.head())

    if output:
        actual_output = _resolve_path(paths.root, output) or output
        actual = export_series(
            actual_output,
            paths,
            csv_path,
            dataset,
            buffer_meters=buffer_meters,
            basins_geojson=basins_path,
            start_date=start_date,
            end_date=end_date,
            dataframe=df,
            geometry_mode=geometry_mode,
        )
        console.print(f"Séries sauvegardées dans {actual}")


@app.command()
def site_masks(
    root: Optional[Path] = typer.Option(None, "--root", help="Chemin vers la racine du dépôt"),
    sites_csv: Optional[Path] = typer.Option(None, "--sites", help="CSV des sites"),
    output: Path = typer.Option(Path("results/site_masks.geojson"), help="Fichier GeoJSON de sortie"),
    buffer_meters: float = typer.Option(500.0, help="Rayon du buffer pour chaque réservoir (m)", show_default=True),
    utm_epsg: Optional[int] = typer.Option(
        None,
        help="EPSG UTM pour les buffers (auto par défaut en fonction des coordonnées)",
        show_default=True,
    ),
) -> None:
    """Exporte les buffers de chaque site PHES sous forme de GeoJSON."""

    paths = get_paths(root)
    csv_path = _resolve_path(paths.root, sites_csv) if sites_csv else (paths.data_dir / DEFAULT_SITES_FILE)
    df = load_sites(csv_path)
    masks = build_site_masks(df, buffer_meters=buffer_meters, utm_epsg=utm_epsg)
    metadata = {
        "buffer_meters": buffer_meters,
        "utm_epsg": utm_epsg or "auto",
        "site_count": len(masks),
        "source_csv": str(csv_path),
    }
    output_path = _resolve_path(paths.root, output) or output
    geojson_path = export_masks_geojson(output_path, masks, metadata=metadata)
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
    csv_path = _resolve_path(paths.root, sites_csv) if sites_csv else (paths.data_dir / DEFAULT_SITES_FILE)
    actual_output = _resolve_path(paths.root, output) or output
    actual_work_dir = _resolve_path(paths.root, work_dir)

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
    runoff_min: float = typer.Option(0.3, help="Borne basse du coefficient de ruissellement"),
    runoff_max: float = typer.Option(0.8, help="Borne haute du coefficient de ruissellement"),
    runoff_alpha: float = typer.Option(3.5, help="Paramètre alpha de la loi Beta du ruissellement"),
    runoff_beta: float = typer.Option(4.0, help="Paramètre beta de la loi Beta du ruissellement"),
    infiltration_min: float = typer.Option(0.05, help="Borne basse du coefficient d'infiltration"),
    infiltration_max: float = typer.Option(0.25, help="Borne haute du coefficient d'infiltration"),
    infiltration_alpha: float = typer.Option(2.0, help="Paramètre alpha de la loi Beta d'infiltration"),
    infiltration_beta: float = typer.Option(5.0, help="Paramètre beta de la loi Beta d'infiltration"),
    initial_storage_fraction: float = typer.Option(0.6, help="Taux de remplissage initial du réservoir"),
    evap_mean: float = typer.Option(1.0, help="Moyenne du multiplicateur appliqué à l'ETP"),
    evap_std: float = typer.Option(0.1, help="Écart-type du multiplicateur ETP"),
    evap_min: float = typer.Option(0.5, help="Borne basse du multiplicateur ETP"),
    evap_max: float = typer.Option(1.5, help="Borne haute du multiplicateur ETP"),
    leakage_min: float = typer.Option(0.0005, help="Borne basse des fuites mensuelles (fraction du stock)"),
    leakage_max: float = typer.Option(0.002, help="Borne haute des fuites mensuelles (fraction du stock)"),
) -> None:
    """Lance la simulation hydrologique stochastique pour les 12 sites."""

    paths = get_paths(root)
    climate_path = _resolve_path(paths.root, climate) or climate
    csv_path = _resolve_path(paths.root, sites_csv) if sites_csv else (paths.data_dir / DEFAULT_SITES_FILE)
    climate_df = load_climate_series(climate_path)
    basins_path = _resolve_path(paths.root, basins)
    if basins_path is None:
        raise typer.BadParameter("Ce workflow requiert un GeoJSON de bassins versants. Fournissez --basins.", param_hint="--basins")
    basin_map = load_site_basins(basins_path)
    basin_areas = {pair: basin.area_m2 for pair, basin in basin_map.items()}
    params = load_site_parameters(csv_path, basin_areas_m2=basin_areas)
    missing_basin_pairs = [pair for pair, param in params.items() if param.basin_area_km2 is None]
    if missing_basin_pairs:
        raise typer.BadParameter(
            "Les bassins suivants sont absents du GeoJSON fourni : " + ", ".join(sorted(missing_basin_pairs))
        )
    runoff_range = _ensure_range("runoff", runoff_min, runoff_max)
    infiltration_range = _ensure_range("infiltration", infiltration_min, infiltration_max)
    evap_bounds = _ensure_range("multiplicateur ETP", evap_min, evap_max)
    leakage_fraction = _ensure_range("fuites", leakage_min, leakage_max)
    if not 0 < initial_storage_fraction <= 1:
        raise typer.BadParameter("--initial-storage-fraction doit être compris entre 0 et 1.")
    if evap_std < 0:
        raise typer.BadParameter("--evap-std doit être positif.")
    if runoff_range[1] > 1 or infiltration_range[1] > 1:
        raise typer.BadParameter("Les coefficients de ruissellement/infiltration doivent rester ≤ 1.")

    config = HydrologyModelConfig(
        iterations=iterations,
        seed=seed,
        runoff_range=runoff_range,
        runoff_alpha=runoff_alpha,
        runoff_beta=runoff_beta,
        infiltration_range=infiltration_range,
        infiltration_alpha=infiltration_alpha,
        infiltration_beta=infiltration_beta,
        evap_mean=evap_mean,
        evap_std=evap_std,
        evap_bounds=evap_bounds,
        leakage_fraction=leakage_fraction,
        initial_storage_fraction=initial_storage_fraction,
    )
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
        actual = _resolve_path(paths.root, output) or output
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

            output_path = _resolve_path(paths.root, sensitivity_output)
            if output_path is None and output:
                actual = _resolve_path(paths.root, output) or output
                suffix = actual.suffix or ".csv"
                output_path = actual.with_name(f"{actual.stem}_sensitivity{suffix}")

            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if output_path.suffix == ".parquet":
                    sensitivity_df.to_parquet(output_path)
                else:
                    sensitivity_df.to_csv(output_path, index=False)
                console.print(f"Indices sauvegardés dans {output_path}")


@app.command("hydro-figures")
def hydro_figures(
    root: Optional[Path] = typer.Option(None, "--root", help="Chemin vers la racine du dépôt"),
    summary: Path = typer.Option(Path("results/hydrology_summary.parquet"), help="Fichier de synthèse hydrologique"),
    sites_csv: Optional[Path] = typer.Option(None, "--sites", help="CSV des sites PHES"),
    figure_dir: Path = typer.Option(Path("results/figures"), help="Dossier de sortie pour les figures"),
    reviewer_csv: Path = typer.Option(Path("results/hydrology_summary_excerpt.csv"), help="CSV réduit pour la revue"),
) -> None:
    """Génère les figures hydrologiques et un CSV synthétique pour la revue."""

    paths = get_paths(root)
    summary_path = _resolve_path(paths.root, summary) or summary
    sites_path = _resolve_path(paths.root, sites_csv) if sites_csv else (paths.data_dir / DEFAULT_SITES_FILE)
    figure_dir_path = _resolve_path(paths.root, figure_dir) or figure_dir
    reviewer_csv_path = _resolve_path(paths.root, reviewer_csv) or reviewer_csv

    outputs = generate_all_figures(summary_path, sites_path, figure_dir_path, reviewer_csv_path)
    console.print("Figures et extrait CSV générés :")
    for label, generated_path in outputs.items():
        console.print(f" • {label}: {generated_path}")


@app.command("article-figures")
def article_figures(
    root: Optional[Path] = typer.Option(None, "--root", help="Chemin vers la racine du dépôt"),
    climate: Path = typer.Option(Path("results/climate_series.csv"), help="Séries climatiques agrégées"),
    summary: Path = typer.Option(Path("results/hydrology_summary.parquet"), help="Fichier de synthèse hydrologique"),
    sites_csv: Optional[Path] = typer.Option(None, "--sites", help="CSV des sites PHES"),
    ahp_rankings: Path = typer.Option(Path("results/ahp_rankings.parquet"), help="Classement AHP calculé"),
    figure_dir: Path = typer.Option(Path("results/figures"), help="Dossier de sortie pour les figures"),
    deterministic: Optional[Path] = typer.Option(
        None,
        "--deterministic",
        help="Fichier CSV/parquet contenant les bilans déterministes (facultatif)",
    ),
    figure7_pair: str = typer.Option(
        "n10_e001_RES31412 & n10_e001_RES31520",
        help="Identifiant du site utilisé pour la figure 7",
    ),
) -> None:
    """Génère les figures 1, 2, 3, 7, 7b, 8 et 9 décrites dans l'article."""

    paths = get_paths(root)
    climate_path = _resolve_path(paths.root, climate) or climate
    summary_path = _resolve_path(paths.root, summary) or summary
    sites_path = _resolve_path(paths.root, sites_csv) if sites_csv else (paths.data_dir / DEFAULT_SITES_FILE)
    ahp_path = _resolve_path(paths.root, ahp_rankings) or ahp_rankings
    figure_dir_path = _resolve_path(paths.root, figure_dir) or figure_dir
    deterministic_path = _resolve_path(paths.root, deterministic) if deterministic else None

    outputs = generate_article_figures(
        climate_path,
        summary_path,
        sites_path,
        ahp_path,
        figure_dir_path,
        figure7_pair=figure7_pair,
        deterministic_path=deterministic_path,
    )

    console.print("Figures générées :")
    for label, generated_path in outputs.items():
        console.print(f" • {label}: {generated_path}")


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
    weights_config: Optional[Path] = typer.Option(None, "--weights-config", help="JSON/YAML définissant uniquement les poids AHP"),
) -> None:
    """Classe les sites via un AHP simplifié reliant classe économique et hydrologie."""

    paths = get_paths(root)
    sites_path = _resolve_path(paths.root, sites_csv) if sites_csv else (paths.data_dir / DEFAULT_SITES_FILE)
    hydro_path = _resolve_path(paths.root, hydrology_summary) or hydrology_summary

    sites_df = load_sites(sites_path)
    hydro_df = load_ahp_summary(hydro_path)
    config_values = {
        "economic_weight": economic_weight,
        "hydrology_weight": hydrology_weight,
        "infrastructure_weight": infrastructure_weight,
    }

    if weights_config:
        config_path = _resolve_path(paths.root, weights_config) or weights_config
        overrides = _load_weights_config(config_path)
        unknown = [key for key in overrides if key not in config_values]
        if unknown:
            raise typer.BadParameter(
                "Clés non supportées dans --weights-config : " + ", ".join(sorted(unknown)),
                param_hint="--weights-config",
            )
        config_values.update(overrides)

    weights = AHPWeights(
        economic=float(config_values["economic_weight"]),
        hydrology=float(config_values["hydrology_weight"]),
        infrastructure=float(config_values["infrastructure_weight"]),
    )
    scores = compute_ahp_scores(
        sites_df,
        hydro_df,
        weights=weights,
    )

    table = Table(title="Classement AHP", header_style="bold green")
    table.add_column("Rang", justify="right")
    table.add_column("Site")
    table.add_column("Classe éco", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Score éco", justify="right")
    table.add_column("Score hydro", justify="right")
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
            f"{row['prob_positive_annual_balance']:.2f}" if not math.isnan(row["prob_positive_annual_balance"]) else "-",
            f"{row['dry_season_prob_positive']:.2f}" if not math.isnan(row["dry_season_prob_positive"]) else "-",
        )

    console.print(table)

    if output:
        actual = _resolve_path(paths.root, output) or output
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
