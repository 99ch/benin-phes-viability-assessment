"""Interface en ligne de commande pour l'inventaire des données."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .catalog import summarize_directory
from .config import get_paths
from .fabdem import DEFAULT_SITES_FILE, sample_sites, summarize_head_gaps

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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
