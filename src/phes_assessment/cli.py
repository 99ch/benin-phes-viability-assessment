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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
