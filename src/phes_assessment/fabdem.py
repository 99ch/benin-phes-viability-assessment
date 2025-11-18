"""Utilitaires pour exploiter les tuiles FABDEM (30 m)."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import rasterio

from .config import DataPaths, get_paths

FABDEM_SUFFIX = "_FABDEM_V1-2.tif"
DEFAULT_SITES_FILE = "n10_e001_12_sites_complete.csv"


def _hemisphere_token(value: float, positive_token: str, negative_token: str, digits: int) -> str:
    hemisphere = positive_token if value >= 0 else negative_token
    magnitude = abs(int(math.floor(value)))
    return f"{hemisphere}{magnitude:0{digits}d}"


def tile_name(lat: float, lon: float) -> str:
    """Retourne le nom de la tuile FABDEM contenant la coordonnée fournie."""

    lat_token = _hemisphere_token(lat, "N", "S", 2)
    lon_token = _hemisphere_token(lon, "E", "W", 3)
    return f"{lat_token}{lon_token}{FABDEM_SUFFIX}"


def tile_path(paths: DataPaths, lat: float, lon: float) -> Path:
    """Calcule le chemin vers la tuile FABDEM correspondante."""

    candidate = paths.fabdem_dir / tile_name(lat, lon)
    if not candidate.exists():
        raise FileNotFoundError(f"Tuile FABDEM introuvable pour ({lat}, {lon}) : {candidate}")
    return candidate


def sample_elevation(lat: float, lon: float, tile: Path) -> float:
    """Lit l'altitude (en mètres) pour la coordonnée donnée."""

    with rasterio.open(tile) as dataset:
        value = next(dataset.sample([(lon, lat)]))[0]
        if value == dataset.nodata:
            raise ValueError(f"Valeur NoData rencontrée pour ({lat}, {lon}) dans {tile.name}")
        return float(value)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def sample_sites(paths: DataPaths | None = None, sites_csv: Path | None = None) -> pd.DataFrame:
    """Retourne le tableau des sites enrichi des altitudes FABDEM."""

    paths = paths or get_paths()
    csv_path = sites_csv or (paths.data_dir / DEFAULT_SITES_FILE)
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [col.strip() for col in df.columns]

    heads = _numeric(df["Head (m)"])
    upper_lats = _numeric(df["Upper latitude"])
    upper_lons = _numeric(df["Upper longitude"])
    lower_lats = _numeric(df["Lower latitude"])
    lower_lons = _numeric(df["Lower longitude"])

    upper_elevations: list[float] = []
    lower_elevations: list[float] = []

    for lat, lon in zip(upper_lats, upper_lons):
        tile = tile_path(paths, lat, lon)
        upper_elevations.append(sample_elevation(lat, lon, tile))

    for lat, lon in zip(lower_lats, lower_lons):
        tile = tile_path(paths, lat, lon)
        lower_elevations.append(sample_elevation(lat, lon, tile))

    df["Upper DEM elevation (m)"] = upper_elevations
    df["Lower DEM elevation (m)"] = lower_elevations
    df["DEM head (m)"] = df["Upper DEM elevation (m)"] - df["Lower DEM elevation (m)"]
    df["Head delta (m)"] = df["DEM head (m)"] - heads

    return df


def summarize_head_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Renvoie un résumé des différences entre le head catalogue et celui issu de FABDEM."""

    subset = df[[
        "Pair Identifier",
        "Head (m)",
        "DEM head (m)",
        "Head delta (m)",
    ]].copy()
    subset.rename(columns={"Head (m)": "Catalog head (m)"}, inplace=True)
    return subset
