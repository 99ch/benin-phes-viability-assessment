"""Utilitaires liés aux sites PHES (géométries, buffers, etc.)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform, unary_union

SITE_COORD_COLUMNS = {
    "upper": ("Upper latitude", "Upper longitude"),
    "lower": ("Lower latitude", "Lower longitude"),
}


@dataclass
class SiteMask:
    """Représente un polygone (buffer) autour d'un site PHES."""

    pair_identifier: str
    geometry: BaseGeometry

    def as_geojson(self) -> Dict:
        return {
            "type": "Feature",
            "properties": {"pair_identifier": self.pair_identifier},
            "geometry": self.geometry.__geo_interface__,
        }


def load_sites(csv_path: Path) -> pd.DataFrame:
    """Charge le CSV des sites en nettoyant les entêtes."""

    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [col.strip() for col in df.columns]
    return df


def _buffer_point(lat: float, lon: float, buffer_meters: float, to_proj: Transformer, to_geo: Transformer) -> BaseGeometry:
    point = Point(lon, lat)
    projected = transform(to_proj.transform, point)
    buffered = projected.buffer(buffer_meters)
    return transform(to_geo.transform, buffered)


def build_site_masks(df: pd.DataFrame, buffer_meters: float = 500.0, utm_epsg: int = 32631) -> List[SiteMask]:
    """Construit des buffers simples autour des réservoirs upper/lower et les fusionne par paire."""

    to_proj = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    to_geo = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)

    masks: List[SiteMask] = []
    for _, row in df.iterrows():
        buffers = []
        for lat_col, lon_col in SITE_COORD_COLUMNS.values():
            lat = float(row[lat_col])
            lon = float(row[lon_col])
            buffers.append(_buffer_point(lat, lon, buffer_meters, to_proj, to_geo))
        geometry = unary_union(buffers)
        masks.append(SiteMask(pair_identifier=str(row["Pair Identifier"]), geometry=geometry))
    return masks


def feature_collection_from_masks(
    masks: Iterable[SiteMask],
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Construit une FeatureCollection GeoJSON optionnellement enrichie de métadonnées."""

    mask_list = list(masks)
    features = [mask.as_geojson() for mask in mask_list]
    collection: Dict[str, Any] = {
        "type": "FeatureCollection",
        "features": features,
    }
    if features:
        union = unary_union([mask.geometry for mask in mask_list])
        if not union.is_empty:
            collection["bbox"] = list(union.bounds)
    if metadata:
        collection["metadata"] = metadata
    return collection


def export_masks_geojson(
    output: Path,
    masks: List[SiteMask],
    metadata: Dict[str, Any] | None = None,
    indent: int = 2,
) -> Path:
    """Exporte une liste de masques vers un fichier GeoJSON."""

    enriched_metadata = dict(metadata) if metadata else {}
    enriched_metadata.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
    collection = feature_collection_from_masks(masks, metadata=enriched_metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(collection, ensure_ascii=False, indent=indent), encoding="utf-8")
    return output
