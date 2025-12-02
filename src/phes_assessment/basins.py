"""Delineation des bassins versants autour des sites PHES."""
from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio import windows
from rasterio.features import shapes
from shapely.geometry import shape as shapely_shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform, unary_union
from pyproj import Transformer

from .config import DataPaths, get_paths
from .crs import utm_epsg_for_point
from .fabdem import DEFAULT_SITES_FILE, tile_path
from .sites import load_sites

try:  # pragma: no cover - dépendance optionnelle
    from whitebox.whitebox_tools import WhiteboxTools
    _HAS_WHITEBOX = True
except ImportError:  # pragma: no cover
    WhiteboxTools = None  # type: ignore[assignment,misc]
    _HAS_WHITEBOX = False


def _ensure_whitebox():  # type: ignore[no-untyped-def]
    if not _HAS_WHITEBOX:  # pragma: no cover - dépendance manquante
        raise RuntimeError(
            "Le module 'whitebox' est requis pour dériver les bassins versants. "
            "Installez-le via 'pip install whitebox' ou 'pip install -e .[full]'."
        )
    return WhiteboxTools()


def _lat_lon_buffer(lat: float, margin_km: float) -> Tuple[float, float]:
    margin_m = margin_km * 1000.0
    deg_lat = margin_m / 111_320.0
    cos_lat = max(math.cos(math.radians(lat)), 1e-3)
    deg_lon = margin_m / (111_320.0 * cos_lat)
    return deg_lat, deg_lon


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value)
    slug = slug.strip("_")
    return slug or "site"


def _clip_dem(tile: Path, bounds: Tuple[float, float, float, float], output: Path) -> Path:
    left, bottom, right, top = bounds
    with rasterio.open(tile) as src:
        tile_bounds = src.bounds
        if left < tile_bounds.left or right > tile_bounds.right or bottom < tile_bounds.bottom or top > tile_bounds.top:
            raise RuntimeError(
                f"La marge demandée dépasse l'emprise de la tuile FABDEM ({tile.name}). Réduisez --dem-margin-km ou ajoutez les tuiles adjacentes."
            )
        window = windows.from_bounds(left, bottom, right, top, transform=src.transform)
        window = window.round_offsets().round_lengths()
        window = window.intersection(windows.Window(col_off=0, row_off=0, width=src.width, height=src.height))
        data = src.read(1, window=window, boundless=True, fill_value=src.nodata or -9999)
        transform = src.window_transform(window)
        meta = src.meta.copy()
        meta.update(
            {
                "height": data.shape[0],
                "width": data.shape[1],
                "transform": transform,
                "count": 1,
                "dtype": "float32",
                "nodata": src.nodata if src.nodata is not None else -9999.0,
            }
        )
        data = data.astype("float32")
        output.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output, "w", **meta) as dst:
            dst.write(data, 1)
    return output


def _write_pour_points(reference: Path, lon: float, lat: float, output: Path, pour_id: int) -> Path:
    with rasterio.open(reference) as src:
        meta = src.meta.copy()
        meta.update({"dtype": "int32", "nodata": 0, "compress": "lzw"})
        arr = np.zeros((src.height, src.width), dtype=np.int32)
        row, col = rasterio.transform.rowcol(src.transform, lon, lat)
        row = int(np.clip(row, 0, src.height - 1))
        col = int(np.clip(col, 0, src.width - 1))
        arr[row, col] = pour_id
        with rasterio.open(output, "w", **meta) as dst:
            dst.write(arr, 1)
    return output


@dataclass
class SiteBasin:
    pair_identifier: str
    geometry: BaseGeometry  # EPSG:4326
    area_m2: float
    utm_epsg: int
    source_tile: str
    upper_lat: float
    upper_lon: float
    dem_margin_km: float

    @property
    def area_km2(self) -> float:
        return self.area_m2 / 1_000_000.0

    def as_feature(self) -> Dict[str, Any]:
        return {
            "type": "Feature",
            "properties": {
                "pair_identifier": self.pair_identifier,
                "basin_area_km2": self.area_km2,
                "utm_epsg": self.utm_epsg,
                "source_tile": self.source_tile,
                "upper_lat": self.upper_lat,
                "upper_lon": self.upper_lon,
                "dem_margin_km": self.dem_margin_km,
            },
            "geometry": self.geometry.__geo_interface__,
        }


def _watershed_geometry(watershed_raster: Path) -> BaseGeometry:
    with rasterio.open(watershed_raster) as src:
        data = src.read(1)
        mask = data > 0
        if not np.any(mask):
            raise RuntimeError("Aucun pixel de bassin versant détecté dans la sortie Whitebox")
        geom_list = []
        for geom, value in shapes(data.astype(np.int32), mask=mask, transform=src.transform):
            if value > 0:
                geom_list.append(shapely_shape(geom))
        if not geom_list:
            raise RuntimeError("Impossible de vectoriser le bassin versant (géométrie vide)")
        union = unary_union(geom_list)
        return union.buffer(0)


def delineate_site_basin(
    paths: DataPaths,
    pair_identifier: str,
    lat: float,
    lon: float,
    margin_km: float = 15.0,
    work_dir: Path | None = None,
) -> SiteBasin:
    wbt = _ensure_whitebox()
    dem_dir = Path(tempfile.mkdtemp()) if work_dir is None else Path(work_dir)
    dem_dir = dem_dir.resolve()
    cleanup_dir = work_dir is None
    dem_dir.mkdir(parents=True, exist_ok=True)
    try:
        slug = _slugify(pair_identifier)
        site_dir = dem_dir / slug
        if site_dir.exists():
            shutil.rmtree(site_dir)
        site_dir.mkdir(parents=True, exist_ok=True)
        wbt.work_dir = str(site_dir)
        tile = tile_path(paths, lat, lon)
        deg_lat, deg_lon = _lat_lon_buffer(lat, margin_km)
        bounds = (lon - deg_lon, lat - deg_lat, lon + deg_lon, lat + deg_lat)
        dem_clip = site_dir / "dem_clip.tif"
        filled = site_dir / "dem_filled.tif"
        pointer = site_dir / "pointer.tif"
        pour = site_dir / "pour.tif"
        watershed = site_dir / "watershed.tif"

        _clip_dem(tile, bounds, dem_clip)
        _write_pour_points(dem_clip, lon, lat, pour, pour_id=1)

        if wbt.fill_depressions(dem=str(dem_clip), output=str(filled), fix_flats=True) is False:  # pragma: no cover
            raise RuntimeError(f"Whitebox fill_depressions a échoué pour {pair_identifier}")
        if wbt.d8_pointer(dem=str(filled), output=str(pointer), esri_pntr=False) is False:  # pragma: no cover
            raise RuntimeError(f"Whitebox d8_pointer a échoué pour {pair_identifier}")
        if wbt.watershed(d8_pntr=str(pointer), pour_pts=str(pour), output=str(watershed)) is False:  # pragma: no cover
            raise RuntimeError(f"Whitebox watershed a échoué pour {pair_identifier}")
        if not watershed.exists():
            raise RuntimeError(f"Whitebox n'a pas généré de raster de bassin versant pour {pair_identifier}")

        geometry = _watershed_geometry(watershed)
        utm_epsg = utm_epsg_for_point(lat, lon)
        to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
        geom_utm = shapely_transform(to_utm.transform, geometry)
        
        # Calcul de l'aire en projection UTM
        # LIMITATION : La projection UTM introduit une distorsion d'aire qui augmente
        # avec la distance au méridien central de la zone. Pour le Bénin (6-12°N),
        # l'erreur est typiquement < 1% près du méridien central mais peut atteindre
        # 2-3% aux bords de la zone UTM. Pour une précision maximale, une projection
        # équivalente (Equal Area) comme Albers ou Lambert Azimuthal pourrait être
        # utilisée, mais l'erreur UTM reste acceptable pour une étude de faisabilité.
        # Référence : Snyder (1987), Map Projections - A Working Manual, USGS PP 1395
        area_m2 = geom_utm.area
        if area_m2 <= 0:
            raise RuntimeError(f"Surface du bassin non valide pour {pair_identifier}")

        return SiteBasin(
            pair_identifier=pair_identifier,
            geometry=geometry,
            area_m2=float(area_m2),
            utm_epsg=utm_epsg,
            source_tile=tile.name,
            upper_lat=lat,
            upper_lon=lon,
            dem_margin_km=margin_km,
        )
    finally:
        if cleanup_dir:
            shutil.rmtree(dem_dir, ignore_errors=True)


def delineate_basins(
    paths: DataPaths | None = None,
    sites_csv: Path | None = None,
    margin_km: float = 15.0,
    work_dir: Path | None = None,
) -> List[SiteBasin]:
    paths = paths or get_paths()
    csv_path = sites_csv or (paths.data_dir / DEFAULT_SITES_FILE)
    df = load_sites(csv_path)
    upper_lat = pd.to_numeric(df["Upper latitude"], errors="coerce")
    upper_lon = pd.to_numeric(df["Upper longitude"], errors="coerce")
    cleanup_base = False
    if work_dir is None:
        tmp_dir = Path(tempfile.mkdtemp())
        cleanup_base = True
    else:
        tmp_dir = Path(work_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

    basins: List[SiteBasin] = []
    try:
        for idx, row in df.iterrows():
            pair = str(row["Pair Identifier"]).strip()
            lat = float(upper_lat.iloc[idx])
            lon = float(upper_lon.iloc[idx])
            if math.isnan(lat) or math.isnan(lon):
                raise ValueError(f"Coordonnées upper manquantes pour {pair}")
            basin = delineate_site_basin(paths, pair, lat, lon, margin_km=margin_km, work_dir=tmp_dir)
            basins.append(basin)
    finally:
        if cleanup_base:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    return basins


def basins_feature_collection(
    basins: Iterable[SiteBasin],
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    features = [basin.as_feature() for basin in basins]
    collection: Dict[str, Any] = {
        "type": "FeatureCollection",
        "features": features,
    }
    if metadata:
        collection["metadata"] = metadata
    return collection


def export_basins_geojson(
    output: Path,
    basins: List[SiteBasin],
    metadata: Dict[str, Any] | None = None,
    indent: int = 2,
) -> Path:
    enriched = dict(metadata) if metadata else {}
    enriched.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
    enriched.setdefault("feature_count", len(basins))
    collection = basins_feature_collection(basins, metadata=enriched)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(collection, ensure_ascii=False, indent=indent), encoding="utf-8")
    return output


def load_site_basins(path: Path) -> Dict[str, SiteBasin]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier GeoJSON introuvable : {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    basins: Dict[str, SiteBasin] = {}
    for feature in data.get("features", []):
        geometry_dict = feature.get("geometry")
        if not geometry_dict:
            continue
        geometry = shapely_shape(geometry_dict)
        props = feature.get("properties", {})
        pair = str(props.get("pair_identifier", ""))
        if not pair:
            continue
        area_km2 = float(props.get("basin_area_km2", 0.0) or 0.0)
        utm_epsg = int(props.get("utm_epsg", 0) or 0)
        upper_lat = float(props.get("upper_lat", geometry.centroid.y))
        upper_lon = float(props.get("upper_lon", geometry.centroid.x))
        if utm_epsg == 0:
            utm_epsg = utm_epsg_for_point(upper_lat, upper_lon)
        area_m2 = area_km2 * 1_000_000.0
        if area_m2 <= 0:
            to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
            area_m2 = shapely_transform(to_utm.transform, geometry).area
        basins[pair] = SiteBasin(
            pair_identifier=pair,
            geometry=geometry,
            area_m2=area_m2,
            utm_epsg=utm_epsg,
            source_tile=str(props.get("source_tile", "")),
            upper_lat=upper_lat,
            upper_lon=upper_lon,
            dem_margin_km=float(props.get("dem_margin_km", 0.0) or 0.0),
        )
    if not basins:
        raise ValueError(f"Aucun bassin valide trouvé dans {path}")
    return basins
