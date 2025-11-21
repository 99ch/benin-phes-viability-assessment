"""Helpers for coordinate reference system selections (UTM zones, etc.)."""
from __future__ import annotations

import math


def utm_epsg_for_point(lat: float, lon: float) -> int:
    """Return the EPSG code of the UTM zone covering the given coordinate."""

    zone = int((lon + 180.0) / 6.0) + 1
    if lat >= 0:
        return 32600 + zone
    return 32700 + zone
