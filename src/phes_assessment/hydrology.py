"""Modèle hydrologique stochastique pour les 12 sites PHES identifiés.

Le module applique un bilan hydrique mensuel simple basé sur les séries
climatiques agrégées (CHIRPS/ERA5) et simule des coefficients de ruissellement,
pertes par infiltration, évapotranspiration et fuites linéaires via des lois
aléatoires (Beta/Normal/Uniform). L'objectif est d'estimer la distribution du
volume net annuel et la probabilité d'autonomie hydrique (>0) pour chaque site.

Hypothèses principales :
- Les précipitations sont converties en mètres puis en volume à partir de
  l'emprise (ha) des réservoirs upper + lower.
- Les coefficients de ruissellement et d'infiltration sont tirés par mois et par
  simulation (500-5000 tirages typiques).
- L'évaporation potentielle ERA5 est appliquée via un multiplicateur normal
  (±10 %) pour représenter les incertitudes locales.
- Des fuites linéaires (0.05-0.2 % du volume) sont soustraites mensuellement.
- On suit l'évolution du stock (borné entre 0 et la capacité GL) et on calcule
  des indicateurs annuels/saisonniers.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .sites import load_sites

GL_IN_M3 = 1_000_000.0
HA_IN_M2 = 10_000.0
DRY_MONTHS = {11, 12, 1, 2, 3}


@dataclass
class SiteHydrologyParams:
    pair_identifier: str
    capacity_gl: float
    area_ha: float
    slope_percent: float
    head_m: float
    basin_area_km2: float | None = None

    @property
    def reservoir_area_m2(self) -> float:
        return self.area_ha * HA_IN_M2

    @property
    def catchment_area_m2(self) -> float:
        if self.basin_area_km2 and self.basin_area_km2 > 0:
            return self.basin_area_km2 * 1_000_000.0
        return self.reservoir_area_m2


@dataclass
class HydrologyModelConfig:
    iterations: int = 10000
    seed: int | None = 42
    runoff_range: Tuple[float, float] = (0.3, 0.8)
    runoff_alpha: float = 3.5
    runoff_beta: float = 4.0
    infiltration_range: Tuple[float, float] = (0.05, 0.25)
    infiltration_alpha: float = 2.0
    infiltration_beta: float = 5.0
    evap_mean: float = 1.0
    evap_std: float = 0.1
    evap_bounds: Tuple[float, float] = (0.5, 1.5)
    leakage_fraction: Tuple[float, float] = (0.0005, 0.002)
    initial_storage_fraction: float = 0.6


@dataclass
class HydrologySimulationResult:
    pair_identifier: str
    capacity_gl: float
    median_annual_balance_gl: float
    p10_annual_balance_gl: float
    p90_annual_balance_gl: float
    prob_positive_annual_balance: float
    prob_storage_never_empty: float
    dry_season_prob_positive: float
    dry_season_p10_gl: float
    dry_season_median_balance_gl: float
    dry_season_median_deficit_gl: float
    dry_season_p90_deficit_gl: float


def load_site_parameters(
    csv_path: Path,
    basin_areas_m2: Dict[str, float] | None = None,
) -> Dict[str, SiteHydrologyParams]:
    df = load_sites(csv_path)
    params: Dict[str, SiteHydrologyParams] = {}
    for _, row in df.iterrows():
        upper_area = float(row.get("Upper reservoir area (ha)", 0) or 0)
        lower_area = float(row.get("Lower reservoir area (ha)", 0) or 0)
        total_area = max(upper_area + lower_area, 1.0)
        capacity = float(row.get("Volume (GL)", row.get("Upper reservoir volume (GL)", 0)))
        pair_id = str(row["Pair Identifier"]).strip()
        basin_area_km2 = None
        if basin_areas_m2:
            catchment_m2 = float(basin_areas_m2.get(pair_id, 0.0) or 0.0)
            if catchment_m2 > 0:
                basin_area_km2 = catchment_m2 / 1_000_000.0
        params[pair_id] = SiteHydrologyParams(
            pair_identifier=pair_id,
            capacity_gl=max(capacity, 1.0),
            area_ha=total_area,
            slope_percent=float(row.get("Slope (%)", 0) or 0),
            head_m=float(row.get("Head (m)", 0) or 0),
            basin_area_km2=basin_area_km2,
        )
    return params


def load_climate_series(climate_path: Path) -> pd.DataFrame:
    df = pd.read_csv(climate_path)
    df["date"] = pd.to_datetime(df["date"])  # type: ignore[assignment]
    df.sort_values(["pair_identifier", "date"], inplace=True)
    return df


def _to_meters(series: pd.Series) -> np.ndarray:
    values = series.fillna(0).astype(float).to_numpy()
    max_abs = np.nanmax(np.abs(values)) if len(values) else 0.0
    if max_abs > 10:
        # Valeurs en millimètres -> convertir en mètres
        return values / 1000.0
    return values


def _scale_beta(rng: np.random.Generator, shape: Tuple[int, ...], low: float, high: float, alpha: float, beta: float) -> np.ndarray:
    samples = rng.beta(alpha, beta, size=shape)
    return low + samples * (high - low)


def _annual_groups(dates: np.ndarray) -> Dict[int, np.ndarray]:
    years = pd.to_datetime(dates).year
    groups: Dict[int, np.ndarray] = {}
    for year in np.unique(years):
        groups[int(year)] = np.where(years == year)[0]
    return groups


def _season_mask(dates: np.ndarray, months: Iterable[int]) -> np.ndarray:
    months_arr = pd.to_datetime(dates).month
    mask = np.isin(months_arr, list(months))
    return np.where(mask)[0]


def simulate_site(
    site: SiteHydrologyParams,
    climate_df: pd.DataFrame,
    config: HydrologyModelConfig,
) -> HydrologySimulationResult:
    df_site = climate_df[climate_df["pair_identifier"] == site.pair_identifier]
    if df_site.empty:
        raise ValueError(f"Aucune série climatique pour {site.pair_identifier}")

    precip_m = _to_meters(df_site["precip_mm"])
    etp_m = np.abs(_to_meters(df_site["etp_mm"]))
    dates = df_site["date"].to_numpy()

    area_m2 = site.catchment_area_m2
    precip_gl = (precip_m * area_m2) / GL_IN_M3
    etp_gl = (etp_m * area_m2) / GL_IN_M3

    n_months = len(df_site)
    n_sim = config.iterations
    rng = np.random.default_rng(config.seed)

    runoff = _scale_beta(
        rng,
        (n_sim, n_months),
        config.runoff_range[0],
        config.runoff_range[1],
        config.runoff_alpha,
        config.runoff_beta,
    )
    infiltration_potential = _scale_beta(
        rng,
        (n_sim, n_months),
        config.infiltration_range[0],
        config.infiltration_range[1],
        config.infiltration_alpha,
        config.infiltration_beta,
    )
    available_fraction = np.clip(1.0 - runoff, 0.0, 1.0)
    infiltration = np.minimum(infiltration_potential, available_fraction)
    evap_multiplier = np.clip(rng.normal(config.evap_mean, config.evap_std, size=(n_sim, n_months)), config.evap_bounds[0], config.evap_bounds[1])
    leakage = rng.uniform(config.leakage_fraction[0], config.leakage_fraction[1], size=(n_sim, n_months)) * site.capacity_gl

    runoff_gl = runoff * precip_gl
    infiltration_gl = infiltration * precip_gl
    evap_gl = evap_multiplier * etp_gl
    net_gl = runoff_gl - infiltration_gl - evap_gl - leakage

    storage = np.full(n_sim, site.capacity_gl * config.initial_storage_fraction)
    never_empty = np.ones(n_sim, dtype=bool)

    for idx in range(n_months):
        storage = storage + net_gl[:, idx]
        storage = np.clip(storage, 0.0, site.capacity_gl)
        never_empty &= storage > 0

    annual_indices = _annual_groups(dates)
    annual_balances = []
    for idx in annual_indices.values():
        annual_balances.append(net_gl[:, idx].sum(axis=1))
    annual_balances_arr = np.stack(annual_balances, axis=1)
    annual_flat = annual_balances_arr.flatten()

    dry_idx = _season_mask(dates, DRY_MONTHS)
    if dry_idx.size:
        dry_balances = net_gl[:, dry_idx].sum(axis=1)
        dry_prob_positive = float((dry_balances > 0).mean())
        dry_p10 = float(np.percentile(dry_balances, 10))
        dry_median = float(np.median(dry_balances))
        dry_deficits = np.clip(-dry_balances, 0.0, None)
        dry_median_deficit = float(np.median(dry_deficits))
        dry_p90_deficit = float(np.percentile(dry_deficits, 90))
    else:
        dry_prob_positive = float("nan")
        dry_p10 = float("nan")
        dry_median = float("nan")
        dry_median_deficit = float("nan")
        dry_p90_deficit = float("nan")

    result = HydrologySimulationResult(
        pair_identifier=site.pair_identifier,
        capacity_gl=site.capacity_gl,
        median_annual_balance_gl=float(np.median(annual_flat)),
        p10_annual_balance_gl=float(np.percentile(annual_flat, 10)),
        p90_annual_balance_gl=float(np.percentile(annual_flat, 90)),
        prob_positive_annual_balance=float((annual_flat > 0).mean()),
        prob_storage_never_empty=float(never_empty.mean()),
        dry_season_prob_positive=dry_prob_positive,
        dry_season_p10_gl=dry_p10,
        dry_season_median_balance_gl=dry_median,
        dry_season_median_deficit_gl=dry_median_deficit,
        dry_season_p90_deficit_gl=dry_p90_deficit,
    )
    return result


def run_hydrology_simulation_from_data(
    climate_df: pd.DataFrame,
    params: Dict[str, SiteHydrologyParams],
    config: HydrologyModelConfig,
) -> pd.DataFrame:
    results: List[dict] = []
    for site in params.values():
        result = simulate_site(site, climate_df, config)
        results.append(asdict(result))
    return pd.DataFrame(results)


def run_hydrology_simulation(
    climate_path: Path,
    sites_csv: Path,
    iterations: int = 10000,
    seed: int | None = 42,
    basin_areas_m2: Dict[str, float] | None = None,
) -> pd.DataFrame:
    climate_df = load_climate_series(climate_path)
    params = load_site_parameters(sites_csv, basin_areas_m2=basin_areas_m2)
    config = HydrologyModelConfig(iterations=iterations, seed=seed)
    return run_hydrology_simulation_from_data(climate_df, params, config)


def scaled_config(
    config: HydrologyModelConfig,
    runoff_scale: float = 1.0,
    infiltration_scale: float = 1.0,
    evap_scale: float = 1.0,
    leakage_scale: float = 1.0,
) -> HydrologyModelConfig:
    def _scale_bounds(bounds: Tuple[float, float], scale: float, upper: float = 1.0) -> Tuple[float, float]:
        low = max(0.0, min(upper, bounds[0] * scale))
        high = max(low + 1e-6, min(upper, bounds[1] * scale))
        return low, high

    return HydrologyModelConfig(
        iterations=config.iterations,
        seed=config.seed,
        runoff_range=_scale_bounds(config.runoff_range, runoff_scale),
        runoff_alpha=config.runoff_alpha,
        runoff_beta=config.runoff_beta,
        infiltration_range=_scale_bounds(config.infiltration_range, infiltration_scale),
        infiltration_alpha=config.infiltration_alpha,
        infiltration_beta=config.infiltration_beta,
        evap_mean=config.evap_mean * evap_scale,
        evap_std=config.evap_std,
        evap_bounds=_scale_bounds(config.evap_bounds, evap_scale, upper=config.evap_bounds[1] * 1.5),
        leakage_fraction=_scale_bounds(config.leakage_fraction, leakage_scale, upper=config.leakage_fraction[1] * 5),
        initial_storage_fraction=config.initial_storage_fraction,
    )