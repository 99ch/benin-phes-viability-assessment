"""Classement AHP combinant critères économiques et hydrologiques."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_MULTIPLIERS: dict[str, float] = {
    "A": 1.00,
    "B": 1.25,
    "C": 1.50,
    "D": 1.75,
    "E": 2.00,
}
BASE_COST_PER_MW_USD = 530_000  # $/MW pour la classe A
BASE_COST_PER_MWH_USD = 47_000  # $/MWh pour la classe A
FIXED_OM_PER_MW_PER_YEAR = 8_210.0
VARIABLE_OM_PER_MWH = 0.30
DEFAULT_CYCLES_PER_YEAR = 300
DEFAULT_LIFETIME_YEARS = 60
DEFAULT_DISCOUNT_RATE = 0.05
DEFAULT_ROUND_TRIP_EFFICIENCY = 0.81


@dataclass(frozen=True)
class AHPWeights:
    """Poids des critères dans l'agrégation AHP."""

    economic: float = 0.4
    hydrology: float = 0.4
    infrastructure: float = 0.2


def load_hydrology_summary(path: Path) -> pd.DataFrame:
    """Charge le fichier de synthèse hydrologique (parquet ou CSV)."""

    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Fichier hydrologique introuvable: {resolved}")
    if resolved.suffix == ".parquet":
        return pd.read_parquet(resolved)
    return pd.read_csv(resolved)


def _normalize_series(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    values = pd.Series(series, dtype="float64")
    mask = values.notna()
    result = pd.Series(np.nan, index=values.index, dtype="float64")
    if mask.sum() <= 1:
        result.loc[mask] = 1.0
        return result

    subset = values[mask]
    min_val = float(subset.min())
    max_val = float(subset.max())
    if math.isclose(max_val, min_val):
        result.loc[mask] = 1.0
    else:
        result.loc[mask] = (subset - min_val) / (max_val - min_val)
    if not higher_is_better:
        result.loc[mask] = 1 - result.loc[mask]
    return result


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator in (0, 0.0):
        return float("nan")
    return numerator / denominator


def estimate_lcos(
    power_mw: float,
    energy_mwh: float,
    cost_per_mw_usd: float,
    cost_per_mwh_usd: float,
    *,
    cycles_per_year: int = DEFAULT_CYCLES_PER_YEAR,
    lifetime_years: int = DEFAULT_LIFETIME_YEARS,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
    round_trip_efficiency: float = DEFAULT_ROUND_TRIP_EFFICIENCY,
    fixed_om_per_mw: float = FIXED_OM_PER_MW_PER_YEAR,
    variable_om_per_mwh: float = VARIABLE_OM_PER_MWH,
) -> float:
    """Calcule le LCOS simplifié pour un site donné."""

    if power_mw <= 0 or energy_mwh <= 0:
        return float("nan")

    capex = cost_per_mw_usd * power_mw + cost_per_mwh_usd * energy_mwh
    annual_energy = energy_mwh * cycles_per_year * round_trip_efficiency
    if annual_energy <= 0:
        return float("nan")

    annual_fixed_om = fixed_om_per_mw * power_mw
    annual_variable_om = variable_om_per_mwh * annual_energy

    r = discount_rate
    n = lifetime_years
    crf = r * (1 + r) ** n / ((1 + r) ** n - 1)
    annualized_capex = capex * crf

    return (annualized_capex + annual_fixed_om + annual_variable_om) / annual_energy


def compute_ahp_scores(
    sites_df: pd.DataFrame,
    hydrology_df: pd.DataFrame,
    *,
    weights: AHPWeights | None = None,
    cycles_per_year: int = DEFAULT_CYCLES_PER_YEAR,
    lifetime_years: int = DEFAULT_LIFETIME_YEARS,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
    round_trip_efficiency: float = DEFAULT_ROUND_TRIP_EFFICIENCY,
) -> pd.DataFrame:
    """Fusionne les données sites/hydrologie et calcule les scores AHP."""

    weights = weights or AHPWeights()
    total_weight = weights.economic + weights.hydrology + weights.infrastructure
    if total_weight <= 0:
        raise ValueError("La somme des poids doit être strictement positive.")

    sites = sites_df.copy()
    sites.columns = [col.strip() for col in sites.columns]
    hydrology = hydrology_df.copy()
    hydrology.columns = [col.strip() for col in hydrology.columns]

    merged = sites.merge(
        hydrology,
        left_on="Pair Identifier",
        right_on="pair_identifier",
        how="left",
        suffixes=("", "_hydro"),
    )

    merged["class"] = merged["Class"].astype(str).str.strip().str.upper()
    merged["class_multiplier"] = merged["class"].map(CLASS_MULTIPLIERS)
    if merged["class_multiplier"].isna().any():
        missing = merged.loc[merged["class_multiplier"].isna(), "class"].unique()
        raise ValueError(f"Classes économiques inconnues: {missing}")

    merged["energy_capacity_mwh"] = merged["Energy (GWh)"].astype(float) * 1_000.0
    merged["storage_hours"] = merged["Storage time (h)"].astype(float)
    merged["power_rating_mw"] = merged.apply(
        lambda row: _safe_divide(row["energy_capacity_mwh"], row["storage_hours"]),
        axis=1,
    )

    merged["cost_per_mw_usd"] = BASE_COST_PER_MW_USD * merged["class_multiplier"]
    merged["cost_per_mwh_usd"] = BASE_COST_PER_MWH_USD * merged["class_multiplier"]
    merged["capex_total_usd"] = (
        merged["cost_per_mw_usd"] * merged["power_rating_mw"].fillna(0)
        + merged["cost_per_mwh_usd"] * merged["energy_capacity_mwh"].fillna(0)
    )
    merged["lcos_usd_per_mwh"] = merged.apply(
        lambda row: estimate_lcos(
            power_mw=row["power_rating_mw"],
            energy_mwh=row["energy_capacity_mwh"],
            cost_per_mw_usd=row["cost_per_mw_usd"],
            cost_per_mwh_usd=row["cost_per_mwh_usd"],
            cycles_per_year=cycles_per_year,
            lifetime_years=lifetime_years,
            discount_rate=discount_rate,
            round_trip_efficiency=round_trip_efficiency,
        ),
        axis=1,
    )

    merged["economic_score"] = _normalize_series(merged["lcos_usd_per_mwh"], higher_is_better=False)

    prob_positive = merged["prob_positive_annual_balance"].astype(float)
    dry_prob = merged["dry_season_prob_positive"].astype(float).fillna(prob_positive)
    hydro_metric = 0.6 * prob_positive + 0.4 * dry_prob
    merged["hydrology_score"] = _normalize_series(hydro_metric, higher_is_better=True)

    sep_score = _normalize_series(merged["Separation (km)"].astype(float), higher_is_better=False)
    slope_score = _normalize_series(merged["Slope (%)"].astype(float), higher_is_better=False)
    head_score = _normalize_series(merged["Head (m)"].astype(float), higher_is_better=True)
    water_rock_score = _normalize_series(
        merged["Combined water to rock ratio"].astype(float),
        higher_is_better=True,
    )

    infra_metric = (
        0.35 * head_score.fillna(0.5)
        + 0.35 * water_rock_score.fillna(0.5)
        + 0.2 * sep_score.fillna(0.5)
        + 0.1 * slope_score.fillna(0.5)
    )
    merged["infrastructure_score"] = _normalize_series(infra_metric, higher_is_better=True)

    merged["final_score"] = (
        weights.economic * merged["economic_score"].fillna(0)
        + weights.hydrology * merged["hydrology_score"].fillna(0)
        + weights.infrastructure * merged["infrastructure_score"].fillna(0)
    ) / total_weight

    merged["rank"] = merged["final_score"].rank(ascending=False, method="dense").astype(int)
    merged.sort_values(["final_score", "pair_identifier"], ascending=[False, True], inplace=True)
    merged.reset_index(drop=True, inplace=True)
    merged["capex_musd"] = merged["capex_total_usd"] / 1_000_000

    return merged[
        [
            "rank",
            "pair_identifier",
            "class",
            "final_score",
            "economic_score",
            "hydrology_score",
            "infrastructure_score",
            "lcos_usd_per_mwh",
            "capex_musd",
            "power_rating_mw",
            "energy_capacity_mwh",
            "prob_positive_annual_balance",
            "dry_season_prob_positive",
        ]
    ]
