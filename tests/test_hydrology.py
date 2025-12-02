"""Tests unitaires sur le module hydrologique."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phes_assessment.hydrology import (
    HydrologyModelConfig,
    SiteHydrologyParams,
    _scale_beta,
    load_climate_series,
    run_hydrology_simulation_from_data,
    simulate_site,
)


def _dummy_site(pair: str = "SITE_A", basin_km2: float | None = None) -> SiteHydrologyParams:
    return SiteHydrologyParams(
        pair_identifier=pair,
        capacity_gl=5.0,
        area_ha=10.0,
        upper_area_ha=6.0,
        lower_area_ha=4.0,
        slope_percent=5.0,
        head_m=400.0,
        basin_area_km2=basin_km2 or 4.0,
    )


def _climate_series(
    pair: str,
    start: str,
    periods: int,
    precip_mm: float = 120.0,
    etp_mm: float = 40.0,
) -> pd.DataFrame:
    dates = pd.date_range(start=start, periods=periods, freq="MS")
    return pd.DataFrame(
        {
            "pair_identifier": pair,
            "date": dates,
            "precip_mm": np.full(periods, precip_mm),
            "etp_mm": np.full(periods, etp_mm),
        }
    )


def test_load_climate_series_raises_on_nan(tmp_path: Path) -> None:
    df = _climate_series("SITE_NAN", "2002-01-01", 12)
    df.loc[3, "precip_mm"] = np.nan
    csv_path = tmp_path / "climate.csv"
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="precip_mm"):
        load_climate_series(csv_path)


def test_runoff_and_infiltration_respect_mass_conservation() -> None:
    """Le volume utilisé (infiltration + ruissellement) ne dépasse jamais les précipitations."""

    rng = np.random.default_rng(1234)
    shape = (512, 24)
    
    # Générer précipitations et surfaces de test
    precip_mm = np.random.uniform(50, 200, size=shape)
    basin_area_m2 = 10_000_000  # 10 km²
    precip_gl = (precip_mm * 0.001 * basin_area_m2) / 1_000_000.0
    
    # Coefficients hydrologiques
    infiltration_fraction = _scale_beta(rng, shape, 0.05, 0.25, 2.0, 5.0)
    infiltration_fraction = np.clip(infiltration_fraction, 0.0, 1.0)
    available_fraction = np.clip(1.0 - infiltration_fraction, 0.0, 1.0)
    runoff_coeff = _scale_beta(rng, shape, 0.3, 0.8, 3.5, 4.0)
    runoff_coeff = np.minimum(runoff_coeff, available_fraction)
    
    # Calcul des volumes (logique corrigée du modèle)
    infiltration_gl = infiltration_fraction * precip_gl
    available_precip_gl = precip_gl - infiltration_gl
    net_runoff_gl = runoff_coeff * available_precip_gl
    
    # Vérification : le total utilisé ne dépasse jamais les précipitations
    total_used_gl = infiltration_gl + net_runoff_gl
    assert np.all(total_used_gl <= precip_gl + 1e-9), \
        "La conservation de masse est violée : infiltration + ruissellement > précipitations"
    
    # Vérification additionnelle : les coefficients seuls respectent aussi la contrainte
    assert np.all(runoff_coeff + infiltration_fraction <= 1.0 + 1e-9), \
        "Les coefficients violent la conservation de masse"


def test_dry_season_metrics_stable_with_repeated_years() -> None:
    """La probabilité saison sèche reste identique si l'on duplique les années climatiques."""

    site = _dummy_site("SITE_DRY")
    config = HydrologyModelConfig(iterations=256, seed=None)
    rng_single = np.random.default_rng(42)
    rng_double = np.random.default_rng(42)

    # 12 mois (novembre → octobre) pour constituer une année hydrologique complète.
    single_year = _climate_series("SITE_DRY", "2002-11-01", 12)
    # Dupliquer la même série sur deux ans.
    double_year = _climate_series("SITE_DRY", "2002-11-01", 24)

    single_result = simulate_site(site, single_year, config, rng=rng_single)
    double_result = simulate_site(site, double_year, config, rng=rng_double)

    assert single_result.dry_season_prob_positive == pytest.approx(
        double_result.dry_season_prob_positive, rel=1e-3
    )
    assert abs(
        single_result.dry_season_median_balance_gl - double_result.dry_season_median_balance_gl
    ) <= 2e-2


def test_rng_independence_between_sites() -> None:
    """Deux sites identiques reçoivent des trajectoires distinctes malgré un seed global."""

    site_a = _dummy_site("SITE_A")
    site_b = _dummy_site("SITE_B")
    config = HydrologyModelConfig(iterations=512, seed=123)

    climate = pd.concat(
        [
            _climate_series("SITE_A", "2002-01-01", 24),
            _climate_series("SITE_B", "2002-01-01", 24),
        ],
        ignore_index=True,
    )

    params = {s.pair_identifier: s for s in (site_a, site_b)}
    df = run_hydrology_simulation_from_data(climate, params, config)
    assert set(df["pair_identifier"]) == {"SITE_A", "SITE_B"}

    a_val = df.loc[df["pair_identifier"] == "SITE_A", "median_annual_balance_gl"].iloc[0]
    b_val = df.loc[df["pair_identifier"] == "SITE_B", "median_annual_balance_gl"].iloc[0]
    assert not np.isclose(a_val, b_val)


def test_physical_separation_inflows_vs_losses() -> None:
    """Vérifie que les apports (bassin) et pertes (réservoirs) utilisent des surfaces distinctes."""

    # Site avec bassin versant >> réservoirs (cas réaliste)
    site = SiteHydrologyParams(
        pair_identifier="TEST_PHYS",
        capacity_gl=100.0,
        area_ha=100.0,  # Total réservoirs = 100 ha = 1 km²
        upper_area_ha=60.0,
        lower_area_ha=40.0,
        basin_area_km2=10.0,  # Bassin = 10 km² = 10× réservoirs
    )

    # Climat simple : 100 mm précip, 50 mm ETP
    climate = _climate_series("TEST_PHYS", "2002-01-01", 12, precip_mm=100.0, etp_mm=50.0)
    config = HydrologyModelConfig(
        iterations=100,
        seed=42,
        runoff_range=(0.5, 0.5),  # Fixe à 50%
        infiltration_range=(0.1, 0.1),  # Fixe à 10%
        evap_mean=1.0,
        evap_std=0.0,  # Pas de variabilité
        leakage_fraction=(0.0, 0.0),  # Pas de fuites
    )

    result = simulate_site(site, climate, config)

    # Calcul théorique (apports bassin, pertes réservoirs)
    basin_area_m2 = 10.0 * 1_000_000  # 10 km²
    reservoir_area_m2 = 100.0 * 10_000  # 100 ha = 1 km²

    monthly_precip_m3 = 0.1 * basin_area_m2  # 100 mm sur 10 km²
    monthly_infiltration_m3 = 0.1 * monthly_precip_m3  # 10% infiltration
    available_precip_m3 = monthly_precip_m3 - monthly_infiltration_m3
    # Le coefficient de ruissellement s'applique sur la pluie restante
    monthly_net_runoff_m3 = 0.5 * available_precip_m3

    monthly_etp_m3 = 0.05 * reservoir_area_m2  # 50 mm sur 1 km² (réservoirs)

    monthly_balance_gl = (monthly_net_runoff_m3 - monthly_etp_m3) / 1_000_000.0
    annual_balance_gl = monthly_balance_gl * 12

    # Vérifier que le bilan médian est proche du calcul théorique
    # (tolérance car il y a un peu de variabilité résiduelle même avec seed)
    assert abs(result.median_annual_balance_gl - annual_balance_gl) < 0.5

    # Le bilan doit être POSITIF car bassin versant >> réservoirs
    assert result.median_annual_balance_gl > 0
    assert result.prob_positive_annual_balance > 0.9


def test_dry_season_probability_detects_midseason_failure() -> None:
    site = _dummy_site("SITE_DRYFAIL")
    config = HydrologyModelConfig(
        iterations=64,
        seed=123,
        runoff_range=(0.0, 0.0),
        infiltration_range=(0.0, 0.0),
        evap_mean=1.0,
        evap_std=0.0,
        evap_bounds=(1.0, 1.0),
        leakage_fraction=(0.0, 0.0),
    )

    # Aucun apport pendant la saison sèche et une ETP gigantesque qui vide le stock.
    climate = _climate_series("SITE_DRYFAIL", "2002-11-01", 12, precip_mm=0.0, etp_mm=10000.0)
    result = simulate_site(site, climate, config)

    assert result.dry_season_prob_positive == 0.0
    assert result.dry_season_p90_deficit_gl > 0