"""Tests unitaires sur le module hydrologique."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phes_assessment.hydrology import (
    HydrologyModelConfig,
    SiteHydrologyParams,
    _scale_beta,
    run_hydrology_simulation_from_data,
    simulate_site,
)


def _dummy_site(pair: str = "SITE_A") -> SiteHydrologyParams:
    return SiteHydrologyParams(
        pair_identifier=pair,
        capacity_gl=5.0,
        area_ha=10.0,
        slope_percent=5.0,
        head_m=400.0,
        basin_area_km2=4.0,
    )


def _climate_series(pair: str, start: str, periods: int, precip_mm: float = 120.0, etp_mm: float = 40.0) -> pd.DataFrame:
    dates = pd.date_range(start=start, periods=periods, freq="MS")
    return pd.DataFrame(
        {
            "pair_identifier": pair,
            "date": dates,
            "precip_mm": np.full(periods, precip_mm),
            "etp_mm": np.full(periods, etp_mm),
        }
    )


def test_runoff_and_infiltration_respect_mass_conservation() -> None:
    """runoff + infiltration ne dépasse jamais 100 % de la lame d'eau."""

    rng = np.random.default_rng(1234)
    shape = (512, 24)
    runoff = _scale_beta(rng, shape, 0.3, 0.8, 3.5, 4.0)
    infiltration_potential = _scale_beta(rng, shape, 0.05, 0.25, 2.0, 5.0)
    available_fraction = np.clip(1.0 - runoff, 0.0, 1.0)
    infiltration = np.minimum(infiltration_potential, available_fraction)
    assert np.all(runoff + infiltration <= 1.0 + 1e-9)


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
    ) <= 1e-2


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