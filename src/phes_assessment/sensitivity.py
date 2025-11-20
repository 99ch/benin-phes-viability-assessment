from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd

from .hydrology import (
    HydrologyModelConfig,
    SiteHydrologyParams,
    scaled_config,
    simulate_site,
)

try:  # pragma: no cover - optional dependency
    from SALib.analyze import morris as morris_analyze
    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import morris as morris_sample
    from SALib.sample import saltelli

    HAS_SALIB = True
except ImportError:  # pragma: no cover - handled at runtime
    HAS_SALIB = False


SENSITIVITY_PROBLEM = {
    "num_vars": 4,
    "names": [
        "runoff_scale",
        "infiltration_scale",
        "evap_scale",
        "leakage_scale",
    ],
    "bounds": [
        [0.7, 1.3],
        [0.7, 1.4],
        [0.8, 1.2],
        [0.6, 1.5],
    ],
}

MetricType = Literal[
    "median_balance",
    "prob_positive",
    "dry_median_balance",
    "dry_p90_deficit",
]
MethodType = Literal["sobol", "morris"]


@dataclass
class SensitivityRecord:
    pair_identifier: str
    method: MethodType
    metric: MetricType
    values: Dict[str, float]
    totals: Optional[Dict[str, float]] = None
    mu_star: Optional[Dict[str, float]] = None
    sigma: Optional[Dict[str, float]] = None


def _metric_value(result, metric: MetricType) -> float:
    if metric == "median_balance":
        return result.median_annual_balance_gl
    if metric == "prob_positive":
        return result.prob_positive_annual_balance
    if metric == "dry_median_balance":
        return result.dry_season_median_balance_gl
    if metric == "dry_p90_deficit":
        return result.dry_season_p90_deficit_gl
    raise ValueError(f"Metric inconnue: {metric}")


def _run_single_site(
    site: SiteHydrologyParams,
    climate_df: pd.DataFrame,
    base_config: HydrologyModelConfig,
    method: MethodType,
    samples: int,
    metric: MetricType,
    iterations: Optional[int] = None,
    seed: Optional[int] = None,
) -> SensitivityRecord:
    if not HAS_SALIB:
        raise ImportError("SALib n'est pas installé, impossible de calculer la sensibilité.")

    config = base_config
    if iterations is not None:
        config = replace(config, iterations=iterations)

    rng_seed = seed if seed is not None else (config.seed or 0)

    if method == "sobol":
        param_values = saltelli.sample(SENSITIVITY_PROBLEM, samples, calc_second_order=False)
    else:
        param_values = morris_sample.sample(
            SENSITIVITY_PROBLEM,
            N=samples,
            num_levels=4,
            seed=seed,
        )

    outputs: list[float] = []
    for idx, sample in enumerate(param_values):
        scaled = scaled_config(
            config,
            runoff_scale=sample[0],
            infiltration_scale=sample[1],
            evap_scale=sample[2],
            leakage_scale=sample[3],
        )
        scaled = replace(scaled, seed=rng_seed + idx)
        result = simulate_site(site, climate_df, scaled)
        outputs.append(_metric_value(result, metric))

    outputs_arr = np.asarray(outputs)

    if method == "sobol":
        analysis = sobol_analyze.analyze(
            SENSITIVITY_PROBLEM,
            outputs_arr,
            calc_second_order=False,
        )
        first_order = {
            name: float(val) for name, val in zip(SENSITIVITY_PROBLEM["names"], analysis["S1"], strict=True)
        }
        total_order = {
            name: float(val) for name, val in zip(SENSITIVITY_PROBLEM["names"], analysis["ST"], strict=True)
        }
        return SensitivityRecord(
            pair_identifier=site.pair_identifier,
            method=method,
            metric=metric,
            values=first_order,
            totals=total_order,
        )

    analysis = morris_analyze.analyze(
        SENSITIVITY_PROBLEM,
        param_values,
        outputs_arr,
        num_levels=4,
        grid_jump=1,
    )
    mu_star = {
        name: float(val) for name, val in zip(SENSITIVITY_PROBLEM["names"], analysis["mu_star"], strict=True)
    }
    sigma = {
        name: float(val) for name, val in zip(SENSITIVITY_PROBLEM["names"], analysis["sigma"], strict=True)
    }
    return SensitivityRecord(
        pair_identifier=site.pair_identifier,
        method=method,
        metric=metric,
        values=mu_star,
        sigma=sigma,
    )


def run_sensitivity_analysis(
    climate_df: pd.DataFrame,
    params: Dict[str, SiteHydrologyParams],
    config: HydrologyModelConfig,
    method: Optional[MethodType],
    samples: int,
    metric: MetricType,
    iterations: Optional[int] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    if method is None:
        raise ValueError("Le paramètre method ne peut pas être None pour une analyse de sensibilité.")

    records: list[dict[str, float | str]] = []
    for site in params.values():
        summary = _run_single_site(
            site,
            climate_df,
            config,
            method=method,
            samples=samples,
            metric=metric,
            iterations=iterations,
            seed=seed,
        )
        base: dict[str, float | str] = {
            "pair_identifier": summary.pair_identifier,
            "method": summary.method,
            "metric": summary.metric,
        }
        prefix = "S1" if method == "sobol" else "mu_star"
        for name, value in summary.values.items():
            base[f"{name}_{prefix}"] = value
        if summary.totals:
            for name, value in summary.totals.items():
                base[f"{name}_ST"] = value
        if summary.sigma:
            for name, value in summary.sigma.items():
                base[f"{name}_sigma"] = value
        records.append(base)

    return pd.DataFrame(records)
