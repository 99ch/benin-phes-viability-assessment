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

# Définition du problème d'analyse de sensibilité globale
# Les bornes représentent des facteurs multiplicatifs appliqués aux paramètres
# de base du modèle hydrologique. Ces plages sont des HYPOTHÈSES D'EXPERT
# reflétant l'incertitude sur chaque processus hydrologique.
#
# JUSTIFICATION DÉTAILLÉE DES BORNES :
#
# 1. runoff_scale [0.7, 1.3] (±30%) :
#    - Descroix et al. (2010, Tableau 2, p.178) rapportent coefficients de
#      ruissellement de 0.3 à 0.8 sur bassins sahéliens avec forte variabilité
#      spatiale liée à l'usage des sols (±25-35% observé entre sous-bassins).
#    - La borne ±30% reflète cette hétérogénéité spatiale.
#
# 2. infiltration_scale [0.7, 1.4] (±40%) :
#    - Kamagaté et al. (2007, Figure 5, p.100) mesurent infiltration 5-24%
#      (ratio ~5×) avec forte sensibilité à l'occupation des sols et compaction.
#    - Azuka & Igué (2020) confirment variabilité ±35-45% entre parcelles.
#    - La borne ±40% capture cette incertitude liée aux pratiques agricoles.
#
# 3. evap_scale [0.8, 1.2] (±20%) :
#    - Hersbach et al. (2020, §4.3.1, p.1987) estiment incertitude ERA5 à
#      ±15-20% pour variables de flux surface en zone tropicale.
#    - Simon et al. (2023, SI-Table S3) utilisent ±20% pour ACV PHES.
#    - La borne reflète l'incertitude combinée modèle + validation locale.
#
# 4. leakage_scale [0.6, 1.5] (±50%) :
#    - Simon et al. (2023, p.4208) rapportent fuites 0.05-0.2% avec forte
#      dépendance à la qualité de construction (liner, fondations).
#    - Pracheil et al. (2025, §3.4) préconisent gamme large en phase faisabilité
#      avant conception détaillée.
#    - La borne ±50% reflète l'absence de specs techniques finales.
#
# LIMITATION : Ces bornes sont interprétées à partir de la littérature mais ne
# citent pas toujours des gammes explicites de sensibilité. Elles constituent
# des hypothèses prudentes en l'absence de données terrain. Validation requise.
SENSITIVITY_PROBLEM = {
    "num_vars": 4,
    "names": [
        "runoff_scale",
        "infiltration_scale",
        "evap_scale",
        "leakage_scale",
    ],
    "bounds": [
        [0.7, 1.3],   # runoff_scale : ±30% (variabilité spatiale)
        [0.7, 1.4],   # infiltration_scale : ±40% (sensible usage sols)
        [0.8, 1.2],   # evap_scale : ±20% (incertitude ERA5)
        [0.6, 1.5],   # leakage_scale : ±50% (dépend construction)
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
