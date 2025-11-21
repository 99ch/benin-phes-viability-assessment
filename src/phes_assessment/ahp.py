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


def compute_ahp_scores(
    sites_df: pd.DataFrame,
    hydrology_df: pd.DataFrame,
    *,
    weights: AHPWeights | None = None,
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

    required_site_cols = {"Pair Identifier", "Class"}
    missing_site_cols = required_site_cols - set(sites.columns)
    if missing_site_cols:
        raise ValueError(
            "Le catalogue de sites ne contient pas les colonnes requises: "
            + ", ".join(sorted(missing_site_cols))
        )

    if sites["Class"].isna().any():
        raise ValueError("La colonne 'Class' contient des valeurs manquantes nécessaires au calcul AHP.")

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

    merged["economic_score"] = _normalize_series(merged["class_multiplier"], higher_is_better=False)

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

    return merged[
        [
            "rank",
            "pair_identifier",
            "class",
            "final_score",
            "economic_score",
            "hydrology_score",
            "infrastructure_score",
            "prob_positive_annual_balance",
            "dry_season_prob_positive",
        ]
    ]
