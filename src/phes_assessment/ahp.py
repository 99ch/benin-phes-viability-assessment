"""Classement AHP combinant critères économiques et hydrologiques.

NOTE MÉTHODOLOGIQUE :
Ce module implémente une version SIMPLIFIÉE de l'AHP (Analytic Hierarchy Process)
qui utilise des poids directs plutôt qu'une matrice de comparaison par paires
complète (Saaty, 1980). Cette approche est acceptable pour une étude de faisabilité
préliminaire mais diffère de l'AHP classique sur les points suivants :

1. Les poids sont assignés DIRECTEMENT (economic=0.4, hydrology=0.4, etc.) sans
   passer par une matrice de jugements par paires et sans calcul de ratio de
   cohérence (CR).

2. Cette méthode est parfois appelée "weighted scoring" ou "weighted sum model"
   plutôt qu'AHP pur.

3. Pour une publication dans une revue AHP rigoureuse, il faudrait :
   - Matrice de comparaison par paires (échelle 1-9 de Saaty)
   - Calcul du vecteur propre principal pour les poids
   - Vérification CR < 0.10 pour la cohérence

4. Les poids actuels (0.4/0.4/0.2) reflètent un consensus d'experts mais n'ont
   pas été dérivés formellement d'une matrice de comparaisons.

Références :
- Saaty, T.L. (1980). The Analytic Hierarchy Process. McGraw-Hill.
- Saaty, T.L. (2008). Decision making with the AHP. Int. J. Services Sciences, 1(1).
"""
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
    """Poids des critères dans l'agrégation AHP (weighted scoring).
    
    Poids par défaut basés sur un consensus d'experts pour une étude de faisabilité
    PHES en contexte tropical :
    - economic (0.4) : Coût du stockage est critique pour la viabilité financière
    - hydrology (0.4) : Autonomie hydrique détermine la fiabilité opérationnelle
    - infrastructure (0.2) : Complexité technique influence les délais et risques
    
    Ces poids DOIVENT être ajustés selon le contexte décisionnel et les priorités
    des parties prenantes. Pour une approche AHP rigoureuse, utiliser une matrice
    de comparaison par paires et vérifier le ratio de cohérence (CR < 0.10).
    """
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
    # Combinaison pondérée : 60% annuelle + 40% saison sèche
    # JUSTIFICATION DES SOUS-POIDS (0.6/0.4) :
    # - Performance annuelle (0.6) : Reflète viabilité long-terme sur cycle complet
    # - Performance saison sèche (0.4) : Capture le stress hydrique critique
    # Ces poids sont des HYPOTHÈSES D'EXPERT basées sur l'importance relative
    # de chaque période pour l'opérabilité PHES. Idéalement, ils devraient être
    # dérivés d'une matrice de comparaison par paires (Saaty, 1980) avec CR < 0.10.
    # Une analyse de sensibilité (±20% sur chaque poids) est RECOMMANDÉE pour
    # valider la robustesse du classement final.
    hydro_metric = 0.6 * prob_positive + 0.4 * dry_prob
    merged["hydrology_score"] = _normalize_series(hydro_metric, higher_is_better=True)

    sep_score = _normalize_series(merged["Separation (km)"].astype(float), higher_is_better=False)
    slope_score = _normalize_series(merged["Slope (%)"].astype(float), higher_is_better=False)
    head_score = _normalize_series(merged["Head (m)"].astype(float), higher_is_better=True)
    water_rock_score = _normalize_series(
        merged["Combined water to rock ratio"].astype(float),
        higher_is_better=True,
    )

    # Combinaison pondérée des sous-critères d'infrastructure
    # JUSTIFICATION DES SOUS-POIDS (0.35/0.35/0.2/0.1) :
    # - Head (0.35) : Détermine le rendement énergétique (ΔE ∝ H)
    # - Water/rock ratio (0.35) : Minimise coûts excavation (volume barrage)
    # - Separation (0.2) : Influence coûts tunnels/pénéloppes
    # - Slope (0.1) : Impact mineur sur complexité construction
    # Ces poids sont des HYPOTHÈSES D'EXPERT basées sur analyses coûts PHES
    # (Stocks et al., 2021). Pour un AHP rigoureux, utiliser matrice Saaty.
    #
    # LIMITATION IMPUTATION : fillna(0.5) attribue un score neutre (médian)
    # aux valeurs manquantes. Cette imputation est ARBITRAIRE et peut biaiser
    # le classement. RECOMMANDATION : Signaler explicitement les sites avec
    # données incomplètes dans la sortie (colonne 'data_completeness') ou
    # utiliser imputation par régression (ex: head ~ separation + slope).
    infra_metric = (
        0.35 * head_score.fillna(0.5)
        + 0.35 * water_rock_score.fillna(0.5)
        + 0.2 * sep_score.fillna(0.5)
        + 0.1 * slope_score.fillna(0.5)
    )
    merged["infrastructure_score"] = _normalize_series(infra_metric, higher_is_better=True)

    # Calculer complétude des données (% de colonnes sans NaN)
    required_cols = ["Head (m)", "Separation (km)", "Slope (%)", "Combined water to rock ratio"]
    completeness = merged[required_cols].notna().sum(axis=1) / len(required_cols) * 100
    merged["data_completeness_pct"] = completeness.round(0).astype(int)

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
            "data_completeness_pct",
            "prob_positive_annual_balance",
            "dry_season_prob_positive",
        ]
    ]
