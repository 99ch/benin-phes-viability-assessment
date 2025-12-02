"""Outils pour générer des figures et exports hydrologiques."""
from __future__ import annotations

from textwrap import fill
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .sites import load_sites

matplotlib.use("Agg")  # Garantit un backend hors-écran pour les rendus batch


HYDRO_COLUMNS: tuple[str, ...] = (
    "pair_identifier",
    "capacity_gl",
    "median_annual_balance_gl",
    "p10_annual_balance_gl",
    "p90_annual_balance_gl",
    "prob_positive_annual_balance",
    "dry_season_prob_positive",
    "dry_season_p10_gl",
    "dry_season_median_balance_gl",
    "dry_season_median_deficit_gl",
)

MONTH_LABELS: tuple[str, ...] = ("Jan", "Fév", "Mar", "Avr", "Mai", "Jun", "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc")
RAINY_MONTHS: tuple[int, ...] = (6, 7, 8, 9)  # Juin à Septembre
DEFAULT_FIGURE7_PAIR = "n10_e001_RES31412 & n10_e001_RES31520"


REVIEWER_COLUMNS: tuple[str, ...] = (
    "pair_identifier",
    "Class",
    "Head (m)",
    "capacity_gl",
    "median_annual_balance_gl",
    "p10_annual_balance_gl",
    "p90_annual_balance_gl",
    "prob_positive_annual_balance",
    "dry_season_prob_positive",
    "dry_season_p10_gl",
    "dry_season_median_balance_gl",
    "dry_season_median_deficit_gl",
)


def _load_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Format non supporté pour {path}")


def _load_climate_dataframe(path: Path) -> pd.DataFrame:
    df = _load_dataframe(path)
    required = {"pair_identifier", "date", "precip_mm", "etp_mm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Le fichier climatique doit contenir les colonnes suivantes : "
            + ", ".join(sorted(required))
            + f". Colonnes absentes : {', '.join(sorted(missing))}."
        )
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    return df


def _normalize_values(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    values = pd.Series(series, dtype="float64")
    mask = values.notna()
    normalized = pd.Series(np.nan, index=values.index, dtype="float64")
    if mask.sum() <= 1:
        normalized.loc[mask] = 1.0
        return normalized
    subset = values[mask]
    min_val = float(subset.min())
    max_val = float(subset.max())
    if np.isclose(max_val, min_val):
        normalized.loc[mask] = 1.0
    else:
        normalized.loc[mask] = (subset - min_val) / (max_val - min_val)
    if not higher_is_better:
        normalized.loc[mask] = 1.0 - normalized.loc[mask]
    return normalized


def _monthly_climate_stats(
    climate_df: pd.DataFrame,
    *,
    site_identifier: str | None = None,
) -> pd.DataFrame:
    df = climate_df
    if site_identifier:
        df = df[df["pair_identifier"] == site_identifier]
        if df.empty:
            raise ValueError(f"Aucune donnée climatique pour {site_identifier}")
    df = df.copy()
    df["month"] = df["date"].dt.month
    grouped = (
        df.groupby("month")[["precip_mm", "etp_mm"]]
        .mean()
        .reindex(range(1, 13))
    )
    grouped.index.name = "month"
    grouped["precip_mm"] = grouped["precip_mm"].clip(lower=0).fillna(0)
    grouped["etp_mm"] = grouped["etp_mm"].abs().fillna(0)
    if grouped["etp_mm"].max() < 10:
        grouped["etp_mm"] = grouped["etp_mm"] * 1000  # ERA5 stocké en mètres
    grouped["month_label"] = [MONTH_LABELS[idx - 1] for idx in grouped.index]
    return grouped


def plot_monthly_cycle(
    monthly_stats: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    subtitle: str | None = None,
) -> Path:
    x = np.arange(1, 13)
    precip = monthly_stats["precip_mm"].to_numpy()
    etp = monthly_stats["etp_mm"].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 5))
    bar = ax.bar(x - 0.2, precip, width=0.4, color="#3182bd", label="Précipitations (mm/mois)")
    ax.plot(x + 0.2, etp, color="#d73027", marker="o", linewidth=2.0, label="Évapotranspiration (mm/mois)")

    for month in RAINY_MONTHS:
        ax.axvspan(month - 0.5, month + 0.5, color="#66c2a5", alpha=0.15)
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(monthly_stats["month_label"].tolist(), rotation=0)
    ax.set_ylabel("mm par mois")
    ax.set_title(title, fontsize=14, fontweight="bold")
    if subtitle:
        ax.text(0.01, -0.18, subtitle, transform=ax.transAxes, fontsize=10, color="#444444")
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def build_hydrology_dataframe(summary_path: Path, sites_csv: Path) -> pd.DataFrame:
    summary_df = _load_dataframe(summary_path)
    missing_columns = set(HYDRO_COLUMNS) - set(summary_df.columns)
    if missing_columns:
        raise ValueError(
            "Le fichier hydrologique doit contenir les colonnes suivantes : "
            + ", ".join(HYDRO_COLUMNS)
            + f". Colonnes absentes : {', '.join(sorted(missing_columns))}."
        )

    sites_df = load_sites(sites_csv)
    site_cols = [
        "Pair Identifier",
        "Class",
        "Head (m)",
        "Upper latitude",
        "Upper longitude",
        "Lower latitude",
        "Lower longitude",
    ]
    for col in site_cols:
        if col not in sites_df.columns:
            raise ValueError(f"Colonne '{col}' absente du CSV des sites {sites_csv}")

    subset = sites_df[site_cols].rename(columns={"Pair Identifier": "pair_identifier"})
    subset["mean_latitude"] = subset[["Upper latitude", "Lower latitude"]].astype(float).mean(axis=1)
    subset["mean_longitude"] = subset[["Upper longitude", "Lower longitude"]].astype(float).mean(axis=1)

    merged = summary_df.merge(subset, on="pair_identifier", how="inner")
    return merged


def export_reviewer_csv(df: pd.DataFrame, output_path: Path, float_format: str = "%.3f") -> Path:
    export_cols = [col for col in REVIEWER_COLUMNS if col in df.columns]
    export_df = df[export_cols].copy()
    export_df_sorted = export_df.sort_values("median_annual_balance_gl", ascending=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_df_sorted.to_csv(output_path, index=False, float_format=float_format)
    return output_path


def plot_median_balance(df: pd.DataFrame, output_path: Path) -> Path:
    sorted_df = df.sort_values("median_annual_balance_gl", ascending=False)
    colors = np.where(sorted_df["median_annual_balance_gl"] >= 0, "#1a9850", "#d73027")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(sorted_df["pair_identifier"], sorted_df["median_annual_balance_gl"], color=colors)
    ax.set_ylabel("Médiane annuelle (GL/an)")
    ax.set_title("Bilan hydrologique médian par site (10k tirages)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_deterministic_balances(df: pd.DataFrame, output_path: Path) -> Path:
    required = {
        "pair_identifier",
        "median_annual_balance_gl",
        "dry_season_p10_deficit_gl",
    }
    if "dry_season_p10_deficit_gl" not in df.columns and "dry_season_p10_gl" in df.columns:
        df = df.rename(columns={"dry_season_p10_gl": "dry_season_p10_deficit_gl"})
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Le dataframe doit contenir les colonnes : "
            + ", ".join(sorted(required))
            + f". Colonnes absentes : {', '.join(sorted(missing))}."
        )

    sorted_df = df.sort_values("median_annual_balance_gl", ascending=False).reset_index(drop=True)
    x = np.arange(len(sorted_df))
    width = 0.42

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(
        x - width / 2,
        sorted_df["median_annual_balance_gl"],
        width,
        color="#1f78b4",
        label="Médiane annuelle (GL/an)",
    )
    ax.bar(
        x + width / 2,
        sorted_df["dry_season_p10_deficit_gl"],
        width,
        color="#e31a1c",
        label="Déficit saison sèche P10 (GL)",
    )
    ax.axhline(0, color="#222222", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_label(pid) for pid in sorted_df["pair_identifier"]], rotation=45, ha="right")
    ax.set_ylabel("Volume (GL)")
    ax.set_title("Référence déterministe : bilans par site", fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def _short_label(pair_identifier: str) -> str:
    if "&" in pair_identifier:
        return pair_identifier.split("&")[0].strip()
    return pair_identifier


def plot_spatial_scatter(df: pd.DataFrame, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        df["mean_longitude"],
        df["mean_latitude"],
        c=df["median_annual_balance_gl"],
        cmap="RdYlBu",
        s=180,
        edgecolor="black",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Gradients spatiaux du bilan hydrologique médian")
    for _, row in df.iterrows():
        ax.text(
            row["mean_longitude"] + 0.01,
            row["mean_latitude"] + 0.01,
            _short_label(str(row["pair_identifier"])),
            fontsize=8,
            color="black",
        )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Médiane annuelle (GL/an)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_site_selection_flow(output_path: Path) -> Path:
    steps: Sequence[tuple[str, str]] = (
        ("FABDEM 30 m", "Prétraitement du MNT, suppression des artefacts anthropiques"),
        ("Détection des paires", "ΔH 100–600 m, distance <10 km, réservoirs 2–500 GWh"),
        ("Modélisation des réservoirs", "Remplissage, surfaces et barrages + calcul volumes"),
        ("Bassins versants", "Delineation WhiteboxTools + aire contributive dédiée"),
        ("Contrôles boucle fermée", "Aucune rivière à <2 km, exclusion des chevauchements"),
        ("Filtres SIG", "Aires protégées, densité urbaine, ratio bassin:réservoir ≥5:1, H<100 m"),
        ("Classement coût", "Classes A→E via modèle Stocks et al. (CAPEX/MWh)"),
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")
    colors = ["#2c7fb8", "#00a6ca", "#00ccbc", "#90eb9d", "#f9d057", "#f29e2e", "#e76818"]
    y_positions = np.linspace(0.9, 0.1, len(steps))

    for idx, ((title, description), y) in enumerate(zip(steps, y_positions)):
        text = f"{title}\n{fill(description, width=45)}"
        bbox = dict(boxstyle="round,pad=0.6", facecolor=colors[idx % len(colors)], edgecolor="#08306b", alpha=0.9)
        ax.text(
            0.5,
            y,
            text,
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            fontweight="bold",
            bbox=bbox,
        )
        if idx < len(steps) - 1:
            ax.annotate(
                "",
                xy=(0.5, y_positions[idx + 1] + 0.05),
                xytext=(0.5, y - 0.05),
                arrowprops=dict(arrowstyle="->", color="#08306b", linewidth=2.0),
            )

    ax.set_title("Algorithme d'identification des sites PHES", fontsize=16, fontweight="bold")
    ax.text(
        0.5,
        0.02,
        "Chaque étape applique automatiquement les contraintes décrites en Section 2.3",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        color="#444444",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_ahp_weights(output_path: Path) -> Path:
    main_weights = {
        "Hydrologie": 0.4,
        "Économie": 0.4,
        "Infrastructure": 0.2,
    }
    infra_weights = {
        "Head": 0.35,
        "Water:Rock": 0.35,
        "Séparation": 0.2,
        "Pente": 0.1,
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax in axes:
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.set_xlim(0, 0.9)

    axes[0].barh(list(main_weights.keys()), list(main_weights.values()), color="#3182bd")
    axes[0].set_title("Critères principaux", fontweight="bold")
    axes[0].set_xlabel("Poids (fraction)")
    for idx, (label, weight) in enumerate(main_weights.items()):
        axes[0].text(weight + 0.02, idx, f"{weight*100:.0f}%", va="center", fontsize=10)

    axes[1].barh(list(infra_weights.keys()), list(infra_weights.values()), color="#de2d26")
    axes[1].set_title("Sous-critères infrastructure", fontweight="bold")
    axes[1].set_xlabel("Poids (fraction)")
    for idx, (label, weight) in enumerate(infra_weights.items()):
        axes[1].text(weight + 0.02, idx, f"{weight*100:.0f}%", va="center", fontsize=10)

    fig.suptitle("Pondérations utilisées dans l'AHP (Section 2.6)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_monte_carlo_percentiles(summary_df: pd.DataFrame, output_path: Path) -> Path:
    required = {
        "pair_identifier",
        "median_annual_balance_gl",
        "p10_annual_balance_gl",
        "p90_annual_balance_gl",
    }
    missing = required - set(summary_df.columns)
    if missing:
        raise ValueError(
            "Le fichier hydrologique doit contenir les colonnes : "
            + ", ".join(sorted(required))
            + f". Colonnes absentes : {', '.join(sorted(missing))}."
        )

    ranked = summary_df.sort_values("median_annual_balance_gl", ascending=False).reset_index(drop=True)
    y_pos = np.arange(len(ranked))
    med = ranked["median_annual_balance_gl"].to_numpy()
    low = med - ranked["p10_annual_balance_gl"].to_numpy()
    high = ranked["p90_annual_balance_gl"].to_numpy() - med

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        med,
        y_pos,
        xerr=[low, high],
        fmt="o",
        color="#1a9850",
        ecolor="#636363",
        elinewidth=1.5,
        capsize=4,
    )
    ax.axvline(0, color="#252525", linewidth=1, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_short_label(pid) for pid in ranked["pair_identifier"]])
    ax.set_xlabel("Bilan annuel (GL/an)")
    ax.set_title("Distribution Monte Carlo (P10-P90) des 12 sites", fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_ahp_heatmap(
    ahp_df: pd.DataFrame,
    sites_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    site_subset = sites_df[[
        "Pair Identifier",
        "Head (m)",
        "Combined water to rock ratio",
        "Separation (km)",
        "Slope (%)",
    ]].rename(columns={"Pair Identifier": "pair_identifier"})

    merged = ahp_df.merge(site_subset, on="pair_identifier", how="left")
    merged.sort_values("rank", inplace=True)

    merged["head_score"] = _normalize_values(merged["Head (m)"], higher_is_better=True)
    merged["water_rock_score"] = _normalize_values(
        merged["Combined water to rock ratio"], higher_is_better=True
    )
    merged["separation_score"] = _normalize_values(
        merged["Separation (km)"], higher_is_better=False
    )
    merged["slope_score"] = _normalize_values(merged["Slope (%)"], higher_is_better=False)

    heatmap_df = merged[[
        "pair_identifier",
        "final_score",
        "economic_score",
        "hydrology_score",
        "infrastructure_score",
        "prob_positive_annual_balance",
        "dry_season_prob_positive",
        "head_score",
        "water_rock_score",
        "separation_score",
        "slope_score",
    ]].copy()

    heatmap_df.rename(
        columns={
            "final_score": "Score final",
            "economic_score": "Économie",
            "hydrology_score": "Hydrologie",
            "infrastructure_score": "Infrastructure",
            "prob_positive_annual_balance": "P(>0) annuel",
            "dry_season_prob_positive": "P(>0) saison sèche",
            "head_score": "Head",
            "water_rock_score": "Water:Rock",
            "separation_score": "Séparation",
            "slope_score": "Pente",
        },
        inplace=True,
    )

    values = heatmap_df.drop(columns=["pair_identifier"]).to_numpy()
    fig, ax = plt.subplots(figsize=(16, 9))
    im = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_xticklabels(heatmap_df.columns[1:], rotation=35, ha="right", fontsize=12)
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_yticklabels([_short_label(pid) for pid in heatmap_df["pair_identifier"]], fontsize=12)
    ax.set_title("Scores AHP et sous-critères normalisés", fontweight="bold", fontsize=16)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color="#08306b", fontsize=11)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.25)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_ylabel("Score normalisé [0-1]", fontsize=12)
    cbar.ax.tick_params(labelsize=11)
    fig.subplots_adjust(left=0.14, right=0.92, top=0.94, bottom=0.12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def generate_all_figures(
    summary_path: Path,
    sites_csv: Path,
    figure_dir: Path,
    reviewer_csv: Path,
) -> dict[str, Path]:
    df = build_hydrology_dataframe(summary_path, sites_csv)
    outputs: dict[str, Path] = {}
    outputs["reviewer_csv"] = export_reviewer_csv(df, reviewer_csv)
    outputs["median_bar"] = plot_median_balance(df, figure_dir / "hydrology_median_balance.png")
    outputs["spatial_map"] = plot_spatial_scatter(df, figure_dir / "hydrology_median_balance_map.png")
    return outputs


def generate_article_figures(
    climate_path: Path,
    summary_path: Path,
    sites_csv: Path,
    ahp_rankings_path: Path,
    figure_dir: Path,
    *,
    figure7_pair: str = DEFAULT_FIGURE7_PAIR,
    deterministic_path: Path | None = None,
) -> dict[str, Path]:
    climate_df = _load_climate_dataframe(climate_path)
    summary_df = _load_dataframe(summary_path)
    sites_df = load_sites(sites_csv)
    ahp_df = _load_dataframe(ahp_rankings_path)
    deterministic_df = _load_dataframe(deterministic_path) if deterministic_path else summary_df

    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    outputs["figure01_flow"] = plot_site_selection_flow(figure_dir / "figure01_algorithme.png")

    national_stats = _monthly_climate_stats(climate_df)
    outputs["figure02_climat"] = plot_monthly_cycle(
        national_stats,
        figure_dir / "figure02_cycle_climatique.png",
        title="Cycle saisonnier moyen (2002-2023)",
        subtitle="Moyenne CHIRPS + ERA5 sur les 12 sites identifiés",
    )

    outputs["figure03_ahp_weights"] = plot_ahp_weights(figure_dir / "figure03_ahp_poids.png")

    site_stats = _monthly_climate_stats(climate_df, site_identifier=figure7_pair)
    outputs["figure07_site_cycle"] = plot_monthly_cycle(
        site_stats,
        figure_dir / "figure07_cycle_site_res31412.png",
        title="Cycle saisonnier moyen – n10_e001_RES31412 & RES31520",
        subtitle="Climat local moyen 2002-2023, CHIRPS vs ERA5",
    )

    outputs["figure07b_deterministe"] = plot_deterministic_balances(
        deterministic_df,
        figure_dir / "figure07b_bilan_deterministe.png",
    )

    outputs["figure08_monte_carlo"] = plot_monte_carlo_percentiles(
        summary_df,
        figure_dir / "figure08_monte_carlo.png",
    )

    outputs["figure09_ahp_heatmap"] = plot_ahp_heatmap(
        ahp_df,
        sites_df,
        figure_dir / "figure09_ahp_heatmap.png",
    )

    return outputs
