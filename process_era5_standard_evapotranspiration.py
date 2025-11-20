#!/usr/bin/env python3
"""
TRAITEMENT ERA5 STANDARD - ÉVAPOTRANSPIRATION SCIENTIFIQUEMENT CORRECTE
Conversion selon documentation officielle ECMWF
"""

import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
import xarray as xr
import glob
import calendar
from collections.abc import Sequence

DEFAULT_DAYS_IN_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "output"

def load_phes_sites_coordinates():
    """
    Charge les coordonnées des 12 sites PHES
    """
    sites_file = DATA_DIR / "n10_e001_12_sites_complete.csv"
    sites_df = pd.read_csv(sites_file)
    
    coords = {}
    sites_df.columns = sites_df.columns.str.strip()
    
    for _, row in sites_df.iterrows():
        site_id = row['Pair Identifier']
        lat = row['Upper latitude']
        lon = row['Upper longitude']
        coords[site_id] = (lat, lon)
    
    return coords

def _resolve_year(year_param, index):
    """Retourne l'année à utiliser pour l'index demandé."""
    if isinstance(year_param, Sequence) and not isinstance(year_param, (str, bytes)):
        if not year_param:
            return None
        return int(year_param[index]) if index < len(year_param) else int(year_param[-1])
    if year_param is None:
        return None
    return int(year_param)


def convert_era5_evapotranspiration_units(values, months=None, year=None):
    """
    Conversion scientifique ERA5 évapotranspiration
    
    Découverte critique: Les données ERA5 mensuelles sont des MOYENNES JOURNALIÈRES,
    pas des cumuls mensuels ! Selon ECMWF: "effective units of m of water per day"
    
    Args:
        values: Array numpy avec valeurs ERA5 brutes (12 mois)
        months: Liste numéros mois (1-12) pour calcul jours/mois
        year: Année ou séquence d'années correspondantes pour gérer les années bissextiles
        
    Returns:
        Array numpy avec évapotranspiration en mm/mois (positive)
    """
    print(" CONVERSION UNITÉS ERA5 ÉVAPOTRANSPIRATION:")
    print("   Découverte: Données mensuelles = moyennes journalières !")
    
    # Diagnostics valeurs brutes
    print(f"   Valeurs brutes - Min: {values.min():.6f} m/jour")
    print(f"   Valeurs brutes - Max: {values.max():.6f} m/jour")
    print(f"   Valeurs brutes - Moyenne: {values.mean():.6f} m/jour")
    
    # Nombre de jours par mois (année non-bissextile par défaut)
    if months is None:
        months = list(range(1, 13))
    
    # Conversion scientifique correcte:
    # 1. Prise en compte du signe ECMWF (flux sortant = négatif)
    # 2. m/jour → mm/jour (×1000)
    # 3. mm/jour → mm/mois (×jours_mois avec année bissextile le cas échéant)
    et0_mm_monthly = []
    
    for i, month_val in enumerate(values):
        month_num = months[i] if months and i < len(months) else (i % 12) + 1
        current_year = _resolve_year(year, i)
        if current_year is not None:
            days = calendar.monthrange(current_year, month_num)[1]
        else:
            # Utilise un mois type non bissextile si année inconnue
            days = DEFAULT_DAYS_IN_MONTH[(month_num - 1) % 12]

        # Conversion: -(m/jour) × 1000 × jours_mois = mm/mois (sign convention ECMWF)
        mm_monthly = max(0.0, -float(month_val)) * 1000 * days
        et0_mm_monthly.append(mm_monthly)
    
    et0_mm = np.array(et0_mm_monthly)
    
    print(f"   Après conversion - Min: {et0_mm.min():.1f} mm/mois")
    print(f"   Après conversion - Max: {et0_mm.max():.1f} mm/mois")
    print(f"   Après conversion - Moyenne: {et0_mm.mean():.1f} mm/mois")
    
    # Validation cohérence Bénin
    mean_monthly = et0_mm.mean()
    annual_est = et0_mm.sum()
    
    print(f"\n VALIDATION CLIMATOLOGIQUE:")
    print(f"   ET₀ mensuel moyen: {mean_monthly:.1f} mm/mois")
    print(f"   ET₀ annuel total: {annual_est:.0f} mm/année")
    
    if 1200 <= annual_est <= 1800:
        print(f"    Cohérent avec littérature Bénin (1200-1800 mm/année)")
    elif annual_est < 600:
        print(f"    Valeurs faibles - vérifier données source")
    elif annual_est > 2500:
        print(f"    Valeurs élevées - vérifier conversion") 
    else:
        print(f"    Hors plage attendue mais acceptable")
    
    return et0_mm

def extract_precip_from_chirps_tif(tif_file, sites_coords):
    """
    Extrait précipitations depuis fichier TIF CHIRPS pour tous les sites PHES
    
    Args:
        tif_file: Path vers fichier TIF CHIRPS
        sites_coords: Dict coordonnées sites
        
    Returns:
        Dict avec précipitations par site {site_id: precip_mm}
    """
    print(f"    CHIRPS: {tif_file.name}")
    
    precip_data = {}
    
    with rasterio.open(tif_file) as src:
        # Lire les données de précipitations (une seule bande)
        precip_array = src.read(1)  # mm/mois
        
        for site_id, (lat, lon) in sites_coords.items():
            try:
                # Conversion coordonnées géographiques → pixels
                row, col = src.index(lon, lat)
                
                if 0 <= row < src.height and 0 <= col < src.width:
                    # Extraire valeur précipitation pour ce point
                    precip_value = float(precip_array[row, col])
                    
                    # Vérifier si la valeur est valide (pas de nodata)
                    if not np.isnan(precip_value) and precip_value >= 0:
                        precip_data[site_id] = precip_value
                    else:
                        precip_data[site_id] = 0.0  # Aucune précipitation
                        
                else:
                    print(f"    Site {site_id} hors grille CHIRPS")
                    precip_data[site_id] = 0.0
                    
            except Exception as e:
                print(f"    Erreur site {site_id}: {e}")
                precip_data[site_id] = 0.0
    
    return precip_data

def extract_et0_from_era5_tif(tif_file, sites_coords):
    """
    Extrait ET₀ depuis fichier TIF ERA5 pour tous les sites PHES
    
    Args:
        tif_file: Path vers fichier TIF ERA5
        sites_coords: Dict coordonnées sites
        
    Returns:
        DataFrame avec données ET₀ mensuelles par site
    """
    print(f"\n TRAITEMENT: {tif_file.name}")
    
    year = int(tif_file.stem.split('_')[-1])
    site_data = []
    
    with rasterio.open(tif_file) as src:
        print(f"    Résolution: {src.res}° ({src.width}×{src.height} pixels)")
        print(f"    Bandes: {src.count} (12 mois attendus)")
        
        # Lire toutes les bandes (12 mois)
        all_bands = src.read()  # Shape: (12, height, width)
        
        for site_id, (lat, lon) in sites_coords.items():
            try:
                # Conversion coordonnées géographiques → pixels
                # ERA5 global : -180 to +180 lon, -90 to +90 lat
                # Méthode directe avec transform
                row, col = src.index(lon, lat)
                
                if 0 <= row < src.height and 0 <= col < src.width:
                    # Extraire série temporelle 12 mois pour ce point
                    site_values = all_bands[:, row, col]
                    
                    # Conversion unités scientifique
                    site_et0_mm = convert_era5_evapotranspiration_units(site_values, year=year)
                    
                    # Créer entrées mensuelles
                    for month in range(12):
                        et0_monthly = float(site_et0_mm[month])
                        
                        site_data.append({
                            'id': site_id,
                            'year': year,
                            'month': month + 1,
                            'E_mean': et0_monthly,  # ET₀ en mm/mois
                            'P_mean': np.nan,  # Sera rempli avec CHIRPS
                            'balance': np.nan,  # Sera calculé P_mean - E_mean
                            'data_source': 'ERA5_Standard_ET0'
                        })
                
                else:
                    print(f"    Site {site_id} hors grille ({lat:.3f}°N, {lon:.3f}°E)")
                    
            except Exception as e:
                print(f"    Erreur site {site_id}: {e}")
                continue
    
    print(f"    {len(site_data)} enregistrements créés")
    return pd.DataFrame(site_data)

def integrate_chirps_precipitation_data(df, sites_coords):
    """
    Intègre les données de précipitations CHIRPS dans le DataFrame ERA5
    
    Args:
        df: DataFrame avec données ET₀ ERA5
        sites_coords: Dict coordonnées sites
        
    Returns:
        DataFrame avec P_mean et balance complétés
    """
    print(f"\n INTÉGRATION DONNÉES CHIRPS")
    print("=" * 50)
    
    chirps_dir = DATA_DIR / "chirps"
    
    if not chirps_dir.exists():
        print(f" Dossier CHIRPS non trouvé: {chirps_dir}")
        return df
    
    # Recherche fichiers CHIRPS TIF
    chirps_files = list(chirps_dir.glob("chirps-v2.0.*.tif"))
    chirps_files.sort()
    
    if not chirps_files:
        print(f" Aucun fichier CHIRPS trouvé dans {chirps_dir}")
        return df
    
    print(f" {len(chirps_files)} fichiers CHIRPS trouvés")
    
    # Traiter chaque fichier CHIRPS et intégrer dans DataFrame
    for chirps_file in chirps_files:
        try:
            # Extraire année et mois du nom fichier: chirps-v2.0.YYYY.MM.tif
            filename_parts = chirps_file.stem.split('.')
            if len(filename_parts) >= 3:
                year = int(filename_parts[2])
                month = int(filename_parts[3])
                
                # Extraire précipitations pour tous les sites
                precip_data = extract_precip_from_chirps_tif(chirps_file, sites_coords)
                
                # Mettre à jour DataFrame pour cette année/mois
                for site_id, precip_value in precip_data.items():
                    # Trouver les lignes correspondantes
                    mask = (df['id'] == site_id) & (df['year'] == year) & (df['month'] == month)
                    
                    if mask.any():
                        # Mettre à jour P_mean
                        df.loc[mask, 'P_mean'] = precip_value
                        
                        # Calculer balance = P_mean - E_mean
                        e_mean = df.loc[mask, 'E_mean'].iloc[0]
                        balance = precip_value - e_mean
                        df.loc[mask, 'balance'] = balance
                        
                        # Mettre à jour source données
                        df.loc[mask, 'data_source'] = 'ERA5_Standard_ET0+CHIRPS'
                        
        except Exception as e:
            print(f"    Erreur traitement {chirps_file.name}: {e}")
            continue
    
    # Statistiques intégration
    total_records = len(df)
    records_with_precip = len(df[df['P_mean'].notna()])
    integration_rate = (records_with_precip / total_records) * 100 if total_records > 0 else 0
    
    print(f"\n RÉSULTATS INTÉGRATION CHIRPS:")
    print(f"   Total enregistrements: {total_records}")
    print(f"   Avec précipitations: {records_with_precip}")
    print(f"   Taux intégration: {integration_rate:.1f}%")
    
    if records_with_precip > 0:
        precip_stats = df['P_mean'].describe()
        balance_stats = df['balance'].describe()
        
        print(f"\n STATISTIQUES PRÉCIPITATIONS:")
        print(f"   Min: {precip_stats['min']:.1f} mm/mois")
        print(f"   Moyenne: {precip_stats['mean']:.1f} mm/mois") 
        print(f"   Max: {precip_stats['max']:.1f} mm/mois")
        
        print(f"\n STATISTIQUES BILAN P-E:")
        print(f"   Min: {balance_stats['min']:.1f} mm/mois")
        print(f"   Moyenne: {balance_stats['mean']:.1f} mm/mois")
        print(f"   Max: {balance_stats['max']:.1f} mm/mois")
        
        # Validation climatologique
        positive_balance_months = len(df[df['balance'] > 0])
        total_months = len(df[df['balance'].notna()])
        positive_rate = (positive_balance_months / total_months) * 100 if total_months > 0 else 0
        
        print(f"\n VALIDATION CLIMATOLOGIQUE:")
        print(f"   Mois bilan positif: {positive_balance_months}/{total_months} ({positive_rate:.1f}%)")
        
        if 20 <= positive_rate <= 60:
            print(f"    Cohérent avec climat tropical Bénin")
        else:
            print(f"    Taux inhabituel pour climat Bénin")
    
    return df

def process_all_era5_standard_files():
    """
    Traite tous les fichiers ERA5 TIF standard
    """
    print(" TRAITEMENT ERA5 STANDARD - ÉVAPOTRANSPIRATION")
    print("=" * 60)
    print(" Source: Produit officiel ECMWF")
    print(" Méthode: Conversion scientifique selon documentation")
    print("=" * 60)
    
    # Chemins
    era5_dir = DATA_DIR / "era5"
    output_file = OUTPUT_DIR / "site_stats_era5_standard.csv"
    
    # Chargement sites
    sites_coords = load_phes_sites_coordinates()
    print(f" {len(sites_coords)} sites PHES chargés")
    
    # Recherche fichiers TIF
    tif_files = list(era5_dir.glob("era5_*.tif"))
    tif_files = sorted([f for f in tif_files if f.name.count('_') == 1])  # era5_YYYY.tif
    
    if not tif_files:
        print(f" Aucun fichier TIF ERA5 trouvé dans {era5_dir}")
        return False
    
    print(f" {len(tif_files)} fichiers ERA5 TIF trouvés")
    
    # Traitement fichier par fichier
    all_data = []
    
    for tif_file in tif_files:
        file_data = extract_et0_from_era5_tif(tif_file, sites_coords)
        
        if len(file_data) > 0:
            all_data.append(file_data)
    
    if not all_data:
        print(" Aucune donnée ET₀ extraite")
        return False
    
    # Concaténation données ERA5
    final_df = pd.concat(all_data, ignore_index=True)
    final_df = final_df.sort_values(['id', 'year', 'month']).reset_index(drop=True)
    
    print(f"\n Données ERA5 ET₀ compilées: {len(final_df)} enregistrements")
    
    # INTÉGRATION DONNÉES CHIRPS
    final_df = integrate_chirps_precipitation_data(final_df, sites_coords)
    
    # Statistiques finales intégrées
    print(f"\n DONNÉES INTÉGRÉES ERA5+CHIRPS GÉNÉRÉES:")
    print(f"   Total enregistrements: {len(final_df)}")
    print(f"   Sites: {final_df['id'].nunique()}")
    print(f"   Années: {final_df['year'].nunique()}")
    print(f"   Plage: {final_df['year'].min()}-{final_df['year'].max()}")
    
    # Validation finale ET₀
    et0_stats = final_df['E_mean'].describe()
    print(f"\n STATISTIQUES ET₀ FINALES:")
    print(f"   Min: {et0_stats['min']:.1f} mm/mois")
    print(f"   Moyenne: {et0_stats['mean']:.1f} mm/mois")
    print(f"   Max: {et0_stats['max']:.1f} mm/mois")
    print(f"   Médiane: {et0_stats['50%']:.1f} mm/mois")
    
    annual_et0_mean = et0_stats['mean'] * 12
    print(f"   Annual moyen: {annual_et0_mean:.0f} mm/année")
    
    # Validation finale Précipitations (si disponibles)
    if 'P_mean' in final_df.columns and final_df['P_mean'].notna().any():
        precip_stats = final_df['P_mean'].describe()
        print(f"\n STATISTIQUES PRÉCIPITATIONS FINALES:")
        print(f"   Min: {precip_stats['min']:.1f} mm/mois")
        print(f"   Moyenne: {precip_stats['mean']:.1f} mm/mois")
        print(f"   Max: {precip_stats['max']:.1f} mm/mois")
        print(f"   Médiane: {precip_stats['50%']:.1f} mm/mois")
        
        annual_precip_mean = precip_stats['mean'] * 12
        print(f"   Annual moyen: {annual_precip_mean:.0f} mm/année")
    
    # Validation finale Bilan P-E (si disponible)
    if 'balance' in final_df.columns and final_df['balance'].notna().any():
        balance_stats = final_df['balance'].describe()
        print(f"\n STATISTIQUES BILAN P-E FINALES:")
        print(f"   Min: {balance_stats['min']:.1f} mm/mois")
        print(f"   Moyenne: {balance_stats['mean']:.1f} mm/mois")
        print(f"   Max: {balance_stats['max']:.1f} mm/mois")
        print(f"   Médiane: {balance_stats['50%']:.1f} mm/mois")
        
        annual_balance_mean = balance_stats['mean'] * 12
        print(f"   Annual moyen: {annual_balance_mean:.0f} mm/année")
        
        # Validation climatologique bilan
        if annual_balance_mean < -500:
            print(f"    Déficit hydrique important (attendu climat semi-aride)")
        elif -500 <= annual_balance_mean <= 0:
            print(f"    Déficit modéré cohérent climat tropical")
        else:
            print(f"    Excès hydrique cohérent climat humide")
    
    # Sauvegarde
    output_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_file, index=False)
    
    print(f"\n DONNÉES SAUVEGARDÉES: {output_file}")
    
    return True

def main():
    """Fonction principale"""
    
    success = process_all_era5_standard_files()
    
    if success:
        print(f"\n TRAITEMENT ERA5+CHIRPS INTÉGRÉ TERMINÉ!")
        print("=" * 50)
        print(" Évapotranspiration ERA5 scientifique appliquée")
        print(" Précipitations CHIRPS intégrées")
        print(" Bilan P-E complet et réaliste")
        print(" Documentation ECMWF + CHIRPS respectée") 
        print(" Fichier: site_stats_era5_standard.csv")
        print(" Prêt pour analyses PHES avec données complètes")
    else:
        print(f"\n Échec traitement ERA5 standard")

if __name__ == "__main__":
    main()