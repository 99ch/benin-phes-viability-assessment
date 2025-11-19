# Étude de faisabilité des systèmes PHES en boucle fermée – Bénin

Ce dépôt regroupe les données et scripts nécessaires pour évaluer la faisabilité technique,
économique et hydrologique de sites PHES (Pumped Hydro Energy Storage) identifiés dans
la région de l’Atacora.

## Objectifs du projet

- **Détection des sites** à partir du modèle numérique de terrain FABDEM (30 m).
- **Modélisation du bilan hydrique (2002-2023)** via les jeux de données CHIRPS et ERA5.
- **Simulation Monte Carlo (10 000 itérations/site)** pour estimer la probabilité d’autonomie hydrique.
- **Classement multicritère (AHP)** intégrant dimensions économiques, hydrologiques et environnementales.

## Structure actuelle

```
.
├── data/                       # Données brutes (FABDEM, CHIRPS, ERA5, fiches sites)
├── n10_e001/                   # Exports détaillés pour différents niveaux de stockage
├── src/phes_assessment/        # Code Python (ingestion, analyse, pipelines)
├── pyproject.toml              # Dépendances et configuration du projet
└── README.md                   # Ce fichier
```

## Installation

1. Créez un environnement virtuel Python (>=3.10) et activez-le.
2. Installez les dépendances de base :

```bash
pip install -e .
```

3. Pour disposer des outils avancés (géotraitement complet) :

```bash
pip install -e .[full]
```

## Premier outil disponible

Le module `phes-data` (alias `phes_assessment.cli`) inventorie les jeux de
données présents localement et vérifie leur couverture temporelle.

Exemple d’exécution :

```bash
python -m phes_assessment.cli
```

ou, après installation, simplement :

```bash
phes-data
```

La commande renvoie le nombre de rasters détectés, les dates de début/fin
et les métadonnées de projection/résolution. Si une colonne affiche `-`,
vérifiez que les fichiers (ex. FABDEM) ont bien été décompressés dans
`data/`.

### Comparer les altitudes catalogue vs FABDEM

Une seconde commande permet d’échantillonner les altitudes FABDEM aux
coordonnées des 12 sites et de comparer la hauteur de chute calculée à
partir du MNT avec celle publiée dans le catalogue.

```bash
python -m phes_assessment.cli fabdem-sample --output results/fabdem_sites.csv
```

Le tableau affiché détaille, pour chaque paire de réservoirs, le head du
catalogue, le head issu de FABDEM et l’écart entre les deux. L’option
`--output` (facultative) permet de sauvegarder le tableau complet
enrichi des altitudes dans un CSV.

### Générer les séries climatiques CHIRPS/ERA5

```bash
# période complète 2002-2023, sortie CSV par défaut
python -m phes_assessment.cli climate-series --output results/climate_series.csv

# exemple : ERA5 uniquement sur 2008-2010 avec buffers de 1.5 km
python -m phes_assessment.cli climate-series \
    --dataset era5 \
    --start-year 2008 --end-year 2010 \
    --buffer-meters 1500 \
    --output results/era5_2008_2010.parquet
```

La commande :

- construit automatiquement des **buffers circulaires** (rayon configurable, 500 m par défaut)
	autour des réservoirs upper/lower décrits dans le CSV, fusionne les polygones
	par paire et applique systématiquement un paramètre `all_touched` pour s’assurer
	que les pixels de bord sont intégrés ;
- parcourt uniquement les rasters CHIRPS (précipitations) et/ou ERA5 (évapotranspiration)
	qui tombent dans la fenêtre temporelle demandée et calcule la moyenne spatiale
	à l’intérieur de chaque polygone (progress bar Rich affichée pendant le traitement) ;
- exporte un tableau `pair_identifier × date` avec les colonnes `precip_mm` et `etp_mm`.

Paramètres principaux :

- `--dataset {both,chirps,era5}` pour choisir la ou les sources à agréger ;
- `--start-year` et `--end-year` pour borner l’analyse (2002‑2023 par défaut) ;
- `--buffer-meters` pour ajuster le rayon spatial (en mètres) autour de chaque réservoir ;
- `--sites` pour pointer vers un CSV spécifique si besoin.

La sortie est écrite au format demandé : `.parquet` si un moteur
(`pyarrow`/`fastparquet`) est disponible, sinon CSV. Par défaut, le fichier
`results/climate_series.csv` est généré.

## Étapes suivantes

- Implémenter les fonctions de lecture et de validation des jeux FABDEM/CHIRPS/ERA5.
- Automatiser la génération des indicateurs hydrologiques intermédiaires.
- Documenter les pipelines (Snakemake/Prefect) pour la reproductibilité.
