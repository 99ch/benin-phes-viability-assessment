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

## Étapes suivantes
- Implémenter les fonctions de lecture et de validation des jeux FABDEM/CHIRPS/ERA5.
- Automatiser la génération des indicateurs hydrologiques intermédiaires.
- Documenter les pipelines (Snakemake/Prefect) pour la reproductibilité.
