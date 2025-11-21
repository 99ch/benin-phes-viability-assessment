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

3. Pour disposer des outils avancés (géotraitement complet : GeoPandas, WhiteboxTools, RichDEM) :

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
- `--basins` pour fournir un GeoJSON de bassins versants généré par `site-basins` (le
  rayon des buffers est alors ignoré et seules les limites hydrologiques sont utilisées).

> ⚠️ `climate-series` refuse désormais de s'exécuter sans GeoJSON `--basins`. Exécutez
> `python -m phes_assessment.cli site-basins --output results/site_basins.geojson` avant
> toute agrégation et recyclez ce fichier pour `climate-series` puis `hydro-sim`.

Pour un workflow **CHIRPS seul** ou **ERA5 seul** (cas de diagnostics incrémentaux) :

1. Lancez d'abord `climate-series --dataset chirps --output results/chirps_only.parquet --basins results/site_basins.geojson` afin de générer `precip_mm`.
2. Lancez ensuite `climate-series --dataset era5 --output results/era5_only.parquet --basins results/site_basins.geojson` pour obtenir `etp_mm`.
3. Fusionnez les deux fichiers sur `pair_identifier` + `date` (ex. via `pandas.merge`). Le
  fichier combiné doit contenir les deux colonnes pour alimenter `hydro-sim`.

La sortie est écrite au format demandé : `.parquet` si un moteur
(`pyarrow`/`fastparquet`) est disponible, sinon CSV. Par défaut, le fichier
`results/climate_series.csv` est généré.

### Préparer les rasters ERA5 (évapotranspiration)

Les GeoTIFF multibandes utilisés par `climate-series --dataset era5` peuvent être
reconstitués depuis la réanalyse ERA5 en conservant toute l’emprise de l’Atacora.
Le script `process_era5_standard_evapotranspiration.py` automatise le
**téléchargement via l’API Copernicus** (cdsapi) puis la conversion NetCDF →
GeoTIFF (1 fichier par année, 12 bandes mensuelles en mm/mois).

```bash
python process_era5_standard_evapotranspiration.py \
  --start-year 2002 --end-year 2023 \
  --output-dir data/era5 \
  --cache-dir data/era5_raw \
  --north 12.8 --south 6.0 --west -1.5 --east 3.5
```

Pré-requis :

- créer le fichier `~/.cdsapirc` contenant votre `url` et `key` Copernicus ;
- installer la dépendance `cdsapi` (déjà incluse dans `pip install -e .`).

Chaque GeoTIFF compressé (`era5_<année>.tif`) conserve la résolution d’origine
de 0,25° et une métadonnée par bande (`YYYY-MM`). Pour « convertir uniquement »
des NetCDF déjà téléchargés, ajoutez `--skip-download` et placez vos fichiers
dans le dossier indiqué par `--cache-dir`.

### Contrôler la complétude des datasets

La commande `data-qa` exécute un diagnostic rapide sur les dossiers CHIRPS, ERA5
et FABDEM : couverture temporelle réelle vs attendue, mois/années manquants,
doublons potentiels et emprise FABDEM par rapport aux coordonnées des sites.

```bash
python -m phes_assessment.cli data-qa --start-year 2002 --end-year 2023
```

Options utiles : `--sites` pour utiliser un CSV spécifique (vérification
emprise), `--start-year` / `--end-year` pour ajuster la fenêtre attendue. La
sortie affiche un tableau synthétique facilitant les contrôles avant d’alimenter
les pipelines. Relancez `data-qa` après la décompression pour confirmer
qu’aucun `.tif.gz` ne subsiste et que la couverture temporelle est complète.

### Simuler le bilan hydrique stochastique

Une fois les séries climatiques disponibles, la commande `hydro-sim` exécute un
Monte Carlo (désormais **10 000 tirages par site**) qui applique des coefficients
aléatoires de ruissellement/infiltration/évaporation tout en conservant la masse
(runoff + infiltration ≤ 1). La sortie inclut la distribution annuelle, la
probabilité d’autonomie (>0) ainsi que des diagnostics saison sèche (médiane et
P90 du déficit → volume d’appoint nécessaire).

```bash
python -m phes_assessment.cli hydro-sim \
	--climate results/climate_series.csv \
  --iterations 10000 \
	--output results/hydrology_summary.parquet
```

Ajoutez `--basins results/site_basins.geojson` pour que le modèle utilise les aires
contributives issues de `site-basins` au lieu des surfaces des réservoirs.

La table affiche, pour chaque site, la médiane/p10/p90 annuels, la probabilité
que le stock reste positif, la probabilité de survivre à la saison sèche, la
**médiane du bilan saison sèche** et le **P90 du déficit saison sèche** (volume
d’appoint à prévoir). L’export `parquet` est automatique grâce à `pyarrow` (et
tombe en CSV seulement en cas d’exception).

#### Analyse de sensibilité globale (Sobol/Morris)

L’option `--sensitivity-method {sobol|morris}` déclenche une analyse globale via
SALib sur quatre facteurs (runoff, infiltration, ETP, fuites) avec un nombre
d’échantillons configurable (`--sensitivity-samples`) et la métrique de votre
choix (`median_balance`, `prob_positive`, `dry_median_balance`,
`dry_p90_deficit`). Un fichier `*_sensitivity.parquet` (ou CSV) est produit pour
alimenter les analyses de robustesse.

### Classer les sites via l’AHP

Les classes économiques (A → E) fournies par `n10_e001_12_sites_complete.csv`
reprennent la méthodologie de Stocks et al. (2021) : coût unitaire `Classe A`
= 530 000 $/MW et 47 000 $/MWh, puis majoration par pas de 25 % jusqu’à la
classe E (×2). La commande `ahp-rank` combine ces coûts avec les diagnostics
hydrologiques issus de `hydro-sim` et quelques indicateurs d’infrastructure
(head, séparation, pente, ratio eau/roche) pour produire un classement AHP
pondéré.

```bash
python -m phes_assessment.cli ahp-rank \
    --hydrology-summary results/hydrology_summary.parquet \
    --output results/ahp_rankings.parquet \
    --economic-weight 0.45 --hydrology-weight 0.4 --infrastructure-weight 0.15
```

Il est possible de définir ces paramètres (ainsi que `cycles_per_year`, `lifetime_years`,
`discount_rate`, `round_trip_efficiency`) dans un fichier JSON/YAML et de les injecter via
`--weights-config` :

```json
{
  "economic_weight": 0.5,
  "hydrology_weight": 0.35,
  "infrastructure_weight": 0.15,
  "cycles_per_year": 320,
  "discount_rate": 0.045
}
```

```bash
python -m phes_assessment.cli ahp-rank \
    --hydrology-summary results/hydrology_summary.parquet \
    --weights-config configs/ahp_weights.json
```

Sous le capot :

- les coûts par MW/MWh sont annualisés sur 60 ans (taux 5 %, 300 cycles/an,
  rendement 81 %) pour calculer un LCOS simplifié (hypothèses Stocks et al.
  2021 + Lopez et al. 2024), normalisé comme critère économique ;
- les probabilités `prob_positive_annual_balance` et
  `dry_season_prob_positive` servent au critère hydrologique ;
- `Head (m)`, `Separation (km)`, `Slope (%)` et `Combined water to rock ratio`
  fournissent un score d’infrastructure.

L’export (CSV ou Parquet) contient les sous-scores, le LCOS estimé et le rang
final pour audit ou intégration dans une matrice multicritère plus large.

#### Références scientifiques des paramètres

| Paramètre | Valeur par défaut | Source / justification |
| --- | --- | --- |
| Coefficient de ruissellement (`runoff_min`–`runoff_max`) | 0,30 → 0,80 | Descroix et al. (2010) documentent le « paradoxe sahélien » et des pics de ruissellement pouvant atteindre 0,8 dans les bassins béninois ; aligné sur les observations régionales Donga/Koupendri. |
| Coefficient d’infiltration (`infiltration_min`–`infiltration_max`) | 0,05 → 0,25 | Mesures de Kamagaté et al. (2007) (5–24 % d’infiltration directe) complétées par Azuka & Igué (2020) sur la variabilité inter-sites. |
| Stock initial (`initial_storage_fraction`) | 60 % de la capacité | Recommandation PNNL (Pracheil et al., 2025) pour les études préliminaires PHES fermés afin de couvrir la saison sèche de départ. |
| Multiplicateur ETP (`evap_mean`, `evap_std`, `evap_min`, `evap_max`) | 1,0 ± 0,1 borné à [0,5 ; 1,5] | Incertitude locale sur ERA5 (Hersbach et al., 2020) + marges utilisées par Simon et al. (2023) dans les ACV PHES. |
| Fuites linéaires (`leakage_min`–`leakage_max`) | 0,05 → 0,20 % du stock / mois | Fourchette tirée des évaluations DOE/PNNL (Pracheil et al., 2025 ; Simon et al., 2023) sur les pertes structures béton/liner. |

Ces sources sont détaillées dans `docs/methodology.md` et les références RDF associées (`Close_loop_Ben.rdf`).

### Exporter les masques GeoJSON des sites

Pour faciliter la visualisation ou le post-traitement (SIG/QGIS), il est désormais
possible d’exporter directement les buffers circulaires générés autour de chaque
réservoir au format GeoJSON :

```bash
python -m phes_assessment.cli site-masks --output results/site_masks.geojson
```

Ceci crée un fichier `FeatureCollection` contenant un polygone par `pair_identifier`
avec, dans les métadonnées, le rayon utilisé et l’EPSG de projection choisi pour le
buffer.

Options disponibles :

- `--sites` pour pointer vers un CSV alternatif ;
- `--buffer-meters` afin de redéfinir le rayon (mètres) appliqué autour de chaque
  réservoir ;
- `--utm-epsg` pour choisir l’EPSG de la projection métrique temporaire ;
- `--output` pour définir le fichier GeoJSON cible (par défaut `results/site_masks.geojson`).

### Dériver les bassins versants à partir du FABDEM

Le workflow hydrologique utilise désormais l’aire contributive réelle de chaque paire de
réservoirs au lieu d’un simple buffer. La commande `site-basins` appelle WhiteboxTools pour
remplir les dépressions du FABDEM, calculer les directions d’écoulement (`D8`) et extraire
le bassin amont associé au point `Upper` de chaque site. Le résultat est un GeoJSON qui
contient les polygones, l’aire en km² et les métadonnées de calcul.

```bash
python -m phes_assessment.cli site-basins \
    --output results/site_basins.geojson \
    --dem-margin-km 15
```

Conseils pratiques :

- installez WhiteboxTools via `pip install -e .[full]` ou `pip install whitebox` avant
  d’exécuter cette commande ;
- utilisez `--work-dir tmp/whitebox` si vous souhaitez conserver les rasters intermédiaires
  pour inspection SIG ;
- le fichier GeoJSON peut servir directement aux autres commandes (`--basins`).

### Pipeline complet (ordre recommandé)

1. **Préparer les rasters climatiques**
   - ERA5 : `python process_era5_standard_evapotranspiration.py --start-year 2002 --end-year 2023 --output-dir data/era5` (requiert `cdsapi` + `~/.cdsapirc`).
   - CHIRPS : décompresser tous les `.tif.gz` dans `data/chirps/` (voir section QA ci-dessous).

     ```bash
     for f in data/chirps/*.tif.gz*; do
       gunzip -f "$f"
     done
     ```

     (Adapter la commande si les fichiers proviennent d’une archive `*.tif.gz.1`.)
2. **Dériver les bassins** : `python -m phes_assessment.cli site-basins --output results/site_basins.geojson` (WhiteboxTools requis).
3. **Agrégations climatiques** : `python -m phes_assessment.cli climate-series --dataset both --basins results/site_basins.geojson --output results/climate_series.parquet` (répéter par sous-période si besoin).
4. **Simulation hydrologique** : `python -m phes_assessment.cli hydro-sim --climate results/climate_series.parquet --basins results/site_basins.geojson --output results/hydrology_summary.parquet`.
5. **Classement AHP** : `python -m phes_assessment.cli ahp-rank --hydrology-summary results/hydrology_summary.parquet --output results/ahp_rankings.parquet`.

Chaque étape est traçable via `docs/methodology.md`, et les fichiers générés dans `results/`
permettent de documenter la reproductibilité du pipeline.

## Méthodologie et références scientifiques

- Les séries **CHIRPS** et **ERA5** mobilisées par `climate-series` sont justifiées par les
  travaux de Funk et al. (2015) sur le suivi des extrêmes tropicaux et Hersbach et al. (2020)
  concernant la réanalyse ERA5.
- La génération de masques et l’échantillonnage d’altitudes s’appuient sur le MNT **FABDEM**
  (Hawker et al., 2022) qui retire l’effet des forêts/bâtis—condition nécessaire pour
  estimer correctement les heads.
- La sélection des sites et le paramétrage multi-critères suivent les recommandations du
  _Global Atlas of Closed-Loop PHES_ (Stocks et al., 2021) et l’approche GIS+AHP étudiée
  par Lopez et al. (2024) pour le Bénin.
- Les gammes de ruissellement/infiltration et le diagnostic du « paradoxe sahélien » utilisés
  dans `hydro-sim` proviennent des travaux de Kamagaté et al. (2007), Azuka & Igué (2020) et
  Descroix et al. (2010) sur les bassins voisins (Donga, Koupendri, Sahel occidental).
- Les sections « QA énergétique » et « enjeux climatiques » du projet reprennent les analyses
  régionales de Mensah et al. (2023) ainsi que les évaluations environnementales et ACV de
  Simon et al. (2023) et Pracheil et al. (2025) pour replacer les sites PHES dans les
  trajectoires nationales.

Une table de correspondance détaillée entre chaque module CLI et les références citées est
disponible dans `docs/methodology.md`.

## Étapes suivantes

- Industrialiser la suite de tests (jeu synthétique + `pytest`) afin de valider la conservation de masse, les métriques saison sèche et l’indépendance RNG.
- Ajouter une orchestration (Snakemake/Prefect) reliant `process_era5_standard_evapotranspiration.py`, `site-basins`, `climate-series`, `hydro-sim` et `ahp-rank`.
- Étendre la QA (`data-qa`) pour signaler automatiquement les fichiers CHIRPS compressés, proposer commandes de décompression et vérifier la complétude des GeoJSON.
