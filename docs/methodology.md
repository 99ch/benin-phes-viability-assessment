# Méthodologie scientifique et références

Ce document relie chaque composant du workflow PHES en boucle fermée aux publications scientifiques et sources institutionnelles listées dans `Close_loop_Ben.rdf`. Il sert d'annexe méthodologique pour démontrer la conformité du pipeline avec les pratiques décrites dans la littérature récente.

## Correspondance workflow ↔ littérature

| Module / donnée                              | Finalité dans ce dépôt                                        | Choix méthodologiques clés                                                                                                                                          | Références principales                                                                                                                                                                                                                                                               |
| -------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `climate-series` (CHIRPS)                    | Agréger les précipitations mensuelles 0.05° sur buffers sites | Produit CHIRPS v2.0 mêlant CCD + stations, latence 2-3 semaines ; recommandé pour suivi des extrêmes tropicaux                                                      | Funk et al., 2015 [[doi](https://www.nature.com/articles/sdata201566)]                                                                                                                                                                                                               |
| `climate-series` (ERA5)                      | Estimer l'évapotranspiration potentielle annuelle             | Réanalyse ERA5 (31 km, horaire) offrant meilleure corrélation avec GPCP que ERA-Interim                                                                             | Hersbach et al., 2020 [[doi](https://onlinelibrary.wiley.com/doi/10.1002/qj.3803)]                                                                                                                                                                                                   |
| `site-masks`, `site-basins`, `fabdem-sample` | Générer buffers/bassins/altitudes pour 12 sites               | Utilisation de FABDEM (MNT 30 m « bare-earth ») pour éliminer biais forêt/bâti + dérivation auto des bassins (remplissage, D8, watershed via WhiteboxTools)         | Hawker et al., 2022 [[doi](https://doi.org/10.1088/1748-9326/ac4d4f)] ; Neal & Hawker, 2023 [[dataset](https://data.bris.ac.uk/data/dataset/s5hqmjcdj8yo2ibzi9b4ew3sn)] ; Lindsay, 2016 [[proc.](https://scholar.uwindsor.ca/cgi/viewcontent.cgi?article=1008&context=uwill-cmluc)]  |
| `ahp-rank`                                   | AHP sur classes Stocks + hydrologie                           | Classes économiques A→E de Stocks et al. (2021) utilisées telles quelles pour l’axe économique, combinaison hydrologie (probabilités) + infrastructure              | Stocks et al., 2021 [[doi](<https://www.cell.com/joule/abstract/S2542-4351(20)30559-6>)]; Lopez et al., 2024 [[RG](https://www.researchgate.net/publication/385803558_GIS_and_AHP_method_to_identify_favorable_sites_for_P2_PHES_installation_in_Benin)]                             |
| Site screening                               | Justifier l’approche GIS + AHP                                | Approches multi-critères pour PHES fermés (atlas mondial, AHP Benin)                                                                                                | Stocks et al., 2021 [[doi](<https://www.cell.com/joule/abstract/S2542-4351(20)30559-6>)]; Lopez et al., 2024 [[RG](https://www.researchgate.net/publication/385803558_GIS_and_AHP_method_to_identify_favorable_sites_for_P2_PHES_installation_in_Benin)]                             |
| `hydro-sim`                                  | Monte Carlo (2k–10k tirages) sur bilans mensuels              | Gammes de ruissellement/infiltration alignées sur études Donga/Koupendri (5–25 %) ; prise en compte du « paradoxe sahélien » (hausse ruissellement liée aux usages) | Kamagaté et al., 2007 [[doi](https://www.sciencedirect.com/science/article/pii/S163107130700096X)]; Azuka & Igué, 2020 [[doi](https://doi.org/10.1080/02626667.2020.1729360)]; Descroix et al., 2010 [[doi](https://www.mdpi.com/2073-4441/2/2/170)]                                 |
| QA énergétique & contexte                    | Dimensionner besoins Benin et trajectoires RE                 | Analyse critique du mix (2010-2018) + politiques nationales pour contextualiser la demande                                                                          | Mensah et al., 2023 [[doi](https://www.sciencedirect.com/science/article/pii/S0960148122017256)]; Communiqués Gouvernement du Bénin (Illoulofin/FORSUN)                                                                                                                              |
| Environnement & ACV                          | Qualifier bénéfices / risques du PHES fermé                   | ACV (58–530 gCO2e/kWh) + guide PNNL sur impacts et mitigations + comparaison DOE open/closed loop                                                                   | Simon et al., 2023 [[doi](https://doi.org/10.1021/acs.est.2c09189)]; Pracheil et al., 2025 [[rapport](https://www.osti.gov/biblio/2500381)]; DOE 2024 [[lien](https://www.energy.gov/eere/water/articles/comparison-environmental-effects-open-loop-and-closed-loop-pumped-storage)] |

## Détails par composant

### Données climatiques

- **CHIRPS v2.0** fournit un enregistrement 1981-présent combinant observations IR et stations, démontré comme fiable pour surveiller les extrêmes dans la Corne de l'Afrique et d'autres régions tropicales [[Funk et al., 2015](https://www.nature.com/articles/sdata201566)]. Notre agrégateur applique exactement les résolutions et latences décrites (0.05° et mise à jour mensuelle), puis convertit en volumes via l'aire des buffers.
- Depuis novembre 2025, les polygones d'échantillonnage proviennent de `site-basins`, qui calcule automatiquement les bassins versants en remplissant les dépressions FABDEM puis en appliquant un routage D8/`watershed` (WhiteboxTools [[Lindsay, 2016](https://scholar.uwindsor.ca/cgi/viewcontent.cgi?article=1008&context=uwill-cmluc)]). Les séries CHIRPS/ERA5 sont donc agrégées sur l'emprise hydrologique réelle avant conversion en volume.
- **ERA5** offre une réanalyse supérieure (résolution 31 km, incertitude horaire) permettant d'estimer l'évapotranspiration potentielle avec un biais réduit vs ERA-Interim [[Hersbach et al., 2020](https://onlinelibrary.wiley.com/doi/10.1002/qj.3803)]. L'option `--dataset era5` applique ces séries selon la fenêtre temporelle 2002-2023.

### Modèle numérique de terrain, buffers et bassins

- **FABDEM v1.2** supprime les effets bâtiments/forêts du Copernicus DEM, réduisant l'erreur absolue moyenne à 1.12 m en zone urbanisée [[Hawker et al., 2022](https://doi.org/10.1088/1748-9326/ac4d4f)]. Les commandes `fabdem-sample` et `site-masks` s'alignent sur cette ressource en reprojetant localement en UTM avant de calculer les buffers.
- La commande `site-basins` applique ensuite WhiteboxTools (`FillDepressions` → `D8Pointer` → `Watershed`) avec un tampon DEM configurable pour extraire l'aire contributive réelle associée à chaque point `Upper`. Ces polygones servent de référence unique pour l'échantillonnage CHIRPS/ERA5 et pour la surface efficace du modèle hydrologique.
- La stratégie de **screening AHP** reprend le flux décrit par Lopez et al. (2024) pour le Bénin, combiné au _Global Atlas of Closed-Loop PHES_ [[Stocks et al., 2021](<https://www.cell.com/joule/pdf/S2542-4351(20)30559-6.pdf>)] pour fixer des critères de pente, head et proximité réseau.

### Paramétrage hydrologique

- Nos coefficients de ruissellement/infiltration (Beta 0.3–0.8 et 0.05–0.25) reflètent les mesures de la Donga (5–24 % d'infiltration directe) [[Kamagaté et al., 2007](https://www.sciencedirect.com/science/article/pii/S163107130700096X)] ainsi que les expériences micro-parcelles de Koupendri (effet limité du relief mais forte sensibilité à l'usage des terres) [[Azuka & Igué, 2020](https://doi.org/10.1080/02626667.2020.1729360)].
- Les coefficients d'infiltration tirés sont systématiquement tronqués pour que `C_runoff + C_inf ≤ 1` (l'eau qui reste pouvant représenter les autres pertes ou le ruissellement local non modélisé), garantissant ainsi la conservation de masse signalée dans les travaux précités.
- Le modèle prend en compte l'augmentation des ruissellements sahéliens liée aux changements d'occupation des sols ("paradoxe sahélien") reportée par Descroix et al. [[2010](https://www.mdpi.com/2073-4441/2/2/170)], ce qui justifie une borne haute de ruissellement à 0.8.
- Les pertes par fuites linéaires (0.05–0.2 % du volume/mois) et la variabilité de l'ETP (±10 %) couvrent les incertitudes locales décrites dans les revues PHES [[Blakers et al., 2021](https://doi.org/10.1088/2516-1083/abeb5b)].
- Les diagnostics saison sèche suivent la période (novembre–mars) et produisent la médiane du bilan, ainsi que les fractiles (P10/P90) et le déficit médian/P90 interprété comme volume d’appoint à mobiliser.
- Les volumes entrants sont désormais calculés en multipliant les hauteurs CHIRPS/ERA5 par la surface des bassins `site-basins`, ce qui remplace l'approximation antérieure basée sur la somme des surfaces des réservoirs upper/lower.

#### Tableau des coefficients et références

| Paramètre                                               | Distribution / bornes               | Références principales                      | Notes                                                                                                              |
| ------------------------------------------------------- | ----------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Ruissellement (`runoff_range`, Beta α=3.5 / β=4.0)      | 0,30 → 0,80                         | Descroix et al., 2010 ; Stocks et al., 2021 | Borne haute alignée sur le « paradoxe sahélien » (hausse des coefficients après changement d’occupation des sols). |
| Infiltration (`infiltration_range`, Beta α=2.0 / β=5.0) | 0,05 → 0,25                         | Kamagaté et al., 2007 ; Azuka & Igué, 2020  | Tronqué pour garantir `runoff + infiltration ≤ 1`, cohérent avec les mesures Donga/Koupendri.                      |
| Multiplicateur ETP (`evap_mean`, `evap_std`, bornes)    | Normal 1,0 ± 0,1, borné [0,5 ; 1,5] | Hersbach et al., 2020 ; Simon et al., 2023  | Convertit ERA5 en mm/mois et ajoute ±10 % d’incertitude locale utilisée en ACV PHES.                               |
| Fuites (`leakage_fraction`)                             | Uniforme 0,0005 → 0,002             | Pracheil et al., 2025 ; Simon et al., 2023  | Reflète les pertes linéaires observées sur ouvrages béton + liners polymères.                                      |
| Stock initial (`initial_storage_fraction`)              | 0,6 de la capacité                  | Pracheil et al., 2025 (PNNL)                | Préconisé pour simuler une saison sèche complète dès la première année d’exploitation.                             |

Les valeurs sont exposées dans `hydro-sim` via des options CLI et peuvent être modifiées ou externalisées dans un fichier de configuration.

### Contexte énergétique et environnemental

- L'analyse du mix béninois souligne la croissance rapide de la demande et la dépendance aux importations (1320 GWh en 2018) [[Mensah et al., 2023](https://www.sciencedirect.com/science/article/pii/S0960148122017256)], ce qui motive nos scénarios de besoins longue durée.
- Les projets solaires Illoulofin/FORSUN (2024) et la centrale Maria-Gléta 2 sont cités pour justifier les hypothèses d'intégration EnR dans l'AHP (sources gouvernementales listées dans le RDF).
- Les bénéfices climatiques et les risques environnementaux des PHES fermés sont documentés via l'ACV de Simon et al. (58–530 gCO2e/kWh) et le guide PNNL 2025, complétés par la note DOE sur les différences open/closed loop.

### Analyses de sensibilité

- Une analyse de sensibilité globale (méthodes de Sobol et de Morris, implémentées via SALib) explore l’influence des facteurs `runoff`, `infiltration`, `evaporation` et `fuites` sur différentes métriques (médiane annuelle, probabilité d’autonomie, bilan saison sèche, déficit saison sèche). Ces méthodes, couramment utilisées pour les bilans hydrologiques multi-paramètres, permettent d’identifier les coefficients dominants avant le couplage techno-économique.

## Chaîne de traitement reproductible

1. **ERA5** – Script `process_era5_standard_evapotranspiration.py` (API Copernicus) → GeoTIFF multi-bandes (`era5_<année>.tif`). Réf. : Hersbach et al., 2020.
2. **CHIRPS** – Téléchargement + décompression `.tif.gz`, suivi d’un contrôle `phes-data data-qa`. Réf. : Funk et al., 2015.
3. **Bassins** – `phes-data site-basins --output results/site_basins.geojson` (FABDEM + WhiteboxTools). Réf. : Hawker et al., 2022 ; Lindsay, 2016.
4. **Agrégation climatique** – `phes-data climate-series --dataset both --basins results/site_basins.geojson` (volumes `precip_mm` + `etp_mm`). Réf. : Funk et al., 2015 ; Hersbach et al., 2020.
5. **Hydrologie** – `phes-data hydro-sim --climate results/climate_series.parquet --basins results/site_basins.geojson`. Réf. : Kamagaté et al., 2007 ; Descroix et al., 2010 ; Azuka & Igué, 2020 ; Simon et al., 2023 ; Pracheil et al., 2025.
6. **Classement** – `phes-data ahp-rank --hydrology-summary results/hydrology_summary.parquet`. Réf. : Stocks et al., 2021 ; Lopez et al., 2024.

Chaque étape renvoie aux identifiants RDF du dossier `Close_loop_Ben.rdf` afin d’assurer la traçabilité scientifique.

## Hypothèses et limites

1. **Stationnarité 2002-2023** : faute de projections CMIP6 locales, la simulation suppose que les distributions observées restent valides sur l'horizon d'étude.
2. **Hydrologie simplifiée** : pas de routage rivière ; on applique un bilan réservoir unique conforme aux évaluations préliminaires recommandées par PNNL (2025) pour les dossiers FERC.
3. **Données socio-économiques** : l'AHP s'appuie sur publications publiques (Mensah 2023, communiqués gouvernementaux). Des enquêtes locales seraient nécessaires pour affiner les poids économiques.

Consultez `README.md` pour la description opérationnelle des commandes et ce document pour citer la méthodologie dans vos rapports.
