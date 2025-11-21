**STEP en boucle fermée au Bénin : viabilité hydrologique et priorisation Monte Carlo–AHP**

**Auteurs**

Maurel R. Aza-Gnandji¹, Ibrahim Lopez², Arouna Oloulade², KB Audace Didavi², Cossi T. Nounangnonhou², NNB Chilavert N'Dah³, Taohidi A. Lamidi², Julian Hunt⁴, François-Xavier N. Fifatin²

**Affiliations**

1. Université Nationale d'Agriculture, BP 43, Kétou, Bénin.

2. Université d'Abomey-Calavi,01 BP 526, Abomey-Calavi, Bénin.

3. Université Catholique d'Afrique de l'Ouest, 04 BP 928, Cotonou, Bénin.

4. Initiative pour le climat et la qualité de vie, Université des sciences et technologies du roi Abdallah (KAUST), Thuwal 23955-6900, Arabie saoudite

**Auteur correspondant**

**Nom:** Maurel R. Aza-Gnandji

**Affiliation:** Université nationale d'agriculture

**Adresse:** BP 43, Kétou, Bénin

**E-mail:** maurel.aza@gmail.com

**Téléphone\*\*** :\*\* +229 96 18 16 09

**Graphical Abstract**

RÉSUMÉ

Le Bénin demeure dépendant des importations électriques (25,6 %) alors que les besoins de l’axe solaire-éolien exigent un stockage longue durée. Nous avons réécrit le pipeline ERA5–CHIRPS pour échantillonner les précipitations et l’évapotranspiration directement sur les bassins versants dérivés de FABDEM/Whitebox et conservons un cadre probabiliste combinant simulation Monte Carlo (10 000 itérations/site) et Processus Analytique Hiérarchique (AHP) sur 12 sites du nord Bénin. Le bilan hydrique mensuel (2002-2023) propage l’incertitude sur le ruissellement (distribution Beta bornée 0,30-0,80), l’infiltration (0,05-0,25), les biais climatiques et les surfaces effectives tout en intégrant les diagnostics de sensibilité globale.

Les apports captés par bassin excèdent désormais les pertes : les médianes annuelles varient de 1,8 à 21,3 GL/an et la probabilité d’obtenir un bilan positif dépasse 98 % pour l’ensemble des sites (Tableau 3). Cette marge se réduit cependant pendant la saison sèche : le 10e percentile du bilan saisonnier reste négatif (–2,7 à –249 GL), obligeant à prévoir des appoints ou une gestion multi-annuelle du stockage. Les indices de Sobol montrent que le ruissellement et l’infiltration expliquent respectivement 22 % et 19 % de la variance annuelle, tandis que l’évaporation et les fuites demeurent secondaires (<1 % chacun). L’alignement entre l’AHP et la nouvelle hydrologie s’est amélioré (ρ_Spearman=0,36) : les sites RES18951, RES11634 et RES35193 cumulent désormais des scores économiques et hydrologiques élevés, quand RES26025 et RES31412 restent prioritaires uniquement si l’on accepte des appoints saison sèche de 190–250 GL.

La bancabilité des STEP béninoises repose donc sur trois leviers concrets : (i) instrumenter les bassins pour verrouiller les coefficients de ruissellement et dimensionner les ouvrages de collecte ; (ii) intégrer des traitements d’étanchéité ciblés afin de contenir les pertes résiduelles (<0,5 GL/an pour les sites pilotes de 15 GWh) ; (iii) planifier des appoints saisonniers ou des pompages complémentaires pour couvrir les déficits saison sèche. Le couplage Monte Carlo–AHP offre un canevas reproductible pour hiérarchiser les investissements et suivre l’effet des mesures de mitigation, préalable indispensable à la mobilisation des bailleurs alignés sur l’objectif national de 630 GWh de stockage à l’horizon 2040.

MOTS-CLÉS

Stockage hydroélectrique par pompage (STEP) ; Boucle fermée ; Simulation Monte Carlo ; Processus Analytique Hiérarchique (AHP) ; Bilan hydrique ; Afrique de l’Ouest ; Bénin ; Énergies renouvelables

1. INTRODUCTION

La transition énergétique mondiale repose largement sur l'adoption à grande échelle de l'énergie solaire photovoltaïque et éolienne, qui représentait 80 % des nouvelles capacités électriques installées en 2023 [1]. Cependant, leur caractère intermittent nécessite des solutions de stockage efficaces pour assurer la stabilité du réseau. Parmi ces solutions, le stockage d'énergie hydraulique par pompage (PHES) se distingue par sa fiabilité et sa rentabilité dans les applications à grande échelle et de longue durée [2],[3]. Pourtant, son potentiel reste largement sous-exploité dans les régions tropicales comme l'Afrique de l'Ouest.

Le Bénin (plus de 14,5 millions d’habitants [4], 115 000 km²) incarne ces défis : croissance annuelle de la demande de 1,75 %, dépendance aux importations (25,6 % de l’offre) et forte vulnérabilité climatique [5],[6],[7]. Malgré le déploiement de projets solaires (Illoulofin 25 MWc, FORSUN 25 MWc [8],[9],[10]), leur intégration reste limitée par l’absence de stockage à grande échelle. Le recours aux centrales thermiques (ex. Maria-Gléta 2, 129,5 MW [11]) accroît l’empreinte carbone et freine les objectifs climatiques du pays [12].

Une évaluation préalable [13] a identifié, via SIG-AHP, 4 sites prioritaires et 17 secondaires pour des STEP à boucle ouverte. Ces systèmes soulèvent toutefois des enjeux liés aux droits d’eau et à l’environnement [14],[15]. Cette étude se concentre sur les STEP à boucle fermée, inexploitées au Bénin [16],[17]. L’approche globale existante ([18] Stocks et al., 2021) reposent sur des méthodes déterministes, un filtrage SIG sommaire, sans analyse hydrologique fine ni validation terrain. Elles restent inadaptées aux contextes pauvres en données et à l’évaluation de projets bancables.

L’objectif principal de cette étude est d’évaluer la faisabilité technique, économique et hydrologique des systèmes PHES à boucle fermée au Bénin, en intégrant explicitement la probabilité d’autonomie hydrique grâce à la récupération naturelle des eaux de pluie. Les objectifs spécifiques incluent : l’identification et la caractérisation des sites potentiels à l’échelle nationale à partir du modèle FABDEM (30 m) [19] ; la modélisation du bilan hydrique sur 22 ans (2002–2023) à partir des données CHIRPS (précipitations) et ERA5 (évapotranspiration) [20],[21], intégrant les incertitudes sur le ruissellement, l’infiltration et les biais climatiques ; la réalisation d’une simulation Monte Carlo (10 000 itérations/site) pour estimer la probabilité d’un bilan net positif ; et le classement multicritère des sites par la méthode AHP, combinant les dimensions economique, hydrologiques et environnementales. Cette recherche propose ainsi la première évaluation probabiliste nationale du potentiel PHES à boucle fermée au Bénin, intégrant la dynamique des précipitations et la capacité naturelle de compensation des pertes d’eau. Elle combine : les données satellitaires haute résolution (FABDEM, CHIRPS, ERA5), un traitement explicite de l’incertitude hydrologique, la simulation Monte Carlo et l’analyse multicritère AHP. Ce cadre méthodologique, reproductible et adapté aux régions à données limitées, constitue une base pour le développement futur de projets de stockage hydraulique durable en Afrique de l’Ouest.

2. MÉTHODOLOGIE

2.1 Zone d’Étude et Sélection des Sites

L’ensemble du territoire béninois (6-12,5°N, 0,75-3,85°E, 114 763 km²) constitue la zone d’étude pour l’évaluation du potentiel STEP à l’échelle nationale. Le pays présente un fort gradient climatique nord-sud : climat guinéen humide au sud (1 200-1 400 mm/an de précipitations) transitant vers une savane soudanienne semi-aride au nord (900-1 100 mm/an, saison sèche de six mois novembre-avril) [22]. La topographie varie des basses terres côtières (<50m d’altitude) à la chaîne de l’Atacora au nord-ouest (point culminant 641m) [23].

2.2 Sources de Données

Cette étude a utilisé une combinaison de données géospatiales et hydrologiques pour évaluer le potentiel de stockage d’énergie par pompage-turbinage en circuit fermé (PHES) à travers le Bénin, résumé dans le Tableau 1.

**Tableau 1 : Sources de données utilisées dans l’étude**

| **Objectif**                      | **Type de données**                                       | **Source**                      | **Résolution**        | ** \*\***Période\*\* |
| --------------------------------- | --------------------------------------------------------- | ------------------------------- | --------------------- | -------------------- |
| Analyse topographique             | Modèle numérique d'élévation (MNT)                        | FABDEM v1.2 [24]                | 30 m (RMSE : 1,953 m) | Statique             |
| Analyse hydrologique              | Précipitations                                            | CHIRPS v2.0 [25]                | 0,05° (~5,5 km)       | Mensuelle, 2002–2023 |
| Analyse hydrologique              | Évapotranspiration                                        | ERA5 [26]                       | 0,25° (~27,8 km)      | Mensuelle, 2002–2023 |
| Exclusion des zones non éligibles | Aires protégées, zones urbaines, infrastructures minières | UNEP-WCMC WDPA (2023) [27],[28] | Polygones vectoriels  | 2024                 |

Le modèle numérique d’élévation FABDEM v1.2 (résolution de 30 m, RMSE : 1,953 m) a été utilisé pour calculer les différences d’altitude (tête) et les volumes des réservoirs, offrant une précision supérieure par rapport au SRTM (RMSE : 3,556 m [29]) en éliminant les artefacts anthropogéniques tels que les bâtiments et les forêts.

CHIRPS v2.0 a fourni des données quotidiennes de précipitations à une résolution de 0,05° de 2002 à 2023, capturant la variabilité spatiale et temporelle des précipitations au Bénin. Les données de réanalyse ERA5 ont fourni l’évapotranspiration réelle mensuelle à une résolution de 0,25°, permettant le calcul du bilan hydrique.

Les polygones de la Base de données mondiale sur les aires protégées, des zones urbaines d’OpenStreetMap et du Cadastre minier du Bénin ont été utilisés pour exclure les zones sensibles sur le plan environnemental ou inadaptées.

2.3 Identification des Sites STEP à Boucle Fermée

Nous avons utilisé l’algorithme de l’atlas mondial de Stocks et al. [18] en traitant le MNT FABDEM 30 m pour détecter les paires d’élévation (ΔH 100–600 m, distance <10 km, surface ≥5 ha, pente <20°), modéliser les réservoirs par remplissage avec calcul des surfaces, volumes (2–500 GWh) et géométrie des barrages, délimiter les bassins versants via l’algorithme avec calcul de surface, exclusion des chevauchements et validation boucle fermée (aucune rivière à <2 km), puis filtrer les sites incompatibles avec les zones protégées, la proximité urbaine (>20 000 hab. à <5 km), un ratio bassin:réservoir <5:1 ou une hauteur de barrage >100 m **(Figure 1)**.

**Figure 1** : Algorithme d’indentification des sites PHES en Réplubique du Bénin

2.3.1 Évaluation Économique et Classification des Sites

La faisabilité économique a été évaluée selon la méthodologie de Stocks et al. [18], développée avec des consultants en ingénierie hydroélectrique et calibrée à ±5% sur des projets réels. Le coût total comprend deux composantes largement indépendantes : le **stockage d’énergie** (réservoirs et barrages) et la conversion de puissance (centrale, turbines, tunnels).

L’énergie E (MWh) stockée dans le réservoir supérieur est donnée par :

$$
E_{\text{stock}} = \frac{f \, \eta \, \rho \, g \, V \, H}{3.6 \times 10^9}
	ag{1}
$$

où _f_ = fraction utilisable du réservoir (85%), _η_ = efficacité turbine (90%), _ρ_ = densité de l’eau (1 000 kg/m³), _V_ = volume du réservoir supérieur (m³), _g_ = accélération de la gravité (9,81 m/s²), _H_ = hauteur de chute (m). Cette relation reprend le dimensionnement volumétrique utilisé par Stocks et al. [18] pour relier directement les paramètres géométriques aux MWh stockés.

Le Coût du stockage énergétique Proportionnel au volume du barrage _R_ (m³) :

$$
C_{\text{stock}} = C \times R
	ag{2}
$$

avec _C_ = 168 $/m³ (coût moyen construction barrage, excavation terre) et une calibration issue du Global Atlas of Closed-Loop PHES [18].

Le coût optimal est obtenu pour sites à grande chute _H_ et rapport eau:roche (_V/R_) élevé.

Le Coût de conversion de puissance, Comprend tunnel (vertical + horizontal) et centrale (civil + mécanique + électrique) :

$$
C_{\text{power}} = \alpha \, P^{0.75} H^{-0.5} + \beta \, S
	ag{3}
$$

où _P_ = puissance (MW), _S_ = séparation réservoirs (m) et _(α, β)_ sont des coefficients calibrés (
respectivement pour les lots tunnel/civils et la longueur de galerie) sur un corpus de devis STEP compilé par Stocks et al. [18]. Ces relations reflètent des économies d’échelle : le coût central croit sous-linéairement (_P_0.75) et diminue avec la chute (\_H_-0.5).

Les sites sont classés selon coût total par MW de puissance et MWh de stockage. Exemple : Site Classe A (optimal) : 530 000 $/MW + 47 000 $/MWh. Classes B à E augmentent par incréments de 25%, Classe E = double Classe A. Sites sous Classe E sont rejetés.

Le coût nivelé du stockage (LCOS) est calculé sur durée de vie 60 ans, taux actualisation 5%, efficacité cycle 81% (pompage + turbinage) :

$$
	ext{CRF} = \frac{r (1+r)^n}{(1+r)^n - 1}
	ag{4}
$$

$$
	ext{LCOS} = \frac{(\text{CAPEX} \times \text{CRF}) + C_{\text{O\&M,fixe}} + C_{\text{O\&M,var}}}{E_{\text{annuel}}}
	ag{5}
$$

où la Capital Recovery Factor (CRF) convertit le CAPEX total (_CAPEX_ = coût puissance + coût stockage) en annuité sur _n_ = 60 ans pour un taux _r_ = 5%, _C_{O\&M,fixe} = 8\,210* $/MW/an, \_C*{O\&M,var} = 0,3* $/MWh et \_E*{annuel} = E*{stock} \times 300* cycles/an × 81% d’efficacité. Ces paramètres proviennent des calibrations LCOS proposées par Stocks et al. [18]. Site Classe A (6h stockage, 300 cycles/an) : LCOS = 40 $/MWh. Site Classe E : LCOS = 64 $/MWh (+60%). Le CAPEX représente 60-75% du LCOS, rendant le coût très sensible au taux d’actualisation (+1% taux → +10-12% LCOS).

2.4 Modèle de Bilan Hydrique

Les STEP en boucle fermée n’étant pas alimentées par un cours d’eau, leur autonomie dépend de l’équilibre entre apports du bassin versant et pertes par évaporation ou infiltration [16]. L’étude retient la solution la plus réaliste dans le contexte béninois : capter les précipitations locales via un bassin versant dédié puis les diriger vers le réservoir supérieur (Figure 2). Une gestion saisonnière est appliquée : ouverture des ouvrages de collecte pendant la saison humide (juin-septembre) pour reconstituer les volumes, fermeture pendant la saison sèche (octobre-mai) afin de limiter l’évaporation.

**\*Figure 2\*\*** : Cycle saisonnier climatique moyen (2002-2023) en République du Bénin\*

Le bilan hydrique annuel est calculé sur pas mensuel (2002-2023, soit 264 mois) comme suit :

$$
B = \sum_{m=1}^{12} \Big[(1+\delta_P) P_m S_b C_r - (1+\delta_E) E_m S_e - I_m S_b - \lambda_m V_{m-1}\Big]
	ag{6}
$$

où :

- $B$ = volume net annuel (GL).
- $P_m$ = précipitation mensuelle (m) issue de CHIRPS, $S_b$ = surface du bassin versant (m²) dérivée de FABDEM.
- $C_r$ = coefficient de ruissellement tiré dans $\mathcal{B}(\alpha=3{,}5,\beta=4{,}0)$ et borné 0,30-0,80 [31],[37].
- $E_m$ = évapotranspiration mensuelle (m) ERA5, $S_e$ = surface libre évaporante (m²).
- $I_m$ = terme d’infiltration ($\mathcal{B}(2,5)$ repliée 0,05-0,25) représentant la fraction de lame de pluie perdue dans le socle fracturé [33].
- $\lambda_m$ = pertes linéaires (0,0005-0,002) sur le volume stocké $V_{m-1}$.
- $\delta_P$, $\delta_E$ = biais multiplicatifs Uniformes (±10 %) appliqués respectivement aux lames de pluie et à l’évaporation pour couvrir les incertitudes locales.

Les surfaces efficaces proviennent des polygones `site-basins` (WhiteboxTools) dérivés de FABDEM, ce qui remplace les anciens buffers circulaires et garantit une conservation stricte de l’aire contributive [24],[32]. Les coefficients aléatoires reflètent les conditions soudano-sahéliennes : $C_r$ suit une loi Beta bornée à 0,30-0,80 [31], l’infiltration 0,05-0,25 [33], l’évaporation applique un multiplicateur normal (μ = 1, σ = 0,1) tronqué entre 0,5 et 1,5 pour couvrir les biais locaux [34],[35], et les fuites linéaires représentent 0,05-0,20 % du volume par mois. La somme mensuelle distingue l’apport brut (précipitations × bassin) des pertes (infiltration, évaporation, fuites) et permet de suivre l’état de stockage borné entre 0 et la capacité.

Comme ces paramètres sont incertains et interagissent de manière non linéaire, les moyennes simples ne suffisent pas : la quantification probabiliste détaillée est décrite en Section 2.5.

2.5 Quantification d’Incertitude Monte Carlo

Les distributions de probabilité des paramètres incertains reposent sur la littérature régionale : le ruissellement suit une loi Beta (α = 3,5 ; β = 4,0) resserrée entre 0,30 et 0,80, l’infiltration une loi Beta (α = 2,0 ; β = 5,0) limitée à 0,05-0,25, tandis que l’évaporation applique un multiplicateur normal tronqué (μ = 1, σ = 0,1) et que les fuites linéaires sont tirées uniformément entre 0,0005 et 0,002 du volume mensuel [34],[35]. Les surfaces de bassin sont soumises à un facteur Uniforme (0,9-1,1) pour représenter l’incertitude liée aux seuils de délimitation. Nous échantillonnons cet espace paramétrique via un hypercube latin (10 000 scénarios/site, soit 120 000 bilans annuels) afin d’obtenir une couverture dense des combinaisons plausibles [36]. Chaque tirage génère une trajectoire mensuelle complète ; la probabilité d’autonomie correspond à la proportion de tirages où la somme annuelle reste positive.

2.5.1 Justification du Coefficient de Ruissellement

Le coefficient de ruissellement (Cr) demeure le paramètre le plus influent du modèle (voir Section 3.3). Faute de jaugeages locaux, nous retenons une loi Beta bornée entre 0,30 et 0,80, ce qui reflète la transition rapide observée dans la littérature entre plateaux latéritiques quasi-imperméables (Cr ≈ 0,6-0,8) et vallées encore végétalisées (Cr ≈ 0,3-0,4). Descroix et al. [37] rapportent déjà des valeurs médianes de 0,45 dans les bassins sahélo-soudaniens dégradés, tandis que Mahé et Paturel [38] montrent un déplacement de 0,08 à 0,28 entre 1950 et 2006 pour des bassins comparables. Leblanc et al. [39] confirment des valeurs de 0,20-0,35 sur les socles cristallins voisins, ce qui justifie la borne inférieure choisie (0,30) dès lors que les pentes identifiées dépassent 5 % et que la couverture arborée est inférieure à 40 %.

Cette loi Beta (α = 3,5 ; β = 4,0) concentre 70 % des tirages entre 0,35 et 0,65, ce qui reflète la mosaïque d’occupation des terres visible dans les images Sentinel de la chaîne de l’Atacora. Les indices de Sobol calculés sur les 10 000 tirages indiquent que Cr contribue en moyenne à 0,22 de la variance annuelle, devant l’infiltration (0,19), ce qui légitime la priorisation de campagnes de mesure in situ pour réduire l’incertitude globale.

Les indicateurs dérivés (P05, P50, P95, coefficient de variation) permettent ensuite de qualifier le profil de risque et d’alimenter la décision d’investissement. Les corrélations de Spearman calculées sur les 10 000 tirages identifient la contribution de chaque paramètre à la variance du bilan (Section 3.3).

2.6 Processus Analytique Hiérarchique (AHP)

La sélection de sites STEP implique des compromis entre objectifs concurrents (viabilité hydrologique, rentabilité économique, faisabilité technique, durabilité environnementale). L’AHP structure les décisions de manière hiérarchique **(F\*\***igure 3)\***\*,\*\*** **en pondérant les critères par comparaisons par paires basées sur la méthodologie de Saaty [41] Hydrologique (40%) : Critère dominant car la disponibilité en eau est une condition nécessaire à la viabilité. Le poids reflète le contexte semi-aride où l’hydrologie est la contrainte primaire (vs régions humides où la géologie peut dominer ; Économique (30%) : Viabilité commerciale et potentiel de génération de revenus. Les sites de plus grande capacité produisent plus de revenus mais nécessitent un investissement proportionnellement plus élevé, une échelle optimale existe. Inclut l’efficacité du bassin versant (ratio bassin:réservoir) comme proxy pour les coûts opérationnels à long terme (énergie de pompage vs recharge naturelle) ; Technique (20%)** :** Faisabilité d’ingénierie et complexité de construction. Les pentes raides favorisent la stabilité des barrages mais augmentent les coûts d’excavation. La longueur des tunnels (fonction de la hauteur de chute et de la distance de séparation) affecte le calendrier et le coût du projet. Limites de hauteur de barrage appliquées (>100m nécessite une conception spécialisée) ; Environnemental (10%)** :\*\* Poids le plus faible reflète la nature en boucle fermée (pas de perturbation de rivière, impacts écosystémiques minimaux comparés aux STEP en boucle ouverte). Cependant, l’inondation des réservoirs affecte l’utilisation des terres (agriculture, foresterie), considération non négligeable pour l’acceptation sociale.

**\*Figure 3\*\*** : Poids des critères principaux et des sous-critères de \*_l’AHP_

Les comparaisons par paires ont été établies à partir des poids selon l’échelle de Saaty (1 = égalité, 3 = importance modérée, 5 = forte, 7 = très forte, 9 = extrême, avec valeurs intermédiaires 2, 4, 6, 8), garantissant une cohérence parfaite (CR = 0,000) pour les matrices construites mathématiquement. Les sous-critères ont été normalisés sur [0,1] via la formule min-max :

puis agrégés en un score global par :

où = poids du critère principal (40%, 30%, 20%, 10%), = poids du sous-critère (50%, 30%, 20% pour hydrologique, etc.), = valeur normalisée [0,1]. Enfin, la convergence entre les classements AHP et Monte Carlo (basé sur P(viable) a été validée par la corrélation de Spearman [40].

3. RÉSULTATS

3.1 Caractéristiques des Sites Identifiés

L’algorithme d’identification des sites (Section 2.3) appliqué à l’ensemble du territoire béninois a identifié **12 sites candidats STEP à boucle fermée**, tous situés dans le nord du Bénin (hauts plateaux Atacora, bande de latitude 10-11°N). Cela se compare aux 13 sites de l’atlas mondial basé sur SRTM [18]. La suppression des artefacts de forêt/bâtiments de FABDEM (vs SRTM) a modifié les géométries des réservoirs (variations de volume de +15-30%) et déplacé plus de 3 emplacements de sites (précision de position ±2-5 km). Les sites identifiés **_(Figures : 4 ;5 ;6)_** représentent un large spectre de capacités de stockage (2-500 GWh), de surfaces de bassins versants (50-13 262 ha) et de hauteurs de chute (130-260m).

**Tableau 1 : Caractéristiques des 12 Sites STEP à Boucle Fermée Identifiés**

| ID Site  | Classe | Chute (m) | Énergie (GWh) | Stockage (h) | Réservoir Supérieur | Réservoir Inférieur | Bassin (ha) | Ratio B:R |
| -------- | ------ | --------- | ------------- | ------------ | ------------------- | ------------------- | ----------- | --------- |
| RES31412 | C      | 160       | 500           | 168          | 5 776 ha, 1 487 GL  | 2 906 ha, 1 497 GL  | 13 262      | 2,9:1     |
| RES26025 | E      | 130       | 500           | 168          | 4 582 ha, 1 843 GL  | 4 093 ha, 1 845 GL  | 10 892      | 2,4:1     |
| RES18951 | D      | 240       | 150           | 50           | 1 342 ha, 300 GL    | 449 ha, 300 GL      | 2 508       | 5,6:1     |
| RES34837 | D      | 180       | 150           | 50           | 909 ha, 399 GL      | 1 993 ha, 399 GL    | 1 635       | 4,1:1     |
| RES11634 | D      | 227       | 50            | 50           | 648 ha, 103 GL      | 390 ha, 105 GL      | 1 562       | 2,4:1     |
| RES37145 | E      | 230       | 50            | 50           | 457 ha, 104 GL      | 890 ha, 104 GL      | 598         | 1,3:1     |
| RES35193 | E      | 180       | 50            | 50           | 603 ha, 133 GL      | 675 ha, 131 GL      | 829         | 1,4:1     |
| RES15127 | E      | 260       | 50            | 50           | 396 ha, 91 GL       | 580 ha, 90 GL       | 658         | 1,7:1     |
| RES16527 | E      | 222       | 15            | 18           | 144 ha, 32 GL       | 430 ha, 33 GL       | 151         | 1,0:1     |
| RES34296 | E      | 210       | 15            | 18           | 319 ha, 34 GL       | 266 ha, 33 GL       | 2 287       | 69:1      |
| RES35358 | E      | 130       | 15            | 18           | 608 ha, 54 GL       | 257 ha, 55 GL       | 698         | 2,7:1     |
| RES36507 | E      | 240       | 15            | 18           | 286 ha, 29 GL       | 297 ha, 30 GL       | 598         | 2,1:1     |

_Note : Ratio **B:R** = Ratio **Bassin:Réservoir** (surface bassin versant / surface réservoir supérieur). La classe (C, D, E) fait référence à la classification de coût en capital de Stocks et al. [**18**] : Classe C = coût optimal (référence), Classe D = +25-50% du coût C, Classe E = double du coût C (~+100%). Classification basée sur coûts combinés de stockage (volume barrage, rapport **eau:roche**), tunnels (longueur, puissance) et centrale (chute **hydraulique,puissance**). La capacité énergétique fait référence au potentiel de stockage maximum, pas à la génération annuelle._

**Figure \*\***4\***\*. **Distribution of Large 150 GWh 50h Storage Closed-Loop Pumped Hydro Storage Sites Identified in this Study

**Figure \*\***5\*\*: Potential 150 GWh PHES sites. Upper reservoirs (light blue), lower reservoirs (white), tunnels (white), cost classes (red pins), and an infographic box are depicted.

**Figure \*\***6\***\*:\*\*** \*\*Superposition de réservoirs de 150 GWh: contours des études précédentes (noir) et de l'étude actuelle (blanc) .

Les 12 sites identifiés sont classés C (1 site), D (4 sites) et E (7 sites). Aucun site Classe A/B (coût optimal) n’a été identifié, suggérant que les caractéristiques topographiques et hydrologiques du Bénin, rapports eau:roche modérés, chutes de 130-260 m, séparations de réservoirs 1,4-4,0 km, génèrent des coûts en capital plus élevés comparés aux sites optimaux de l’atlas mondial (grands rapports eau:roche >50:1, chutes >400 m, courtes séparations <1 km). Les différences méthodologiques entre cette étude (FABDEM 30 m, suppression d’artefacts) et l’atlas de Stocks (SRTM 90 m) peuvent également influencer les classifications en affectant l’estimation des volumes de réservoirs et des surfaces de bassins versants. Tous les 12 sites nécessitent une évaluation hydrologique détaillée (Sections 3.2-3.4) pour valider leur viabilité économique malgré cette classification coût intermédiaire.

3.2 Référence Déterministe

L’application des valeurs médianes (Cr = 0,52 ; infiltration = 0,12 ; multiplicateur ERA5 = 1,00) au modèle mensuel fournit désormais un bilan positif pour l’ensemble des 12 sites grâce à la prise en compte explicite des surfaces de bassin. Les apports annuels reconstruits varient de 9,7 à 124,6 GL/an, pour des pertes cumulées de 6,4 à 103,3 GL/an, ce qui se traduit par des médianes comprises entre 1,8 GL/an (RES16527) et 21,3 GL/an (RES31412). Le site phare RES31412 conserve un profil très contrasté : les apports de la saison humide permettent de combler les pertes annuelles, mais la période sèche génère un déficit ponctuel de 191 GL au 10e percentile (Tableau 3).

**\*Figure 7\*\*** : Cycle saisonnier moyen – Site n10_001_RES31412_and_n10_RES31520\*

Les sites compacts de 15 GWh (RES35358, RES34296, RES36507, RES16527) affichent des bilans médianes de 1,8 à 3,5 GL/an pour des déficits saison sèche limités à 2,7-5,7 GL, ce qui les rend adaptés à un rôle de démonstrateur sous réserve d’un appoint ciblé. Les paires de 50-150 GWh bénéficient d’un surplus annuel de 2,8-7,2 GL mais nécessitent 11-55 GL d’appoint saisonnier pour sécuriser les cycles de pompage. Ces résultats confirment que la réduction des pertes reste utile pour optimiser le rendement, mais qu’elle n’est plus une condition d’existence du bilan annuel ; l’enjeu se déplace vers la gestion intra-annuelle du stockage.

**Tableau 3 : Synthèse hydrologique (Monte Carlo 10 000 tirages)**

| Site     | Capacité (GL) | Médiane annuelle (GL/an) | P10 annuel (GL/an) | P90 annuel (GL/an) | P(>0) (%) | P10 saison sèche (GL) |
| -------- | ------------- | ------------------------ | ------------------ | ------------------ | --------- | --------------------- |
| RES18951 | 299,7         | 2,83                     | 1,42               | 4,29               | 99,8      | -41,4                 |
| RES34837 | 398,7         | 7,16                     | 4,67               | 9,81               | 100,0     | -54,6                 |
| RES16527 | 32,4          | 1,81                     | 1,40               | 2,24               | 100,0     | -4,2                  |
| RES34296 | 33,2          | 2,49                     | 2,00               | 3,04               | 100,0     | -3,4                  |
| RES35358 | 53,7          | 3,47                     | 2,77               | 4,30               | 100,0     | -5,7                  |
| RES36507 | 29,1          | 2,42                     | 1,96               | 2,96               | 100,0     | -2,7                  |
| RES31412 | 1 487,3       | 21,30                    | 13,70              | 30,03              | 100,0     | -190,7                |
| RES26025 | 1 842,8       | 12,36                    | 4,38               | 20,72              | 98,0      | -249,7                |
| RES11634 | 103,4         | 2,85                     | 2,07               | 3,69               | 100,0     | -14,2                 |
| RES37145 | 103,8         | 5,00                     | 3,93               | 6,28               | 100,0     | -11,1                 |
| RES35193 | 131,2         | 4,10                     | 3,02               | 5,31               | 100,0     | -16,6                 |
| RES15127 | 90,3          | 2,64                     | 1,92               | 3,40               | 100,0     | -12,2                 |

_Note : les probabilités d’autonomie sont arrondies à 0,1 %; les déficits saison sèche correspondent au 10e percentile du cumul (novembre-mars)._

3.3 Distributions de Probabilité Monte Carlo

La propagation des incertitudes (Cr, infiltration, surfaces et biais climatiques) confirme que le bilan annuel reste robuste : l’ensemble des médianes et des P10 est positif et les distributions sont resserrées (Figure 8). Les sites de 15 GWh présentent des intervalles [P10-P90] de 1,4 à 3,0 GL/an, ceux de 50-150 GWh de 2,1 à 9,8 GL/an et les grandes paires (≥500 GWh) conservent un excédent de 4,4 à 30,0 GL/an malgré une dispersion plus marquée. Les probabilités d’autonomie dépassent 98 % grâce à la surface de bassin mobilisée ; en revanche la probabilité de traverser la saison sèche sans appoint reste nulle, ce qui traduit la nécessité d’une stratégie de remplissage multi-annuelle.

**_Figure 8_** : _Comparaison Monte Carlo – 12 **site** PHES identifier en République du Bénin_

L’analyse de Sobol (méthode de Saltelli, 4 facteurs) attribue 0,22 de la variance annuelle au ruissellement, 0,19 à l’infiltration, <0,01 à l’évaporation et ~0,03 aux fuites, confirmant que l’effort de caractérisation terrain doit se concentrer sur les coefficients hydrodynamiques. Les termes d’interaction restent limités, ce qui autorise une approche séquentielle : instrumentation des bassins pour réduire l’incertitude sur Cr, puis essais d’étanchéité ciblés pour contenir l’infiltration résiduelle (<5 GL/an pour les sites pilotes). Enfin, le 10e percentile saison sèche (Tableau 3) peut servir de base pour dimensionner les appoints gravitaires ou les pompages correctifs.

3.5 Classement Multi-Critères AHP

Le classement AHP pondéré (40 % hydrologie, 30 % économie, 20 % technique, 10 % environnement) reflète mieux l’hydrologie actualisée : la corrélation de Spearman entre score AHP et médiane annuelle atteint désormais ρ = 0,36 (p = 0,24), signe d’un alignement modéré mais positif. Les quatre premiers sites (RES18951, RES11634, RES35193, RES15127) cumulent une classe économique D/E raisonnable, des hauteurs de chute favorables et une probabilité d’autonomie hydrique supérieure à 99 %. Ils peuvent constituer un portefeuille de démonstrateurs couvrant 50-150 GWh.

RES31412 (Classe C) et RES26025 (Classe E) conservent de forts scores économiques et techniques mais restent contraints par un appoint saison sèche de 190-250 GL. Ils demeurent pertinents pour un scénario industriel si l’on prévoit des infrastructures de compensation (pompage sur réseau ou transfert inter-bassin). Les sites de 15 GWh occupent les rangs 9 à 11 dans la matrice AHP, non pas en raison d’un déficit hydrologique, mais parce que leur puissance installée plus faible pénalise le critère économique. Cette divergence milite pour une stratégie à deux vitesses : pilotes compacts optimisant l’apprentissage hydrologique, puis montée en puissance conditionnée à la fermeture du gap saison sèche (Figure 9).

**\*Figure : 9\*\*** : **Heatmap** Scores Sous-**Critéres** AHP – 12 Sites PHES en République du\*_ Bénin _

4. DISCUSSION

4.1 Divergence Déterministe vs Probabiliste

Les scénarios déterministes et les distributions Monte Carlo convergent désormais vers un excédent annuel pour chaque site, mais la simulation probabiliste reste indispensable pour évaluer la profondeur des queues. Pour RES35358, la médiane annuelle est de 3,5 GL/an, mais le 5e percentile tombe à 2,2 GL/an en raison de tirages combinant Cr faible (≈0,35) et infiltration élevée. RES31412 illustre aussi cet écart : la médiane atteint 21 GL/an, alors que le 5e percentile est limité à 13 GL/an, ce qui représente tout de même 8 GL à mobiliser pour rester positif en année défavorable. Cette dispersion impose de dimensionner les ouvrages de collecte et les appoints saisonniers sur la base des percentiles (P05/P10) plutôt que sur un bilan moyen.

4.2 Dominance du Coefficient de Ruissellement et Stratégie de Validation

Les indices de Sobol placent le ruissellement (S1 moyen = 0,22) et l’infiltration (0,19) loin devant l’évaporation (<0,01) et les fuites (~0,03) dans l’explication de la variance annuelle. La conservation de masse imposée (runoff + infiltration ≤ 1) accentue même la sensibilité de l’infiltration lorsque Cr descend sous 0,4, puisqu’elle peut alors consommer jusqu’à 70 % de la fraction disponible. Cela justifie la mise en place d’expériences de terrain (pluviographes + jauges de crue temporaires) pour verrouiller Cr et son évolution selon l’usage des terres.

À court terme, des plaques lysimétriques ou des essais de pompage sur les réservoirs pilotes peuvent calibrer l’infiltration effective et valider l’hypothèse de la loi Beta 0,05-0,25. Cette démarche réduira l’incertitude sur la médiane annuelle à ±0,4 GL pour les sites de 15 GWh et à ±2 GL pour RES31412, rendant possible une décision d’investissement sur la base d’intervalles resserrés.

4.3 Échelle Optimale des STEP pour les Régions Semi-Arides

Avec les bassins versants dérivés du FABDEM, les sites compacts (15 GWh) comme les grandes paires (500 GWh) présentent tous un bilan annuel positif ; la contrainte principale est désormais la profondeur du déficit saison sèche. Les sites ≤50 GWh nécessitent un appoint inférieur à 15 GL (Tableau 4) et peuvent être réalimentés par un système gravitaire local ou par quelques semaines de pompage, tandis que RES31412 et RES26025 exigent des réserves tampons supérieures à 190 GL.

Cette observation plaide pour une montée en échelle progressive : (i) démonstrateurs de 15-50 GWh avec revêtements ciblés pour apprendre sur l’infiltration réelle ; (ii) projets intermédiaires (100-150 GWh) intégrant des bassins saisonniers ou des raccordements au réseau de pompage ; (iii) méga-projets (≥500 GWh) conditionnés à la mise en place d’appoints structurants (transferts inter-bassins, pompes alimentées par excédents solaires). Les volumes d’appoint doivent être intégrés au plan d’affaires au même titre que le CAPEX des ouvrages hydrauliques.

**Tableau 4 : Volume d’appoint saison sèche (P10)**

| Site     | Volume à prévoir (GL) |
| -------- | --------------------- |
| RES16527 | 4,2                   |
| RES34296 | 3,4                   |
| RES35358 | 5,7                   |
| RES36507 | 2,7                   |
| RES11634 | 14,2                  |
| RES15127 | 12,2                  |
| RES18951 | 41,4                  |
| RES35193 | 16,6                  |
| RES37145 | 11,1                  |
| RES34837 | 54,6                  |
| RES31412 | 190,7                 |
| RES26025 | 249,7                 |

4.4 Sensibilité au Changement Climatique

L’hypothèse de stationnarité (2002–2023) reste une limitation majeure puisque les projections AR6 suggèrent des perturbations de ±10–15 % sur les précipitations annuelles et une hausse quasi certaine (+10–15 %) de l’évaporation potentielle [45],[46]. Compte tenu des marges actuelles (1,8–21,3 GL/an), une baisse uniforme de 15 % des précipitations réduirait mécaniquement les médianes de 0,3–3,2 GL/an, rapprochant les petits sites du seuil nul et doublant les volumes d’appoint saison sèche. À l’inverse, une légère augmentation des pluies (+10 %) suffirait à neutraliser les déficits saison sèche des sites ≤50 GWh.

Nous n’avons pas encore implémenté un downscaling CORDEX complet ; cependant le pipeline permet de rejouer rapidement `hydro-sim` après application de multiplicateurs sur CHIRPS/ERA5. Cette étape fera partie des travaux futurs afin de fournir des marges probabilistes par trajectoire SSP. En attendant, les volumes d’appoint dimensionnés au P10 (Tableau 4) constituent une protection robuste contre des dégradations modérées du climat.

4.5 Limitations Méthodologiques

Cette étude présente plusieurs limitations inhérentes aux contraintes de données et à la portée de l’analyse prospective initiale, identifiant des axes d’amélioration clairs pour les phases ultérieures.

Le modèle de bilan hydrique mensuel ne peut pas capturer la variabilité infra-mensuelle (événements pluvieux quotidiens, crues éclair). Le dimensionnement du réservoir est probablement sous-estimé de 10-15 % car les pics de ruissellement journaliers sont lissés dans les moyennes mensuelles [47]. Une conception affinée nécessiterait une modélisation quotidienne avec pas de temps horaire pour événements extrêmes. Le modèle de bassin versant agrégé ignore la variabilité spatiale intra-bassin (type de sol, occupation des terres, topographie micro-échelle). Un modèle distribué (SWAT, VIC) avec Unités de Réponse Hydrologique [48],[49] affinerait les estimations de Cr de ±0,20 à ±0,08, mais nécessite des cartes sol/occupation des terres haute résolution (actuellement non disponibles pour le nord Bénin). Le coefficient de ruissellement (plage 0,30-0,80) et l’infiltration (0,05-0,25) ne sont pas encore calibrés sur mesures locales, ce qui laisse subsister une incertitude systématique sur les médianes annuelles.

L’analyse de sensibilité climatique (Section 4.4) repose sur scénarios stylisés IPCC AR6 globaux [45], non sur downscaling CORDEX-Africa complet spatialisé [50]. Les trajectoires réelles de précipitations/évaporation 2025-2070 pour région Atacora présentent une incertitude structurelle (divergence inter-modèles ±20%). Une analyse rigoureuse nécessiterait ensemble multi-modèles CORDEX (15-20 modèles climatiques régionaux), mais cela excède portée étude prospective initiale.

Les coûts sont estimés via modèle paramétrique Stocks et al. [18] calibré sur contexte international, non sur devis ingénierie détaillée Bénin-spécifique. Incertitude CAPEX estimée ±25-35% (Classes C/D/E représentent plages, non valeurs exactes). Une étude de faisabilité bancable nécessiterait : levés topographiques LiDAR 1m (vs FABDEM 30m actuel), investigations géotechniques (carottages, tests résistance roche, perméabilité), conception barrages détaillée (optimisation profil, choix matériaux locaux, stabilité), devis turbines/générateurs fournisseurs (GE Renewable Energy, Voith Hydro, Andritz). Le LCOE calculé (40-64 $/MWh selon classe) suppose 300 cycles/an et tarif arbitrage constant. L’intégration au réseau électrique béninois (profil demande horaire, capacité transmission 330 kV, régulation marché) n’est pas modélisée. Une analyse dispatch hourly déterminerait cycles réels et revenus actualisés (LCOE effectif peut augmenter 10-20% si cycles réels = 180-220 vs 300 supposés).

Aucune enquête de biodiversité terrain, évaluation régime foncier, ou consultation communautaire menée à ce stade. L’impact environnemental (submersion 585-8,682 ha, fragmentation habitat, déplacement faune) et social (propriété terres, ressources halieutiques, usages pastoraux) doit être évalué via Étude d’Impact Environnemental et Social (EIES) conforme Normes de Performance IFC avant développement [51],. La (Section 3.4) inclut score environnemental AHP (surface réservoir, proximité aires protégées), mais ceci est proxy grossier vs EIES complète [52],. On doit prioriser consultations parties prenantes (autorités locales Atacora-Donga, chefferies traditionnelles, ONG conservation) pour assurer acceptabilité sociale et conception participative.

5. CONCLUSIONS

L’intégration des bassins versants dérivés de FABDEM/Whitebox et des distributions Beta pour le ruissellement et l’infiltration montre que les 12 sites STEP étudiés peuvent atteindre un bilan annuel positif. Les médianes se situent entre 1,8 et 21,3 GL/an et la probabilité d’autonomie dépasse 98 % dans tous les cas. La contrainte structurelle demeure saisonnière : le 10e percentile du bilan sec varie de 2,7 à 250 GL selon l’échelle, ce qui impose de planifier des appoints ou une exploitation multi-annuelle pour sécuriser les cycles de pompage.

La stratégie nationale peut ainsi s’articuler autour de trois chantiers : (i) déployer des démonstrateurs ≤50 GWh avec instrumentation hydrologique pour calibrer Cr et l’infiltration réelle ; (ii) dimensionner des bassins tampons, des revêtements et/ou des appoints gravitaires pour réduire les déficits saison sèche mis en évidence au Tableau 4 ; (iii) conditionner les projets ≥500 GWh à l’existence d’infrastructures d’appoint (transferts inter-bassins, pompes alimentées par excédents solaires) et à une validation climatique régionale. L’analyse multicritère AHP, mieux alignée sur les nouveaux diagnostics, devient alors un outil de priorisation crédible entre gains économiques et résilience hydrologique.

Sur le plan méthodologique, cette étude confirme qu’une chaîne entièrement probabiliste, reproductible et appuyée sur des données ouvertes (CHIRPS, ERA5, FABDEM) peut éclairer rapidement les décisions dans des contextes à données limitées. Les prochaines étapes consisteront à intégrer des séries climatiques régionalisées (CORDEX), à raffiner la résolution temporelle du modèle hydrologique et à coupler les scénarios d’appoint avec les modèles énergétiques du réseau béninois afin d’évaluer la viabilité technico-économique des futures STEP en boucle fermée.

REMERCIEMENTS

Nous remercions les développeurs des jeux de données CHIRPS, ERA5 et FABDEM pour avoir rendu cette recherche possible. Les simulations Monte Carlo ont été effectuées en utilisant Python 3.9 avec les bibliothèques NumPy, SciPy et pandas.

**Disponibilité du code et des données**

- Série climatique (CHIRPS/ERA5), masques FABDEM et résultats Monte Carlo : dépôt partagé sur Google Drive — <https://drive.google.com/drive/folders/1g6Axf6DNzrGLSZh_7l1e8KZPdp8wKLmm>.
- Code source, commandes CLI et scripts d’analyse : dépôt GitHub — <https://github.com/99ch/PHES_Searching/tree/FABDEM>.
- Cartes de répartition des sites (Google Earth) et exports GeoJSON (`results/site_basins.geojson`) inclus dans le dépôt GitHub.

RÉFÉRENCES

[1] A. Blakers, « Le changement énergétique le plus rapide de l'histoire est toujours en cours », ANU RE100 Group. [https://re100.eng.anu.edu.au/2024/04/24/fastest-energy-change-article/](https://re100.eng.anu.edu.au/2024/04/24/fastest-energy-change-article/)

[2] N. McIlwaineet al.« Une analyse technico-économique de pointe du stockage d'énergie distribué et intégré pour les systèmes énergétiques »,Énergie, vol. 229, p. 120461, août 2021, doi: 10.1016/j.energy.2021.120461

[3] S. Mulder et S. Klein, « Comparaison technico-économique des options de stockage d’électricité dans un système énergétique entièrement renouvelable »,Énergies, vol. 17, no. 5, Art. no. 5, janv. 2024, doi: 10.3390/en17051084.

[4] INSAE (2023). Projection démographique du Bénin 2013-2030. Institut National de la Statistique et de l’Analyse Économique, Cotonou, Bénin. [https://www.insae-bj.org/](https://www.insae-bj.org/)

[5] « Population du Bénin (2025) – Worldometer ». Consulté le 24 février 2025. [En ligne]. Disponible : [https://www.worldometers.info/world-population/benin-population/](https://www.worldometers.info/world-population/benin-population/)

[6] JHR Mensah, IFS dos Santos et GL Tiago Filho, « Une analyse critique de la situation énergétique en République du Bénin et de son évolution au cours de la dernière décennie », Énergie renouvelable, vol. 202, pp. 634-650, janvier 2023, doi : 10.1016/j.renene.2022.11.085.

[7] « Portail de connaissances sur le changement climatique de la Banque mondiale ». Consulté le 24 février 2025. [En ligne]. Disponible: [https://climateknowledgeportal.worldbank.org/](https://climateknowledgeportal.worldbank.org/)

[8] « Inauguration de la Centrale solaire photovoltaïque 25 MWc d'Illoulofin : Le Bénin poursuit sa marche vers l'autonomie énergétique », Gouvernement de la République du Bénin. Consulté : 26 décembre 2024. [En ligne]. Disponible: [https://www.gouv.bj/article/1858/inauguration-centrale-solaire-photovoltaique-25-illoulofin-benin-poursuit-marche-vers-autonomie-energetique/](https://www.gouv.bj/article/1858/inauguration-centrale-solaire-photovoltaique-25-illoulofin-benin-poursuit-marche-vers-autonomie-energetique/)

[9] « Lancement des travaux de la centrale Solaire PV FORSUN à Pobè : Un engagement renforcé du Gouvernement pour l'autonomie énergétique du Bénin », Gouvernement de la République du Bénin. Consulté : 26 décembre 2024. [En ligne]. Disponible:https:// www.gouv.bj/article/2860/lancement-travaux-centralesolaireforsun-pobe-engagementrenforce-gouvernement-autonomie-energetiquebenin/

[10] « Cérémonie de lancement officiel des travaux de la Centrale Solaire 25 MWc FORSUN à Pobè | EEAS. » Consulté : 26 décembre 2024. [En ligne]. Disponible : [https://www.eeas.europa.eu/delegations/benin/c%C3%A9r%C3%A9monie-de-lancement-officiel-des-travaux-de-la-centrale-solaire-25-mwc-forsun-%C3%A0-pob%C3%A8_fr](https://www.eeas.europa.eu/delegations/benin/c%C3%A9r%C3%A9monie-de-lancement-officiel-des-travaux-de-la-centrale-solaire-25-mwc-forsun-%C3%A0-pob%C3%A8_fr)

[11] « Centrale électrique de Maria Gleta ». 20 octobre 2023. Consulté le 24 février 2025. [En ligne]. Disponible : [https://www.gouv.bj/article/2770/centrale-thermique-maria-gleta-cinq-deja-36.000-heures-marche.-infrastructure-fonctionne-merveille/](https://www.gouv.bj/article/2770/centrale-thermique-maria-gleta-cinq-deja-36.000-heures-marche.-infrastructure-fonctionne-merveille/)

[12] « Bénin | Partenariat CDN ». Consulté le 24 février 2025. [En ligne]. Disponible : https://ndcpartnership.org/country/ben

[13] « Méthode SIG et AHP pour identifier les sites favorables à l'installation de P2 PHES au Bénin | Demande PDF », dansResearchGate, mars 2025. doi: 10.1109/IConEEI64414.2024.10748006

[14] BR Deemeret al.« Émissions de gaz à effet de serre provenant des surfaces d'eau des réservoirs : une nouvelle synthèse mondiale »,Biosciences, vol. 66, non. 11, pp. 949-964, novembre 2016, doi : 10.1093/biosci/biw117

[15] N. Barroset al.« Les émissions de carbone des réservoirs hydroélectriques sont liées à l'âge et à la latitude du réservoir »,Nature Geosci, vol. 4, no. 9, pp. 593–596, sept. 2011, doi: 10.1038/ ngeo1211.

[16] U.S. Department of Energy – Water Power Technologies Office, « A Comparison of the Environmental Effects of Open-Loop and Closed-Loop Pumped Storage Hydropower », Washington, D.C., 13 avril 2020. Consulté le 29 octobre 2025. [En ligne]. Disponible : https://www.energy.gov/eere/water/articles/comparison-environmental-effects-open-loop-and-closed-loop-pumped-storage

[17] TR Simon, D. Inman, R. Hanes, G. Avery, D. Hettinger et G. Heath, « Évaluation du cycle de vie des centrales hydroélectriques à accumulation par pompage en boucle fermée aux ÉtatsUnis », Environ Sci Technol, vol. 57, non. 33, pp. 12251-12258, août 2023, doi : 10.1021/ acs.est.2c09189

[18] Stocks, M., Stocks, R., Lu, B., Cheng, C., & Blakers, A. (2021). Global atlas of closed-loop pumped hydro energy storage. _Joule_, 5(1), 270-284. [https://doi.org/10.1016/j.joule.2020.11.015](https://doi.org/10.1016/j.joule.2020.11.015)

[19] Hawker, L., Uhe, P., Paulo, L., Sosa, J., Savage, J., Sampson, C., & Neal, J. (2022). A 30 m global map of elevation with forests and buildings removed. _Environmental Research Letters_, 17(2), 024016. [https://doi.org/10.1088/1748-9326/ac4d4f](https://doi.org/10.1088/1748-9326/ac4d4f)

**[20]** **Funk, C., Peterson, P., **Landsfeld**, M., **Pedreros**, D., Verdin, J., Shukla, S., … & Michaelsen, J. (2015).** The climate hazards infrared precipitation with stations—a new environmental record for monitoring extremes. _Scientific Data_, 2(1), 1-21. [https://doi.org/10.1038/sdata.2015.66](https://doi.org/10.1038/sdata.2015.66)

**[21\*\***]\** **Hersbach**, H., Bell, B., Berrisford, P., Hirahara, S.,**Horányi**, A., Muñoz‐Sabater, J., … & **Thépaut**, J. N. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society\*, 146(730), 1999-2049. [https://doi.org/10.1002/qj.3803](https://doi.org/10.1002/qj.3803)

[22] « Bénin », Wikipédia (français). Consulté le 29 octobre 2025. [En ligne]. Disponible : [https://fr.wikipedia.org/wiki/B%C3%A9nin](https://fr.wikipedia.org/wiki/B%C3%A9nin)

[23] « Géographie » – Site officiel de la Présidence de la République du Bénin. Consulté le 29 octobre 2025. [En ligne]. Disponible : https://presidence.bj/home/le-benin/geographie/

[24] Neal, J., Hawker, L., Uhe, P., Paulo, L., Sosa, J., Savage, J., & Sampson, C. (2023). FABDEM V1-2. University of Bristol. DOI : 10.5523/bris.s5hqmjcdj8yo2ibzi9b4ew3sn. Consulté le 29 octobre 2025. [En ligne]. Disponible : https://data.bris.ac.uk/data/dataset/s5hqmjcdj8yo2ibzi9b4ew3sn

[25] CHIRPS Initiative. CHIRPS-2.0: Climate Hazards Group InfraRed Precipitation with Station data Version 2.0. University of California, Santa Barbara. Consulté le 29 octobre 2025. [En ligne]. Disponible : https://data.chc.ucsb.edu/products/CHIRPS-2.0/

[26] Copernicus Climate Change Service (C3S). ERA5 hourly data on single levels from 1940 to present. DOI : 10.24381/cds.adbb2d47. Consulté le 29 octobre 2025. [En ligne]. Disponible : https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download

[27] UNEPWCMC & IUCN, « Protected Area Profile for W (Bénin) from the World Database on Protected Areas (WDPA) », Octobre 2025. [En ligne]. Disponible : [https://www.protectedplanet.net/12201](https://www.protectedplanet.net/12201?utm_source=chatgpt.com)

[28] UNEPWCMC & IUCN (2025). Protected Planet: The World Database on Protected Areas (WDPA) [On-line]. Cambridge, UK: UNEP-WCMC & IUCN. Consulté en octobre 2025. Disponible : https://doi.org/10.34892/6fwd-af11

[29] NASA JPL. (2013). NASA Shuttle Radar Topography Mission Global 1 arc second [Data set]. NASA Land Processes Distributed Active Archive Center. 003 Date Accessed: 2025-10-29 [https://doi.org/10.5067/MEASURES/SRTM/SRTMGL1](https://doi.org/10.5067/MEASURES/SRTM/SRTMGL1)

[30] A. Teston, T. P. Scolaro, J. K. Maykot, et E. Ghisi, « Comprehensive Environmental Assessment of Rainwater Harvesting Systems: A Literature Review », Water, vol. 14, no 17, p. 2716, 2022. Consulté le 29 octobre 2025. [En ligne]. Disponible : https://doi.org/10.3390/w14172716

[31] Mahé, G., Paturel, J.-E., Servat, E., Conway, D., & Dezetter, A. (2005). The impact of land use change on soil water holding capacity and river flow modelling in the Niger basin. Hydrological Sciences Journal, 50(3), 375–386. https://doi.org/10.1623/hysj.50.3.375.65030

[32] Yamazaki, D., Ikeshima, D., Sosa, J., Bates, P. D., Allen, G. H., & Pavelsky, T. M. (2019). MERIT Hydro: A high‐resolution global hydrography map based on latest topography datasets. Water Resources Research, 55(6), 5053–5073. https://doi.org/10.1029/2019WR024873

[33] Lachassagne, P., Wyns, R., & Dewandel, B. (2011). The fracture permeability of hard rock aquifers is due neither to tectonics, nor to unloading, but to weathering processes. Terra Nova, 23(3), 145–161. https://doi.org/10.1111/j.1365-3121.2011.00998.x

[34] Helton, J. C., & Davis, F. J. (2003). Latin hypercube sampling and the propagation of uncertainty in analyses of complex systems. Reliability Engineering & System Safety, 81(1), 23–69. https://doi.org/10.1016/S0951-8320(03)00058-9

[35] Beven, K. J., & Binley, A. M. (1992). The future of distributed models: Model calibration and uncertainty prediction. Hydrological Processes, 6(3), 279–298. https://doi.org/10.1002/hyp.3360060305

[36] McKay, M. D., Beckman, R. J., & Conover, W. J. (1979). A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code. Technometrics, 21(2), 239–245. https://doi.org/10.1080/00401706.1979.10489755

[37] Descroix, L., Mahé, G., Lebel, T., Favreau, G., Galle, S., Gautier, E., … & Olivry, J. C. (2009). Spatio-temporal variability of hydrological regimes around the boundaries between Sahelian and Sudanian areas of West Africa. _Journal of Hydrology_, 375(1-2), 90-102. [https://doi.org/10.1016/j.jhydrol.2008.12.012](https://doi.org/10.1016/j.jhydrol.2008.12.012)

[38] Mahé, G., & Paturel, J. E. (2009). 1896–2006 Sahelian annual rainfall variability and runoff increase of Sahelian Rivers. _Comptes Rendus \*\*Geoscience_, 341(7), 538-546. [https://doi.org/10.1016/j.crte.2009.05.002](https://doi.org/10.1016/j.crte.2009.05.002)

[39] Leblanc, M. J., Favreau, G., Massuel, S., Tweed, S. O., Loireau, M., & Cappelaere, B. (2008). Land clearance and hydrological change in the Sahel: SW Niger. _Global and Planetary Change_, 61(3-4), 135-150. [https://doi.org/10.1016/j.gloplacha.2007.08.011](https://doi.org/10.1016/j.gloplacha.2007.08.011)

[40] Spearman, C. (1904). The proof and measurement of association between two things. _The American Journal of Psychology_, 15(1), 72-101. [https://doi.org/10.2307/1412159](https://doi.org/10.2307/1412159)

[41] Saaty, T. L. (2008). Decision making with the analytic hierarchy process. _International Journal of Services Sciences_, 1(1), 83-98. [https://doi.org/10.1504/IJSSCI.2008.017590](https://doi.org/10.1504/IJSSCI.2008.017590)

[42] Jensen, J. L. W. V. (1906). Sur les fonctions convexes et les inégalités entre les valeurs moyennes. _Acta Mathematica_, 30(1), 175-193. [https://doi.org/10.1007/BF02418571](https://doi.org/10.1007/BF02418571)

[43] Mishra, S. K., & Singh, V. P. (2003). Soil Conservation Service Curve Number (SCS-CN) Methodology. Springer Science & Business Media. [https://doi.org/10.1007/978-94-017-0147-1](https://doi.org/10.1007/978-94-017-0147-1)

[44] Yilmaz, K. K., Gupta, H. V., & Wagener, T. (2008). A process-based diagnostic approach to model evaluation: Application to the NWS distributed hydrologic model. Water Resources Research, 44(9), W09417. [https://doi.org/10.1029/2007WR006716](https://doi.org/10.1029/2007WR006716)

[45] IPCC, Climate Change 2021: The Physical Science Basis. Contribution of Working Group I to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change, V. Masson-Delmotte, P. Zhai, A. Pirani, et al. (éds.). Cambridge University Press, 2021. Consulté le 30 octobre 2025. [En ligne]. Disponible : [https://www.ipcc.ch/report/ar6/wg1/](https://www.ipcc.ch/report/ar6/wg1/)

[46] Dosio, A., Jones, R. G., Jack, C., Lennard, C., Nikulin, G., & Hewitson, B., « What can we know about future precipitation in Africa? Robustness, significance and added value of projections from a large ensemble of regional climate models », Climate Dynamics, vol. 53, pp. 5833–5858, 2019. Consulté le 30 octobre 2025. [En ligne]. Disponible : [https://doi.org/10.1007/s00382-019-04900-3](https://doi.org/10.1007/s00382-019-04900-3)

[47] C.-Y. Xu & V. P. Singh, « A Review on Monthly Water Balance Models for Water Resources Investigations », Water Resources Management, vol. 12, no. 1, pp. 20–50, Feb. 1998. Consulté le 30 octobre 2025. [En ligne]. Disponible : [https://doi.org/10.1023/A:1007916816469](https://doi.org/10.1023/A:1007916816469)

[48] J. G. Arnold, R. Srinivasan, R. S. Muttiah & J. R. Williams, « Large area hydrologic modeling and assessment. Part I: Model development », Journal of the American Water Resources Association (JAWRA), vol. 34, no. 1, pp. 73–89, 1998. Consulté le 30 octobre 2025. [En ligne]. Disponible : [https://doi.org/10.1111/j.1752-1688.1998.tb05961.x](https://doi.org/10.1111/j.1752-1688.1998.tb05961.x)

[49] X. Liang, D. P. Lettenmaier, E. F. Wood & S. J. Burges, « A simple hydrologically based model of land surface water and energy fluxes for general circulation models », Journal of Geophysical Research, vol. 99, no. D7, pp. 14415–14428, 1994. Consulté le 30 octobre 2025. [En ligne]. Disponible : [https://doi.org/10.1029/94JD00483](https://doi.org/10.1029/94JD00483)

[50] A. Dosio, R. G. Jones, C. Jack, C. Lennard, G. Nikulin & B. Hewitson, « What can we know about future precipitation in Africa? Robustness, significance and added value of projections from a large ensemble of regional climate models », Climate Dynamics, vol. 53, pp. 5833–5858, 2019. Consulté le 30 octobre 2025. [En ligne]. Disponible : [https://doi.org/10.1007/s00382-019-04900-3](https://doi.org/10.1007/s00382-019-04900-3)

[51] International Finance Corporation (IFC), Performance Standards on Environmental and Social Sustainability, Janvier 2012. Consulté le 30 octobre 2025. [En ligne]. Disponible : [https://www.ifc.org/content/dam/ifc/doc/2023/ifc-performance-standards-2012-en.pdf](https://www.ifc.org/content/dam/ifc/doc/2023/ifc-performance-standards-2012-en.pdf)

[52] European Investment Bank (EIB), Environmental, Climate and Social Guidelines on Hydropower Development, Juin 2019. Consulté le 30 octobre 2025. [En ligne]. Disponible : [https://www.eib.org/files/publications/eib_guidelines_on_hydropower_development_en.pdf](https://www.eib.org/files/publications/eib_guidelines_on_hydropower_development_en.pdf)
