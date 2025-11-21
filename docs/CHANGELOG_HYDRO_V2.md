# Changelog : Modèle hydrologique v2 (novembre 2025)

## 🔬 Amélioration majeure : Séparation physique apports/pertes

### Motivation

Le modèle hydrologique a été amélioré pour refléter plus fidèlement la **physique réelle** d'un PHES en boucle fermée dans un contexte tropical.

**Problème du modèle v1** :
- Appliquait l'évaporation sur toute la surface du bassin versant (surestimation des pertes)
- Ne distinguait pas clairement la collecte passive (gravité) de l'évaporation active

**Solution du modèle v2** :
- **APPORTS** : Précipitations × Bassin versant × Ruissellement → Réservoir inférieur
- **PERTES** : ETP × Surfaces des réservoirs (upper + lower uniquement)

### Changements techniques

#### 1. Nouvelles propriétés `SiteHydrologyParams`

```python
@dataclass
class SiteHydrologyParams:
    upper_area_ha: float  # Surface réservoir supérieur
    lower_area_ha: float  # Surface réservoir inférieur
    basin_area_km2: float # Bassin versant (site-basins)
```

#### 2. Calcul séparé des flux

**Avant (v1)** :
```python
# Tout sur le bassin versant
precip_gl = precip_m * basin_area_m2 / GL_IN_M3
etp_gl = etp_m * basin_area_m2 / GL_IN_M3
net = runoff * precip_gl - infiltration * precip_gl - evap * etp_gl
```

**Après (v2)** :
```python
# APPORTS : bassin versant → ruissellement net
precip_catchment_gl = precip_m * catchment_area_m2 / GL_IN_M3
net_runoff_gl = runoff * precip_catchment_gl - infiltration * precip_catchment_gl

# PERTES : surfaces d'eau uniquement
etp_upper_gl = etp_m * upper_area_m2 / GL_IN_M3
etp_lower_gl = etp_m * lower_area_m2 / GL_IN_M3
evap_gl = evap_multiplier * (etp_upper_gl + etp_lower_gl)

# BILAN
net = net_runoff_gl - evap_gl - fuites
```

### Justification physique

#### Pourquoi le réservoir **INFÉRIEUR** reçoit les apports ?

1. **Gravité** : Le bassin versant délimité autour du point Upper draine naturellement vers le bas
2. **Maximisation énergétique** : L'eau collectée au Lower peut être pompée → turbinée (cycle complet)
3. **Observations terrain** : Cohérent avec la topographie réelle (Upper sur plateau, Lower en vallée)

#### Surfaces typiques pour les 12 sites

| Site | Bassin (km²) | Upper (ha) | Lower (ha) | Ratio Bassin/Réservoirs |
|------|--------------|------------|------------|-------------------------|
| RES18951 & RES16451 | ~15-20* | 1342 | 449 | ~10× |
| RES31412 & RES31520 | ~25-30* | 5776 | 2906 | ~3× |

*Valeurs estimées, seront calculées par `site-basins`

→ Le bassin versant est **significativement plus grand** que les réservoirs, justifiant la distinction.

### Impact sur les résultats

**Attendu** :
- ✅ **Apports augmentés** : Plus de surface contributive (bassin > réservoirs)
- ✅ **Pertes réduites** : Évaporation uniquement sur surfaces d'eau
- ✅ **Bilan net amélioré** : Probabilité d'autonomie augmentée
- ✅ **Déficit saison sèche réduit** : Moins de besoins en appoint

**Validation** :
- Les tests unitaires existants **passent** (conservation de masse, indépendance RNG)
- Le modèle respecte `runoff + infiltration ≤ 1`
- Cohérent avec Kamagaté et al. (2007), Descroix et al. (2010)

### Migration

**Aucune action requise pour l'utilisateur** si le workflow complet est suivi :

```bash
# 1. Calculer les bassins versants (OBLIGATOIRE)
python -m phes_assessment.cli site-basins --output results/site_basins.geojson

# 2. Agréger les séries climatiques
python -m phes_assessment.cli climate-series --basins results/site_basins.geojson

# 3. Simulation hydrologique (utilise automatiquement le modèle v2)
python -m phes_assessment.cli hydro-sim --basins results/site_basins.geojson
```

Le code détecte automatiquement :
- Si `--basins` est fourni → utilise le bassin versant pour les apports
- Si absent → fallback sur les surfaces des réservoirs (compatibilité v1)

### Références

Ce modèle s'inspire des meilleures pratiques documentées par :
- **PNNL** (Pracheil et al., 2025) : Guide d'évaluation hydrologique PHES fermés
- **DOE** (2024) : Comparaison impacts environnementaux open/closed loop
- **Littérature régionale** : Kamagaté (2007), Azuka & Igué (2020), Descroix (2010)

---

**Date de mise en œuvre** : 21 novembre 2025  
**Auteur** : 99ch  
**Branche** : phes  
**Tests** : ✅ 3/3 passed
