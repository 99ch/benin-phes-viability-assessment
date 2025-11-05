# Closed-Loop Pumped Hydro Storage Potential in Benin

## Description

This project assesses the hydrological viability of closed-loop Pumped Hydro Energy Storage (PHES) sites in Benin. It employs a probabilistic approach combining Monte Carlo simulations and Analytic Hierarchy Process (AHP) to identify and prioritize candidate sites in a semi-arid context.

## Repository Contents

- **process_era5_standard_evapotranspiration.py** : ERA5 data processing script
- **data/** : Climate and geospatial data directory
  - `era5/` : ERA5 reanalysis data (evapotranspiration)
  - `chirps/` : CHIRPS precipitation data
- **output/** : Statistical analysis and Monte Carlo results


## Methodology

1. **Site Identification** : FABDEM-based algorithm to detect viable reservoir pairs
2. **Hydrological Modeling** : Monthly water balance (2002-2023) integrating runoff, evaporation, and seepage
3. **Monte Carlo Simulation** : 10,000 iterations per site to quantify uncertainty
4. **AHP Prioritization** : Multi-criteria weighting (hydrology 40%, economics 30%, technical 20%, environmental 10%)

## Key Results

- 12 PHES sites identified in northern Benin (15-500 GWh)
- Structural hydrological deficit for all sites (negative deterministic balances)
- Maximum viability probability: 36.6% (15 GWh sites)
- Seepage (5-20% of volume) accounts for 49-82% of total losses

## Requirements

```bash
# Python 3.9+
pip install numpy pandas xarray cfgrib
```

## Usage

```bash
# Process ERA5 data
python process_era5_standard_evapotranspiration.py


## Data Sources

Climate (ERA5, CHIRPS) and topographic (FABDEM) data are available from:

- ERA5 : [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)
- CHIRPS : [Climate Hazards Center](https://www.chc.ucsb.edu/data/chirps)
- FABDEM : [University of Bristol](https://data.bris.ac.uk/data/dataset/s5hqmjcdj8yo2ibzi9b4ew3sn)


