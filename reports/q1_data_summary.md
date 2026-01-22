# Question 1: Data Acquisition and Preprocessing Report

## Executive Summary

This report documents the acquisition and preprocessing of CO2 emissions data and 
socio-economic indicators from the World Bank Climate Change database.

**Dataset Overview:**
- **Time Period**: 1990 - 2024
- **Countries**: 199
- **Total Observations**: 6,965
- **Variables**: 28

**Important Note**: This analysis uses the **updated World Bank indicator codes** (EN.GHG.CO2.* 
series based on IPCC AR5 methodology). The older EN.ATM.CO2E.* indicators have been archived.

---

## 1. Data Source and Methodology

### 1.1 Theory and Rationale:

The World Bank provides comprehensive global development data through their API.
For CO2 emissions analysis, we need predictors that capture:

1. ECONOMIC FACTORS:
   - GDP and GDP per capita: Economic activity drives energy consumption
   - GDP growth: Rapid growth often means increased emissions
   
2. ENERGY FACTORS:
   - Energy use per capita: Direct measure of energy consumption
   - Fossil fuel consumption: Primary source of CO2
   - Renewable energy share: Cleaner alternatives
   - Electric power consumption: Modernization proxy
   
3. DEMOGRAPHIC FACTORS:
   - Population: Scale factor for total emissions
   - Urban population %: Urbanization affects energy patterns
   - Population density: Concentration effects
   
4. STRUCTURAL FACTORS:
   - Industry value added (% GDP): Manufacturing is energy-intensive
   - Agriculture value added: Different emission profile
   - Services value added: Generally lower emissions intensity
   
5. DEVELOPMENT INDICATORS:
   - Access to electricity: Infrastructure development
   - Forest area: Carbon sinks


### 1.2 Data Source
- **Source**: World Bank Climate Change Database
- **API**: World Bank API v2 (wbgapi Python package)
- **Methodology**: IPCC AR5 (5th Assessment Report) for CO2 indicators


### 1.2 Indicator Codes Used

| Category | Code | Variable Name | Description |
|----------|------|---------------|-------------|
| **CO2 - Target** | EN.GHG.CO2.MT.CE.AR5 | `co2_emissions_mt` | CO2 emissions total (Mt CO2e) |
| **CO2 - Target** | EN.GHG.CO2.PC.CE.AR5 | `co2_emissions_per_capita` | CO2 per capita (t CO2e/cap) |
| **CO2 - Sectoral** | EN.GHG.CO2.TR.MT.CE.AR5 | `co2_transport_mt` | Transport sector emissions |
| **CO2 - Sectoral** | EN.GHG.CO2.PI.MT.CE.AR5 | `co2_power_industry_mt` | Power industry emissions |
| **CO2 - Sectoral** | EN.GHG.CO2.BU.MT.CE.AR5 | `co2_buildings_mt` | Buildings sector emissions |
| **CO2 - Sectoral** | EN.GHG.CO2.IC.MT.CE.AR5 | `co2_industrial_mt` | Industrial combustion emissions |
| **Economic** | NY.GDP.MKTP.CD | `gdp_current_usd` | GDP (current US$) |
| **Economic** | NY.GDP.PCAP.CD | `gdp_per_capita` | GDP per capita (current US$) |
| **Economic** | NY.GDP.MKTP.KD.ZG | `gdp_growth` | GDP growth (annual %) |
| **Energy** | EG.USE.PCAP.KG.OE | `energy_use_per_capita` | Energy use (kg oil equiv. per capita) |
| **Energy** | EG.USE.ELEC.KH.PC | `electric_power_consumption` | Electric power consumption (kWh per capita) |
| **Energy** | EG.FEC.RNEW.ZS | `renewable_energy_pct` | Renewable energy (% of total final consumption) |
| **Energy** | EG.ELC.FOSL.ZS | `fossil_fuel_electricity_pct` | Electricity from fossil fuels (%) |
| **Demographic** | SP.POP.TOTL | `population` | Population, total |
| **Demographic** | SP.URB.TOTL.IN.ZS | `urban_pop_pct` | Urban population (% of total) |
| **Demographic** | EN.POP.DNST | `population_density` | Population density (people/sq km) |
| **Structural** | NV.IND.TOTL.ZS | `industry_value_added_pct` | Industry value added (% of GDP) |
| **Structural** | NV.AGR.TOTL.ZS | `agriculture_value_added_pct` | Agriculture value added (% of GDP) |
| **Structural** | NV.SRV.TOTL.ZS | `services_value_added_pct` | Services value added (% of GDP) |
| **Development** | EG.ELC.ACCS.ZS | `access_to_electricity_pct` | Access to electricity (% of population) |
| **Development** | AG.LND.FRST.ZS | `forest_area_pct` | Forest area (% of land area) |

**Derived Features (created during preprocessing):**
- `log_gdp` - Natural log of GDP
- `log_population` - Natural log of population  
- `log_co2` - Natural log of CO2 emissions
- `co2_intensity` - CO2 emissions per billion USD GDP

---

## 2. Data Preprocessing

### 2.1 Cleaning Steps

1. **Removed Regional Aggregates**: Excluded World Bank regional groupings (WLD, EUU, etc.) to focus on country-level data
2. **Dropped Countries Without CO2 Data**: Removed countries that have no CO2 emissions data (target variable must be present)
3. **Missing Predictor Handling**: 
   - Dropped countries with >50% missing predictor data
   - Applied linear interpolation for gaps ≤3 years
   - Used forward/backward fill for edge values
4. **Derived Features**: Created log transforms and CO2 intensity metrics

### 2.2 Data Quality

| Metric | Value |
|--------|-------|
| Initial rows | 6,965 |
| Countries retained | 199 |
| Year coverage | 1990 - 2024 |

---

## 3. Key Statistics

### 3.1 Correlation with CO2 Emissions

Correlations with `co2_emissions_mt`:

| Indicator | Correlation |
|-----------|-------------|
| co2_power_industry_mt | 0.992 |
| co2_industrial_mt | 0.929 |
| co2_buildings_mt | 0.918 |
| gdp_current_usd | 0.795 |
| co2_transport_mt | 0.795 |
| population | 0.734 |
| log_co2 | 0.458 |
| log_gdp | 0.381 |
| log_population | 0.330 |


---

## 4. Files Generated

| File | Description |
|------|-------------|
| `data/raw/co2_data_raw.csv` | Raw downloaded data |
| `data/processed/co2_data_clean.csv` | Cleaned dataset |
| `figures/correlation_matrix.png` | Correlation heatmap |
| `figures/co2_trends.png` | CO2 time series |
| `figures/gdp_vs_co2.png` | GDP vs emissions scatter |

---

## 5. Notes for Subsequent Analysis

1. **CO2 units**: The new indicators use Mt CO2e (megatonnes CO2 equivalent)
2. **Sectoral breakdown available**: Transport, Power, Buildings, Industrial - useful for Q3 (EV analysis)
3. **Log transforms recommended**: GDP, population, and CO2 show log-normal distributions

---

*Report generated: 2026-01-21 22:20:35*
