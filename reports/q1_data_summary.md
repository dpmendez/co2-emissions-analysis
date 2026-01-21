# Question 1: Data Acquisition and Preprocessing Report

## Executive Summary

This report documents the acquisition and preprocessing of CO2 emissions data and 
socio-economic indicators from the World Bank Climate Change database.

**Dataset Overview:**
- **Time Period**: 1990 - 2024
- **Countries**: 203
- **Total Observations**: 7,105
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


### 1.3 Indicator Codes Used

| Category | Code | Description |
|----------|------|-------------|
| **CO2 Total** | EN.GHG.CO2.MT.CE.AR5 | CO2 emissions total (Mt CO2e) |
| **CO2 Per Capita** | EN.GHG.CO2.PC.CE.AR5 | CO2 per capita (t CO2e/cap) |
| **CO2 Transport** | EN.GHG.CO2.TR.MT.CE.AR5 | Transport sector emissions |
| **CO2 Power** | EN.GHG.CO2.PI.MT.CE.AR5 | Power industry emissions |
| **GDP** | NY.GDP.MKTP.CD | GDP (current US$) |
| **GDP per Capita** | NY.GDP.PCAP.CD | GDP per capita |
| **Population** | SP.POP.TOTL | Total population |
| **Renewable Energy** | EG.FEC.RNEW.ZS | Renewable energy share (%) |

---

## 2. Data Preprocessing

### 2.1 Cleaning Steps

1. **Removed Regional Aggregates**: Excluded World Bank regional groupings to focus on country-level data
2. **Missing Data Handling**: 
   - Dropped countries with >50% missing data
   - Applied linear interpolation for gaps ≤3 years
   - Used forward/backward fill for edge values
3. **Derived Features**: Created log transforms and CO2 intensity metrics

### 2.2 Data Quality

| Metric | Value |
|--------|-------|
| Initial rows | 7,105 |
| Countries retained | 203 |
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
| log_co2 | 0.456 |
| log_gdp | 0.381 |
| log_population | 0.327 |


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

*Report generated: 2026-01-21 12:39:35*
