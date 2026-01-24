# CO2 Emissions Analysis

A comprehensive data science project analyzing global CO2 emissions using World Bank data. This project covers the full ML pipeline: data acquisition, predictive modeling, Fermi estimation, classification, and strategic recommendations.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Question Summaries](#question-summaries)
- [Data Sources](#data-sources)
- [Methodology Notes](#methodology-notes)

---

## Project Overview

This project addresses five analytical questions about global CO2 emissions:

| Question | Topic | Method |
|----------|-------|--------|
| Q1 | Data Acquisition & Preprocessing | World Bank API, pandas |
| Q2 | Predictive Modeling | Ridge Regression (log-log elasticity) |
| Q3 | Fermi Problem: EV Impact | Scenario analysis, sensitivity testing |
| Q4 | Classification | Binary classifier for emission reduction success |
| Q5 | Strategic Recommendations | Country prioritization, policy analysis |

---

## Key Findings

### Q2: GDP-CO2 Relationship
- **GDP Elasticity: 0.67** — A 10% increase in GDP is associated with a 6.6% increase in CO2 emissions
- This indicates **relative decoupling**: emissions grow slower than GDP, but still grow

### Q3: Electric Vehicle Impact
- **50% global EV adoption -> 168 Mt CO2 reduction (only 0.4% of global emissions)**
- **88 countries (45%)** have electricity grids so carbon-intensive that EVs would *increase* emissions
- Grid decarbonization must precede or accompany EV adoption

### Q4: What Predicts Successful Decoupling?
- Countries that reduce emissions while growing GDP tend to be:
  - **Higher GDP per capita** ($34,243 vs $12,352)
  - **Lower fossil fuel dependence** (53.5% vs 65.1%)
  - **Larger services sector** (63.0% vs 54.5%)
- Best model: Logistic Regression (F1 = 0.67, AUC = 0.92)

### Q5: Strategic Priorities
- Top 15 emitters account for **70%+ of global emissions** — focus policy here
- Tailor interventions to country archetypes (high-income vs developing)
- Support developing countries to leapfrog fossil fuels

---

## Project Structure

```
co2-emissions-analysis/
│
├── README.md                 # This file
│
├── src/                      # Source code
│   ├── q1_data_acquisition.py
│   ├── q2_predictive_modeling.py
│   ├── q3_fermi_analysis.py
│   ├── q4_classification.py
│   └── q5_strategic_recommendations.py
│
├── data/
│   ├── raw/                  # Raw data from World Bank API
│   │   └── co2_data_raw.csv
│   └── processed/            # Cleaned data
│       └── co2_data_clean.csv
│
├── figures/                  # Generated visualizations
│   ├── correlation_matrix.png
│   ├── co2_trends.png
│   ├── gdp_vs_co2.png
│   ├── model_predictions.png
│   ├── ridge_coefficients.png
│   ├── residuals.png
│   ├── ev_impact_by_country.png
│   ├── ev_benefit_vs_grid.png
│   ├── sensitivity_tornado.png
│   ├── model_comparison_q4.png
│   ├── feature_importance_q4.png
│   ├── confusion_matrices_q4.png
│   ├── priority_countries.png
│   └── emissions_by_country_type.png
│
└── reports/                  # Generated markdown reports
    ├── q1_data_summary.md
    ├── q2_modeling_results.md
    ├── q3_ev_analysis.md
    ├── q4_classification_report.md
    └── q5_strategic_recommendations.md
```

---


## Usage

Run each question script in order:

```bash
# Q1: Download and preprocess data
python src/q1_data_acquisition.py

# Q2: Predictive modeling (GDP-CO2 elasticity)
python src/q2_predictive_modeling.py

# Q3: Fermi estimation (EV adoption impact)
python src/q3_fermi_analysis.py

# Q4: Classification (emission reduction prediction)
python src/q4_classification.py

# Q5: Strategic recommendations
python src/q5_strategic_recommendations.py
```

Each script will:
1. Load the required data
2. Perform analysis
3. Generate visualizations in `figures/`
4. Create a markdown report in `reports/`

---

## Question Summaries

### Question 1: Data Acquisition & Preprocessing

**Objective**: Acquire and clean CO2 emissions data from the World Bank API.

**Indicators Downloaded** (21 total):
- **CO2 Emissions**: Total (Mt), per capita, by sector (transport, power, buildings, industrial)
- **Economic**: GDP, GDP per capita, GDP growth
- **Energy**: Energy use, renewable %, fossil fuel electricity %
- **Demographic**: Population, urban %, population density
- **Structural**: Industry %, agriculture %, services %
- **Development**: Electricity access %, forest area %

**Preprocessing Steps**:
1. Remove aggregate regions (World, EU, etc.)
2. Drop countries without CO2 data
3. Handle missing values (interpolation, forward/backward fill)
4. Create derived features (log transforms, CO2 intensity)

**Output**: `data/processed/co2_data_clean.csv` (~7,000 observations, 200 countries, 1990-2024)

---

### Question 2: Predictive Modeling

**Objective**: Build a model to predict CO2 emissions and answer: *"If a country increases GDP by 10%, how much will CO2 emissions change?"*

**Methodology**:
- **Log-log regression**: Coefficients represent elasticities
- **Ridge Regression**: Handles multicollinearity, interpretable
- **Temporal split**: Train <=2017, Test >=2018 (avoids data leakage)

**Key Result**:
```
GDP Elasticity (beta) = 0.67

10% GDP increase -> 6.6% CO2 increase
```

**Interpretation**: Elasticity < 1 indicates **relative decoupling** — CO2 grows slower than GDP, but still grows in absolute terms.

---

### Question 3: Fermi Problem — EV Adoption Impact

**Objective**: Estimate global CO2 reduction if 50% of passenger vehicle travel switched to electric vehicles.

**Interpretation of "50% of population adopts EVs"**: We interpret this as 50% of passenger vehicle kilometers traveled become electric, since:
- Not everyone owns a car (ownership varies from 20 to 800 per 1000 people)
- Emissions come from driving, not owning
- This is directly applicable to transport emissions data

**Methodology**:
```
CO2 Reduction = Transport_CO2 * Passenger_Share * Adoption_Rate * Net_Benefit_Factor
```

**Key Insight**: EVs are not zero-emission — they shift emissions from tailpipe to power plant. The benefit depends on grid carbon intensity.

**Integrated Indicators**:
- **Vehicles per capita**: Estimated from GDP per capita
- **Energy mix**: Grid carbon intensity from fossil fuel electricity %
- **Current EV adoption**: Estimated by region (2-20% baseline)

**Key Results**:
| Metric | Value |
|--------|-------|
| Global CO2 reduction | 168 Mt |
| As % of global emissions | 0.4% |
| Countries where EVs increase emissions | 88 (45%) |
| Best case scenario | 1,064 Mt reduction |
| Worst case scenario | -226 Mt (increase!) |

**Critical Finding**: In 88 countries, the electricity grid is so carbon-intensive that EVs emit more CO2 per km than gasoline vehicles. Grid decarbonization must precede or accompany EV adoption.

---

### Question 4: Classification — Predicting Emission Reduction Success

**Objective**: Build a classifier to identify countries likely to reduce emissions while growing their economy (true decoupling).

**Business Case Question**: *"What are the common characteristics of countries that successfully reduce emissions, and how can policymakers in other nations apply these insights?"*

**Target Variable Definition**:
```
Success = 1 if:
  - CO2 emissions decreased >10% over 10 years
  - AND real GDP grew (cumulative growth > 0%)
```

**Why Real GDP Growth?** We use cumulative real GDP growth (from `gdp_growth` indicator) instead of nominal USD change because nominal GDP is distorted by:
- Inflation
- Exchange rate fluctuations  
- Currency crises

This excludes countries that reduced emissions due to economic collapse (e.g., Venezuela -69% GDP, Yemen -35% GDP, Ukraine -30% GDP, Lebanon -31% GDP).

**Results**:
| Model | Accuracy | F1 Score | AUC-ROC |
|-------|----------|----------|---------|
| Random Forest | 0.86 | 0.46 | 0.74 |
| Gradient Boosting | 0.84 | 0.43 | 0.70 |
| **Logistic Regression** | **0.88** | **0.67** | **0.92** |

**Characteristics of Successful Decouplers** (based on group averages):
| Feature | Successful | Non-Successful | Difference |
|---------|------------|----------------|------------|
| GDP per capita | $34,243 | $12,352 | +$21,891 |
| Fossil fuel electricity % | 53.5% | 65.1% | -11.6% |
| Services sector % | 63.0% | 54.5% | +8.5% |
| Urban population % | 73.6% | 55.5% | +18.1% |

**Examples of Successful Decouplers**: UK (-32% CO2, +19% GDP), Germany (-26% CO2, +12% GDP), Denmark (-29% CO2, +25% GDP), Finland (-37% CO2, +9% GDP), USA (-12% CO2, +31% GDP), Japan (-24% CO2, +6% GDP)

---

### Question 5: Strategic Recommendations

**Objective**: Synthesize findings into actionable policy recommendations.

**Country Prioritization Framework**:

| Priority | Criteria | Examples |
|----------|----------|----------|
| **High Impact** | Largest absolute emitters | China, USA, India |
| **High Potential** | Significant emissions + favorable conditions | France, UK, Germany |
| **Quick Wins** | Already declining emissions | UK (-32%), Germany (-26%) |

**Policy Recommendations by Country Type**:

| Archetype | Examples | Key Actions |
|-----------|----------|-------------|
| High-Income High-Emitter | USA, Australia | Aggressive targets, carbon pricing $100+/ton |
| High-Income Low-Emitter | France, Sweden | Maintain momentum, export expertise |
| Clean Energy Leader | Brazil, Norway | Protect renewables, focus on transport |
| Middle-Income Fossil-Dependent | India, Indonesia | Leapfrog to renewables, just transition |
| Low-Income | Ethiopia, Kenya | Avoid carbon lock-in, distributed solar |

---

## Data Sources

### World Bank Indicators

| Category | Indicator Code | Description |
|----------|---------------|-------------|
| CO2 | EN.GHG.CO2.MT.CE.AR5 | Total CO2 emissions (Mt) |
| CO2 | EN.GHG.CO2.PC.CE.AR5 | CO2 per capita |
| CO2 Sectors | EN.CO2.TRAN.MT.CE.AR5 | Transport CO2 |
| CO2 Sectors | EN.CO2.BLDG.MT.CE.AR5 | Buildings CO2 |
| Economic | NY.GDP.MKTP.CD | GDP (current USD) |
| Economic | NY.GDP.PCAP.CD | GDP per capita |
| Economic | NY.GDP.MKTP.KD.ZG | GDP growth (annual %) |
| Energy | EG.USE.PCAP.KG.OE | Energy use per capita |
| Energy | EG.FEC.RNEW.ZS | Renewable energy % |
| Energy | EG.ELC.FOSL.ZS | Fossil fuel electricity % |

**Note**: CO2 indicators use IPCC AR5 methodology (updated from older deprecated codes like EN.ATM.CO2E.KT).

---

## Methodology Notes

### Log-Log Regression (Q2)

Economic relationships are often multiplicative:
```
CO2 = alpha × GDP^beta_1 × Population^beta_2 × ...
```

Taking logs linearizes this:
```
log(CO2) = log(alpha) + beta_1×log(GDP) + beta_2×log(Population) + ...
```

The coefficient beta_1 is the **elasticity**: % change in CO2 for 1% change in GDP.

For a 10% GDP increase:
```
CO2_new / CO2_old = 1.10^beta = 1.10^0.67 = 1.066 -> 6.6% increase
```

### Fermi Estimation (Q3)

Break complex problems into estimable components:
1. Total transport CO2 (from data)
2. Passenger vehicle share (~45%)
3. EV adoption rate (50%)
4. Net benefit factor (depends on grid)

Net benefit = 1 - (EV emissions / ICE emissions)

Where:
- ICE emissions ≈ 120 g CO2/km
- EV emissions = 0.2 kWh/km × grid intensity (g CO2/kWh)

| Grid Type | Grid Intensity | EV Emissions | Net Benefit |
|-----------|---------------|--------------|-------------|
| Very clean (hydro/nuclear) | 50 g/kWh | 10 g/km | +92% cleaner |
| Mixed | 400 g/kWh | 80 g/km | +33% cleaner |
| Dirty (coal) | 800 g/kWh | 160 g/km | **-33% dirtier** |

### Real GDP Growth (Q4)

We use **cumulative real GDP growth** instead of nominal USD change:
```python
cumulative_growth = ∏(1 + annual_growth_rate) - 1
```

Example: If annual growth rates are 3%, 2%, -1%, 4%:
```
cumulative = (1.03)(1.02)(0.99)(1.04) - 1 = 8.2%
```

This avoids distortions from inflation and exchange rate fluctuations.
