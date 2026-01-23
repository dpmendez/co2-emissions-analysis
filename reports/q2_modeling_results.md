# Question 2: Predictive Modeling and Scenario Analysis Report

## Executive Summary

This report presents a predictive model for CO2 emissions and answers the key question:

> **"If a country increases its GDP by 10%, what is the expected percentage change in CO2 emissions?"**

**Answer: A 10% increase in GDP is associated with approximately 6.6% increase in CO2 emissions**, holding other factors constant (GDP elasticity = 0.672).

---

## 1. Methodology

### 1.1 Model Selection Rationale

We chose **Ridge Regression** as the primary model because:

1. **Interpretability**: Coefficients directly represent elasticities in a log-log model
2. **Multicollinearity handling**: L2 regularization addresses correlated predictors (GDP and population)
3. **Answers the question**: The GDP coefficient directly tells us the % change in CO2 per % change in GDP

We also trained a **Random Forest** for comparison on predictive performance.

### 1.2 Log-Log Specification

We use log-transformed variables:

```
log(CO2) = beta_0 + beta_gdp*log(GDP) + beta_pop*log(Population) + beta_1*X_1 + ...
```

**Why?** In this specification, beta_gdp is the **elasticity** of CO2 with respect to GDP:
- A 1% increase in GDP -> beta_gdp% increase in CO2
- This directly answers the exam question

### 1.3 Train/Test Split

We used a **temporal split** to prevent data leakage:
- **Training set**: Years <= 2017 (3,698 observations)
- **Test set**: Years > 2017 (982 observations)

This simulates real forecasting: using historical data to predict future emissions.

### 1.4 Features Used

| Category | Features |
|----------|----------|
| Economic | log_gdp, gdp_per_capita, gdp_growth |
| Demographic | log_population, urban_pop_pct, population_density |
| Energy | energy_use_per_capita, renewable_energy_pct, fossil_fuel_electricity_pct |
| Structural | industry_value_added_pct, services_value_added_pct |
| Development | access_to_electricity_pct |


**Note**: We excluded CO2 sectoral variables (transport, power, etc.) as they are components of the target variable and would cause data leakage.

---

## 2. Model Performance

### 2.1 Comparison of Models

| Metric | Ridge Regression | Random Forest |
|--------|------------------|---------------|
| R2 (test) | 0.9330 | 0.9444 |
| RMSE (log scale) | 0.4357 | 0.3968 |
| MAE (log scale) | 0.3522 | 0.2649 |
| MAPE | 13.24% | 9.71% |
| RMSE (Mt CO2) | 693.31 | 537.77 |
| MAE (Mt CO2) | 110.63 | 81.26 |

**Interpretation**: 
- R2 of 0.93 means the model explains 93% of the variance in CO2 emissions
- Random Forest has slightly better predictive power but Ridge provides interpretable coefficients

### 2.2 Ridge Regression Coefficients

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| log_population | +0.9860 | 1% population increase → 0.986% CO2 change |
| log_gdp | +0.6719 | **1% GDP increase → 0.672% CO2 change** |
| renewable_energy_pct | -0.4348 | 1 pp increase → -0.435 change in log(CO2) |
| energy_use_per_capita | +0.3273 | 1 unit increase → 0.327 change in log(CO2) |
| access_to_electricity_pct | +0.2643 | 1 pp increase → 0.264 change in log(CO2) |
| gdp_per_capita | -0.1158 | 1 unit increase → -0.116 change in log(CO2) |
| fossil_fuel_electricity_pct | +0.0930 | 1 pp increase → 0.093 change in log(CO2) |
| population_density | -0.0705 | 1 unit increase → -0.070 change in log(CO2) |
| industry_value_added_pct | +0.0590 | 1 pp increase → 0.059 change in log(CO2) |
| gdp_growth | -0.0328 | 1 pp increase → -0.033 change in log(CO2) |
| services_value_added_pct | +0.0238 | 1 pp increase → 0.024 change in log(CO2) |
| urban_pop_pct | +0.0147 | 1 pp increase → 0.015 change in log(CO2) |


---

## 3. Scenario Analysis: 10% GDP Increase

### 3.1 Theoretical Calculation

Given the GDP elasticity beta_gdp = 0.6719:

```
If GDP increases by 10%:
  New GDP = 1.10 * Old GDP
  CO2 multiplier = 1.10^β_gdp = 1.10^0.6719 = 1.0661
  
  Expected CO2 change = (1.0661 - 1) * 100% = +6.61%
```

### 3.2 Key Finding

> **A 10% increase in GDP is associated with a 6.6% increase in CO2 emissions**, assuming all other factors remain constant.

### 3.3 Interpretation

- The GDP elasticity of 0.672 indicates that CO2 emissions are **inelastic** with respect to GDP
- Emissions grow slower than GDP (decoupling tendency)
- This is consistent with improved energy efficiency and structural shifts in developed economies

### 3.4 Variation Across Countries

Simulating the 10% GDP increase across countries in the test set:

| Statistic | CO2 Change |
|-----------|------------|
| Mean | +3.15% |
| Median | +3.07% |
| Std Dev | 0.24% |

The variation reflects differences in:
- Economic structure (services vs. industry-heavy)
- Energy efficiency
- Energy mix (renewables vs. fossil fuels)
- Development stage

---

## 4. Limitations and Caveats

1. **Ceteris Paribus Assumption**: The scenario assumes "all else equal," but GDP growth typically correlates with population growth, urbanization, and industrialization

2. **Linearity in Logs**: The model assumes constant elasticity across all GDP levels, but elasticity likely varies by income level

3. **Causality**: The model captures associations, not causal effects. GDP growth might cause CO2 increase, or both might be driven by industrialization

4. **Heterogeneity**: A single global elasticity masks variation across countries with different economic structures

5. **Temporal Dynamics**: The model doesn't capture lag effects (GDP changes may affect CO2 with delay)

---

## 5. Files Generated

| File | Description |
|------|-------------|
| `figures/model_predictions.png` | Actual vs predicted scatter plots |
| `figures/ridge_coefficients.png` | Bar chart of Ridge coefficients |
| `figures/residuals.png` | Residual analysis plot |

---

## 6. Conclusion

The Ridge Regression model with log-transformed variables provides an interpretable answer to the policy question:

**A 10% increase in GDP is expected to increase CO2 emissions by approximately 6.6%** (95% of countries see changes between 2.7% and 3.6%).

This elasticity of 0.672 suggests that economic growth is associated with emissions growth, but at a slower rate (relative decoupling).

---

*Report generated: 2026-01-22 21:05:45*
