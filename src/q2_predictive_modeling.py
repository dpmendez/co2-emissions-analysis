"""
Question 2: Predictive Modeling and Scenario Analysis
=====================================================

This script builds a predictive model for CO2 emissions and answers:
"If a country increases its GDP by 10%, what is the expected percentage 
change in CO2 emissions, assuming all other factors remain constant?"

Author: Diana Patricia Mendez Mendez
Date: January 21 2026

THEORETICAL FOUNDATION
======================

1. WHY LOG-LOG REGRESSION?
   
   Economic relationships are typically multiplicative, not additive:
   - CO2 = alpha * GDP^beta_1 * Population^beta_2 * ...
   
   Taking logs linearizes this:
   - log(CO2) = log(alpha) + beta_1*log(GDP) + beta_1*log(Population) + ...
   
   The coefficients (beta) become ELASTICITIES:
   - beta_1 = 0.75 means "1% increase in GDP -> 0.75% increase in CO2"
   - This directly answers the exercise question

2. WHY RIDGE REGRESSION?
   
   - Handles multicollinearity (GDP and population are correlated)
   - Coefficients remain interpretable (unlike tree-based models)
   - Regularization prevents overfitting
   - We can directly read elasticities from coefficients

3. WHY TEMPORAL SPLIT?
   
   Panel data (countries * years) requires careful splitting:
   - Random split would leak future information into training
   - Temporal split: train on past (<=2017), test on future (>=2018)
   - This simulates real forecasting scenarios

4. ANSWERING THE "10% GDP INCREASE" QUESTION
   
   With log-log model and coefficient beta_gdp:
   - % change in CO2 ~= beta_gdp * % change in GDP
   - For 10% GDP increase: % change in CO2 ~= beta_gdp * 10%
   
   More precisely: CO2_new/CO2_old = (GDP_new/GDP_old)^beta_gdp = 1.10^beta_gdp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings

# Machine Learning
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Features to use (excluding target-derived variables to avoid leakage)
FEATURES = [
    # Economic
    'log_gdp',              # Log of GDP - KEY PREDICTOR
    'gdp_per_capita',       # GDP per capita
    'gdp_growth',           # Annual GDP growth %
    
    # Demographic
    'log_population',       # Log of population - KEY PREDICTOR
    'urban_pop_pct',        # Urbanization rate
    'population_density',   # People per sq km
    
    # Energy
    'energy_use_per_capita',       # Energy consumption
    'renewable_energy_pct',        # Renewable energy share
    'fossil_fuel_electricity_pct', # Fossil fuel share in electricity
    
    # Structural
    'industry_value_added_pct',    # Industry share of GDP
    'services_value_added_pct',    # Services share of GDP
    
    # Development
    'access_to_electricity_pct',   # Electrification rate
]

# Target variable (we'll use log transform)
TARGET = 'co2_emissions_mt'
LOG_TARGET = 'log_co2'

# Temporal split year
SPLIT_YEAR = 2017  # Train on ≤2017, test on ≥2018


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_and_prepare_data(data_path: Path) -> pd.DataFrame:
    """
    Load the cleaned data and prepare it for modeling.
    
    Key steps:
    1. Load the preprocessed data from Q1
    2. Create log-transformed target if not present
    3. Handle any remaining missing values
    """
    print("=" * 50)
    print("LOADING AND PREPARING DATA")
    print("=" * 50)
    
    df = pd.read_csv(data_path)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Ensure log_co2 exists
    if LOG_TARGET not in df.columns and TARGET in df.columns:
        df[LOG_TARGET] = np.log1p(df[TARGET])
        print(f"Created {LOG_TARGET} from {TARGET}")
    
    # Ensure log_gdp exists
    if 'log_gdp' not in df.columns and 'gdp_current_usd' in df.columns:
        df['log_gdp'] = np.log1p(df['gdp_current_usd'])
        print("Created log_gdp from gdp_current_usd")
    
    # Ensure log_population exists
    if 'log_population' not in df.columns and 'population' in df.columns:
        df['log_population'] = np.log1p(df['population'])
        print("Created log_population from population")
    
    # Check which features are available
    available_features = [f for f in FEATURES if f in df.columns]
    missing_features = [f for f in FEATURES if f not in df.columns]
    
    print(f"\nAvailable features: {len(available_features)}/{len(FEATURES)}")
    if missing_features:
        print(f"Missing features: {missing_features}")
    
    # Check for year column
    if 'year' not in df.columns:
        raise ValueError("Data must have 'year' column for temporal split")
    
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    
    return df, available_features


def create_temporal_split(df: pd.DataFrame, features: list, split_year: int = SPLIT_YEAR):
    """
    Create train/test split based on time.
    
    WHY TEMPORAL SPLIT:
    - Panel data has both cross-sectional (countries) and time dimensions
    - Random split would leak future information into training
    - Temporal split simulates real forecasting: use past to predict future
    
    Parameters:
    -----------
    df : DataFrame
    features : list of feature column names
    split_year : int, last year to include in training set
    
    Returns:
    --------
    X_train, X_test, y_train, y_test, train_df, test_df
    """
    print("\n" + "=" * 50)
    print("CREATING TEMPORAL TRAIN/TEST SPLIT")
    print("=" * 50)
    
    # Split by year
    train_df = df[df['year'] <= split_year].copy()
    test_df = df[df['year'] > split_year].copy()
    
    print(f"Split year: {split_year}")
    print(f"Training set: {train_df['year'].min()}-{train_df['year'].max()} ({len(train_df)} rows)")
    print(f"Test set: {test_df['year'].min()}-{test_df['year'].max()} ({len(test_df)} rows)")
    
    # Drop rows with missing target
    train_df = train_df.dropna(subset=[LOG_TARGET])
    test_df = test_df.dropna(subset=[LOG_TARGET])
    
    print(f"After dropping missing target - Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Prepare feature matrices
    # Handle missing values in features by dropping rows with any NaN
    # (Alternative: imputation, but simpler to drop for now)
    train_complete = train_df.dropna(subset=features)
    test_complete = test_df.dropna(subset=features)
    
    print(f"After dropping missing features - Train: {len(train_complete)}, Test: {len(test_complete)}")
    
    X_train = train_complete[features].values
    X_test = test_complete[features].values
    y_train = train_complete[LOG_TARGET].values
    y_test = test_complete[LOG_TARGET].values
    
    print(f"\nFinal shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, train_complete, test_complete


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_ridge_regression(X_train, y_train, feature_names: list):
    """
    Train Ridge Regression model with cross-validation for alpha selection.
    
    WHY RIDGE:w
    - L2 regularization handles multicollinearity
    - Coefficients remain interpretable as elasticities
    - RidgeCV automatically selects optimal regularization strength
    
    Returns:
    --------
    model : fitted RidgeCV model
    coefficients : dict mapping feature names to coefficients
    """
    print("\n" + "=" * 50)
    print("TRAINING RIDGE REGRESSION")
    print("=" * 50)
    
    # Standardize features for regularization to work properly
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Use cross-validation to select alpha (regularization strength)
    alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    ridge_cv = RidgeCV(alphas=alphas, cv=5)
    ridge_cv.fit(X_train_scaled, y_train)
    
    print(f"Optimal alpha (regularization): {ridge_cv.alpha_}")
    print(f"Training R2: {ridge_cv.score(X_train_scaled, y_train):.4f}")
    
    # Get coefficients
    coefficients = dict(zip(feature_names, ridge_cv.coef_))
    
    print("\nFeature Coefficients (Elasticities for log variables):")
    print("-" * 50)
    for name, coef in sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True):
        interpretation = ""
        if name == 'log_gdp':
            interpretation = f" 1% GDP increase -> {coef:.3f}% CO2 change"
        elif name == 'log_population':
            interpretation = f" 1% population increase -> {coef:.3f}% CO2 change"
        print(f"  {name:30s}: {coef:+.4f}{interpretation}")
    
    return ridge_cv, scaler, coefficients


def train_random_forest(X_train, y_train, feature_names: list):
    """
    Train Random Forest for comparison (better predictive power, less interpretable).
    """
    print("\n" + "=" * 50)
    print("TRAINING RANDOM FOREST (for comparison)")
    print("=" * 50)
    
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    print(f"Training R2: {rf.score(X_train, y_train):.4f}")
    
    # Feature importance
    importance = dict(zip(feature_names, rf.feature_importances_))
    print("\nFeature Importance:")
    print("-" * 50)
    for name, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {name:30s}: {imp:.4f}")
    
    return rf, importance


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(model, X_test, y_test, scaler=None, model_name="Model"):
    """
    Evaluate model performance on test set.
    
    Metrics:
    - R2 (coefficient of determination): variance explained
    - RMSE (root mean squared error): average error magnitude
    - MAE (mean absolute error): average absolute error
    - MAPE (mean absolute percentage error): percentage error
    """
    print(f"\n--- {model_name} Evaluation ---")
    
    # Scale if necessary (for Ridge)
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # MAPE (careful with zeros)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"  R2 Score:  {r2:.4f}  (variance explained)")
    print(f"  RMSE:      {rmse:.4f}  (in log scale)")
    print(f"  MAE:       {mae:.4f}  (in log scale)")
    print(f"  MAPE:      {mape:.2f}%")
    
    # Convert back from log scale for interpretability
    y_test_original = np.expm1(y_test)
    y_pred_original = np.expm1(y_pred)
    
    rmse_original = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    mae_original = mean_absolute_error(y_test_original, y_pred_original)
    
    print(f"\n  In original scale (Mt CO2):")
    print(f"  RMSE:      {rmse_original:.2f} Mt")
    print(f"  MAE:       {mae_original:.2f} Mt")
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'rmse_original': rmse_original,
        'mae_original': mae_original,
        'y_pred': y_pred,
        'y_pred_original': y_pred_original
    }


# =============================================================================
# SCENARIO ANALYSIS: 10% GDP INCREASE
# =============================================================================

def analyze_gdp_scenario(coefficients: dict, test_df: pd.DataFrame):
    """
    Answer the key question: "If GDP increases by 10%, what happens to CO2?"
    
    METHODOLOGY:
    With log-log model: log(CO2) = beta_0 + beta_gdp * log(GDP) + ...
    
    If GDP increases by 10%:
    - New GDP = 1.10 * Old GDP
    - log(New GDP) = log(Old GDP) + log(1.10)
    - deltalog(CO2) = beta_gdp * log(1.10)
    - New CO2 / Old CO2 = exp(beta_gdp * log(1.10)) = 1.10^beta_gdp
    
    So the percentage change in CO2 = (1.10^beta_gdp - 1) * 100%
    """
    print("\n" + "=" * 50)
    print("SCENARIO ANALYSIS: 10% GDP INCREASE")
    print("=" * 50)
    
    # Get the GDP elasticity coefficient
    beta_gdp = coefficients.get('log_gdp', 0)
    
    print(f"\nGDP Elasticity (beta_gdp): {beta_gdp:.4f}")
    print("\nInterpretation:")
    print("-" * 50)
    
    # Calculate percentage change in CO2 for 10% GDP increase
    gdp_multiplier = 1.10  # 10% increase
    co2_multiplier = gdp_multiplier ** beta_gdp
    co2_pct_change = (co2_multiplier - 1) * 100
    
    print(f"If GDP increases by 10%:")
    print(f"  CO2 multiplier = 1.10^{beta_gdp:.4f} = {co2_multiplier:.4f}")
    print(f"  Expected CO2 change: {co2_pct_change:+.2f}%")
    
    # Quick approximation (valid for small changes)
    approx_change = beta_gdp * 10  # beta × %deltaGDP
    print(f"\n  Quick approximation: {beta_gdp:.4f} * 10% = {approx_change:+.2f}%")
    print(f"  (Approximation error: {abs(co2_pct_change - approx_change):.3f} percentage points)")
    
    # Analysis by country income level
    if 'gdp_per_capita' in test_df.columns:
        print("\n" + "-" * 50)
        print("Analysis by Country Income Level:")
        print("-" * 50)
        
        # Categorize countries by GDP per capita
        test_recent = test_df[test_df['year'] == test_df['year'].max()].copy()
        
        if len(test_recent) > 0:
            # Create income groups
            test_recent['income_group'] = pd.cut(
                test_recent['gdp_per_capita'],
                bins=[0, 1000, 4000, 12000, float('inf')],
                labels=['Low', 'Lower-Middle', 'Upper-Middle', 'High']
            )
            
            # Show distribution
            print(f"\nCountries in test set by income group:")
            income_counts = test_recent['income_group'].value_counts().sort_index()
            for group, count in income_counts.items():
                print(f"  {group}: {count} countries")
            
            print(f"\nNote: The elasticity of {beta_gdp:.3f} is an AVERAGE across all countries.")
            print("In reality, elasticity likely varies:")
            print("  - Lower in developed countries (more efficient economies)")
            print("  - Higher in developing countries (more carbon-intensive growth)")
    
    return {
        'beta_gdp': beta_gdp,
        'gdp_change_pct': 10,
        'co2_change_pct': co2_pct_change,
        'co2_multiplier': co2_multiplier
    }


def simulate_gdp_scenarios(model, scaler, features: list, test_df: pd.DataFrame, 
                           feature_names: list):
    """
    Simulate various GDP increase scenarios and show range of impacts.
    """
    print("\n" + "=" * 50)
    print("GDP SCENARIO SIMULATIONS")
    print("=" * 50)
    
    # Get a sample of countries from test set
    latest_year = test_df['year'].max()
    sample_df = test_df[test_df['year'] == latest_year].copy()
    sample_df = sample_df.dropna(subset=features)
    
    if len(sample_df) == 0:
        print("No complete data available for simulation")
        return None
    
    # Get baseline predictions
    X_baseline = sample_df[features].values
    X_baseline_scaled = scaler.transform(X_baseline)
    y_baseline_log = model.predict(X_baseline_scaled)
    y_baseline = np.expm1(y_baseline_log)
    
    # Simulate 10% GDP increase
    gdp_idx = features.index('log_gdp') if 'log_gdp' in features else None
    
    if gdp_idx is None:
        print("log_gdp not in features, cannot simulate")
        return None
    
    # Create modified feature matrix with 10% higher GDP
    X_modified = X_baseline.copy()
    X_modified[:, gdp_idx] = X_baseline[:, gdp_idx] + np.log(1.10)  # log(1.10*GDP) = log(GDP) + log(1.10)
    
    X_modified_scaled = scaler.transform(X_modified)
    y_modified_log = model.predict(X_modified_scaled)
    y_modified = np.expm1(y_modified_log)
    
    # Calculate changes
    pct_changes = ((y_modified - y_baseline) / y_baseline) * 100
    
    print(f"\nSimulated impact of 10% GDP increase across {len(sample_df)} countries:")
    print("-" * 50)
    print(f"  Mean CO2 change:   {pct_changes.mean():+.2f}%")
    print(f"  Median CO2 change: {np.median(pct_changes):+.2f}%")
    print(f"  Std deviation:     {pct_changes.std():.2f}%")
    print(f"  Range:             {pct_changes.min():+.2f}% to {pct_changes.max():+.2f}%")
    
    # Show by country if country_code available
    if 'country_code' in sample_df.columns:
        sample_df = sample_df.copy()
        sample_df['co2_baseline'] = y_baseline
        sample_df['co2_after_gdp_increase'] = y_modified
        sample_df['pct_change'] = pct_changes
        
        print(f"\nTop 5 countries with LARGEST CO2 increase:")
        top_increase = sample_df.nlargest(5, 'pct_change')[['country_code', 'co2_baseline', 'pct_change']]
        for _, row in top_increase.iterrows():
            print(f"  {row['country_code']}: {row['pct_change']:+.2f}% (baseline: {row['co2_baseline']:.1f} Mt)")
        
        print(f"\nTop 5 countries with SMALLEST CO2 increase:")
        top_decrease = sample_df.nsmallest(5, 'pct_change')[['country_code', 'co2_baseline', 'pct_change']]
        for _, row in top_decrease.iterrows():
            print(f"  {row['country_code']}: {row['pct_change']:+.2f}% (baseline: {row['co2_baseline']:.1f} Mt)")
    
    return {
        'pct_changes': pct_changes,
        'mean_change': pct_changes.mean(),
        'median_change': np.median(pct_changes),
        'std_change': pct_changes.std()
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualizations(y_test, ridge_results, rf_results, coefficients, 
                         test_df, output_dir: Path):
    """Create visualizations for the modeling results."""
    print("\n" + "=" * 50)
    print("CREATING VISUALIZATIONS")
    print("=" * 50)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Actual vs Predicted (both models)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Ridge
    ax1 = axes[0]
    ax1.scatter(np.expm1(y_test), ridge_results['y_pred_original'], alpha=0.5, s=20)
    max_val = max(np.expm1(y_test).max(), ridge_results['y_pred_original'].max())
    ax1.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
    ax1.set_xlabel('Actual CO2 Emissions (Mt)')
    ax1.set_ylabel('Predicted CO2 Emissions (Mt)')
    ax1.set_title(f'Ridge Regression (R2 = {ridge_results["r2"]:.3f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Random Forest
    ax2 = axes[1]
    ax2.scatter(np.expm1(y_test), rf_results['y_pred_original'], alpha=0.5, s=20)
    ax2.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
    ax2.set_xlabel('Actual CO2 Emissions (Mt)')
    ax2.set_ylabel('Predicted CO2 Emissions (Mt)')
    ax2.set_title(f'Random Forest (R2 = {rf_results["r2"]:.3f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: model_predictions.png")
    
    # 2. Coefficient plot for Ridge
    fig, ax = plt.subplots(figsize=(10, 8))
    
    coef_df = pd.DataFrame({
        'feature': list(coefficients.keys()),
        'coefficient': list(coefficients.values())
    }).sort_values('coefficient')
    
    colors = ['green' if c > 0 else 'red' for c in coef_df['coefficient']]
    ax.barh(coef_df['feature'], coef_df['coefficient'], color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Coefficient (Elasticity for log variables)')
    ax.set_title('Ridge Regression Coefficients')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Highlight log_gdp
    for i, (feat, coef) in enumerate(zip(coef_df['feature'], coef_df['coefficient'])):
        if feat == 'log_gdp':
            ax.annotate(f'GDP Elasticity: {coef:.3f}', 
                       xy=(coef, feat), xytext=(coef + 0.1, i),
                       fontsize=10, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='black'))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ridge_coefficients.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: ridge_coefficients.png")
    
    # 3. Residual plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    residuals = y_test - ridge_results['y_pred']
    ax.scatter(ridge_results['y_pred'], residuals, alpha=0.5, s=20)
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_xlabel('Predicted log(CO2)')
    ax.set_ylabel('Residuals')
    ax.set_title('Ridge Regression Residuals')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'residuals.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: residuals.png")


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_markdown_report(ridge_results: dict, rf_results: dict, 
                            coefficients: dict, scenario_results: dict,
                            simulation_results: dict,
                            train_size: int, test_size: int,
                            features_used: list,
                            output_path: Path):
    """Generate comprehensive Markdown report for Q2."""
    
    beta_gdp = scenario_results['beta_gdp']
    co2_change = scenario_results['co2_change_pct']
    
    report = f"""# Question 2: Predictive Modeling and Scenario Analysis Report

## Executive Summary

This report presents a predictive model for CO2 emissions and answers the key question:

> **"If a country increases its GDP by 10%, what is the expected percentage change in CO2 emissions?"**

**Answer: A 10% increase in GDP is associated with approximately {co2_change:.1f}% increase in CO2 emissions**, holding other factors constant (GDP elasticity = {beta_gdp:.3f}).

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
- **Training set**: Years <= 2017 ({train_size:,} observations)
- **Test set**: Years > 2017 ({test_size:,} observations)

This simulates real forecasting: using historical data to predict future emissions.

### 1.4 Features Used

| Category | Features |
|----------|----------|
"""
    
    # Group features by category
    economic = [f for f in features_used if f in ['log_gdp', 'gdp_per_capita', 'gdp_growth']]
    demographic = [f for f in features_used if f in ['log_population', 'urban_pop_pct', 'population_density']]
    energy = [f for f in features_used if f in ['energy_use_per_capita', 'renewable_energy_pct', 'fossil_fuel_electricity_pct', 'electric_power_consumption']]
    structural = [f for f in features_used if f in ['industry_value_added_pct', 'services_value_added_pct', 'agriculture_value_added_pct']]
    development = [f for f in features_used if f in ['access_to_electricity_pct', 'forest_area_pct']]
    
    if economic:
        report += f"| Economic | {', '.join(economic)} |\n"
    if demographic:
        report += f"| Demographic | {', '.join(demographic)} |\n"
    if energy:
        report += f"| Energy | {', '.join(energy)} |\n"
    if structural:
        report += f"| Structural | {', '.join(structural)} |\n"
    if development:
        report += f"| Development | {', '.join(development)} |\n"
    
    report += f"""

**Note**: We excluded CO2 sectoral variables (transport, power, etc.) as they are components of the target variable and would cause data leakage.

---

## 2. Model Performance

### 2.1 Comparison of Models

| Metric | Ridge Regression | Random Forest |
|--------|------------------|---------------|
| R2 (test) | {ridge_results['r2']:.4f} | {rf_results['r2']:.4f} |
| RMSE (log scale) | {ridge_results['rmse']:.4f} | {rf_results['rmse']:.4f} |
| MAE (log scale) | {ridge_results['mae']:.4f} | {rf_results['mae']:.4f} |
| MAPE | {ridge_results['mape']:.2f}% | {rf_results['mape']:.2f}% |
| RMSE (Mt CO2) | {ridge_results['rmse_original']:.2f} | {rf_results['rmse_original']:.2f} |
| MAE (Mt CO2) | {ridge_results['mae_original']:.2f} | {rf_results['mae_original']:.2f} |

**Interpretation**: 
- R2 of {ridge_results['r2']:.2f} means the model explains {ridge_results['r2']*100:.0f}% of the variance in CO2 emissions
- Random Forest has slightly better predictive power but Ridge provides interpretable coefficients

### 2.2 Ridge Regression Coefficients

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
"""
    
    for name, coef in sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True):
        if name == 'log_gdp':
            interp = f"**1% GDP increase → {coef:.3f}% CO2 change**"
        elif name == 'log_population':
            interp = f"1% population increase → {coef:.3f}% CO2 change"
        elif 'pct' in name or 'growth' in name:
            interp = f"1 pp increase → {coef:.3f} change in log(CO2)"
        else:
            interp = f"1 unit increase → {coef:.3f} change in log(CO2)"
        report += f"| {name} | {coef:+.4f} | {interp} |\n"
    
    report += f"""

---

## 3. Scenario Analysis: 10% GDP Increase

### 3.1 Theoretical Calculation

Given the GDP elasticity beta_gdp = {beta_gdp:.4f}:

```
If GDP increases by 10%:
  New GDP = 1.10 * Old GDP
  CO2 multiplier = 1.10^β_gdp = 1.10^{beta_gdp:.4f} = {scenario_results['co2_multiplier']:.4f}
  
  Expected CO2 change = ({scenario_results['co2_multiplier']:.4f} - 1) * 100% = {co2_change:+.2f}%
```

### 3.2 Key Finding

> **A 10% increase in GDP is associated with a {co2_change:.1f}% increase in CO2 emissions**, assuming all other factors remain constant.

### 3.3 Interpretation

- The GDP elasticity of {beta_gdp:.3f} indicates that CO2 emissions are **{"inelastic" if abs(beta_gdp) < 1 else "elastic"}** with respect to GDP
- {"Emissions grow slower than GDP (decoupling tendency)" if beta_gdp < 1 else "Emissions grow proportionally with GDP"}
- This is consistent with {"improved energy efficiency and structural shifts in developed economies" if beta_gdp < 1 else "carbon-intensive growth patterns"}

"""
    
    if simulation_results:
        report += f"""### 3.4 Variation Across Countries

Simulating the 10% GDP increase across countries in the test set:

| Statistic | CO2 Change |
|-----------|------------|
| Mean | {simulation_results['mean_change']:+.2f}% |
| Median | {simulation_results['median_change']:+.2f}% |
| Std Dev | {simulation_results['std_change']:.2f}% |

The variation reflects differences in:
- Economic structure (services vs. industry-heavy)
- Energy efficiency
- Energy mix (renewables vs. fossil fuels)
- Development stage

"""
    
    report += f"""---

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

**A 10% increase in GDP is expected to increase CO2 emissions by approximately {co2_change:.1f}%** (95% of countries see changes between {simulation_results['mean_change'] - 2*simulation_results['std_change']:.1f}% and {simulation_results['mean_change'] + 2*simulation_results['std_change']:.1f}%).

This elasticity of {beta_gdp:.3f} suggests that economic growth is associated with emissions growth, but at a {"slower rate (relative decoupling)" if beta_gdp < 1 else "proportional or faster rate"}.

---

*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"\nSaved report: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("QUESTION 2: PREDICTIVE MODELING AND SCENARIO ANALYSIS")
    print("=" * 60)
    
    # Set up paths
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / 'data' / 'processed' / 'co2_data_clean.csv'
    figures_dir = base_dir / 'figures'
    reports_dir = base_dir / 'reports'
    
    # Load and prepare data
    df, available_features = load_and_prepare_data(data_path)
    
    # Create train/test split
    X_train, X_test, y_train, y_test, train_df, test_df = create_temporal_split(
        df, available_features, SPLIT_YEAR
    )
    
    # Train Ridge Regression
    ridge_model, scaler, coefficients = train_ridge_regression(
        X_train, y_train, available_features
    )
    
    # Train Random Forest for comparison
    rf_model, rf_importance = train_random_forest(
        X_train, y_train, available_features
    )
    
    # Evaluate both models
    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    
    ridge_results = evaluate_model(ridge_model, X_test, y_test, scaler, "Ridge Regression")
    rf_results = evaluate_model(rf_model, X_test, y_test, None, "Random Forest")
    
    # Scenario Analysis
    scenario_results = analyze_gdp_scenario(coefficients, test_df)
    
    # Simulate across countries
    simulation_results = simulate_gdp_scenarios(
        ridge_model, scaler, available_features, test_df, available_features
    )
    
    # Create visualizations
    create_visualizations(
        y_test, ridge_results, rf_results, coefficients, test_df, figures_dir
    )
    
    # Generate report
    report_path = reports_dir / 'q2_modeling_results.md'
    generate_markdown_report(
        ridge_results, rf_results, coefficients, scenario_results,
        simulation_results, len(train_df), len(test_df),
        available_features, report_path
    )
    
    print("\n" + "=" * 50)
    print("QUESTION 2 COMPLETE")
    print("=" * 50)
    print(f"\nKey Finding:")
    print(f"  GDP Elasticity: {scenario_results['beta_gdp']:.4f}")
    print(f"  10% GDP increase → {scenario_results['co2_change_pct']:.2f}% CO2 increase")
    
    return ridge_model, rf_model, coefficients, scenario_results


if __name__ == "__main__":
    ridge_model, rf_model, coefficients, scenario_results = main()