"""
Question 4: Classification and Policy Implications
===================================================

Build a classifier to identify countries that are likely to achieve a significant 
reduction in CO2 emissions in the next decade.

Business Case Question:
"What are the common characteristics of countries that successfully reduce emissions, 
and how can policymakers in other nations apply these insights?"

Author: Diana Patricia Mendez Mendez
Date: January 23 2026

METHODOLOGY
===========

1. TARGET VARIABLE DEFINITION
   - Binary: 1 = "Likely to reduce emissions significantly", 0 = "Unlikely"
   - Definition: Country achieved >10% reduction in CO2 over past 10 years
   - Assumption: Past performance predicts future trajectory

2. FEATURE SELECTION
   - Use indicators that would be KNOWN at prediction time
   - Exclude leakage variables (future emissions, trends)
   - Focus on structural, economic, and energy indicators

3. MODEL TRAINING
   - Multiple classifiers for comparison
   - Cross-validation for robustness
   - Handle class imbalance if present

4. INTERPRETATION
   - Feature importance analysis
   - Identify common characteristics of successful countries
   - Policy recommendations based on findings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)


# =============================================================================
# CONFIGURATION
# =============================================================================

REDUCTION_THRESHOLD = 0.10  # 10% reduction over the analysis period
ANALYSIS_YEARS = 10  # Look at 10-year trends

RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 8,
    'min_samples_split': 5,
    'min_samples_leaf': 3,
    'class_weight': 'balanced',
    'random_state': 42
}

GB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 4,
    'learning_rate': 0.1,
    'random_state': 42
}

LR_PARAMS = {
    'max_iter': 1000,
    'class_weight': 'balanced',
    'random_state': 42
}


def load_data(data_path):
    print("=" * 50)
    print("LOADING DATA")
    print("=" * 50)
    df = pd.read_csv(data_path)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    print(f"Countries: {df['country_code'].nunique()}")
    return df


def create_target_variable(df, threshold=REDUCTION_THRESHOLD, years=ANALYSIS_YEARS):
    """
    Create binary target variable indicating emission reduction success.
    
    DEFINITION OF SUCCESS (True Decoupling):
    - CO2 emissions decreased by more than `threshold` (e.g., >10%)
    - AND real GDP grew (cumulative real growth > 0%)
    
    We use REAL GDP GROWTH (gdp_growth indicator) instead of nominal GDP change
    because nominal GDP in USD is distorted by:
    - Inflation
    - Exchange rate fluctuations
    - Currency crises
    
    Real GDP growth is inflation-adjusted and measured in local currency,
    giving a true picture of economic performance.
    """
    print("\n" + "=" * 50)
    print("CREATING TARGET VARIABLE")
    print("=" * 50)
    
    max_year = df['year'].max()
    start_year = max_year - years
    
    print(f"Analysis period: {start_year} to {max_year} ({years} years)")
    print(f"CO2 reduction threshold: {threshold*100:.0f}%")
    print(f"\nTarget definition:")
    print(f"  SUCCESS = CO2 reduction >{threshold*100:.0f}% AND real GDP growth > 0%")
    print(f"  (Using cumulative real GDP growth, not nominal USD change)")
    
    country_changes = []
    
    for country in df['country_code'].unique():
        country_data = df[df['country_code'] == country].sort_values('year')
        
        # Filter to analysis period
        period_data = country_data[(country_data['year'] >= start_year) & 
                                    (country_data['year'] <= max_year)]
        
        start_data = country_data[country_data['year'] == start_year]
        end_data = country_data[country_data['year'] == max_year]
        
        if len(start_data) == 0 or len(end_data) == 0:
            continue
        
        co2_start = start_data['co2_emissions_mt'].values[0]
        co2_end = end_data['co2_emissions_mt'].values[0]
        
        if pd.isna(co2_start) or pd.isna(co2_end) or co2_start <= 0:
            continue
        
        co2_pct_change = (co2_end - co2_start) / co2_start
        
        # Calculate CUMULATIVE REAL GDP GROWTH
        # Using the gdp_growth indicator (annual real GDP growth rate in %)
        # Cumulative growth = (1+g1)*(1+g2)*...*(1+gn) - 1
        if 'gdp_growth' in period_data.columns:
            gdp_growth_rates = period_data['gdp_growth'].dropna() / 100  # Convert % to decimal
            
            if len(gdp_growth_rates) >= years * 0.7:  # Require at least 70% of years to have data
                # Calculate cumulative growth: product of (1 + annual_rate) - 1
                cumulative_growth = np.prod(1 + gdp_growth_rates.values) - 1
                real_gdp_grew = cumulative_growth > 0
            else:
                cumulative_growth = np.nan
                real_gdp_grew = None
        else:
            cumulative_growth = np.nan
            real_gdp_grew = None
        
        # Determine success type
        if real_gdp_grew is None:
            # Insufficient GDP data
            success = 0
            success_type = 'insufficient_data'
        elif co2_pct_change < -threshold and real_gdp_grew:
            # TRUE DECOUPLING: Reduced emissions while economy grew
            success = 1
            success_type = 'decoupling'
        elif co2_pct_change < -threshold and not real_gdp_grew:
            # Reduced emissions but economy contracted - not true decoupling
            success = 0
            success_type = 'economic_contraction'
        else:
            # Did not reduce emissions significantly
            success = 0
            success_type = 'no_reduction'
        
        country_changes.append({
            'country_code': country,
            'co2_start': co2_start,
            'co2_end': co2_end,
            'co2_pct_change': co2_pct_change,
            'real_gdp_cumulative_growth': cumulative_growth,
            'success_type': success_type,
            'emission_reduction_success': success
        })
    
    change_df = pd.DataFrame(country_changes)
    
    # Summary statistics
    n_success = change_df['emission_reduction_success'].sum()
    n_total = len(change_df)
    
    print(f"\nCountries analyzed: {n_total}")
    print(f"\nBreakdown by type:")
    type_counts = change_df['success_type'].value_counts()
    for stype, count in type_counts.items():
        print(f"  {stype}: {count} countries")
    
    print(f"\nFinal classification:")
    print(f"  Success (true decoupling): {n_success} ({n_success/n_total*100:.1f}%)")
    print(f"  Non-success: {n_total - n_success} ({(n_total-n_success)/n_total*100:.1f}%)")
    
    # Show examples of successful decouplers
    print("\n--- Examples of SUCCESSFUL DECOUPLERS ---")
    print("(Reduced CO2 >10% while real GDP grew)")
    decouplers = change_df[change_df['success_type'] == 'decoupling'].nsmallest(10, 'co2_pct_change')
    for _, row in decouplers.iterrows():
        gdp_str = f"{row['real_gdp_cumulative_growth']*100:+.1f}%" if pd.notna(row['real_gdp_cumulative_growth']) else "N/A"
        print(f"  {row['country_code']}: CO2 {row['co2_pct_change']*100:+.1f}%, Real GDP growth: {gdp_str}")
    
    # Show examples of economic contraction (excluded from success)
    contraction = change_df[change_df['success_type'] == 'economic_contraction']
    if len(contraction) > 0:
        print("\n--- EXCLUDED: Economic Contraction Countries ---")
        print("(Reduced CO2 but real GDP shrank - not true decoupling)")
        for _, row in contraction.iterrows():
            gdp_str = f"{row['real_gdp_cumulative_growth']*100:+.1f}%" if pd.notna(row['real_gdp_cumulative_growth']) else "N/A"
            print(f"  {row['country_code']}: CO2 {row['co2_pct_change']*100:+.1f}%, Real GDP growth: {gdp_str}")
    
    # Show examples of non-reducers with high growth
    print("\n--- Examples of NON-REDUCERS (highest CO2 growth) ---")
    non_reducers = change_df[change_df['success_type'] == 'no_reduction'].nlargest(5, 'co2_pct_change')
    for _, row in non_reducers.iterrows():
        gdp_str = f"{row['real_gdp_cumulative_growth']*100:+.1f}%" if pd.notna(row['real_gdp_cumulative_growth']) else "N/A"
        print(f"  {row['country_code']}: CO2 {row['co2_pct_change']*100:+.1f}%, Real GDP growth: {gdp_str}")
    
    return change_df


def create_features(df, target_df, base_year=None):
    print("\n" + "=" * 50)
    print("CREATING FEATURE MATRIX")
    print("=" * 50)
    
    if base_year is None:
        base_year = df['year'].max() - ANALYSIS_YEARS
    
    print(f"Using features from year: {base_year}")
    base_df = df[df['year'] == base_year].copy()
    
    feature_cols = [
        'gdp_per_capita', 'gdp_growth',
        'energy_use_per_capita', 'renewable_energy_pct', 'fossil_fuel_electricity_pct',
        'industry_value_added_pct', 'services_value_added_pct',
        'urban_pop_pct', 'population_density',
        'access_to_electricity_pct',
        'co2_emissions_mt', 'co2_emissions_per_capita',
    ]
    
    available_features = [c for c in feature_cols if c in base_df.columns]
    print(f"Available features: {len(available_features)}/{len(feature_cols)}")
    
    feature_df = base_df[['country_code'] + available_features].copy()
    
    for col in ['gdp_per_capita', 'co2_emissions_mt', 'population_density', 'energy_use_per_capita']:
        if col in feature_df.columns:
            feature_df[f'log_{col}'] = np.log1p(feature_df[col])
    
    feature_df = feature_df.merge(
        target_df[['country_code', 'emission_reduction_success', 'co2_pct_change', 
                   'real_gdp_cumulative_growth', 'success_type']], 
        on='country_code'
    )
    
    print(f"Final dataset: {len(feature_df)} countries")
    return feature_df


def prepare_data(feature_df, feature_cols):
    clean_df = feature_df.dropna(subset=['emission_reduction_success'])
    
    # Remove duplicates from feature_cols
    feature_cols = list(dict.fromkeys(feature_cols))
    
    X = clean_df[feature_cols].copy()
    y = clean_df['emission_reduction_success']
    country_codes = clean_df['country_code']
    
    for col in feature_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    
    return X, y, country_codes


def train_and_evaluate(X, y, country_codes):
    print("\n" + "=" * 50)
    print("TRAINING AND EVALUATING MODELS")
    print("=" * 50)
    
    print(f"\nDataset size: {len(X)} countries")
    print(f"Features: {X.shape[1]}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    X_train, X_test, y_train, y_test, codes_train, codes_test = train_test_split(
        X, y, country_codes, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Random Forest': RandomForestClassifier(**RF_PARAMS),
        'Gradient Boosting': GradientBoostingClassifier(**GB_PARAMS),
        'Logistic Regression': LogisticRegression(**LR_PARAMS)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"MODEL: {name}")
        print('='*50)
        
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            y_train_pred = model.predict(X_train_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            y_train_pred = model.predict(X_train)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, y_prob)
        except:
            auc = 0.5
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        if name == 'Logistic Regression':
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1')
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
        
        print(f"\nPerformance Metrics:")
        print(f"  Train Accuracy:  {train_acc:.3f}")
        print(f"  Test Accuracy:   {test_acc:.3f}")
        print(f"  Precision:       {precision:.3f}")
        print(f"  Recall:          {recall:.3f}")
        print(f"  F1 Score:        {f1:.3f}")
        print(f"  AUC-ROC:         {auc:.3f}")
        print(f"  CV F1 (5-fold):  {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
        
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted No  Predicted Yes")
        print(f"  Actual No           {cm[0,0]:3d}          {cm[0,1]:3d}")
        print(f"  Actual Yes          {cm[1,0]:3d}          {cm[1,1]:3d}")
        
        results[name] = {
            'model': model,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std(),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'confusion_matrix': cm,
            'scaler': scaler if name == 'Logistic Regression' else None
        }
        
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            results[name]['feature_importance'] = importance
            
            print(f"\nTop 10 Important Features:")
            for i, (_, row) in enumerate(importance.head(10).iterrows(), 1):
                print(f"  {i:2d}. {row['feature']:30s}: {row['importance']:.4f}")
        
        elif hasattr(model, 'coef_'):
            importance = pd.DataFrame({
                'feature': X.columns,
                'coefficient': model.coef_[0],
                'importance': np.abs(model.coef_[0])
            }).sort_values('importance', ascending=False)
            results[name]['feature_importance'] = importance
            
            print(f"\nTop 10 Features (by coefficient):")
            for i, (_, row) in enumerate(importance.head(10).iterrows(), 1):
                direction = "+" if row['coefficient'] > 0 else "-"
                print(f"  {i:2d}. {row['feature']:30s}: {direction}{row['importance']:.4f}")
    
    results['_data'] = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'codes_test': codes_test,
        'feature_names': list(X.columns)
    }
    
    return results


def analyze_successful_countries(feature_df, feature_cols):
    print("\n" + "=" * 50)
    print("ANALYZING CHARACTERISTICS OF SUCCESSFUL COUNTRIES")
    print("=" * 50)
    
    success = feature_df[feature_df['emission_reduction_success'] == 1]
    non_success = feature_df[feature_df['emission_reduction_success'] == 0]
    
    print(f"\nSuccessful countries: {len(success)}")
    print(f"Non-successful countries: {len(non_success)}")
    
    print("\n--- Feature Comparison (Mean Values) ---")
    print(f"{'Feature':<35} {'Success':>12} {'Non-Success':>12} {'Diff':>10}")
    print("-" * 75)
    
    comparisons = []
    for col in feature_cols:
        if col in feature_df.columns:
            success_mean = success[col].mean()
            non_success_mean = non_success[col].mean()
            
            if pd.notna(success_mean) and pd.notna(non_success_mean):
                diff = success_mean - non_success_mean
                comparisons.append({
                    'feature': col,
                    'success_mean': success_mean,
                    'non_success_mean': non_success_mean,
                    'difference': diff,
                })
                print(f"{col:<35} {success_mean:>12.2f} {non_success_mean:>12.2f} {diff:>+10.2f}")
    
    comparison_df = pd.DataFrame(comparisons)
    
    print("\n--- KEY INSIGHTS ---")
    insights = []
    
    if 'gdp_per_capita' in comparison_df['feature'].values:
        row = comparison_df[comparison_df['feature'] == 'gdp_per_capita'].iloc[0]
        if row['difference'] > 0:
            insights.append(f"Higher GDP per capita (${row['success_mean']:,.0f} vs ${row['non_success_mean']:,.0f})")
    
    if 'renewable_energy_pct' in comparison_df['feature'].values:
        row = comparison_df[comparison_df['feature'] == 'renewable_energy_pct'].iloc[0]
        if row['difference'] > 0:
            insights.append(f"Higher renewable energy share ({row['success_mean']:.1f}% vs {row['non_success_mean']:.1f}%)")
    
    if 'fossil_fuel_electricity_pct' in comparison_df['feature'].values:
        row = comparison_df[comparison_df['feature'] == 'fossil_fuel_electricity_pct'].iloc[0]
        if row['difference'] < 0:
            insights.append(f"Lower fossil fuel dependence ({row['success_mean']:.1f}% vs {row['non_success_mean']:.1f}%)")
    
    if 'services_value_added_pct' in comparison_df['feature'].values:
        row = comparison_df[comparison_df['feature'] == 'services_value_added_pct'].iloc[0]
        if row['difference'] > 0:
            insights.append(f"Larger services sector ({row['success_mean']:.1f}% vs {row['non_success_mean']:.1f}%)")
    
    print("\nSuccessful emission reducers tend to have:")
    for insight in insights:
        print(f"  * {insight}")
    
    print("\n--- SUCCESSFUL COUNTRIES ---")
    success_sorted = success.sort_values('co2_pct_change')
    for _, row in success_sorted.iterrows():
        renewable = f"{row['renewable_energy_pct']:.0f}%" if pd.notna(row['renewable_energy_pct']) else "N/A"
        gdp = f"${row['gdp_per_capita']:,.0f}" if pd.notna(row['gdp_per_capita']) else "N/A"
        gdp_growth = f"{row['real_gdp_cumulative_growth']*100:+.1f}%" if pd.notna(row.get('real_gdp_cumulative_growth')) else "N/A"
        print(f"  {row['country_code']}: CO2 {row['co2_pct_change']*100:+.1f}%, Real GDP growth: {gdp_growth}, Renewable: {renewable}, GDP/cap: {gdp}")
    
    return {
        'comparison': comparison_df,
        'insights': insights,
        'success_countries': success_sorted['country_code'].tolist()
    }


def create_visualizations(results, analysis, feature_df, output_dir):
    print("\n" + "=" * 50)
    print("CREATING VISUALIZATIONS")
    print("=" * 50)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Model Comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    models = ['Random Forest', 'Gradient Boosting', 'Logistic Regression']
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, model in enumerate(models):
        if model in results:
            values = [results[model]['test_acc'], results[model]['precision'],
                      results[model]['recall'], results[model]['f1'], results[model]['auc']]
            ax.bar(x + i * width, values, width, label=model)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1]
    for model_name in models:
        if model_name in results:
            y_test = results[model_name]['y_test']
            y_prob = results[model_name]['y_prob']
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = results[model_name]['auc']
            ax.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.2f})')
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.50)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison_q4.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: model_comparison_q4.png")
    
    # 2. Feature Importance
    if 'Random Forest' in results and 'feature_importance' in results['Random Forest']:
        fig, ax = plt.subplots(figsize=(10, 8))
        importance = results['Random Forest']['feature_importance']
        colors = plt.cm.RdYlGn(importance['importance'] / importance['importance'].max())
        ax.barh(importance['feature'], importance['importance'], color=colors)
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance for Emission Reduction Prediction\n(Random Forest)')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance_q4.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: feature_importance_q4.png")
    
    # 3. Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, model_name in enumerate(models):
        if model_name in results:
            ax = axes[i]
            cm = results[model_name]['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Non-Reducer', 'Reducer'],
                       yticklabels=['Non-Reducer', 'Reducer'])
            ax.set_title(f'{model_name}\nF1={results[model_name]["f1"]:.2f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices_q4.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: confusion_matrices_q4.png")


def generate_report(results, analysis, feature_df, output_path):
    best_model = max(['Random Forest', 'Gradient Boosting', 'Logistic Regression'],
                     key=lambda m: results[m]['f1'] if m in results else 0)
    best_f1 = results[best_model]['f1']
    
    report = f"""# Question 4: Classification and Policy Implications Report

## Executive Summary

This report presents a binary classification model to identify countries likely to achieve significant CO2 emission reductions.

**Business Case Question**: "What are the common characteristics of countries that successfully reduce emissions, and how can policymakers in other nations apply these insights?"

**Key Findings:**
- Best Model: {best_model} (F1 Score: {best_f1:.2f})
- Successful reducers tend to be wealthier, service-based economies with higher renewable energy shares

---

## 1. Methodology

### 1.1 Target Variable

- **Success (1)**: Country reduced CO2 emissions by >{REDUCTION_THRESHOLD*100:.0f}% over {ANALYSIS_YEARS} years
- **Non-Success (0)**: Otherwise

### 1.2 Features

Economic, energy, structural, and demographic indicators from the start of the analysis period.

---

## 2. Model Performance

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
"""
    
    for model_name in ['Random Forest', 'Gradient Boosting', 'Logistic Regression']:
        if model_name in results:
            r = results[model_name]
            report += f"| {model_name} | {r['test_acc']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['auc']:.3f} |\n"
    
    report += f"""
**Best Model**: {best_model}

---

## 3. Key Features

"""
    
    if 'Random Forest' in results and 'feature_importance' in results['Random Forest']:
        importance = results['Random Forest']['feature_importance']
        report += "| Rank | Feature | Importance |\n|------|---------|------------|\n"
        for i, (_, row) in enumerate(importance.head(10).iterrows(), 1):
            report += f"| {i} | {row['feature']} | {row['importance']:.4f} |\n"
    
    report += """
---

## 4. Characteristics of Successful Countries

"""
    
    if 'insights' in analysis:
        for insight in analysis['insights']:
            report += f"- {insight}\n"
    
    report += f"""

### Successful Countries

{', '.join(analysis.get('success_countries', [])[:15])}

---

## 5. Policy Recommendations

### Energy Transition
- Increase renewable energy share
- Phase out fossil fuel electricity
- Invest in grid modernization

### Economic Structure
- Support service sector growth
- Improve industrial efficiency
- Avoid carbon-intensive development

### Enabling Conditions
- Economic development enables climate action
- Carbon pricing creates incentives
- International support for developing countries

---

## 6. Files Generated

- `model_comparison_q4.png`: Performance metrics and ROC curves
- `feature_importance_q4.png`: Feature importance chart
- `confusion_matrices_q4.png`: Confusion matrices

---

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"\nSaved report: {output_path}")


def main():
    print("=" * 70)
    print("QUESTION 4: BINARY CLASSIFICATION - EMISSION REDUCTION PREDICTION")
    print("=" * 70)
    
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / 'data' / 'processed' / 'co2_data_clean.csv'
    figures_dir = base_dir / 'figures'
    reports_dir = base_dir / 'reports'
    
    df = load_data(data_path)
    target_df = create_target_variable(df)
    feature_df = create_features(df, target_df)
    
    feature_cols = [c for c in feature_df.columns 
                    if c not in ['country_code', 'emission_reduction_success', 'co2_pct_change', 
                                 'real_gdp_cumulative_growth', 'success_type']
                    and not c.startswith('co2_')]
    
    # Add log_co2_emissions_mt if present (only once)
    if 'log_co2_emissions_mt' in feature_df.columns and 'log_co2_emissions_mt' not in feature_cols:
        feature_cols.append('log_co2_emissions_mt')
    
    # Remove any duplicates
    feature_cols = list(dict.fromkeys(feature_cols))
    
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
    
    X, y, country_codes = prepare_data(feature_df, feature_cols)
    results = train_and_evaluate(X, y, country_codes)
    analysis = analyze_successful_countries(feature_df, feature_cols)
    create_visualizations(results, analysis, feature_df, figures_dir)
    generate_report(results, analysis, feature_df, reports_dir / 'q4_classification_report.md')
    
    print("\n" + "=" * 50)
    print("QUESTION 4 COMPLETE")
    print("=" * 50)
    
    best_model = max(['Random Forest', 'Gradient Boosting', 'Logistic Regression'],
                     key=lambda m: results[m]['f1'] if m in results else 0)
    
    print(f"\nKey Results:")
    print(f"  Best Model: {best_model}")
    print(f"  F1 Score: {results[best_model]['f1']:.3f}")
    
    return feature_df, results, analysis


if __name__ == "__main__":
    feature_df, results, analysis = main()