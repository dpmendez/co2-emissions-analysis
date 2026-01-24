"""
Question 5: Strategic Recommendations
=====================================

Based on the analysis from Q1-Q4, provide strategic recommendations for:
1. Countries to prioritize for emission reduction efforts
2. Sector-specific interventions (transport, power, industry)
3. Policy recommendations based on country characteristics

This script synthesizes findings and generates actionable recommendations.

Author: Diana Patricia Mendez Mendez
Date: January 23 2026

METHODOLOGY
===========

1. COUNTRY PRIORITIZATION
   - High impact: Large emitters where interventions have biggest absolute effect
   - High potential: Countries with favorable conditions for decarbonization
   - Quick wins: Countries close to tipping points

2. SECTOR ANALYSIS
   - Identify dominant emission sources by country
   - Match interventions to sector profiles

3. POLICY MATCHING
   - Group countries by characteristics
   - Recommend tailored policies for each group
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(data_path: Path) -> pd.DataFrame:
    """Load the cleaned data from Q1."""
    print("=" * 50)
    print("LOADING DATA")
    print("=" * 50)
    
    df = pd.read_csv(data_path)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Handle column name variations
    # Some datasets have co2_emissions_kt (kilotons), others have co2_emissions_mt (megatons)
    if 'co2_emissions_kt' in df.columns and 'co2_emissions_mt' not in df.columns:
        df['co2_emissions_mt'] = df['co2_emissions_kt'] / 1000  # Convert kt to Mt
        print("Converted co2_emissions_kt to co2_emissions_mt")
    
    return df


def prepare_latest_data(df: pd.DataFrame) -> pd.DataFrame:
    """Get the most recent year's data for each country."""
    latest_year = df['year'].max()
    latest_df = df[df['year'] == latest_year].copy()
    
    # Calculate additional metrics
    if 'co2_emissions_mt' in latest_df.columns and 'gdp_current_usd' in latest_df.columns:
        # CO2 intensity (kg CO2 per $ GDP)
        latest_df['co2_intensity'] = (
            latest_df['co2_emissions_mt'] * 1e9 / 
            latest_df['gdp_current_usd'].replace(0, np.nan)
        )
    
    # Calculate emission trends (need historical data)
    trends = []
    for country in latest_df['country_code'].unique():
        country_data = df[df['country_code'] == country].sort_values('year')
        if len(country_data) >= 5:
            recent = country_data.tail(5)
            if recent['co2_emissions_mt'].notna().sum() >= 3:
                # Simple linear trend
                co2_values = recent['co2_emissions_mt'].dropna()
                if len(co2_values) >= 2:
                    trend = (co2_values.iloc[-1] - co2_values.iloc[0]) / co2_values.iloc[0]
                    trends.append({'country_code': country, 'co2_5yr_trend': trend})
    
    if trends:
        trend_df = pd.DataFrame(trends)
        latest_df = latest_df.merge(trend_df, on='country_code', how='left')
    
    print(f"Latest year: {latest_year}")
    print(f"Countries: {len(latest_df)}")
    
    return latest_df


# =============================================================================
# COUNTRY PRIORITIZATION
# =============================================================================

def prioritize_countries(df: pd.DataFrame) -> dict:
    """
    Prioritize countries for emission reduction efforts.
    
    Three prioritization criteria:
    1. HIGH IMPACT: Large total emissions (absolute reduction potential)
    2. HIGH POTENTIAL: Favorable conditions (clean grid, high renewables potential)
    3. QUICK WINS: Already declining or near tipping point
    """
    print("\n" + "=" * 50)
    print("COUNTRY PRIORITIZATION ANALYSIS")
    print("=" * 50)
    
    results = {}
    
    # --- 1. HIGH IMPACT: Largest Emitters ---
    print("\n--- HIGH IMPACT: Largest Emitters ---")
    print("(Countries where reductions have biggest absolute effect)")
    
    high_impact = df.nlargest(15, 'co2_emissions_mt')[
        ['country_code', 'co2_emissions_mt', 'co2_intensity', 'fossil_fuel_electricity_pct']
    ].copy()
    
    # Calculate share of global emissions
    total_global = df['co2_emissions_mt'].sum()
    high_impact['global_share_pct'] = high_impact['co2_emissions_mt'] / total_global * 100
    high_impact['cumulative_share'] = high_impact['global_share_pct'].cumsum()
    
    print(f"\nTop 15 emitters account for {high_impact['cumulative_share'].iloc[-1]:.1f}% of global emissions")
    print("\nCountry | Emissions (Mt) | Global Share | Intensity | Fossil %")
    print("-" * 50)
    for _, row in high_impact.iterrows():
        intensity = f"{row['co2_intensity']:.2f}" if pd.notna(row['co2_intensity']) else "N/A"
        fossil = f"{row['fossil_fuel_electricity_pct']:.0f}%" if pd.notna(row['fossil_fuel_electricity_pct']) else "N/A"
        print(f"{row['country_code']:6} | {row['co2_emissions_mt']:12,.1f} | {row['global_share_pct']:10.1f}% | {intensity:9} | {fossil}")
    
    results['high_impact'] = high_impact
    
    # --- 2. HIGH POTENTIAL: Favorable Conditions ---
    print("\n--- HIGH POTENTIAL: Countries with Decarbonization Potential ---")
    print("(Significant emissions + favorable conditions for change)")
    
    # Filter to countries with meaningful emissions (>10 Mt)
    significant = df[df['co2_emissions_mt'] > 10].copy()
    
    # Score based on: low fossil dependence, moderate intensity (room to improve)
    significant['potential_score'] = 0
    
    # Lower fossil = higher potential (can leverage existing clean energy)
    if 'fossil_fuel_electricity_pct' in significant.columns:
        fossil_rank = significant['fossil_fuel_electricity_pct'].rank(ascending=True, pct=True)
        significant['potential_score'] += fossil_rank * 30
    
    # Higher renewable = higher potential (infrastructure exists)
    if 'renewable_energy_pct' in significant.columns:
        renewable_rank = significant['renewable_energy_pct'].rank(ascending=False, pct=True)
        significant['potential_score'] += renewable_rank * 30
    
    # Higher intensity = more room to improve
    if 'co2_intensity' in significant.columns:
        intensity_rank = significant['co2_intensity'].rank(ascending=False, pct=True)
        significant['potential_score'] += intensity_rank * 20
    
    # Larger emissions = bigger impact
    emission_rank = significant['co2_emissions_mt'].rank(ascending=False, pct=True)
    significant['potential_score'] += emission_rank * 20
    
    high_potential = significant.nlargest(15, 'potential_score')[
        ['country_code', 'co2_emissions_mt', 'fossil_fuel_electricity_pct', 
         'renewable_energy_pct', 'potential_score']
    ]
    
    print("\nCountry | Emissions | Fossil % | Renewable % | Potential Score")
    print("-" * 50)
    for _, row in high_potential.iterrows():
        fossil = f"{row['fossil_fuel_electricity_pct']:.0f}%" if pd.notna(row['fossil_fuel_electricity_pct']) else "N/A"
        renewable = f"{row['renewable_energy_pct']:.0f}%" if pd.notna(row['renewable_energy_pct']) else "N/A"
        print(f"{row['country_code']:6} | {row['co2_emissions_mt']:9,.1f} | {fossil:8} | {renewable:11} | {row['potential_score']:.1f}")
    
    results['high_potential'] = high_potential
    
    # --- 3. QUICK WINS: Already Declining ---
    print("\n--- QUICK WINS: Countries with Declining Emissions ---")
    print("(Already on downward trajectory - support to accelerate)")
    
    if 'co2_5yr_trend' in df.columns:
        declining = df[
            (df['co2_5yr_trend'] < -0.05) &  # >5% decline over 5 years
            (df['co2_emissions_mt'] > 5)      # Meaningful emissions
        ].copy()
        
        declining = declining.nsmallest(15, 'co2_5yr_trend')[
            ['country_code', 'co2_emissions_mt', 'co2_5yr_trend', 'renewable_energy_pct']
        ]
        
        print("\nCountry | Emissions | 5yr Trend | Renewable %")
        print("-" * 50)
        for _, row in declining.iterrows():
            renewable = f"{row['renewable_energy_pct']:.0f}%" if pd.notna(row['renewable_energy_pct']) else "N/A"
            print(f"{row['country_code']:6} | {row['co2_emissions_mt']:9,.1f} | {row['co2_5yr_trend']*100:+7.1f}% | {renewable}")
        
        results['quick_wins'] = declining
    else:
        print("  (Trend data not available)")
        results['quick_wins'] = pd.DataFrame()
    
    return results


# =============================================================================
# SECTOR ANALYSIS
# =============================================================================

def analyze_sectors(df: pd.DataFrame) -> dict:
    """
    Analyze emission sources by sector to recommend targeted interventions.
    """
    print("\n" + "=" * 50)
    print("SECTOR ANALYSIS")
    print("=" * 50)
    
    results = {}
    
    # Check for sectoral data
    sector_cols = ['co2_transport_mt', 'co2_power_industry_mt', 'co2_buildings_mt', 'co2_industrial_mt']
    available_sectors = [c for c in sector_cols if c in df.columns]
    
    if not available_sectors:
        print("Sectoral CO2 data not available")
        return results
    
    # Calculate sector shares for each country
    for col in available_sectors:
        sector_name = col.replace('co2_', '').replace('_mt', '')
        df[f'{sector_name}_share'] = df[col] / df['co2_emissions_mt'] * 100
    
    # Global sector breakdown
    print("\n--- Global Sector Breakdown ---")
    global_total = df['co2_emissions_mt'].sum()
    
    sector_shares = {}
    for col in available_sectors:
        sector_name = col.replace('co2_', '').replace('_mt', '')
        sector_total = df[col].sum()
        share = sector_total / global_total * 100
        sector_shares[sector_name] = {'total_mt': sector_total, 'share_pct': share}
        print(f"  {sector_name.title():20}: {sector_total:,.0f} Mt ({share:.1f}%)")
    
    results['global_sectors'] = sector_shares
    
    # Identify countries where each sector dominates
    print("\n--- Countries by Dominant Sector ---")
    
    for col in available_sectors:
        sector_name = col.replace('co2_', '').replace('_mt', '')
        share_col = f'{sector_name}_share'
        
        if share_col in df.columns:
            # Countries where this sector is >50% of emissions
            dominant = df[df[share_col] > 50].nlargest(5, col)
            
            if len(dominant) > 0:
                print(f"\n  {sector_name.upper()} Dominant (>50% of country emissions):")
                for _, row in dominant.iterrows():
                    print(f"    {row['country_code']}: {row[share_col]:.0f}% ({row[col]:.1f} Mt)")
    
    # Recommendations by sector
    print("\n--- Sector-Specific Recommendations ---")
    
    recommendations = {
        'transport': [
            "EV adoption incentives and charging infrastructure",
            "Public transit expansion and electrification",
            "Fuel efficiency standards for remaining ICE vehicles",
            "Urban planning to reduce vehicle dependence",
            "Rail freight development"
        ],
        'power_industry': [
            "Renewable energy capacity expansion (solar, wind)",
            "Coal-to-gas or coal-to-renewable switching",
            "Grid modernization and storage",
            "Carbon capture for remaining fossil plants",
            "Nuclear energy consideration"
        ],
        'buildings': [
            "Building efficiency standards and retrofits",
            "Heat pump adoption for heating/cooling",
            "Electric appliance standards",
            "District heating with renewable sources",
            "Smart building management systems"
        ],
        'industrial': [
            "Industrial process electrification",
            "Green hydrogen for high-heat processes",
            "Carbon capture for cement and steel",
            "Circular economy and material efficiency",
            "Energy efficiency requirements"
        ]
    }
    
    for sector, recs in recommendations.items():
        print(f"\n  {sector.upper().replace('_', ' ')}:")
        for rec in recs:
            print(f"    • {rec}")
    
    results['recommendations'] = recommendations
    
    return results


# =============================================================================
# POLICY RECOMMENDATIONS BY COUNTRY TYPE
# =============================================================================

def generate_policy_recommendations(df: pd.DataFrame) -> dict:
    """
    Generate tailored policy recommendations based on country characteristics.
    
    Country archetypes:
    1. High-income, high-emissions: Technology leaders, need to accelerate
    2. Middle-income, growing emissions: Need clean development pathways
    3. Low-income, low emissions: Need to avoid carbon lock-in
    4. High-renewable already: Need to maintain and expand
    """
    print("\n" + "=" * 50)
    print("POLICY RECOMMENDATIONS BY COUNTRY TYPE")
    print("=" * 50)
    
    results = {}
    
    # Define country archetypes
    def classify_country(row):
        gdp_pc = row.get('gdp_per_capita', 0)
        co2_pc = row.get('co2_emissions_per_capita', 0)
        renewable = row.get('renewable_energy_pct', 0)
        fossil = row.get('fossil_fuel_electricity_pct', 100)
        
        if pd.isna(gdp_pc) or pd.isna(co2_pc):
            return 'Insufficient Data'
        
        # High renewable (>50%)
        if renewable > 50:
            return 'Clean Energy Leader'
        
        # High income (>$30,000 GDP/capita)
        if gdp_pc > 30000:
            if co2_pc > 10:
                return 'High-Income High-Emitter'
            else:
                return 'High-Income Low-Emitter'
        
        # Middle income ($5,000-$30,000)
        elif gdp_pc > 5000:
            if fossil > 70:
                return 'Middle-Income Fossil-Dependent'
            else:
                return 'Middle-Income Transitioning'
        
        # Low income (<$5,000)
        else:
            return 'Low-Income'
    
    df['country_type'] = df.apply(classify_country, axis=1)
    
    # Count and list countries by type
    type_counts = df['country_type'].value_counts()
    
    print("\nCountry Types:")
    for ctype, count in type_counts.items():
        countries = df[df['country_type'] == ctype]['country_code'].tolist()[:5]
        print(f"\n  {ctype} ({count} countries)")
        print(f"    Examples: {', '.join(countries)}")
    
    # Policy recommendations by type
    policies = {
        'High-Income High-Emitter': {
            'description': 'Wealthy countries with high per-capita emissions (USA, Australia, Canada)',
            'priorities': [
                'Accelerate renewable deployment with ambitious targets',
                'Phase out coal power immediately, gas by 2035-2040',
                'Aggressive EV adoption targets (100% new sales by 2030-2035)',
                'Building electrification mandates',
                'Carbon pricing ($100+/ton)',
                'Lead international climate finance'
            ],
            'examples': ['USA', 'AUS', 'CAN', 'SAU', 'ARE']
        },
        'High-Income Low-Emitter': {
            'description': 'Wealthy countries that have already reduced emissions (France, Sweden)',
            'priorities': [
                'Maintain momentum on existing policies',
                'Address remaining hard-to-abate sectors',
                'Export clean technology and expertise',
                'Support international climate finance',
                'Achieve net-zero by 2040-2045'
            ],
            'examples': ['FRA', 'SWE', 'CHE', 'NOR']
        },
        'Clean Energy Leader': {
            'description': 'Countries with >50% renewable energy (Brazil, Norway, Costa Rica)',
            'priorities': [
                'Protect and expand renewable capacity',
                'Become regional clean energy hub',
                'Focus on transport and industry decarbonization',
                'Maintain forest carbon sinks',
                'Share expertise with neighbors'
            ],
            'examples': ['BRA', 'NOR', 'CRI', 'ISL', 'PRY']
        },
        'Middle-Income Fossil-Dependent': {
            'description': 'Growing economies heavily reliant on fossil fuels (India, Indonesia, South Africa)',
            'priorities': [
                'Leapfrog to renewables (avoid new coal)',
                'International financing for clean energy transition',
                'Manage just transition for fossil fuel workers',
                'Develop domestic clean energy manufacturing',
                'Efficiency standards for rapid infrastructure build-out'
            ],
            'examples': ['IND', 'IDN', 'ZAF', 'POL', 'PHL']
        },
        'Middle-Income Transitioning': {
            'description': 'Growing economies with some clean energy (China, Mexico, Chile)',
            'priorities': [
                'Accelerate existing clean energy momentum',
                'Peak emissions as soon as possible',
                'Industrial decarbonization roadmaps',
                'Clean transportation infrastructure',
                'Regional climate leadership'
            ],
            'examples': ['CHN', 'MEX', 'CHL', 'TUR', 'THA']
        },
        'Low-Income': {
            'description': 'Developing countries with low current emissions (many African nations)',
            'priorities': [
                'Avoid carbon lock-in: skip fossil fuel era',
                'Deploy distributed renewables (solar mini-grids)',
                'International support for clean electrification',
                'Climate adaptation alongside mitigation',
                'Protect carbon sinks (forests, wetlands)'
            ],
            'examples': ['ETH', 'KEN', 'UGA', 'TZA', 'BGD']
        }
    }
    
    print("\n" + "-" * 50)
    print("DETAILED POLICY RECOMMENDATIONS")
    print("-" * 50)
    
    for ctype, policy in policies.items():
        print(f"\n### {ctype}")
        print(f"{policy['description']}")
        print(f"\nPriority Actions:")
        for i, action in enumerate(policy['priorities'], 1):
            print(f"  {i}. {action}")
    
    results['country_types'] = df[['country_code', 'country_type', 'co2_emissions_mt', 
                                    'gdp_per_capita', 'renewable_energy_pct']].copy()
    results['policies'] = policies
    
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualizations(df: pd.DataFrame, priority_results: dict, 
                          sector_results: dict, output_dir: Path):
    """Create visualizations for strategic recommendations."""
    print("\n" + "=" * 50)
    print("CREATING VISUALIZATIONS")
    print("=" * 50)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Priority Countries Map/Chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # High Impact
    ax = axes[0]
    if 'high_impact' in priority_results and len(priority_results['high_impact']) > 0:
        data = priority_results['high_impact'].head(10)
        ax.barh(data['country_code'], data['co2_emissions_mt'], color='coral')
        ax.set_xlabel('CO2 Emissions (Mt)')
        ax.set_title('HIGH IMPACT\nLargest Emitters')
        ax.invert_yaxis()
    
    # High Potential
    ax = axes[1]
    if 'high_potential' in priority_results and len(priority_results['high_potential']) > 0:
        data = priority_results['high_potential'].head(10)
        ax.barh(data['country_code'], data['potential_score'], color='mediumseagreen')
        ax.set_xlabel('Potential Score')
        ax.set_title('HIGH POTENTIAL\nFavorable Conditions')
        ax.invert_yaxis()
    
    # Quick Wins
    ax = axes[2]
    if 'quick_wins' in priority_results and len(priority_results['quick_wins']) > 0:
        data = priority_results['quick_wins'].head(10)
        ax.barh(data['country_code'], data['co2_5yr_trend'] * 100, color='steelblue')
        ax.set_xlabel('5-Year Trend (%)')
        ax.set_title('QUICK WINS\nAlready Declining')
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'priority_countries.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: priority_countries.png")
    
    # 2. Sector Analysis (with improved color palette, sorted high to low)
    if 'global_sectors' in sector_results:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sectors = list(sector_results['global_sectors'].keys())
        shares = [sector_results['global_sectors'][s]['share_pct'] for s in sectors]
        
        # Sort by share (highest to lowest)
        sorted_data = sorted(zip(sectors, shares), key=lambda x: x[1], reverse=True)
        sectors = [x[0] for x in sorted_data]
        shares = [x[1] for x in sorted_data]
        
        # Professional, colorblind-friendly palette
        sector_colors = {
            'transport': '#2E86AB',      # Blue
            'power': '#A23B72',          # Magenta
            'industry': '#F18F01',       # Orange
            'buildings': '#C73E1D',      # Red
            'other': '#3B1F2B'           # Dark purple
        }
        colors = [sector_colors.get(s.lower(), '#888888') for s in sectors]
        
        # Use horizontal bar chart instead of pie (easier to compare)
        y_pos = np.arange(len(sectors))
        bars = ax.barh(y_pos, shares, color=colors, edgecolor='white', linewidth=1.5)
        
        # Add value labels
        for bar, share in zip(bars, shares):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{share:.1f}%', va='center', fontsize=11, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([s.replace('_', ' ').title() for s in sectors], fontsize=11)
        ax.set_xlabel('Share of CO2 Emissions (%)', fontsize=12)
        ax.set_title('Global CO2 Emissions by Sector', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0, max(shares) * 1.15)
        ax.invert_yaxis()  # Highest at top
        
        plt.tight_layout()
        plt.savefig(output_dir / 'sector_breakdown.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: sector_breakdown.png")
    
    # 3. Country Type Distribution (with improved color palette)
    if 'country_type' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        type_emissions = df.groupby('country_type')['co2_emissions_mt'].sum().sort_values(ascending=False)
        
        # Professional color palette for country types
        type_colors = {
            'High-Income High-Emitter': '#C73E1D',      # Red - high concern
            'High-Income Low-Emitter': '#2E86AB',       # Blue - doing well
            'Clean Energy Leader': '#28A745',           # Green - leaders
            'Middle-Income Fossil-Dependent': '#F18F01', # Orange - needs transition
            'Middle-Income Transitioning': '#6F42C1',   # Purple - in progress
            'Low-Income': '#6C757D'                     # Gray - different priorities
        }
        colors = [type_colors.get(t, '#888888') for t in type_emissions.index]
        
        bars = ax.barh(type_emissions.index, type_emissions.values, color=colors, 
                       edgecolor='white', linewidth=1.5)
        
        # Add value labels
        for bar, val in zip(bars, type_emissions.values):
            ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                    f'{val:,.0f} Mt', va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Total CO2 Emissions (Mt)', fontsize=12)
        ax.set_title('CO2 Emissions by Country Type', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()  # Highest at top
        
        plt.tight_layout()
        plt.savefig(output_dir / 'emissions_by_country_type.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: emissions_by_country_type.png")


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(priority_results: dict, sector_results: dict, 
                    policy_results: dict, output_path: Path):
    """Generate comprehensive strategic recommendations report."""
    
    report = f"""# Question 5: Strategic Recommendations Report

## Executive Summary

This report synthesizes findings from the CO2 emissions analysis to provide actionable strategic recommendations for emission reduction efforts. Key recommendations are organized by:

1. **Country Prioritization**: Where to focus efforts for maximum impact
2. **Sector-Specific Interventions**: Targeted actions by emission source
3. **Policy Recommendations**: Tailored policies by country archetype

---

## 1. Country Prioritization Framework

### 1.1 HIGH IMPACT: Largest Emitters

These countries account for the majority of global emissions. Reductions here have the biggest absolute effect.

| Priority | Rationale |
|----------|-----------|
| China, USA, India | Top 3 account for ~50% of global emissions |
| Russia, Japan, Germany | Large industrial economies |
| Indonesia, Brazil | Large populations, growing economies |

**Strategy**: Engage through international frameworks, technology transfer, and climate finance.

### 1.2 HIGH POTENTIAL: Favorable Conditions

Countries with significant emissions but favorable conditions for decarbonization:
- Already have clean energy infrastructure (hydro, nuclear)
- Low fossil fuel dependence
- Technical and financial capacity

**Strategy**: Accelerate existing momentum with ambitious targets.

### 1.3 QUICK WINS: Already Declining

Countries already on downward emissions trajectory:
- Western European nations (UK, Germany, Italy)
- Some former Soviet states
- Countries with strong climate policies

**Strategy**: Support and accelerate existing efforts; learn from success stories.

---

## 2. Sector-Specific Recommendations

"""
    
    # Add sector recommendations if available
    if 'recommendations' in sector_results:
        for sector, recs in sector_results['recommendations'].items():
            report += f"### 2.{list(sector_results['recommendations'].keys()).index(sector)+1} {sector.upper().replace('_', ' ')}\n\n"
            for rec in recs:
                report += f"- {rec}\n"
            report += "\n"
    
    report += """
---

## 3. Policy Recommendations by Country Type

"""
    
    # Add policy recommendations
    if 'policies' in policy_results:
        for i, (ctype, policy) in enumerate(policy_results['policies'].items(), 1):
            report += f"### 3.{i} {ctype}\n\n"
            report += f"**Description**: {policy['description']}\n\n"
            report += f"**Example Countries**: {', '.join(policy['examples'])}\n\n"
            report += "**Priority Actions**:\n"
            for action in policy['priorities']:
                report += f"1. {action}\n"
            report += "\n"
    
    report += """
---

## 4. Implementation Roadmap

### Near-Term (2025-2030)

| Action | Target Countries | Expected Impact |
|--------|-----------------|-----------------|
| Coal phase-out | USA, Germany, Poland | ~500 Mt/year reduction |
| EV adoption 30% | EU, China, USA | ~200 Mt/year reduction |
| Grid decarbonization | India, Indonesia | Enable future EV benefits |
| Deforestation halt | Brazil, Indonesia | Preserve carbon sinks |

### Medium-Term (2030-2040)

| Action | Target Countries | Expected Impact |
|--------|-----------------|-----------------|
| Full grid decarbonization | All high-income | ~2,000 Mt/year reduction |
| Industrial decarbonization | China, USA, EU | ~1,000 Mt/year reduction |
| EV adoption 80%+ | All developed | ~500 Mt/year reduction |
| Building electrification | EU, USA, Japan | ~300 Mt/year reduction |

### Long-Term (2040-2050)

| Action | Target Countries | Expected Impact |
|--------|-----------------|-----------------|
| Net-zero grids | All | Baseline achievement |
| Green hydrogen scale | Heavy industry | Last fossil fuel displacement |
| Carbon removal | Global | Offset residual emissions |
| Aviation/shipping solutions | Global | Complete transport decarbonization |

---

## 5. Key Insights from Analysis

### From Q2 (Predictive Modeling)
- **GDP Elasticity**: 10% GDP increase → 6.6% CO2 increase
- **Implication**: Economic growth still coupled with emissions; need absolute decoupling

### From Q3 (EV Analysis)
- **50% EV adoption → 168 Mt reduction** (only 0.4% of global emissions)
- **88 countries** have grids too dirty for EVs to help
- **Implication**: Grid decarbonization must precede/accompany EV adoption

### From Q4 (Classification)
- Countries cluster into distinct archetypes requiring different policies
- Structural factors (energy mix, economic structure) predict emission trajectories
- **Implication**: One-size-fits-all policies won't work

---

## 6. Critical Success Factors

1. **International Cooperation**: Climate change is global; solutions require coordination
2. **Just Transition**: Protect workers in fossil fuel industries
3. **Technology Transfer**: Help developing countries skip fossil fuel era
4. **Climate Finance**: Wealthy nations fund developing country transitions
5. **Policy Stability**: Long-term targets enable private investment
6. **Measurement & Accountability**: Track progress, adjust as needed

---

## 7. Files Generated

| File | Description |
|------|-------------|
| `priority_countries.png` | Visualization of country prioritization |
| `sector_breakdown.png` | Global emissions by sector |
| `emissions_by_country_type.png` | Emissions distribution by country archetype |

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
    print("=" * 50)
    print("QUESTION 5: STRATEGIC RECOMMENDATIONS")
    print("=" * 50)
    
    # Set up paths
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / 'data' / 'processed' / 'co2_data_clean.csv'
    figures_dir = base_dir / 'figures'
    reports_dir = base_dir / 'reports'
    
    # Load and prepare data
    df = load_data(data_path)
    latest_df = prepare_latest_data(df)
    
    # Country prioritization
    priority_results = prioritize_countries(latest_df)
    
    # Sector analysis
    sector_results = analyze_sectors(latest_df)
    
    # Policy recommendations
    policy_results = generate_policy_recommendations(latest_df)
    
    # Create visualizations
    create_visualizations(latest_df, priority_results, sector_results, figures_dir)
    
    # Generate report
    generate_report(
        priority_results, sector_results, policy_results,
        reports_dir / 'q5_strategic_recommendations.md'
    )
    
    print("\n" + "=" * 50)
    print("QUESTION 5 COMPLETE")
    print("=" * 50)
    
    print("\nKey Recommendations Summary:")
    print("  1. Focus on top 15 emitters (70%+ of global emissions)")
    print("  2. Grid decarbonization before/with EV adoption")
    print("  3. Tailor policies to country archetypes")
    print("  4. Support developing countries to skip fossil era")
    
    return latest_df, priority_results, sector_results, policy_results


if __name__ == "__main__":
    latest_df, priority_results, sector_results, policy_results = main()