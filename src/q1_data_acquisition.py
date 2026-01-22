"""
Question 1: Comprehensive Data Acquisition and Preprocessing
============================================================

This script downloads CO2 emissions data and socio-economic/environmental indicators
from the World Bank's Climate Change database, preprocesses the data, and generates
summary statistics.

Author: Diana Patricia Mendez Mendez
Date: January 21 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# =============================================================================
# UPDATED INDICATOR DEFINITIONS (January 2025)
# =============================================================================

"""
IMPORTANT: The World Bank has updated their indicator codes.

OLD (DEPRECATED):
- EN.ATM.CO2E.KT → ARCHIVED
- EN.ATM.CO2E.PC → ARCHIVED

NEW (CURRENT):
- EN.GHG.CO2.MT.CE.AR5 → CO2 emissions total (Mt CO2e)
- EN.GHG.CO2.PC.CE.AR5 → CO2 emissions per capita (t CO2e/capita)

The new indicators use AR5 (IPCC 5th Assessment Report) methodology.
"""

# Updated indicators with NEW codes
INDICATORS = {
    # TARGET VARIABLES
    'EN.GHG.CO2.MT.CE.AR5': 'co2_emissions_mt',           # CO2 total (Megatonnes CO2e)
    'EN.GHG.CO2.PC.CE.AR5': 'co2_emissions_per_capita',   # CO2 per capita (tonnes/person)
    
    # SECTORAL CO2 BREAKDOWN (useful for Q3 - EV analysis)
    'EN.GHG.CO2.TR.MT.CE.AR5': 'co2_transport_mt',        # Transport sector CO2
    'EN.GHG.CO2.PI.MT.CE.AR5': 'co2_power_industry_mt',   # Power industry CO2
    'EN.GHG.CO2.BU.MT.CE.AR5': 'co2_buildings_mt',        # Buildings CO2
    'EN.GHG.CO2.IC.MT.CE.AR5': 'co2_industrial_mt',       # Industrial combustion CO2
    
    # ECONOMIC INDICATORS
    'NY.GDP.MKTP.CD': 'gdp_current_usd',            # GDP (current US$)
    'NY.GDP.PCAP.CD': 'gdp_per_capita',             # GDP per capita (current US$)
    'NY.GDP.MKTP.KD.ZG': 'gdp_growth',              # GDP growth (annual %)

    # ENERGY INDICATORS
    'EG.USE.PCAP.KG.OE': 'energy_use_per_capita',      # Energy use (kg oil equiv. per capita)
    'EG.USE.ELEC.KH.PC': 'electric_power_consumption', # Electric power consumption (kWh per capita)
    'EG.FEC.RNEW.ZS': 'renewable_energy_pct',          # Renewable energy (% of total)
    'EG.ELC.FOSL.ZS': 'fossil_fuel_electricity_pct',   # Electricity from fossil fuels (%)

    # DEMOGRAPHIC INDICATORS
    'SP.POP.TOTL': 'population',                    # Population, total
    'SP.URB.TOTL.IN.ZS': 'urban_pop_pct',           # Urban population (% of total)
    'EN.POP.DNST': 'population_density',            # Population density (people/sq km)

    # STRUCTURAL/SECTORAL INDICATORS
    'NV.IND.TOTL.ZS': 'industry_value_added_pct',    # Industry value added (% of GDP)
    'NV.AGR.TOTL.ZS': 'agriculture_value_added_pct', # Agriculture value added (% of GDP)
    'NV.SRV.TOTL.ZS': 'services_value_added_pct',    # Services value added (% of GDP)

    # DEVELOPMENT INDICATORS
    'EG.ELC.ACCS.ZS': 'access_to_electricity_pct',  # Access to electricity (% of population)
    'AG.LND.FRST.ZS': 'forest_area_pct',            # Forest area (% of land area)
}

# Time range for data collection
START_YEAR = 1990 # Less recent years may have incomplete data
END_YEAR = 2024  # More recent years may also have incomplete data


# =============================================================================
# DATA ACQUISITION
# =============================================================================

def download_world_bank_data(indicators: dict, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Download data from World Bank API using updated indicator codes.
    
    Handles the case where some indicators may not exist or have limited data.
    """
    try:
        import wbgapi as wb
    except ImportError:
        print("ERROR: wbgapi not installed. Install with: pip install wbgapi")
        return None
    
    print(f"Downloading data for {len(indicators)} indicators...")
    print(f"Time range: {start_year} - {end_year}")
    
    # Download each indicator separately to handle failures gracefully
    all_data = []
    successful_indicators = []
    failed_indicators = []
    
    for code, name in indicators.items():
        print(f"  Fetching {code} ({name})...", end=" ")
        try:
            df_temp = wb.data.DataFrame(
                series=code,
                time=range(start_year, end_year + 1),
                labels=True,
                columns='series'
            )
            
            if df_temp is not None and len(df_temp) > 0:
                df_temp = df_temp.reset_index()
                df_temp = df_temp.rename(columns={
                    'economy': 'country_code',
                    'Country': 'country_name',
                    'time': 'year',
                    code: name
                })
                
                # Clean year column
                if 'year' in df_temp.columns:
                    if df_temp['year'].dtype == object:
                        df_temp['year'] = df_temp['year'].str.replace('YR', '').astype(int)
                
                # Keep only relevant columns
                cols_to_keep = ['country_code', 'year', name]
                if 'country_name' in df_temp.columns:
                    cols_to_keep.insert(1, 'country_name')
                df_temp = df_temp[[c for c in cols_to_keep if c in df_temp.columns]]
                
                all_data.append(df_temp)
                successful_indicators.append(name)
                print(f"({len(df_temp)} rows)")
            else:
                failed_indicators.append((code, name, "No data returned"))
                print("(no data)")
                
        except Exception as e:
            failed_indicators.append((code, name, str(e)))
            print(f"({str(e)[:50]})")
    
    if not all_data:
        print("\nERROR: No data could be downloaded!")
        return None
    
    # Merge all dataframes
    print(f"\nMerging {len(all_data)} indicator datasets...")
    
    df = all_data[0]
    for df_temp in all_data[1:]:
        # Determine merge columns
        merge_cols = ['country_code', 'year']
        if 'country_name' in df.columns and 'country_name' in df_temp.columns:
            # Drop country_name from right to avoid duplicates
            df_temp = df_temp.drop(columns=['country_name'], errors='ignore')
        
        df = df.merge(df_temp, on=merge_cols, how='outer')
    
    print(f"\nSuccessfully downloaded {len(successful_indicators)} indicators")
    if failed_indicators:
        print(f"Failed to download {len(failed_indicators)} indicators:")
        for code, name, error in failed_indicators:
            print(f"    - {name}: {error[:60]}")
    
    print(f"\nFinal dataset shape: {df.shape}")
    return df


def search_available_indicators(search_terms: list = None):
    """
    Search for available indicators in the World Bank database.
    Useful for finding updated indicator codes.
    """
    try:
        import wbgapi as wb
    except ImportError:
        print("ERROR: wbgapi not installed")
        return
    
    if search_terms is None:
        search_terms = ['CO2', 'GDP', 'energy', 'population', 'renewable']
    
    print("=" * 50)
    print("AVAILABLE WORLD BANK INDICATORS")
    print("=" * 50)
    
    for term in search_terms:
        print(f"\n--- Searching for '{term}' ---")
        try:
            results = list(wb.series.list(q=term))
            print(f"Found {len(results)} indicators:")
            for r in results[:10]:
                print(f"  {r['id']}: {r['value'][:60]}...")
            if len(results) > 10:
                print(f"  ... and {len(results) - 10} more")
        except Exception as e:
            print(f"  Error: {e}")


# =============================================================================
# DATA PREPROCESSING
# =============================================================================

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the raw data.
    """
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)
    
    df = df.copy()
    initial_shape = df.shape
    print(f"Initial shape: {initial_shape}")
    
    # Step 1: Remove aggregate regions
    aggregate_codes = ['WLD', 'EUU', 'ECS', 'SSF', 'MEA', 'SAS', 'EAS', 'LCN', 
                       'NAC', 'OED', 'LDC', 'HIC', 'MIC', 'LIC', 'PST', 'PRE',
                       'AFE', 'AFW', 'ARB', 'CEB', 'CSS', 'EAP', 'EAR', 'EMU',
                       'ECA', 'FCS', 'HPC', 'IBD', 'IBT', 'IDA', 'IDB', 'IDX',
                       'LAC', 'LMC', 'LMY', 'LTE', 'MNA', 'OGC', 'OSS', 'PSS',
                       'SSA', 'SST', 'TEA', 'TEC', 'TLA', 'TMN', 'TSA', 'TSS',
                       'UMC', 'INX']
    
    if 'country_code' in df.columns:
        before = len(df)
        df = df[~df['country_code'].isin(aggregate_codes)]
        removed = before - len(df)
        if removed > 0:
            print(f"Removed {removed} rows of aggregate/regional data")
    
    # Step 2: Drop countries without CO2 data (target variable is essential)
    co2_target_col = None
    for col in ['co2_emissions_mt', 'co2_emissions_kt']:
        if col in df.columns:
            co2_target_col = col
            break
    
    if co2_target_col and 'country_code' in df.columns:
        print(f"\n--- Checking for missing target variable ({co2_target_col}) ---")
        
        # Find countries that have NO CO2 data at all
        country_co2_coverage = df.groupby('country_code')[co2_target_col].apply(
            lambda x: x.notna().sum() / len(x)
        )
        countries_no_co2 = country_co2_coverage[country_co2_coverage == 0].index.tolist()
        
        if countries_no_co2:
            print(f"Dropping {len(countries_no_co2)} countries with NO CO2 emissions data:")
            print(f"  {countries_no_co2[:10]}{'...' if len(countries_no_co2) > 10 else ''}")
            before = len(df)
            df = df[~df['country_code'].isin(countries_no_co2)]
            print(f"  Removed {before - len(df)} rows")
        
        # Also drop individual rows where CO2 is missing (after country-level filter)
        before = len(df)
        df = df[df[co2_target_col].notna()]
        if before - len(df) > 0:
            print(f"Dropped {before - len(df)} individual rows with missing CO2 values")
    
    # Step 3: Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'year' in numeric_cols:
        numeric_cols.remove('year')
    
    print(f"\nNumeric columns to process: {len(numeric_cols)}")
    
    # Step 4: Missing data analysis
    print("\n--- Missing Data Analysis ---")
    missing_pct = df[numeric_cols].isnull().sum() / len(df) * 100
    print("Missing data percentage by column:")
    for col, pct in missing_pct.sort_values(ascending=False).items():
        if pct > 0:
            print(f"  {col}: {pct:.1f}%")
    
    # Step 5: Drop countries with too much missing predictor data (>50%)
    if 'country_code' in df.columns:
        country_missing = df.groupby('country_code')[numeric_cols].apply(
            lambda x: x.isnull().mean().mean()
        )
        countries_to_drop = country_missing[country_missing > 0.5].index.tolist()
        
        if countries_to_drop:
            print(f"\nDropping {len(countries_to_drop)} countries with >50% missing data")
            df = df[~df['country_code'].isin(countries_to_drop)]
    
    # Step 6: Interpolate missing values within each country
    print("\n--- Interpolating Missing Values ---")
    
    def interpolate_country(group):
        group = group.sort_values('year')
        for col in numeric_cols:
            if col in group.columns:
                group[col] = group[col].interpolate(method='linear', limit=3)
                group[col] = group[col].ffill(limit=2)
                group[col] = group[col].bfill(limit=2)
        return group
    
    if 'country_code' in df.columns:
        df = df.groupby('country_code', group_keys=False).apply(interpolate_country)
    
    remaining_missing = df[numeric_cols].isnull().sum().sum()
    print(f"Remaining missing values: {remaining_missing}")
    
    # Step 7: Create derived features
    print("\n--- Creating Derived Features ---")
    
    # Determine CO2 column name (might be different based on what downloaded)
    co2_col = None
    for col in ['co2_emissions_mt', 'co2_emissions_kt', 'co2_total']:
        if col in df.columns:
            co2_col = col
            break
    
    if 'gdp_current_usd' in df.columns:
        df['log_gdp'] = np.log1p(df['gdp_current_usd'])
        print("  Added: log_gdp")
    
    if 'population' in df.columns:
        df['log_population'] = np.log1p(df['population'])
        print("  Added: log_population")
    
    if co2_col and co2_col in df.columns:
        df['log_co2'] = np.log1p(df[co2_col])
        print(f"  Added: log_co2 (from {co2_col})")
        
        if 'gdp_current_usd' in df.columns:
            # CO2 intensity: emissions per unit GDP
            df['co2_intensity'] = df[co2_col] / (df['gdp_current_usd'] / 1e9)
            print("  Added: co2_intensity (CO2 per billion USD GDP)")
    
    print(f"\nFinal shape: {df.shape}")
    return df

# =============================================================================
# ANALYSIS AND VISUALIZATION
# =============================================================================

def generate_summary_statistics(df: pd.DataFrame) -> dict:
    """Generate comprehensive summary statistics."""
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    
    results = {}
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'year' in numeric_cols:
        numeric_cols.remove('year')
    
    # Basic statistics
    print("\n--- Descriptive Statistics ---")
    desc = df[numeric_cols].describe()
    results['descriptive'] = desc
    print(desc.to_string())
    
    # Find CO2 column
    co2_col = None
    for col in ['co2_emissions_mt', 'co2_emissions_kt', 'co2_emissions_per_capita']:
        if col in df.columns:
            co2_col = col
            break
    
    if co2_col:
        print(f"\n--- Correlations with {co2_col} ---")
        correlations = df[numeric_cols].corr()[co2_col].sort_values(ascending=False)
        results['co2_correlations'] = correlations
        print("Top correlations:")
        for col, corr in correlations.head(10).items():
            if col != co2_col:
                print(f"  {col}: {corr:.3f}")
    
    # Full correlation matrix
    results['correlation_matrix'] = df[numeric_cols].corr()
    
    return results


def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create exploratory visualizations."""
    print("\n" + "=" * 50)
    print("CREATING VISUALIZATIONS")
    print("=" * 50)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'year' in numeric_cols:
        numeric_cols.remove('year')
    
    # 1. Correlation heatmap
    try:
        plt.figure(figsize=(14, 12))
        corr = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
                   center=0, square=True, linewidths=0.5, annot_kws={'size': 8})
        plt.title('Correlation Matrix of Indicators', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: correlation_matrix.png")
    except Exception as e:
        print(f"Error creating correlation heatmap: {e}")
    
    # 2. CO2 trends over time
    co2_col = None
    for col in ['co2_emissions_mt', 'co2_emissions_kt']:
        if col in df.columns:
            co2_col = col
            break
    
    if co2_col and 'year' in df.columns:
        try:
            plt.figure(figsize=(12, 5))
            
            # Global trend
            plt.subplot(1, 2, 1)
            yearly_total = df.groupby('year')[co2_col].sum()
            plt.plot(yearly_total.index, yearly_total.values, 'b-', linewidth=2)
            plt.fill_between(yearly_total.index, yearly_total.values, alpha=0.3)
            plt.xlabel('Year')
            plt.ylabel(f'CO2 Emissions ({co2_col.split("_")[-1].upper()})')
            plt.title('Global CO2 Emissions Over Time')
            plt.grid(True, alpha=0.3)
            
            # Top 5 countries
            plt.subplot(1, 2, 2)
            if 'country_code' in df.columns:
                latest_year = df['year'].max()
                top_5 = df[df['year'] == latest_year].nlargest(5, co2_col)['country_code'].values
                
                for country in top_5:
                    country_data = df[df['country_code'] == country].sort_values('year')
                    plt.plot(country_data['year'], country_data[co2_col], label=country, linewidth=2)
                
                plt.xlabel('Year')
                plt.ylabel(f'CO2 Emissions')
                plt.title('Top 5 Emitters Over Time')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'co2_trends.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: co2_trends.png")
        except Exception as e:
            print(f"Error creating CO2 trends: {e}")
    
    # 3. GDP vs CO2 scatter
    if co2_col and 'gdp_current_usd' in df.columns:
        try:
            plt.figure(figsize=(10, 8))
            latest_year = df['year'].max() if 'year' in df.columns else None
            plot_data = df[df['year'] == latest_year] if latest_year else df
            
            plt.scatter(
                plot_data['gdp_current_usd'] / 1e12,
                plot_data[co2_col],
                alpha=0.6, s=80
            )
            plt.xlabel('GDP (Trillion USD)')
            plt.ylabel(f'CO2 Emissions ({co2_col})')
            plt.title(f'GDP vs CO2 Emissions ({latest_year})')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'gdp_vs_co2.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: gdp_vs_co2.png")
        except Exception as e:
            print(f"Error creating GDP vs CO2 plot: {e}")


def generate_markdown_report(df: pd.DataFrame, stats: dict, output_path: Path):
    """Generate Markdown report."""
    
    # Find CO2 column
    co2_col = None
    for col in ['co2_emissions_mt', 'co2_emissions_kt', 'co2_emissions_per_capita']:
        if col in df.columns:
            co2_col = col
            break
    
    n_countries = df['country_code'].nunique() if 'country_code' in df.columns else 'N/A'
    year_min = df['year'].min() if 'year' in df.columns else 'N/A'
    year_max = df['year'].max() if 'year' in df.columns else 'N/A'
    
    report = f"""# Question 1: Data Acquisition and Preprocessing Report

## Executive Summary

This report documents the acquisition and preprocessing of CO2 emissions data and 
socio-economic indicators from the World Bank Climate Change database.

**Dataset Overview:**
- **Time Period**: {year_min} - {year_max}
- **Countries**: {n_countries}
- **Total Observations**: {len(df):,}
- **Variables**: {len(df.columns)}

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
| Initial rows | {len(df):,} |
| Countries retained | {n_countries} |
| Year coverage | {year_min} - {year_max} |

---

## 3. Key Statistics

### 3.1 Correlation with CO2 Emissions

"""
    
    if 'co2_correlations' in stats and co2_col:
        report += f"Correlations with `{co2_col}`:\n\n"
        report += "| Indicator | Correlation |\n|-----------|-------------|\n"
        for col, corr in stats['co2_correlations'].head(10).items():
            if col != co2_col:
                report += f"| {col} | {corr:.3f} |\n"
    
    report += f"""

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

*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"\n✓ Saved report: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("QUESTION 1: DATA ACQUISITION AND PREPROCESSING")
    print("Using UPDATED World Bank Indicator Codes (2025)")
    print("=" * 70)
    
    # Set up paths
    base_dir = Path(__file__).parent.parent
    data_raw_dir = base_dir / 'data' / 'raw'
    data_processed_dir = base_dir / 'data' / 'processed'
    figures_dir = base_dir / 'figures'
    reports_dir = base_dir / 'reports'
    
    for d in [data_raw_dir, data_processed_dir, figures_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Optional: Show available indicators
    # search_available_indicators(['CO2', 'GDP', 'energy', 'renewable'])
    
    # Download data
    print("\n" + "=" * 60)
    print("STEP 1: DATA ACQUISITION")
    print("=" * 60)
    
    df_raw = download_world_bank_data(INDICATORS, START_YEAR, END_YEAR)
    
    if df_raw is None or len(df_raw) == 0:
        print("\nERROR: Failed to download data. Exiting.")
        return None, None
    
    # Save raw data
    raw_path = data_raw_dir / 'co2_data_raw.csv'
    df_raw.to_csv(raw_path, index=False)
    print(f"\n✓ Raw data saved: {raw_path}")
    
    # Preprocess
    df_clean = preprocess_data(df_raw)
    
    # Save processed data
    clean_path = data_processed_dir / 'co2_data_clean.csv'
    df_clean.to_csv(clean_path, index=False)
    print(f"✓ Cleaned data saved: {clean_path}")
    
    # Statistics
    stats = generate_summary_statistics(df_clean)
    
    # Visualizations
    create_visualizations(df_clean, figures_dir)
    
    # Save correlation matrix
    if 'correlation_matrix' in stats:
        stats['correlation_matrix'].to_csv(data_processed_dir / 'correlation_matrix.csv')
    
    # Generate report
    report_path = reports_dir / 'q1_data_summary.md'
    generate_markdown_report(df_clean, stats, report_path)
    
    print("\n" + "=" * 50)
    print("QUESTION 1 COMPLETE")
    print("=" * 50)
    
    return df_clean, stats


if __name__ == "__main__":
    df, stats = main()