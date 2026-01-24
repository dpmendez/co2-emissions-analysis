# Question 3: Fermi Problem and Sensitivity Analysis Report

## Executive Summary

This report estimates the impact on global CO2 emissions if 50% of the world's population adopted electric vehicles (EVs).

**Key Findings:**
- **Global CO2 reduction: 168 Mt** (base case)
- This represents **0.4% of total global emissions**
- And **2.4% of transport emissions**
- Range (sensitivity analysis): -226 - 1,064 Mt

**Critical Insight**: The benefit of EVs varies dramatically by grid cleanliness. Countries with clean electricity grids see 2-3x more benefit than those relying on coal.

---

## 1. Methodology: Fermi Estimation

### 1.1 Interpreting "50% of Population Adopts EVs"

The question is intentionally ambiguous — this is a Fermi problem. We must make a reasonable interpretation:

| Interpretation | Issue |
|----------------|-------|
| 50% of people own an EV | Not everyone owns a car (ownership ranges from 20 to 800 per 1000 people) |
| 50% of vehicles become EVs | Doesn't account for usage patterns |
| **50% of vehicle-km become EV** | **Our interpretation** — directly tied to emissions |

**Our interpretation**: 50% of all passenger vehicle kilometers traveled globally switch from internal combustion engine (ICE) vehicles to electric vehicles.

This is reasonable because:
- Emissions come from **driving**, not from owning a vehicle
- It's directly applicable to our transport emissions data
- It sidesteps the need for vehicle ownership data per country

### 1.2 Problem Decomposition

We break down the complex question into estimable components:

```
CO2 Reduction = Transport_CO2 * Passenger_Share * Adoption_Rate * Net_Reduction_Factor
```

Where:
- **Transport CO2**: 6,937 Mt (from data)
- **Passenger Share**: 45% of transport (cars, not trucks/ships/planes)
- **Adoption Rate**: 50% of passenger vehicle-km become EV
- **Net Reduction Factor**: Varies by grid cleanliness

### 1.2 Key Insight: EVs Aren't Zero-Emission

EVs shift emissions from tailpipe to power plant:

| Grid Type | Grid Intensity | EV Emissions | vs ICE (120 g/km) |
|-----------|---------------|--------------|-------------------|
| Very Clean (hydro/nuclear) | 50 g CO2/kWh | 10 g/km | **92% cleaner** |
| Clean (20% fossil) | 210 g CO2/kWh | 42 g/km | **65% cleaner** |
| Mixed (50% fossil) | 450 g CO2/kWh | 90 g/km | **25% cleaner** |
| Dirty (80% fossil) | 690 g CO2/kWh | 138 g/km | **15% cleaner** |
| Very Dirty (95%+ coal) | 810 g CO2/kWh | 162 g/km | **35% DIRTIER** |

### 1.3 Assumptions Used

| Parameter | Low | Base | High |
|-----------|-----|------|------|
| Passenger vehicle share | 40% | 45% | 50% |
| EV adoption rate | 30% | 50% | 70% |
| ICE emissions | 100 g/km | 120 g/km | 150 g/km |
| EV consumption | 0.15 kWh/km | 0.20 kWh/km | 0.25 kWh/km |

---

## 2. Results: Base Scenario

### 2.1 Global Impact

| Metric | Value |
|--------|-------|
| Total transport CO2 | 6,937 Mt |
| Passenger vehicle CO2 (45%) | 3,122 Mt |
| CO2 reduced by 50% EV adoption | **168 Mt** |
| As % of transport emissions | 2.4% |
| As % of total emissions | 0.4% |

### 2.2 Top 10 Countries by Absolute Reduction

| Rank | Country | CO2 Reduction (Mt) | Grid Intensity | Reduction Factor |
|------|---------|-------------------|----------------|------------------|
| 1 | USA | 40.0 | 524 g/kWh | 0.13 |
| 2 | BRA | 37.9 | 120 g/kWh | 0.80 |
| 3 | CAN | 20.6 | 207 g/kWh | 0.65 |
| 4 | FRA | 18.4 | 109 g/kWh | 0.82 |
| 5 | ESP | 11.4 | 268 g/kWh | 0.55 |
| 6 | CHN | 10.2 | 565 g/kWh | 0.06 |
| 7 | DEU | 8.9 | 396 g/kWh | 0.34 |
| 8 | GBR | 8.5 | 346 g/kWh | 0.42 |
| 9 | RUS | 5.4 | 548 g/kWh | 0.09 |
| 10 | ITA | 4.4 | 484 g/kWh | 0.19 |


### 2.3 Top 10 Countries by Percentage Reduction

| Rank | Country | % of Total Emissions | CO2 Reduction (Mt) | Grid Intensity |
|------|---------|---------------------|-------------------|----------------|
| 1 | PRY | 16.58% | 1.3 | 50 g/kWh |
| 2 | COD | 15.69% | 1.0 | 50 g/kWh |
| 3 | SWZ | 15.19% | 0.2 | 72 g/kWh |
| 4 | CRI | 13.67% | 1.1 | 90 g/kWh |
| 5 | LSO | 11.53% | 0.1 | 52 g/kWh |
| 6 | LUX | 11.37% | 0.8 | 77 g/kWh |
| 7 | CAF | 10.67% | 0.0 | 78 g/kWh |
| 8 | NAM | 10.58% | 0.4 | 62 g/kWh |
| 9 | UGA | 8.86% | 0.7 | 71 g/kWh |
| 10 | URY | 8.64% | 0.8 | 121 g/kWh |


---

## 3. Sensitivity Analysis

### 3.1 Parameter Impact

| Scenario | CO2 Reduction (Mt) | vs Base |
|----------|-------------------|---------|
| **Base case** | 168 | — |
| Best case | 1,064 | +533% |
| Worst case | -226 | -234% |

### 3.2 Sensitivity to Individual Parameters

| Parameter | Low Value | High Value | Range (Mt) |
|-----------|-----------|------------|------------|
| Passenger Share | 149 | 187 | 37 |
| EV Adoption | 89 | 247 | 158 |
| ICE Emissions | -78 | 414 | 492 |
| EV Efficiency | -140 | 476 | 615 |

**Most sensitive parameter**: EV adoption rate (as expected — it's the "dosage" of the intervention)

---

## 4. Analysis by Grid Cleanliness

### Very Clean (0-20%)
- Countries: 32
- Average EV benefit factor: 0.82
- Total CO2 reduction: 103.4 Mt

### Clean (20-40%)
- Countries: 34
- Average EV benefit factor: 0.53
- Total CO2 reduction: 55.6 Mt

### Mixed (40-60%)
- Countries: 25
- Average EV benefit factor: 0.23
- Total CO2 reduction: 70.6 Mt

### Dirty (60-80%)
- Countries: 29
- Average EV benefit factor: -0.04
- Total CO2 reduction: 5.3 Mt

### Very Dirty (80-100%)
- Countries: 71
- Average EV benefit factor: -0.32
- Total CO2 reduction: -68.3 Mt



**Key Insight**: Countries with cleaner grids benefit disproportionately from EV adoption. Policy implication: Grid decarbonization and EV adoption should be pursued together.

---

## 5. Limitations and Assumptions

1. **Passenger vehicle share (45%)**: This excludes freight, aviation, shipping. True share varies by country (higher in car-dependent nations).

2. **Grid intensity estimation**: We estimate from fossil fuel share. Actual intensity depends on specific fuel mix (coal vs gas) and plant efficiency.

3. **Static analysis**: We don't model grid evolution. If EVs increase electricity demand, grids might become dirtier (or cleaner, if renewables scale).

4. **"50% adoption"**: We interpret this as 50% of vehicle-kilometers, not 50% of people owning EVs. Actual impact depends on driving patterns.

5. **Uniform vehicle assumptions**: Real fleets vary (small cars vs SUVs, old vs new vehicles).

6. **No lifecycle emissions**: We only count operational emissions, not vehicle manufacturing or battery production.

---

## 6. Policy Implications

1. **Prioritize clean grids**: EVs in coal-dependent countries provide minimal benefit. These countries should focus on grid decarbonization first (or simultaneously).

2. **Target high-transport countries**: Large economies with significant transport sectors (USA, China, EU) offer the biggest absolute reduction potential.

3. **Consider transport structure**: Countries with high car dependency see more impact from EV adoption than those with better public transit.

4. **Complement with efficiency**: Smaller, more efficient EVs multiply the benefit.

---

## 7. Files Generated

| File | Description |
|------|-------------|
| `figures/ev_impact_by_country.png` | Top countries by CO2 reduction |
| `figures/ev_benefit_vs_grid.png` | EV benefit vs grid carbon intensity |
| `figures/sensitivity_tornado.png` | Sensitivity analysis tornado chart |

---

*Report generated: 2026-01-23 21:52:11*
*Data year: 2024*
