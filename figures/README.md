## Plot Explanations

### ev_benefit_vs_grid.png
Shows the relationship between grid carbon intensity (x-axis: % fossil fuel in electricity) 
and EV benefit (y-axis: how much cleaner EVs are vs gasoline cars).

- **Red dashed line** = Break-even (EV emissions = ICE emissions)
- **Above the line** (green) = EVs reduce emissions
- **Below the line** (red) = EVs increase emissions
- **Bubble size** = Transport sector emissions (larger = more transport CO2)
- **Bubble color** = CO2 reduction from EV adoption (green = large reduction, red = increase)

Key insight: Countries with >60-70% fossil fuel electricity are BELOW the break-even line, 
meaning EVs would actually increase their emissions.

### ev_impact_by_country.png
Two bar charts showing EV adoption impact:

**Left panel (Absolute reduction in Mt)**:
- Bar length = Total CO2 reduction in megatons
- Bar color = Grid cleanliness (green = clean grid, red = dirty grid)
- USA leads in absolute reduction because of its huge transport sector

**Right panel (Percentage reduction)**:
- Bar length = % reduction of country's total emissions
- Countries with very clean grids (Paraguay, Costa Rica) see highest % impact
- These are often smaller countries where transport is a larger share of emissions

### gdp_vs_co2.png
Scatter plot of GDP vs CO2 emissions:
- Each dot = one country
- Labels show major economies (CHN, USA, IND, etc.)
- Log-log scale version helps visualize the full range of countries
- The positive relationship shows that richer countries emit more

### model_predictions.png
Actual vs Predicted CO2 emissions for the test set:
- Each dot = one country-year observation
- Red dashed line = perfect prediction (if dot is on line, prediction = actual)
- Labels show outliers (countries the model struggled with)
- R² value shows overall model accuracy

**Ridge vs Random Forest**:
- Ridge: Better extrapolation to extreme values (China)
- Random Forest: Tends to "cap" predictions within training range

### residuals.png
Model residuals (Actual - Predicted):
- X-axis = Predicted value
- Y-axis = Error (positive = underpredicted, negative = overpredicted)
- Good model: Residuals randomly scattered around 0
- Bad patterns to watch for:
  - Funnel shape = heteroscedasticity (variance changes with size)
  - Curves = non-linear relationship not captured

### sensitivity_tornado.png
Shows how sensitive the EV impact estimate is to each assumption:
- Each bar = range of outcomes when varying ONE parameter (low to high)
- Longer bars = more sensitive (that assumption matters most)
- Center line = base case result (168 Mt reduction)

Reading example: "ICE emissions" bar goes from -78 Mt to +414 Mt, meaning:
- If ICE cars emit only 100 g/km (efficient), EV benefit is negative
- If ICE cars emit 150 g/km (inefficient), EV benefit jumps to 414 Mt

### sector_breakdown.png
CO2 emissions by sector (if sectoral data available):
- Bar length = % of total emissions
- Colors distinguish sectors
- Helps identify which sectors to prioritize for decarbonization