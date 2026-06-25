# Favorita Demand Forecasting

[![Python](https://img.shields.io/badge/Python-3.10-blue)]()

End-to-end retail demand forecasting from raw sales data to a published dashboard that shows where better forecasts save the most money.

DASHBOARD: [Tableau Dashboard](https://public.tableau.com/app/profile/jordan.andersen3718/viz/favorita_demand_forecasting_dashboard/demand_forecast_dashboard?publish=yes). 

Data source: Corporacion Favorita. (2018). Corporación Favorita Grocery Sales Forecasting [Data set]. Kaggle. https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data.

## Results

**Research Questions:** 

1. How much does LightGBM beat a naive baseline based on the weighted average percent error (WAPE) and root mean squared logarithmic error (RMSLE)?
2. How well does that performance deal with disruptions like the June 2016 Ecuadorian Earthquake?

### Model Results

|Model|RMSLE|WAPE|
|---|---|---|
|seasonal_naive|0.55|0.17|
|ets|0.45|0.15|
|lightgbm|0.40|0.13|

- LightGBM's forecasts improved on the seasonal naive baseline by 27% RMSLE and decreased the WAPE by 4%.
- The LightGBM model decreases the cost of error by $1,705,506 against the seasonal naive model. 
- Beverages, grocery and produce are the highest cost families for over-forecasting. Forecast improvemet efforts should focus on these top categories for the biggest cost saving wins.
- The LightGBM model leans toward underforecasting for both perishable and non-perishable product families, so inventory planners should bias safety stock upward. 

![model_metrics](https://github.com/jandersen12/favorita-demand-forecasting/blob/main/outputs/figures/all_models_metrics_by_family.png)

### Disruption Analysis

- The WAPE is just below 20% before the earthquake, then increases to 30% in the days just around the quake before dropping down to just above 20% after the earthquake. 
- Coastal regions, who were hit hardest by the earthquake, show a higher WAPE at 37% compared to the inland region's 29%. 
- Product families like home appliances, electronics and baby care show the highest excess demand ratios. This is plausible given that during an earthquake things often break or are damaged within houses, so many people may need to purchase new home appliances or items that may have been damaged during the quake. However, we also know from previous EDA that these are generally low volume units, so I also looked at the difference in true versus predicted values by total units, which shows that gorcery, beverages, produce and cleaning were among the families with the highest excess demand in total units, which makes sense after an earthquake that emergency essentials like this would have excess demand compared to that which is forecasted.

![forecast_v_actual_earthquake](https://github.com/jandersen12/favorita-demand-forecasting/blob/main/outputs/figures/forecast_v_actual_earthquake.png)

## Setup

```
favorita-demand-forecasting/
├── data/
│   ├── raw/          
│   ├── interim/      
│   └── processed/   
├── notebooks/
│   ├── 01-eda.ipynb
│   ├── 02-baselines.ipynb
│   ├── 03-lightgbm.ipynb
│   ├── 04-disruption-analysis.ipynb
│   └── 05-business-impact.ipynb
├── outputs/ 
│   ├── figures/
│   └── results/
├── src/              
│   ├── __init__.py
│   ├── evaluate.py
│   ├── features.py
│   ├── make_dataset.py
│   ├── models.py
│   ├── rolling_origin_cv.py 
│   └── training.py
├── .gitignore
├── requirements.txt
└── README.md
```
