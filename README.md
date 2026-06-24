# Favorita Demand Forecasting

Find the Tableau dashboard for this project (here)[https://public.tableau.com/app/profile/jordan.andersen3718/viz/favorita_demand_forecasting_dashboard/demand_forecast_dashboard?publish=yes]. This dashboard utilizes the model prediction outputs exported from notebook 05-business-impact.ipynb.

## Objective

How much does ML beat a naive baseline (WAPE % and $), and how well does that performance deal with disruptions?

Measurable outputs: 
    (1) WAPE: naive vs. ETS vs. LightGBM, overall + by family
    (2) $ cost of error per model under the Phase 5 cost assumptions
    (3) % error degradation during vs. before the disruption. Success threshold: beat seasonal naive by >=15% relative WAPE, or explain why not.

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

## EDA Findings

- Total daily sales trends upward with increasing variance. 
- There are noticeable dips in total daily sales at the end of each year, which could reflect store closures during the holidays.
- The distribution of units sold is heavily right-skewed, suggesting that it may be necessary to model in log format.
- The most frequent number of units sold is 0, which tells us that many store/family combinations do not sell any units on most days.
- Weekends (Saturdays and Sundays) show the highest average units sold by day of the week. This pattern holds across product families, except for the liqour, wine and beer family, which shows highest sales on Saturday, but very low sales on Sunday.
- A heatmap confirms that Saturdays and Sundays are the highest sale days across families, with the exception of Liqour. An ACF plot confirms that a 7-week seasonality period is present in the data from the spikes that are apparent every 7 days.
- December shows the highest average units sold by month.
- The STL decomposition shows the trend moving upward as the years progress, the variance in the seasonality plot becoming greater over time, and the variance in the residuals staying relatively steady around the mean over time, but with a few noticeable outliers which increase in their variance from the mean over time.
- Items on promotion sell 7.2x more on average.
- We see a directional index of +5 units sold on paydays versus non-paydays. Units are not directly comparable across product families, but this indicates a directional increase in overall volume of units sold on paydays.
- The Grocery and Beverages product families have the highest total unit sales.
- The Earthquake in 2016 caused a visibly higher spike in units sold.

## Baseline Findings

### Assumptions

- Modeling inputs are clipped at zero to try and predict "no net sales" days, which excludes negative sales representing returns.
- Missing rows are filled with zero, assuming no unit sales for that item that day and therefore no "demand". Further research could instead model demand and sales seperately with stock data or flag likely stockout days in the training data, but that is out of scope for the current models and available data. 
- The seasonal period is defined as 7 based on the EDA findings in the heatmap and ACF plot. 
- RMSLE evaluation: RMSLE measures error as a percentage rather than a raw amount, because working in log space converts ratios into differences. We are also assuming in this case that over estimating demand is better than underestimating demand. NOTE that RMSLE cannot handle negative values so we have to clip the negative observations to zero. 
- WAPE evaluation: Error is expressed as a fraction of total sales (inclusive of everything sold, not by family).

### Findings

#### Overall

|Model|RMSLE|WAPE|
|---|---|---|
|seasonal_naive|0.546429|0.169253|
|ets|0.450813|0.150874|

ETS improved over the seasonal naive model by ~1.8% on the WAPE score and ~18% on the RMSLE score. This means that a model looking at level, trend and seasonality outperforms a model that only uses data from the previous week. 

For the ETS model, the WAPE tells us that forecasts are still off by about 15% of total volume

#### By Product Family

Predictions by ETS improve over seasonal naive in almost all families except for frozen foods and liqour/wine/beer. This shows robustness with the largest gains on volatile low-volume categories. We had discovered in the EDA that liqour/wine/beer shows irregular weekend patterns compared to the other families. It is possible that The WAPE scores for books, home appliances and baby care also exceed 1, which could be due to the low amounts of units sold (close to zero) in those categories, causing the WAPE scoring method to break down. However, we note that the RMSLE scores for those same categories show the ETS out-performing the seasonal naive model and staying stable. The models struggling to predict these sparse series is an argument for a more advanced model that can generalize to all series while still capturing distinct series specific features. We will try this with the LightGBM model in the next phase. 

## LightGBM Findings

### Assumptions

- LightGBM will natively handle NaN values in the first lags in each series and also in the oil dataset.
- It's critical to check for leakage across dates and ensure the validation/test data are not leaking into the training phases. 

### Findings

#### Overall

|Model|RMSLE|WAPE|
|---|---|---|
|seasonal_naive|0.546429|0.169253|
|ets|0.450813|0.150874|
|lightgbm|0.402587|0.131917|

LightGBM improved overall predictions by 0.05 over ETS and 0.14 over seasonal naive baselines. LightGBM also showed marginal improvements on the WAPE score, with forecasts being off by ~13% compared to ETS' 15% and seasonal naive's 17%.

##### Error Analysis and SHAP

Plotting the RMSLE scores grouped by horizon days shows that the scores increase over time, meaning that forecasts become less accurate as we increase the number of days we are attempting to forecast. There are noticeable drops in the RMSLE on days 6, 10, and 13, suggesting that there may be distinct patterns for those days that the model picks up on to make predictions.

Forecasts are unbiased at the median, but a slight positive mean residual reflects occasional large demand spikes the model under-predicts, which has potential implications for inventory and sales if products are not available.

The SHAP plot tells us that the roll mean and most recent sales lags (16,21) are providing the greatest signal for the model, which is expected considering the past weeks value is a decent predictor of the forecasted value. Products on promotion are also an indicator of the forecasted value, which is on par with the evidence we found in the EDA that unit sales for items on promotion was higher than those not on promotion. Oil prices is a feature that does not largely effect the forecast since it lands towards the bottom of the rankings. Interestingly, holidays also have little predictive power in the model. 

#### By Product Family

LightGBM outperforms ETS and seasonal naive in most product families, with the notable exception of perishable families like bread, dairy, eggs, meats, poultry, prepared foods, and produce still achieving the best performance under the ETS model. My hypothesis is that since these items are perishable, they are normally sold in large quantities every week since consumers often need to replenish. Knowing this, it's possible that the ETS model offers a more simple prediction based on the 7-day seasonal pattern, and LightGBM might be adding too much variance to the model to where it ends up hurting the final predictions. AFter further investigation, I found that the product families where LightGBM loses to the seasonal naive models or ets models are the high-volume staples (dairy, eggs, etc.) whereas the product families where it definitively wins over the other models are those with intermittent or volatile purchases. From here, I can draw the conclusion that LightGBM doesn't improve the predictions for products where volume is high and purchase frequency is predictable because the naive predictions based on seasonality already capture the patterns pretty well. The one exception to this seems to be books, which is not a perishable item. 


## Disruption Analysis Findings

By training the model on data prior to the April 2016 Earthquake and then testing the model on data from two weeks before up until a month after the earthquake, we find that the model shows a large gap between the predicted and actual values on the test set. The WAPE is just below 20% before the earthquake, then increases to 30% in the days just around the quake before dropping down to just above 20% after the earthquake. When we group the forecasts into coastal and inland regions we find that the coastal regions, who were hit hardest by the earthquake, show a higher WAPE at 37% compared to the inland region's 29%. When we observe the ratio of true versus forecasted units by family, interestingly we see families like home appliances, electronics and baby care towards the top of the list. This is plausible given that during an earthquake things often break or are damaged within houses, so many people may need to purchase new home appliances or items that may have been damaged during the quake. However, we also know from previous EDA that these are generally low volume units, so I also looked at the difference in true versus predicted values by total units, which shows that gorcery, beverages, produce and cleaning were among the families with the highest excess demand in total units, which makes sense after an earthquake that emergency essentials like this would have excess demand compared to that which is forecasted.

## Business Impact

- The improved forecast from the LightGBM model reduces the estimated error cost by $1,705,506 over the test window when compared to the seasonal naive model. 
- Beverages, grocery and produce dominate the cost ranking, which is likely due to their volume since their per-unit error is low. Forecasting efforts should focus on these top categories for the biggest cost saving wins.
- The model leans toward underforecasting for both perishable and non-perishable families, so a planner should bias safety stock upward. 
- NOTE: These assumption parameters are illustrative, and in production would be switched out for a company's actual cost parameters.