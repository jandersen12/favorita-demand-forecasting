# Favorita Demand Forecasting

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
│   └── 02-baselines.ipynb
├── src/              
│   ├── __init__.py
│   ├── evaluate.py
│   ├── make_dataset.py
│   ├── models.py
│   └── rolling_origin_cv.py 
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

```
|---|---|---|
|Model|RMSLE|WAPE|
|seasonal_naive|0.546429|0.169253|
|ets|0.450813|0.150874|
```

ETS improved over the seasonal naive model by ~1.8% on the WAPE score and ~18% on the RMSLE score. This means that a model looking at level, trend and seasonality outperforms a model that only uses data from the previous week. 

For the ETS model, the WAPE tells us that forecasts are still off by about 15% of total volume

#### By Product Family

Predictions by ETS improve over seasonal naive in almost all families except for frozen foods and liqour/wine/beer. This shows robustness with the largest gains on volatile low-volume categories. We had discovered in the EDA that liqour/wine/beer shows irregular weekend patterns compared to the other families. It is possible that The WAPE scores for books, home appliances and baby care also exceed 1, which could be due to the low amounts of units sold (close to zero) in those categories, causing the WAPE scoring method to break down. However, we note that the RMSLE scores for those same categories show the ETS out-performing the seasonal naive model and staying stable. 
