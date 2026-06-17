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
    ├── 01-eda.ipynb
    └── 02-baselines.ipynb
├── src/              
│   ├── __init__.py
    ├── evaluate.py
│   ├── make_dataset.py
    ├── models.py
    └── rolling_origin_cv.py 
├── .gitignore
├── requirements.txt
└── README.md
```