
# Experimental Summary

## Best Model
- Experiment Name: v2_hyperparam1
- Feature Version: V2
- R² Score: 0.8442
- RMSE: 37.84
- MAE: 28.13
- Carbon Emissions: 0.000004 kg CO2

## Key findings
1. Comparison of feature versions:
- V1 average R²: 0.4633
- V2 average R²: 0.8407
- Improvement: 81.5%

2. Hyperparameter impact:
- Average R² for n_estimators=100: 0.6797
- Average R² for n_estimators=200: 0.6243

3. Environmental impact:
- Total carbon emissions: 0.000037 kg CO2
- Average per experiment: 0.000009 kg CO2

## Conclusion
I noticed that V2 feature version (including more features and derived features) significantly improves model performance. Also, increasing the number of trees (n_estimators) can improve performance, but it will also increase training time and carbon emissions. Therefore, in practical applications, it is necessary for us to balance model performance and environmental impact.
