import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.metrics import mean_squared_error, r2_score


with open('results/experiment_results.json', 'r') as f:
    results_summary = json.load(f)

results = {}
for exp_name in results_summary.keys():
    with open(f'models/{exp_name}_model.pkl', 'rb') as f:
        model = pickle.load(f)
    results[exp_name] = {
        'model': model,
        'metrics': results_summary[exp_name]['metrics'],
        'carbon_emissions': results_summary[exp_name]['carbon_emissions'],
        'feature_version': results_summary[exp_name]['feature_version'],
        'hyperparams': results_summary[exp_name]['hyperparams']
    }

comparison_data = []
for exp_name, result in results_summary.items():
    row = {
        'Experiment': exp_name,
        'Feature Version': result['feature_version'],
        'n_estimators': result['hyperparams']['n_estimators'],
        'max_depth': result['hyperparams']['max_depth'],
        'MSE': result['metrics']['mse'],
        'MAE': result['metrics']['mae'],
        'R²': result['metrics']['r2'],
        'RMSE': result['metrics']['rmse'],
        'Carbon Emissions (kg CO2)': result['carbon_emissions']
    }
    comparison_data.append(row)

df_comparison = pd.DataFrame(comparison_data)
df_comparison = df_comparison.sort_values('R²', ascending=False)

print(df_comparison.to_string(index=False))

df_comparison.to_csv('results/model_comparison.csv', index=False)


os.makedirs('plots', exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Model Performance Comparison', fontsize=16)

# R²
ax = axes[0, 0]
bars = ax.bar(df_comparison['Experiment'], df_comparison['R²'])
ax.set_title('R² Score Comparison')
ax.set_xlabel('Experiment')
ax.set_ylabel('R² Score')
ax.set_xticks(range(len(df_comparison)))
ax.set_xticklabels(df_comparison['Experiment'], rotation=45, ha='right')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom')

# RMSE
ax = axes[0, 1]
bars = ax.bar(df_comparison['Experiment'], df_comparison['RMSE'])
ax.set_title('RMSE Comparison')
ax.set_xlabel('Experiment')
ax.set_ylabel('RMSE')
ax.set_xticklabels(df_comparison['Experiment'], rotation=45, ha='right')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}', ha='center', va='bottom')

# MAE
ax = axes[1, 0]
bars = ax.bar(df_comparison['Experiment'], df_comparison['MAE'])
ax.set_title('MAE Comparison')
ax.set_xlabel('Experiment')
ax.set_ylabel('MAE')
ax.set_xticklabels(df_comparison['Experiment'], rotation=45, ha='right')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}', ha='center', va='bottom')

# Emissions
ax = axes[1, 1]
bars = ax.bar(df_comparison['Experiment'], df_comparison['Carbon Emissions (kg CO2)'])
ax.set_title('Emissions Comparison')
ax.set_xlabel('Experiment')
ax.set_ylabel('Carbon Emission (kg CO2)')
ax.set_xticklabels(df_comparison['Experiment'], rotation=45, ha='right')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.6f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('plots/model_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature Version
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Feature Version Performance Comparison', fontsize=16)

v1_results = df_comparison[df_comparison['Feature Version'] == 'V1']
v2_results = df_comparison[df_comparison['Feature Version'] == 'V2']

# R²
ax = axes[0]
x = ['V1', 'V2']
y = [v1_results['R²'].mean(), v2_results['R²'].mean()]
bars = ax.bar(x, y, width=0.5)
ax.set_title('Average R² Score')
ax.set_ylabel('R² Score')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom')

# RMSE
ax = axes[1]
y = [v1_results['RMSE'].mean(), v2_results['RMSE'].mean()]
bars = ax.bar(x, y, width=0.5)
ax.set_title('Average RMSE')
ax.set_ylabel('RMSE')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('plots/feature_version_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Hyperparameter
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# n_estimators vs R²，size of the point represents the max_depth
scatter = ax.scatter(df_comparison['n_estimators'], 
                    df_comparison['R²'],
                    s=df_comparison['max_depth'] * 20, 
                    c=['blue' if v == 'V1' else 'red' for v in df_comparison['Feature Version']],
                    alpha=0.6)

ax.set_xlabel('n_estimators')
ax.set_ylabel('R² Score')
ax.set_title('Hyperparameters on Model Performance')

for version, color in [('V1', 'blue'), ('V2', 'red')]:
    ax.scatter([], [], c=color, label=f'Feature {version}')
ax.legend()

for idx, row in df_comparison.iterrows():
    ax.annotate(row['Experiment'], 
                (row['n_estimators'], row['R²']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.savefig('plots/hyperparameter_impact.png', dpi=300, bbox_inches='tight')
plt.show()


best_model = df_comparison.iloc[0]
print(f"Best Model: {best_model['Experiment']}")
print(f"Feature Version: {best_model['Feature Version']}")
print(f"R² Score: {best_model['R²']:.4f}")
print(f"RMSE: {best_model['RMSE']:.2f}")
print(f"Carbon Emissions: {best_model['Carbon Emissions (kg CO2)']:.6f} kg CO2")

print("\nKey Findings:")
print(f"1. Average R² of V2 feature version: {v2_results['R²'].mean():.4f} vs V1: {v1_results['R²'].mean():.4f}")
print(f"2. More trees (n_estimators=200) improve R² on average: {df_comparison[df_comparison['n_estimators']==200]['R²'].mean() - df_comparison[df_comparison['n_estimators']==100]['R²'].mean():.4f}")
print(f"3. Model with the highest carbon emissions: {df_comparison.loc[df_comparison['Carbon Emissions (kg CO2)'].idxmax(), 'Experiment']}")

summary_report = f"""
# Experimental Summary

## Best Model
- Experiment Name: {best_model['Experiment']}
- Feature Version: {best_model['Feature Version']}
- R² Score: {best_model['R²']:.4f}
- RMSE: {best_model['RMSE']:.2f}
- MAE: {best_model['MAE']:.2f}
- Carbon Emissions: {best_model['Carbon Emissions (kg CO2)']:.6f} kg CO2

## Key findings
1. Comparison of feature versions:
- V1 average R²: {v1_results['R²'].mean():.4f}
- V2 average R²: {v2_results['R²'].mean():.4f}
- Improvement: {(v2_results['R²'].mean() - v1_results['R²'].mean()) / v1_results['R²'].mean() * 100:.1f}%

2. Hyperparameter impact:
- Average R² for n_estimators=100: {df_comparison[df_comparison['n_estimators']==100]['R²'].mean():.4f}
- Average R² for n_estimators=200: {df_comparison[df_comparison['n_estimators']==200]['R²'].mean():.4f}

3. Environmental impact:
- Total carbon emissions: {df_comparison['Carbon Emissions (kg CO2)'].sum():.6f} kg CO2
- Average per experiment: {df_comparison['Carbon Emissions (kg CO2)'].mean():.6f} kg CO2

## Recommendations
- V2 feature version (including more features and derived features) significantly improves model performance
- Increasing the number of trees (n_estimators) can improve performance, but it will also increase training time and carbon emissions
- In practical applications, it is necessary to balance model performance and environmental impact
"""

with open('results/summary_report.md', 'w', encoding='utf-8') as f:
    f.write(summary_report)