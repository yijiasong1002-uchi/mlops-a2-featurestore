import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import pickle
import json

# Feast imports
from feast import FeatureStore

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker


store = FeatureStore(repo_path="athletes_feature_store")
df_source = pd.read_parquet("athletes_feature_store/data/athletes_clean.parquet")

df_with_labels = df_source[df_source['deadlift'].notna()].copy()
print(f"with labels: {len(df_with_labels)}")

entity_df = pd.DataFrame({
    'athlete_id': df_with_labels['athlete_id'],
    'event_timestamp': df_with_labels['event_timestamp'],
    'deadlift_actual': df_with_labels['deadlift']
})


# V1
training_df_v1 = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "athletes_stats_v1:age",
        "athletes_stats_v1:height", 
        "athletes_stats_v1:weight",
    ],
).to_df()

# V2
training_df_v2 = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "athletes_stats_v2:age",
        "athletes_stats_v2:height",
        "athletes_stats_v2:weight", 
        "athletes_stats_v2:candj",
        "athletes_stats_v2:snatch",
        "athletes_stats_v2:backsq",
        "athletes_stats_v2:experience_years",

        "calculated_features:bmi",
        "calculated_features:strength_score",
        "calculated_features:strength_to_weight_ratio"
    ],
).to_df()

leak_cols = [c for c in training_df_v2.columns if c.endswith("__deadlift")]
training_df_v2.drop(columns=leak_cols, inplace=True)

print(f"V1: {training_df_v1.shape}")
print(f"V2: {training_df_v2.shape}")


def preprocess_data(df, feature_version):
    df = df.copy()
    
    y = df['deadlift_actual'].values
    X = df.drop(['athlete_id', 'event_timestamp', 'deadlift_actual'], axis=1)
    
    X = X.fillna(X.median())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    preprocessors = {'scaler': scaler, 'feature_names': X.columns.tolist()}
    
    return X_scaled, y, preprocessors

X_v1, y_v1, prep_v1 = preprocess_data(training_df_v1, 'v1')
X_v2, y_v2, prep_v2 = preprocess_data(training_df_v2, 'v2')

X_train_v1, X_test_v1, y_train_v1, y_test_v1 = train_test_split(X_v1, y_v1, test_size=0.2, random_state=42)
X_train_v2, X_test_v2, y_train_v2, y_test_v2 = train_test_split(X_v2, y_v2, test_size=0.2, random_state=42)

print(f"V1 train: {X_train_v1.shape}")
print(f"V2 train: {X_train_v2.shape}")


# Random Forest
hyperparams_set1 = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42
}

hyperparams_set2 = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42
}


results = {}

experiments = [
    ('v1_hyperparam1', X_train_v1, X_test_v1, y_train_v1, y_test_v1, hyperparams_set1, 'V1', prep_v1),
    ('v1_hyperparam2', X_train_v1, X_test_v1, y_train_v1, y_test_v1, hyperparams_set2, 'V1', prep_v1),
    ('v2_hyperparam1', X_train_v2, X_test_v2, y_train_v2, y_test_v2, hyperparams_set1, 'V2', prep_v2),
    ('v2_hyperparam2', X_train_v2, X_test_v2, y_train_v2, y_test_v2, hyperparams_set2, 'V2', prep_v2),
]

for exp_name, X_train, X_test, y_train, y_test, hyperparams, feature_version, preprocessor in experiments:
    print(f"experiment: {exp_name}")
    print(f"features: {X_train.shape[1]}")
    
    tracker = EmissionsTracker()
    tracker.start()
    
    model = RandomForestRegressor(**hyperparams)
    model.fit(X_train, y_train)
    
    emissions = tracker.stop()
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    results[exp_name] = {
        'model': model,
        'hyperparams': hyperparams,
        'feature_version': feature_version,
        'metrics': {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': rmse
        },
        'carbon_emissions': emissions,
        'y_test': y_test,
        'y_pred': y_pred,
        'feature_importance': dict(zip(
            preprocessor['feature_names'],
            model.feature_importances_
        ))
    }
    
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"Emission: {emissions:.6f} kg CO2")

# save
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

for exp_name, result in results.items():
    model_path = f'models/{exp_name}_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(result['model'], f)
    print(f"model saved: {model_path}")

results_summary = {}
for exp_name, result in results.items():
    results_summary[exp_name] = {
        'hyperparams': result['hyperparams'],
        'feature_version': result['feature_version'],
        'metrics': result['metrics'],
        'carbon_emissions': result['carbon_emissions'],
        'feature_importance': result['feature_importance']
    }

with open('results/experiment_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)


fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Feature Importance', fontsize=16)

for idx, (exp_name, result) in enumerate(results.items()):
    ax = axes[idx // 2, idx % 2]
    
    importance_dict = result['feature_importance']
    features = list(importance_dict.keys())
    importances = list(importance_dict.values())
    
    indices = np.argsort(importances)[::-1]
    features_sorted = [features[i] for i in indices]
    importances_sorted = [importances[i] for i in indices]
    
    bars = ax.bar(range(len(features_sorted)), importances_sorted)
    ax.set_title(f'{exp_name}')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Importance')
    ax.set_xticks(range(len(features_sorted)))
    ax.set_xticklabels(features_sorted, rotation=45, ha='right')
    
    for bar, imp in zip(bars, importances_sorted):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')