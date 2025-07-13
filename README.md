python prepare_data.py

feast -c athletes_feature_store apply
feast -c athletes_feature_store materialize-incremental $(date +%F)

python ml_pipeline.py
python compare_results.py
