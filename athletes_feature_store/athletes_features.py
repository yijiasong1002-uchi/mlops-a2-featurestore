from datetime import timedelta
import pandas as pd
from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.types import Float32, Int64, Float64
from feast.on_demand_feature_view import on_demand_feature_view

athletes_source = FileSource(
    name="athletes_source",
    path="data/athletes_clean.parquet",
    timestamp_field="event_timestamp",
)

athlete = Entity(name="athlete", join_keys=["athlete_id"])

athletes_stats_v1 = FeatureView(
    name="athletes_stats_v1",
    entities=[athlete],
    ttl=timedelta(days=365),
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="height", dtype=Float32),
        Field(name="weight", dtype=Float32)
    ],
    online=True,
    source=athletes_source,
    tags={"version": "v1", "type": "basic_features"},
)

athletes_stats_v2 = FeatureView(
    name="athletes_stats_v2",
    entities=[athlete],
    ttl=timedelta(days=365),
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="height", dtype=Float32),
        Field(name="weight", dtype=Float32),
        # V2 new
        Field(name="candj", dtype=Float32),
        Field(name="snatch", dtype=Float32),
        Field(name="backsq", dtype=Float32), 
        Field(name="experience_years", dtype=Float32),
    ],
    online=True,
    source=athletes_source,
    tags={"version": "v2", "type": "advanced_features"},
)

@on_demand_feature_view(
    sources=[athletes_stats_v2],
    schema=[
        Field(name="bmi", dtype=Float64),
        Field(name="strength_score", dtype=Float64),
        Field(name="strength_to_weight_ratio", dtype=Float64),
    ],
)
def calculated_features(inputs: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    height_m = inputs["height"] * 0.0254
    weight_kg = inputs["weight"] * 0.453592
    df["bmi"] = weight_kg / (height_m ** 2)
    df["strength_score"] = inputs[["backsq","snatch","candj"]].mean(axis=1)
    df["strength_to_weight_ratio"] = inputs["backsq"] / inputs["weight"]
    return df

athlete_features_v1 = FeatureService(name="athlete_model_v1", features=[athletes_stats_v1])
athlete_features_v2 = FeatureService(name="athlete_model_v2", features=[athletes_stats_v2, calculated_features])