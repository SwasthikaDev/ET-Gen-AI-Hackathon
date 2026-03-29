"""
Feature engineering pipeline.
Merges crime records, weather, POI density, and temporal signals
into a model-ready feature matrix.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

INDIAN_HOLIDAYS = {
    (1, 26): "Republic Day",
    (8, 15): "Independence Day",
    (10, 2): "Gandhi Jayanti",
    (11, 1): "Kannada Rajyotsava",
    (12, 25): "Christmas",
}


def is_public_holiday(dt: datetime) -> int:
    return int((dt.month, dt.day) in INDIAN_HOLIDAYS)


def shift_slot(hour: int) -> int:
    """0=morning (6–14), 1=afternoon (14–22), 2=night (22–6)."""
    if 6 <= hour < 14:
        return 0
    if 14 <= hour < 22:
        return 1
    return 2


def build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lag and rolling features per (city, zone_id).
    Expects columns: city, zone_id, timestamp, incident_count.
    """
    df = df.copy().sort_values(["city", "zone_id", "timestamp"])
    grp = df.groupby(["city", "zone_id"])["incident_count"]

    df["lag_1h"] = grp.shift(1).fillna(0)
    df["lag_2h"] = grp.shift(2).fillna(0)
    df["lag_24h"] = grp.shift(24).fillna(0)
    df["lag_48h"] = grp.shift(48).fillna(0)
    df["lag_7d"] = grp.shift(24 * 7).fillna(0)
    df["rolling_7d_mean"] = (
        grp.transform(lambda x: x.shift(1).rolling(24 * 7, min_periods=1).mean()).fillna(0)
    )
    df["ema_14d"] = (
        grp.transform(lambda x: x.shift(1).ewm(span=24 * 14, adjust=False).mean()).fillna(0)
    )
    df["days_since_last_incident"] = (
        df.groupby(["city", "zone_id"])["incident_count"]
        .transform(lambda s: s.shift(1).eq(0).cumsum())
        .fillna(0)
    )
    return df


def aggregate_to_hourly(records_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw event records to zone-hour incident counts.
    """
    records_df["timestamp"] = pd.to_datetime(records_df["timestamp"])
    records_df["hour_ts"] = records_df["timestamp"].dt.floor("h")

    agg = (
        records_df.groupby(["city", "zone_id", "hour_ts"])
        .agg(
            incident_count=("crime_type", "count"),
            top_crime_type=("crime_type", lambda x: x.value_counts().idxmax()),
            temperature_c=("temperature_c", "mean"),
            precipitation_mm=("precipitation_mm", "mean"),
            wind_speed_kmh=("wind_speed_kmh", "mean"),
            is_rainy=("is_rainy", "max"),
        )
        .reset_index()
        .rename(columns={"hour_ts": "timestamp"})
    )
    return agg


def attach_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["hour"] = ts.dt.hour
    df["weekday"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    df["is_public_holiday"] = ts.dt.to_pydatetime()
    df["is_public_holiday"] = [is_public_holiday(d) for d in pd.to_datetime(df["timestamp"])]
    df["shift_slot"] = df["hour"].apply(shift_slot)
    return df


def attach_zone_features(df: pd.DataFrame, zones_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join spatial/POI and enriched NCRB features from zones lookup."""
    base_cols = [
        "zone_id",
        "zone_type",
        "population_density",
        "bar_count_500m",
        "atm_count_500m",
        "market_count_500m",
        "bus_stop_count_500m",
        "nearest_police_station_km",
        "road_density",
        "lighting_score",
    ]
    # Include enriched features if present in zones_df
    enriched_cols = [
        c for c in [
            "women_safety_index",
            "vulnerability_index",
            "police_coverage_ratio",
            "property_value_stolen_lakh",
            "state_auto_theft_count",
        ] if c in zones_df.columns
    ]
    merge_cols = base_cols + enriched_cols
    return df.merge(zones_df[[c for c in merge_cols if c in zones_df.columns]], on="zone_id", how="left")


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    crime_dummies = pd.get_dummies(df.get("top_crime_type", pd.Series(dtype=str)), prefix="prev_crime")
    zone_type_dummies = pd.get_dummies(df.get("zone_type", pd.Series(dtype=str)), prefix="zone_type")
    df = pd.concat([df.drop(columns=["top_crime_type", "zone_type"], errors="ignore"), crime_dummies, zone_type_dummies], axis=1)
    return df


FEATURE_COLS = [
    # Temporal
    "hour",
    "weekday",
    "month",
    "is_weekend",
    "is_public_holiday",
    "shift_slot",
    # Lag / rolling
    "days_since_last_incident",
    "rolling_7d_mean",
    "lag_1h",
    "lag_2h",
    "lag_24h",
    "lag_48h",
    "lag_7d",
    "ema_14d",
    # Weather
    "temperature_c",
    "precipitation_mm",
    "wind_speed_kmh",
    "is_rainy",
    # Spatial / POI
    "population_density",
    "bar_count_500m",
    "atm_count_500m",
    "market_count_500m",
    "bus_stop_count_500m",
    "nearest_police_station_km",
    "road_density",
    "lighting_score",
    # Enriched NCRB multi-source features
    "women_safety_index",
    "vulnerability_index",
    "police_coverage_ratio",
    "property_value_stolen_lakh",
    "state_auto_theft_count",
]


def build_feature_matrix(
    records_df: pd.DataFrame,
    zones_df: pd.DataFrame,
    include_lags: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Full pipeline from raw records → (X features, y labels).

    y labels:
      0 = no incident
      1 = low risk (1 incident)
      2 = medium risk (2–3 incidents)
      3 = high risk (4+ incidents)
    """
    hourly = aggregate_to_hourly(records_df)
    hourly = attach_temporal_features(hourly)
    hourly = attach_zone_features(hourly, zones_df)

    if include_lags:
        hourly = build_lag_features(hourly)

    hourly = encode_categoricals(hourly)

    # Risk label
    def label(count: int) -> int:
        if count == 0:
            return 0
        if count == 1:
            return 1
        if count <= 3:
            return 2
        return 3

    y = hourly["incident_count"].apply(label)

    available_cols = [c for c in FEATURE_COLS if c in hourly.columns]
    extra_dummies = [c for c in hourly.columns if c.startswith("prev_crime_") or c.startswith("zone_type_")]
    X = hourly[available_cols + extra_dummies].fillna(0)

    return X, y


def build_inference_row(
    zone: dict[str, Any],
    target_dt: datetime,
    weather: dict[str, float],
    recent_counts: list[int] | None = None,
) -> pd.DataFrame:
    """
    Build a single feature row for real-time inference.

    zone: zone metadata dict (from zones lookup)
    target_dt: prediction target datetime
    weather: {"temperature_c": ..., "precipitation_mm": ..., "wind_speed_kmh": ...}
    recent_counts: [lag_1h, lag_2h, lag_24h, lag_48h, lag_7d]
    """
    recent = recent_counts or [0, 0, 0, 0, 0]
    row = {
        "hour": target_dt.hour,
        "weekday": target_dt.weekday(),
        "month": target_dt.month,
        "is_weekend": int(target_dt.weekday() >= 5),
        "is_public_holiday": is_public_holiday(target_dt),
        "shift_slot": shift_slot(target_dt.hour),
        "days_since_last_incident": 0 if any(c > 0 for c in recent) else 1,
        "rolling_7d_mean": sum(recent) / max(len(recent), 1),
        "lag_1h": recent[0] if len(recent) > 0 else 0,
        "lag_2h": recent[1] if len(recent) > 1 else 0,
        "lag_24h": recent[2] if len(recent) > 2 else 0,
        "lag_48h": recent[3] if len(recent) > 3 else 0,
        "lag_7d": recent[4] if len(recent) > 4 else 0,
        "ema_14d": sum(recent) / max(len(recent), 1),
        "temperature_c": weather.get("temperature_c", 28.0),
        "precipitation_mm": weather.get("precipitation_mm", 0.0),
        "wind_speed_kmh": weather.get("wind_speed_kmh", 10.0),
        "is_rainy": int(weather.get("precipitation_mm", 0) > 1),
        "population_density": zone.get("population_density", 10000),
        "bar_count_500m": zone.get("bar_count_500m", 2),
        "atm_count_500m": zone.get("atm_count_500m", 4),
        "market_count_500m": zone.get("market_count_500m", 5),
        "bus_stop_count_500m": zone.get("bus_stop_count_500m", 3),
        "nearest_police_station_km": zone.get("nearest_police_station_km", 1.5),
        "road_density": zone.get("road_density", 0.6),
        "lighting_score": zone.get("lighting_score", 0.7),
        # Enriched NCRB features (populated from zones metadata)
        "women_safety_index": zone.get("women_safety_index", 0.0),
        "vulnerability_index": zone.get("vulnerability_index", 0.0),
        "police_coverage_ratio": zone.get("police_coverage_ratio", 0.75),
        "property_value_stolen_lakh": zone.get("property_value_stolen_lakh", 0.0),
        "state_auto_theft_count": zone.get("state_auto_theft_count", 0),
    }
    return pd.DataFrame([row])
