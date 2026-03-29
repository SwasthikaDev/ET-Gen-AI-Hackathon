"""
Open-Meteo weather fetcher.
- Historical hourly weather: used to enrich training data
- Current / forecast weather: used at inference time

Free API, no key required.
Docs: https://open-meteo.com/en/docs
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

CACHE_DIR = Path("backend/data/cache/weather")

CITY_COORDS: dict[str, tuple[float, float]] = {
    "Bengaluru": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Chennai": (13.0827, 80.2707),
}

HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARS = "temperature_2m,precipitation,wind_speed_10m"
CURRENT_VARS = "temperature_2m,precipitation,wind_speed_10m"


# ---------------------------------------------------------------------------
# Current weather (for inference)
# ---------------------------------------------------------------------------

async def get_current_weather(city: str) -> dict[str, float]:
    """Fetch current weather for a city. Returns dict with standard keys."""
    lat, lon = CITY_COORDS.get(city, (20.0, 78.0))

    if not HTTPX_AVAILABLE:
        return _defaults()

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                FORECAST_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": CURRENT_VARS,
                    "timezone": "Asia/Kolkata",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            cur = data.get("current", {})
            return {
                "temperature_c": float(cur.get("temperature_2m", 28.0)),
                "precipitation_mm": float(cur.get("precipitation", 0.0)),
                "wind_speed_kmh": float(cur.get("wind_speed_10m", 10.0)),
            }
    except Exception as e:
        print(f"Weather fetch failed for {city}: {e}")
        return _defaults()


def _defaults() -> dict[str, float]:
    return {"temperature_c": 28.0, "precipitation_mm": 0.0, "wind_speed_kmh": 10.0}


# ---------------------------------------------------------------------------
# Historical weather (for training data enrichment)
# ---------------------------------------------------------------------------

def fetch_historical(
    city: str,
    start_date: str,
    end_date: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch hourly historical weather for a city between start_date and end_date.
    Returns DataFrame with columns: timestamp, temperature_c, precipitation_mm, wind_speed_kmh.
    """
    cache_key = f"{city}_{start_date}_{end_date}"
    cache_path = CACHE_DIR / f"{cache_key}.parquet"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if use_cache and cache_path.exists():
        print(f"  Weather cache hit: {cache_key}")
        return pd.read_parquet(cache_path)

    lat, lon = CITY_COORDS.get(city, (20.0, 78.0))

    if not HTTPX_AVAILABLE:
        return _synthetic_weather(city, start_date, end_date)

    print(f"  Fetching historical weather for {city} ({start_date} → {end_date})...")
    try:
        resp = httpx.get(
            HISTORICAL_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": end_date,
                "hourly": HOURLY_VARS,
                "timezone": "Asia/Kolkata",
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        hourly = data["hourly"]

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(hourly["time"]),
            "temperature_c": hourly["temperature_2m"],
            "precipitation_mm": hourly["precipitation"],
            "wind_speed_kmh": hourly["wind_speed_10m"],
        })
        df["city"] = city
        df["is_rainy"] = (df["precipitation_mm"] > 1).astype(int)

        df.to_parquet(cache_path, index=False)
        print(f"  {len(df):,} hourly weather records fetched and cached.")
        return df

    except Exception as e:
        print(f"  Historical weather fetch failed for {city}: {e}. Using synthetic fallback.")
        return _synthetic_weather(city, start_date, end_date)


def _synthetic_weather(city: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fallback: generate realistic synthetic weather when API is unavailable."""
    import numpy as np
    rng = np.random.default_rng(seed=hash(city) % 2**32)

    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    hours = int((end - start).total_seconds() / 3600) + 1
    timestamps = [start + timedelta(hours=i) for i in range(hours)]

    mean_temp, std_temp = {"Bengaluru": (23, 4), "Hyderabad": (27, 6),
                           "Mumbai": (28, 3), "Delhi": (25, 9), "Chennai": (29, 3)}.get(city, (27, 5))

    rows = []
    for ts in timestamps:
        seasonal = -4 if ts.month in {12, 1, 2} else (2 if ts.month in {4, 5} else 0)
        temp = float(rng.normal(mean_temp + seasonal, std_temp))
        precip = float(max(0, rng.normal(3, 5))) if (ts.month in {6,7,8,9} and rng.random() < 0.3) else 0.0
        wind = float(abs(rng.normal(12, 5)))
        rows.append({
            "timestamp": ts,
            "temperature_c": round(temp, 1),
            "precipitation_mm": round(precip, 2),
            "wind_speed_kmh": round(wind, 1),
            "city": city,
            "is_rainy": int(precip > 1),
        })
    return pd.DataFrame(rows)


def enrich_records_with_weather(
    records_df: pd.DataFrame,
    start_date: str = "2001-01-01",
    end_date: str = "2022-12-31",
) -> pd.DataFrame:
    """
    Join real weather data onto crime records by (city, hour).
    Replaces the synthetic temperature/precipitation columns with real values.
    """
    records_df = records_df.copy()
    records_df["timestamp"] = pd.to_datetime(records_df["timestamp"])

    cities = records_df["city"].unique()
    weather_frames = []

    for city in cities:
        print(f"Fetching weather for {city}...")
        w = fetch_historical(city, start_date, end_date)
        weather_frames.append(w)

    weather = pd.concat(weather_frames, ignore_index=True)
    weather["timestamp"] = pd.to_datetime(weather["timestamp"]).dt.floor("h")
    weather = weather.rename(columns={
        "temperature_c": "temperature_c_real",
        "precipitation_mm": "precipitation_mm_real",
        "wind_speed_kmh": "wind_speed_kmh_real",
    })

    records_df["timestamp_h"] = records_df["timestamp"].dt.floor("h")
    merged = records_df.merge(
        weather[["city", "timestamp", "temperature_c_real", "precipitation_mm_real", "wind_speed_kmh_real", "is_rainy"]],
        left_on=["city", "timestamp_h"],
        right_on=["city", "timestamp"],
        how="left",
        suffixes=("", "_w"),
    )

    # Use real values where available, fall back to synthetic
    for col, real_col in [("temperature_c", "temperature_c_real"),
                           ("precipitation_mm", "precipitation_mm_real"),
                           ("wind_speed_kmh", "wind_speed_kmh_real")]:
        if real_col in merged.columns:
            merged[col] = merged[real_col].fillna(merged[col])

    merged.drop(columns=["timestamp_h", "temperature_c_real", "precipitation_mm_real",
                          "wind_speed_kmh_real"], errors="ignore", inplace=True)
    return merged


if __name__ == "__main__":
    # Quick test
    import asyncio
    async def test():
        w = await get_current_weather("Bengaluru")
        print(f"Bengaluru current weather: {w}")
    asyncio.run(test())
