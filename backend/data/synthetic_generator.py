"""
Synthetic crime data generator.
Produces statistically realistic records for 5 Indian pilot cities.
Run: python -m backend.data.synthetic_generator
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

CITIES = {
    "Bengaluru": {
        "center": (12.9716, 77.5946),
        "zones": 25,
        "dominant_crimes": ["vehicle_theft", "chain_snatching", "burglary"],
        "peak_hours": [8, 9, 20, 21, 22],
        "hotspot_zones": [3, 7, 12, 18, 22],
    },
    "Hyderabad": {
        "center": (17.3850, 78.4867),
        "zones": 22,
        "dominant_crimes": ["vehicle_theft", "robbery", "cyber_fraud"],
        "peak_hours": [9, 10, 19, 20, 21],
        "hotspot_zones": [2, 5, 11, 16, 20],
    },
    "Mumbai": {
        "center": (19.0760, 72.8777),
        "zones": 30,
        "dominant_crimes": ["pickpocketing", "chain_snatching", "assault"],
        "peak_hours": [8, 9, 17, 18, 22, 23],
        "hotspot_zones": [4, 8, 14, 19, 25],
    },
    "Delhi": {
        "center": (28.6139, 77.2090),
        "zones": 28,
        "dominant_crimes": ["vehicle_theft", "robbery", "assault", "burglary"],
        "peak_hours": [7, 8, 20, 21, 22],
        "hotspot_zones": [1, 6, 13, 17, 24],
    },
    "Chennai": {
        "center": (13.0827, 80.2707),
        "zones": 20,
        "dominant_crimes": ["vehicle_theft", "chain_snatching", "pickpocketing"],
        "peak_hours": [8, 9, 18, 19, 21],
        "hotspot_zones": [2, 5, 9, 14, 18],
    },
}

CRIME_TYPES = [
    "vehicle_theft",
    "chain_snatching",
    "burglary",
    "robbery",
    "pickpocketing",
    "assault",
    "cyber_fraud",
    "dacoity",
]

POI_PROFILES = {
    "commercial": {"bar_count": (3, 10), "atm_count": (5, 15), "market_count": (8, 20), "bus_stop_count": (3, 8)},
    "residential": {"bar_count": (0, 3), "atm_count": (2, 8), "market_count": (2, 10), "bus_stop_count": (2, 6)},
    "transit": {"bar_count": (1, 5), "atm_count": (4, 12), "market_count": (5, 15), "bus_stop_count": (5, 15)},
    "mixed": {"bar_count": (1, 7), "atm_count": (3, 10), "market_count": (4, 14), "bus_stop_count": (3, 9)},
}

ZONE_TYPES = ["commercial", "residential", "transit", "mixed"]


def _generate_zones(city: str, config: dict) -> list[dict]:
    """Generate realistic zone metadata for a city."""
    rng = np.random.default_rng(seed=hash(city) % 2**32)
    clat, clon = config["center"]
    zones = []
    for i in range(1, config["zones"] + 1):
        zone_type = random.choice(ZONE_TYPES)
        poi = POI_PROFILES[zone_type]
        lat_offset = rng.uniform(-0.12, 0.12)
        lon_offset = rng.uniform(-0.12, 0.12)
        z = {
            "zone_id": f"{city[:3].upper()}_{i:02d}",
            "city": city,
            "zone_type": zone_type,
            "lat": round(clat + lat_offset, 6),
            "lon": round(clon + lon_offset, 6),
            "population_density": int(rng.uniform(3000, 25000)),
            "bar_count_500m": int(rng.integers(*poi["bar_count"])),
            "atm_count_500m": int(rng.integers(*poi["atm_count"])),
            "market_count_500m": int(rng.integers(*poi["market_count"])),
            "bus_stop_count_500m": int(rng.integers(*poi["bus_stop_count"])),
            "nearest_police_station_km": round(rng.uniform(0.3, 4.5), 2),
            "road_density": round(rng.uniform(0.2, 1.0), 2),
            "lighting_score": round(rng.uniform(0.3, 1.0), 2),
            "is_hotspot": i in config["hotspot_zones"],
        }
        zones.append(z)
    return zones


def _hourly_crime_probability(hour: int, zone: dict, city_config: dict) -> float:
    """Compute base crime probability given hour + zone features."""
    base = 0.05
    if hour in city_config["peak_hours"]:
        base += 0.25
    if zone["bar_count_500m"] > 5:
        base += 0.10
    if zone["bus_stop_count_500m"] > 8:
        base += 0.08
    if zone["nearest_police_station_km"] > 3.0:
        base += 0.07
    if zone["lighting_score"] < 0.5:
        base += 0.06
    if zone["is_hotspot"]:
        base += 0.20
    return min(base, 0.95)


def generate_records(
    start_date: str = "2021-01-01",
    end_date: str = "2022-12-31",
    output_dir: str = "backend/data/synthetic",
) -> dict:
    """
    Generate synthetic hourly crime records for all cities.
    Returns a summary dict with file paths.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    summary = {}

    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    total_hours = int((end - start).total_seconds() / 3600)

    all_records: list[dict] = []
    all_zones: list[dict] = []

    for city, config in CITIES.items():
        zones = _generate_zones(city, config)
        all_zones.extend(zones)

        city_records = 0
        for hour_offset in range(total_hours):
            ts = start + timedelta(hours=hour_offset)
            temp_c = rng.gauss(28, 6)
            precip_mm = max(0, rng.gauss(0, 2)) if rng.random() < 0.15 else 0.0
            wind_kmh = abs(rng.gauss(12, 5))

            for zone in zones:
                prob = _hourly_crime_probability(ts.hour, zone, config)
                if precip_mm > 3:
                    prob *= 0.75

                if rng.random() < prob:
                    crime_pool = config["dominant_crimes"] + ["pickpocketing", "assault"]
                    crime_type = rng.choices(
                        crime_pool,
                        weights=[0.35, 0.25, 0.15, 0.10, 0.08, 0.07],
                        k=1,
                    )[0] if len(crime_pool) >= 6 else rng.choice(crime_pool)

                    all_records.append(
                        {
                            "city": city,
                            "zone_id": zone["zone_id"],
                            "timestamp": ts.isoformat(),
                            "year": ts.year,
                            "month": ts.month,
                            "day": ts.day,
                            "hour": ts.hour,
                            "weekday": ts.weekday(),
                            "is_weekend": int(ts.weekday() >= 5),
                            "crime_type": crime_type,
                            "temperature_c": round(temp_c, 1),
                            "precipitation_mm": round(precip_mm, 2),
                            "wind_speed_kmh": round(wind_kmh, 1),
                            "is_rainy": int(precip_mm > 1),
                        }
                    )
                    city_records += 1

        summary[city] = {"zones": len(zones), "records": city_records}
        print(f"  {city}: {len(zones)} zones, {city_records} crime records")

    records_df = pd.DataFrame(all_records)
    zones_df = pd.DataFrame(all_zones)

    records_path = f"{output_dir}/crime_records.csv"
    zones_path = f"{output_dir}/zones.csv"
    summary_path = f"{output_dir}/summary.json"

    records_df.to_csv(records_path, index=False)
    zones_df.to_csv(zones_path, index=False)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTotal records: {len(all_records):,}")
    print(f"Saved to {output_dir}/")
    return {"records": records_path, "zones": zones_path, "summary": summary}


if __name__ == "__main__":
    print("Generating synthetic crime data...")
    result = generate_records()
    print(f"\nDone. Files: {result}")
