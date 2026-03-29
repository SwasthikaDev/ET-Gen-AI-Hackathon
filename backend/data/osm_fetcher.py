"""
OpenStreetMap POI fetcher via Overpass API.
Enriches zone metadata with real counts of bars, ATMs, markets, bus stops, etc.

Free, no API key required. Caches results to avoid repeated queries.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pandas as pd

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
CACHE_DIR = Path("backend/data/cache/osm")

# POI categories: each entry is a list of Overpass tag filters
POI_QUERIES: dict[str, str] = {
    "bar_count_500m": '(node["amenity"~"bar|pub|nightclub"](around:{r},{lat},{lon});'
                      'way["amenity"~"bar|pub|nightclub"](around:{r},{lat},{lon}););',
    "atm_count_500m": '(node["amenity"="atm"](around:{r},{lat},{lon});'
                      'node["amenity"="bank"](around:{r},{lat},{lon}););',
    "market_count_500m": '(node["amenity"~"marketplace|supermarket"](around:{r},{lat},{lon});'
                         'way["shop"~"supermarket|convenience|mall"](around:{r},{lat},{lon}););',
    "bus_stop_count_500m": '(node["highway"="bus_stop"](around:{r},{lat},{lon});'
                           'node["public_transport"="stop_position"](around:{r},{lat},{lon}););',
    "school_count_500m": '(node["amenity"~"school|college|university"](around:{r},{lat},{lon}););',
    "park_count_500m": '(way["leisure"~"park|garden"](around:{r},{lat},{lon}););',
}

RADIUS_M = 500


def _overpass_count_query(poi_query: str, lat: float, lon: float, radius: int = RADIUS_M) -> str:
    q = poi_query.format(r=radius, lat=lat, lon=lon)
    return f"[out:json][timeout:25];\n{q}\nout count;"


def _fetch_count(poi_key: str, lat: float, lon: float) -> int:
    """Fetch POI count from Overpass with caching."""
    cache_key = f"{poi_key}_{lat:.4f}_{lon:.4f}"
    cache_path = CACHE_DIR / f"{cache_key}.json"

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)["count"]

    if not HTTPX_AVAILABLE:
        return 0

    query = _overpass_count_query(POI_QUERIES[poi_key], lat, lon)
    try:
        resp = httpx.post(OVERPASS_URL, data={"data": query}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        count = int(data["elements"][0]["tags"].get("total", 0))
    except Exception as e:
        print(f"    OSM query failed ({poi_key} at {lat:.4f},{lon:.4f}): {e}")
        count = 0

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"count": count}, f)

    return count


def _nearest_police_station_km(lat: float, lon: float) -> float:
    """Query nearest police station from OSM."""
    cache_key = f"police_{lat:.4f}_{lon:.4f}"
    cache_path = CACHE_DIR / f"{cache_key}.json"

    if cache_path.exists():
        with open(cache_path) as f:
            return float(json.load(f)["km"])

    if not HTTPX_AVAILABLE:
        return 1.5

    query = f"""[out:json][timeout:25];
(node["amenity"="police"](around:5000,{lat},{lon}););
out body;"""

    km = 1.5
    try:
        resp = httpx.post(OVERPASS_URL, data={"data": query}, timeout=30)
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
        if elements:
            nearest = elements[0]
            dlat = nearest["lat"] - lat
            dlon = nearest["lon"] - lon
            km = round(math.sqrt(dlat**2 + dlon**2) * 111.0, 2)
    except Exception as e:
        print(f"    Police station query failed: {e}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"km": km}, f)

    return km


def enrich_zones(
    zones_df: pd.DataFrame,
    delay_seconds: float = 1.0,
    max_zones: int | None = None,
) -> pd.DataFrame:
    """
    For each zone row, query OSM for real POI counts.
    Updates the DataFrame in-place and returns it.

    delay_seconds: pause between Overpass queries (be polite to the free API).
    max_zones: limit for testing/demo (None = all zones).
    """
    zones_df = zones_df.copy()
    subset = zones_df if max_zones is None else zones_df.head(max_zones)

    total = len(subset)
    print(f"Enriching {total} zones with OSM POI data...")

    for idx, (row_idx, row) in enumerate(subset.iterrows()):
        lat, lon = float(row["lat"]), float(row["lon"])
        print(f"  [{idx+1}/{total}] Zone {row['zone_id']} ({lat:.4f}, {lon:.4f})")

        for poi_key in POI_QUERIES:
            count = _fetch_count(poi_key, lat, lon)
            zones_df.at[row_idx, poi_key] = count
            time.sleep(0.1)

        zones_df.at[row_idx, "nearest_police_station_km"] = _nearest_police_station_km(lat, lon)
        time.sleep(delay_seconds)

    print("OSM enrichment complete.")
    return zones_df


def enrich_and_save(
    zones_csv: str = "backend/data/processed/zones.csv",
    output_csv: str = "backend/data/processed/zones_osm.csv",
    max_zones: int | None = None,
) -> str:
    """Load zones CSV, enrich with OSM, save enriched version."""
    df = pd.read_csv(zones_csv)
    enriched = enrich_zones(df, max_zones=max_zones)
    enriched.to_csv(output_csv, index=False)
    print(f"Enriched zones saved to {output_csv}")
    return output_csv


if __name__ == "__main__":
    enrich_and_save(max_zones=20)
