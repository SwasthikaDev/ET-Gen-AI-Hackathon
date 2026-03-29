"""
NCRB real data loader — calibrated to actual Kaggle dataset structure.

Dataset: https://www.kaggle.com/datasets/rajanand/crime-in-india
Files used (auto-discovered via rglob):
  01_District_wise_crimes_committed_IPC_2001_2012.csv
  01_District_wise_crimes_committed_IPC_2013.csv
  01_District_wise_crimes_committed_IPC_2014.csv
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column normalisation — ORDER MATTERS: specific patterns before general ones
# ---------------------------------------------------------------------------

# Each tuple: (regex_pattern, canonical_name)
# The regex is matched against the lowercased column name.
# First match wins — put SPECIFIC patterns before GENERAL ones.
COLUMN_PATTERNS: list[tuple[str, str]] = [
    # Identity columns
    (r"state.*ut|states.*ut",          "state"),
    (r"^district$",                     "district"),
    (r"^year$",                         "year"),

    # Specific crime columns — BEFORE generic "murder" / "theft" patterns
    (r"attempt.*murder|attempt to commit murder",       "attempt_murder"),
    (r"culpable homicide",                              "culpable_homicide"),
    (r"custodial rape",                                 "custodial_rape"),
    (r"other rape|rape other",                         "other_rape"),
    (r"attempt.*rape",                                  "attempt_rape"),
    (r"kidnapping.*abduction.*total|kidnapping.*abduction(?!.*women)",  "kidnapping"),
    (r"dacoity with murder",                            "dacoity_murder"),
    (r"preparation.*assembly.*dacoity|making preparation.*dacoity",     "dacoity_prep"),
    (r"other dacoity",                                  "other_dacoity"),
    (r"criminal trespass.*burglary|house trespass|burglary",            "burglary"),
    (r"auto theft",                                     "auto_theft"),
    (r"other theft|other thefts",                      "other_theft"),
    (r"hurt.*grevious|grievous hurt|hurt$",             "hurt"),
    (r"total.*ipc|total cognizable ipc",                "total_ipc"),
    (r"other ipc",                                      "other_ipc"),

    # General columns — AFTER all specific variants
    (r"^murder$",                       "murder"),
    (r"^rape$",                         "rape"),
    (r"^dacoity$",                      "dacoity"),
    (r"^robbery$",                      "robbery"),
    (r"^theft$",                        "theft"),
    (r"^riots$",                        "riots"),
    (r"dowry death",                    "dowry_deaths"),
    (r"cheating",                       "cheating"),
    (r"criminal breach of trust",       "criminal_breach_trust"),
    (r"counterfeit",                    "counterfeiting"),
    (r"arson",                          "arson"),
    (r"causing death by negligence",    "causing_death_negligence"),
]

# Map canonical crime columns → our model's crime_type labels
CRIME_TO_TYPE: dict[str, str] = {
    "murder":                   "assault",
    "attempt_murder":           "assault",
    "culpable_homicide":        "assault",
    "hurt":                     "assault",
    "rape":                     "assault",
    "dacoity":                  "dacoity",
    "dacoity_murder":           "dacoity",
    "robbery":                  "robbery",
    "kidnapping":               "robbery",
    "burglary":                 "burglary",
    "theft":                    "pickpocketing",
    "other_theft":              "pickpocketing",
    "auto_theft":               "vehicle_theft",
    "cheating":                 "cyber_fraud",
    "criminal_breach_trust":    "cyber_fraud",
}

# ---------------------------------------------------------------------------
# City → exact district names as they appear in NCRB CSVs
# ---------------------------------------------------------------------------

CITY_DISTRICTS: dict[str, list[str]] = {
    "Bengaluru": [
        "BANGALORE COMMR.",
        "BANGALORE RURAL",
        "BANGALORE URBAN",
        "BENGALURU URBAN",
        "BENGALURU RURAL",
        "BENGALURU COMMISSIONERATE",
    ],
    "Hyderabad": [
        "HYDERABAD CITY",
        "HYDERABAD",
        "RANGA REDDY",
        "MEDCHAL MALKAJGIRI",
        "CYBERABAD COMMISSIONERATE",
    ],
    "Mumbai": [
        "MUMBAI",
        "MUMBAI RLY.",
        "NAVI MUMBAI",
        "MUMBAI RURAL",
        "MUMBAI SUBURBAN",
        "THANE",
    ],
    "Delhi": [
        "CENTRAL", "EAST", "NEW DELHI", "NORTH", "NORTH EAST", "NORTH WEST",
        "SOUTH", "SOUTH EAST", "SOUTH WEST", "WEST", "OUTER",
        "NORTH-EAST", "NORTH-WEST", "SOUTH-EAST", "SOUTH-WEST",
        "DELHI UT TOTAL",
    ],
    "Chennai": [
        "CHENNAI",
        "CHENNAI RLY.",
        "KANCHEEPURAM",
        "TIRUVALLUR",
    ],
}

# District centroid coordinates — calibrated to real geography
DISTRICT_CENTROIDS: dict[str, tuple[float, float]] = {
    "BANGALORE COMMR.":         (12.9716, 77.5946),
    "BANGALORE RURAL":          (13.1630, 77.4050),
    "BANGALORE URBAN":          (12.9716, 77.5946),
    "BENGALURU URBAN":          (12.9716, 77.5946),
    "BENGALURU RURAL":          (13.1630, 77.4050),
    "BENGALURU COMMISSIONERATE":(12.9716, 77.5946),
    "HYDERABAD CITY":           (17.3850, 78.4867),
    "HYDERABAD":                (17.3850, 78.4867),
    "RANGA REDDY":              (17.2403, 78.3538),
    "MEDCHAL MALKAJGIRI":       (17.6270, 78.5337),
    "CYBERABAD COMMISSIONERATE":(17.4399, 78.3489),
    "MUMBAI":                   (19.0760, 72.8777),
    "MUMBAI RLY.":              (18.9389, 72.8355),
    "NAVI MUMBAI":              (19.0330, 73.0297),
    "MUMBAI SUBURBAN":          (19.1136, 72.8697),
    "THANE":                    (19.2183, 72.9781),
    "CENTRAL":                  (28.6440, 77.2090),
    "EAST":                     (28.6600, 77.2900),
    "NEW DELHI":                (28.6139, 77.2090),
    "NORTH":                    (28.7041, 77.1025),
    "NORTH EAST":               (28.6837, 77.3050),
    "NORTH-EAST":               (28.6837, 77.3050),
    "NORTH WEST":               (28.7218, 77.0960),
    "NORTH-WEST":               (28.7218, 77.0960),
    "SOUTH":                    (28.5245, 77.1855),
    "SOUTH EAST":               (28.5450, 77.2810),
    "SOUTH-EAST":               (28.5450, 77.2810),
    "SOUTH WEST":               (28.5890, 77.0610),
    "SOUTH-WEST":               (28.5890, 77.0610),
    "OUTER":                    (28.7300, 77.1200),
    "WEST":                     (28.6520, 77.0580),
    "CHENNAI":                  (13.0827, 80.2707),
    "CHENNAI RLY.":             (13.0827, 80.2707),
    "KANCHEEPURAM":             (12.8333, 79.7000),
    "TIRUVALLUR":               (13.1433, 79.9083),
}

ZONES_PER_DISTRICT = 4


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names. Specific patterns checked first."""
    rename_map: dict[str, str] = {}
    used_canonical: set[str] = set()

    for col in df.columns:
        col_clean = col.strip().lower()
        for pattern, canonical in COLUMN_PATTERNS:
            if re.search(pattern, col_clean):
                # Avoid clobbering already-mapped canonical names with a later column
                if canonical not in used_canonical:
                    rename_map[col] = canonical
                    used_canonical.add(canonical)
                break

    return df.rename(columns=rename_map)


def _load_single_file(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Cannot decode {path}")

    df = _normalise_columns(df)
    df.columns = [c.strip() for c in df.columns]

    # Ensure core columns exist
    required = {"state", "district", "year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path.name}. Columns found: {list(df.columns)[:10]}")

    # Coerce all crime columns to numeric
    for col in df.columns:
        if col not in ("state", "district", "year"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["state"]    = df["state"].astype(str).str.upper().str.strip()
    df["district"] = df["district"].astype(str).str.upper().str.strip()
    df["year"]     = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    return df


def load_ncrb_raw(data_dir: str = "backend/data/raw/ncrb") -> pd.DataFrame:
    """
    Recursively discover and load all IPC district-wise CSV files.
    Handles nested subdirectory structure from the Kaggle zip.
    """
    root = Path(data_dir)
    files: list[Path] = list(root.rglob("01_District_wise_crimes_committed_IPC*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No NCRB IPC files found under {root} (searched recursively).\n"
            "Expected files matching: 01_District_wise_crimes_committed_IPC*.csv\n"
            "Download from https://www.kaggle.com/datasets/rajanand/crime-in-india"
        )

    dfs: list[pd.DataFrame] = []
    for f in sorted(files):
        print(f"  Loading {f.name}...")
        try:
            dfs.append(_load_single_file(f))
        except Exception as e:
            print(f"  Warning: skipping {f.name}: {e}")

    if not dfs:
        raise ValueError("All NCRB files failed to load.")

    combined = pd.concat(dfs, ignore_index=True)
    # Deduplicate: same district+year may appear in overlapping files
    combined = combined.sort_values(["district", "year"])
    combined = combined.drop_duplicates(subset=["state", "district", "year"], keep="last")
    combined = combined.reset_index(drop=True)

    print(f"Loaded {len(combined):,} district-year records from {len(dfs)} file(s).")
    return combined


def filter_city_districts(df: pd.DataFrame, city: str) -> pd.DataFrame:
    targets = {d.upper() for d in CITY_DISTRICTS.get(city, [])}
    result = df[df["district"].isin(targets)].copy()
    if result.empty:
        print(f"  Warning: No records matched for city='{city}'.")
        print(f"  Districts searched: {targets}")
        available = df["district"].unique()[:20]
        print(f"  Sample available districts: {list(available)}")
    return result


# ---------------------------------------------------------------------------
# District → hourly zone records
# ---------------------------------------------------------------------------

HOUR_WEIGHTS: dict[str, list[float]] = {
    "vehicle_theft":  [0.5,0.4,0.3,0.3,0.3,0.4,0.8,1.5,1.8,1.6,1.4,1.2,
                       1.3,1.2,1.1,1.0,1.1,1.3,1.8,2.5,3.0,2.8,2.0,1.0],
    "burglary":       [2.0,2.5,2.5,2.2,1.5,0.8,0.4,0.3,0.3,0.4,0.6,0.8,
                       0.8,0.7,0.7,0.8,1.0,1.2,1.5,1.8,2.0,2.0,2.0,2.0],
    "robbery":        [1.5,1.5,1.2,0.8,0.5,0.4,0.5,0.8,1.0,1.0,0.9,0.9,
                       1.0,0.9,0.9,1.0,1.2,1.5,2.0,2.5,2.8,2.5,2.0,1.8],
    "pickpocketing":  [0.2,0.1,0.1,0.1,0.2,0.5,1.5,3.0,3.5,3.0,2.5,2.5,
                       2.0,2.0,2.0,2.5,3.0,3.5,3.0,2.0,1.5,1.0,0.5,0.3],
    "assault":        [1.0,0.8,0.7,0.6,0.5,0.5,0.6,0.8,0.9,0.9,1.0,1.1,
                       1.2,1.1,1.1,1.2,1.5,1.8,2.2,2.8,3.0,2.8,2.2,1.5],
    "dacoity":        [1.5,2.0,2.2,2.0,1.5,0.8,0.5,0.4,0.4,0.5,0.6,0.8,
                       0.8,0.8,0.9,1.0,1.2,1.5,1.8,2.0,2.0,1.8,1.5,1.5],
    "cyber_fraud":    [0.3,0.2,0.2,0.2,0.3,0.5,1.0,2.5,3.5,3.8,3.5,3.0,
                       2.5,3.0,3.5,3.5,3.0,2.5,2.0,1.5,1.0,0.7,0.5,0.4],
    "other":          [1.0]*24,
}

CITY_TEMP: dict[str, tuple[float, float]] = {
    "Bengaluru": (23, 4), "Hyderabad": (27, 6),
    "Mumbai": (28, 3),    "Delhi": (25, 9), "Chennai": (29, 3),
}
MONSOON_MONTHS = {6, 7, 8, 9}


def _sample_hour(crime_type: str, rng: np.random.Generator) -> int:
    w = np.array(HOUR_WEIGHTS.get(crime_type, HOUR_WEIGHTS["other"]), dtype=float)
    w /= w.sum()
    return int(rng.choice(24, p=w))


def _sample_temp(city: str, month: int, rng: np.random.Generator) -> float:
    mean, std = CITY_TEMP.get(city, (27, 5))
    seasonal = -4 if month in {12, 1, 2} else (2 if month in {4, 5} else 0)
    return float(rng.normal(mean + seasonal, std))


def _sample_precip(month: int, rng: np.random.Generator) -> float:
    if month in MONSOON_MONTHS:
        return float(max(0, rng.normal(4, 6))) if rng.random() < 0.4 else 0.0
    return float(max(0, rng.normal(0.5, 1))) if rng.random() < 0.1 else 0.0


def district_to_hourly_zones(
    df: pd.DataFrame,
    city: str,
    zones_per_district: int = ZONES_PER_DISTRICT,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    Convert annual district-level crime counts to simulated hourly zone records.
    Each crime in the annual count is distributed to a random hour using
    empirically calibrated hour-of-day weights.
    """
    rng = np.random.default_rng(seed=rng_seed)
    records: list[dict] = []

    # Only use crime columns that exist in this dataframe
    available_crime_cols = {col: ctype for col, ctype in CRIME_TO_TYPE.items() if col in df.columns}

    if not available_crime_cols:
        print(f"  Warning: No crime columns found for {city}. Available columns: {list(df.columns)}")
        return pd.DataFrame()

    print(f"  Crime columns found: {list(available_crime_cols.keys())}")

    for _, row in df.iterrows():
        district = str(row["district"])
        year = int(row["year"])
        base_lat, base_lon = DISTRICT_CENTROIDS.get(district, (20.0, 78.0))

        # Dirichlet split: assign each zone a share of district incidents
        zone_shares = rng.dirichlet(np.ones(zones_per_district) * 2)

        for zone_i in range(1, zones_per_district + 1):
            zone_share = zone_shares[zone_i - 1]
            lat = base_lat + rng.uniform(-0.04, 0.04)
            lon = base_lon + rng.uniform(-0.04, 0.04)
            zone_id = f"{city[:3].upper()}_{district[:6].replace('.','').replace(' ','_')}_{zone_i}"

            for crime_col, crime_type in available_crime_cols.items():
                raw_val = row.get(crime_col, 0)
                if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
                    continue
                annual_count = float(raw_val) * zone_share
                if annual_count < 0.5:
                    continue

                for _ in range(max(1, int(annual_count))):
                    hour = _sample_hour(crime_type, rng)
                    month = int(rng.integers(1, 13))
                    day = int(rng.integers(1, 29))
                    weekday = int(rng.integers(0, 7))
                    temp = _sample_temp(city, month, rng)
                    precip = _sample_precip(month, rng)

                    records.append({
                        "city": city,
                        "zone_id": zone_id,
                        "timestamp": f"{year}-{month:02d}-{day:02d}T{hour:02d}:00:00",
                        "year": year,
                        "month": month,
                        "day": day,
                        "hour": hour,
                        "weekday": weekday,
                        "is_weekend": int(weekday >= 5),
                        "crime_type": crime_type,
                        "district": district,
                        "lat": round(float(lat), 6),
                        "lon": round(float(lon), 6),
                        "temperature_c": round(float(temp), 1),
                        "precipitation_mm": round(float(precip), 2),
                        "wind_speed_kmh": round(float(abs(rng.normal(12, 5))), 1),
                        "is_rainy": int(precip > 1),
                    })

    return pd.DataFrame(records)


def build_zones_from_ncrb(city: str, df_city: pd.DataFrame) -> pd.DataFrame:
    """Build zone metadata from NCRB district data (with heuristic POI counts)."""
    rng = np.random.default_rng(seed=hash(city) % 2**32)
    districts = df_city["district"].unique()
    zones: list[dict] = []

    for district in districts:
        base_lat, base_lon = DISTRICT_CENTROIDS.get(district, (20.0, 78.0))
        for i in range(1, ZONES_PER_DISTRICT + 1):
            zone_id = f"{city[:3].upper()}_{district[:6].replace('.','').replace(' ','_')}_{i}"
            zones.append({
                "zone_id": zone_id,
                "city": city,
                "district": district,
                "zone_type": rng.choice(["commercial", "residential", "transit", "mixed"]),
                "lat": round(float(base_lat + rng.uniform(-0.04, 0.04)), 6),
                "lon": round(float(base_lon + rng.uniform(-0.04, 0.04)), 6),
                "population_density": int(rng.integers(5000, 35000)),
                "bar_count_500m":           int(rng.integers(0, 12)),
                "atm_count_500m":           int(rng.integers(2, 18)),
                "market_count_500m":        int(rng.integers(2, 25)),
                "bus_stop_count_500m":      int(rng.integers(1, 15)),
                "nearest_police_station_km": round(float(rng.uniform(0.2, 4.0)), 2),
                "road_density":             round(float(rng.uniform(0.2, 1.0)), 2),
                "lighting_score":           round(float(rng.uniform(0.3, 1.0)), 2),
                "is_hotspot": False,
            })

    return pd.DataFrame(zones)


# ---------------------------------------------------------------------------
# End-to-end entry point
# ---------------------------------------------------------------------------

def load_and_prepare(
    data_dir: str = "backend/data/raw/ncrb",
    output_dir: str = "backend/data/processed",
    cities: list[str] | None = None,
) -> dict:
    import json
    from pathlib import Path as P

    target_cities = cities or list(CITY_DISTRICTS.keys())
    P(output_dir).mkdir(parents=True, exist_ok=True)

    raw = load_ncrb_raw(data_dir)
    print(f"\nTotal records in raw data: {len(raw):,}")
    print(f"Years covered: {sorted(raw['year'].unique())}")

    all_records: list[pd.DataFrame] = []
    all_zones: list[pd.DataFrame] = []

    for city in target_cities:
        print(f"\nProcessing {city}...")
        city_df = filter_city_districts(raw, city)
        if city_df.empty:
            print(f"  Skipped.")
            continue

        years = sorted(city_df["year"].unique())
        print(f"  {len(city_df)} district-year rows | years: {years[0]}-{years[-1]}")

        records_df = district_to_hourly_zones(city_df, city)
        zones_df = build_zones_from_ncrb(city, city_df)

        if not records_df.empty:
            all_records.append(records_df)
            all_zones.append(zones_df)
            print(f"  Generated {len(records_df):,} hourly records across {len(zones_df)} zones.")

    if not all_records:
        raise ValueError("No records built from NCRB data. Check CITY_DISTRICTS mapping.")

    final_records = pd.concat(all_records, ignore_index=True)
    final_zones   = pd.concat(all_zones,   ignore_index=True)

    rec_path   = f"{output_dir}/crime_records.csv"
    zones_path = f"{output_dir}/zones.csv"
    summary    = {
        c: {
            "records": int(len(final_records[final_records["city"] == c])),
            "zones":   int(len(final_zones[final_zones["city"] == c])),
        }
        for c in target_cities if c in final_records["city"].values
    }

    final_records.to_csv(rec_path, index=False)
    final_zones.to_csv(zones_path, index=False)
    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTotal: {len(final_records):,} records | {len(final_zones)} zones")
    print(f"Saved to {output_dir}/")
    return {"records": rec_path, "zones": zones_path, "summary": summary}


if __name__ == "__main__":
    load_and_prepare()
