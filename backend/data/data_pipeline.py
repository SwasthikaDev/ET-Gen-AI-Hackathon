"""
Full data pipeline orchestration — MULTI-SOURCE NCRB EDITION.

Run this script to go from raw NCRB CSVs → trained model in one command:

  python -m backend.data.data_pipeline

Steps:
  1. Load ALL NCRB sources (IPC + Women + SC/ST + Children + Police strength +
     Property value + Auto theft) and build enriched zone profiles
  2. Enrich zones with OSM POI data (optional, takes ~10 min)
  3. Enrich crime records with real weather from Open-Meteo
  4. Save processed datasets
  5. Train model on 12 crime types including domestic violence, sexual assault,
     child crime, and property crime with 5 enriched features

Set --skip-osm to skip OSM enrichment (uses heuristic POI counts instead).
Set --skip-weather to skip weather API calls (uses NCRB synthetic weather).
Set --synthetic to bypass all external APIs and use fully synthetic data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROCESSED_DIR = "backend/data/processed"
SYNTHETIC_DIR = "backend/data/synthetic"
RAW_NCRB_DIR = "backend/data/raw/ncrb"


def run_pipeline(
    skip_osm: bool = True,
    skip_weather: bool = False,
    use_synthetic: bool = False,
    cities: list[str] | None = None,
    train: bool = True,
) -> dict:
    import pandas as pd

    if use_synthetic:
        print("=" * 60)
        print("MODE: Fully synthetic data (no external dependencies)")
        print("=" * 60)
        from backend.data.synthetic_generator import generate_records
        paths = generate_records(output_dir=SYNTHETIC_DIR)
        data_dir = SYNTHETIC_DIR
    else:
        print("=" * 60)
        print("MODE: Real NCRB data")
        print("=" * 60)

        # Step 1: NCRB multi-source enriched pipeline
        print("\n[1/4] Loading ALL NCRB sources (IPC + Women + SC + Children + Police + Property + Auto)...")
        from backend.data.ncrb_enriched_loader import load_and_prepare_enriched
        paths = load_and_prepare_enriched(
            data_dir=RAW_NCRB_DIR,
            output_dir=PROCESSED_DIR,
            cities=cities,
        )
        data_dir = PROCESSED_DIR

        # Step 2: OSM enrichment
        if not skip_osm:
            print("\n[2/4] Enriching zones with OSM POI data...")
            print("  This queries Overpass API — may take several minutes.")
            from backend.data.osm_fetcher import enrich_and_save
            osm_path = enrich_and_save(
                zones_csv=paths["zones"],
                output_csv=f"{PROCESSED_DIR}/zones_osm.csv",
            )
            paths["zones"] = osm_path
        else:
            print("\n[2/4] Skipping OSM enrichment (--skip-osm).")

        # Step 3: Weather enrichment
        if not skip_weather:
            print("\n[3/4] Enriching crime records with real weather data...")
            from backend.data.weather_fetcher import enrich_records_with_weather
            records_df = pd.read_csv(paths["records"])
            start = str(records_df["year"].min()) + "-01-01"
            end = str(records_df["year"].max()) + "-12-31"
            enriched = enrich_records_with_weather(records_df, start_date=start, end_date=end)
            enriched_path = f"{data_dir}/crime_records_weather.csv"
            enriched.to_csv(enriched_path, index=False)
            paths["records"] = enriched_path
            print(f"  Weather-enriched records saved: {enriched_path}")
        else:
            print("\n[3/4] Skipping weather enrichment (--skip-weather).")

        print("\n[4/4] Data pipeline complete.")

    # Step 4: Train
    if train:
        print("\n[5/5] Training model...")
        from backend.models.train import train_and_evaluate, load_data
        from backend.models.feature_engineering import build_feature_matrix
        import pandas as pd

        records_df = pd.read_csv(paths["records"])
        zones_df = pd.read_csv(paths["zones"])

        print(f"Building feature matrix ({len(records_df):,} records)...")
        X, y = build_feature_matrix(records_df, zones_df)
        print(f"Feature matrix: {X.shape}")

        meta = train_and_evaluate(X, y, output_dir="backend/models/saved")
        paths["model_meta"] = meta
    else:
        print("\nSkipping training (--no-train).")

    print("\nPipeline finished successfully.")
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    return paths


def main():
    parser = argparse.ArgumentParser(description="CrimeWatch AI data pipeline")
    parser.add_argument("--skip-osm", action="store_true", default=True,
                        help="Skip OSM POI enrichment (recommended for first run)")
    parser.add_argument("--with-osm", dest="skip_osm", action="store_false",
                        help="Enable OSM POI enrichment")
    parser.add_argument("--skip-weather", action="store_true", default=False,
                        help="Skip Open-Meteo weather enrichment")
    parser.add_argument("--synthetic", action="store_true", default=False,
                        help="Use fully synthetic data instead of NCRB")
    parser.add_argument("--cities", nargs="+", default=None,
                        help="Limit to specific cities e.g. --cities Bengaluru Mumbai")
    parser.add_argument("--no-train", action="store_true", default=False,
                        help="Skip model training step")
    args = parser.parse_args()

    ncrb_exists = Path(RAW_NCRB_DIR).exists() and any(Path(RAW_NCRB_DIR).glob("*.csv"))

    if not args.synthetic and not ncrb_exists:
        print(f"\nNRCB data not found in {RAW_NCRB_DIR}/")
        print("Options:")
        print("  A) Download NCRB data from:")
        print("     https://www.kaggle.com/datasets/rajanand/crime-in-india")
        print(f"     Place CSV files in: {RAW_NCRB_DIR}/")
        print("  B) Run with synthetic data (no download needed):")
        print("     python -m backend.data.data_pipeline --synthetic")
        print()
        choice = input("Use synthetic data for now? [Y/n]: ").strip().lower()
        if choice in ("", "y", "yes"):
            args.synthetic = True
        else:
            sys.exit(1)

    run_pipeline(
        skip_osm=args.skip_osm,
        skip_weather=args.skip_weather,
        use_synthetic=args.synthetic,
        cities=args.cities,
        train=not args.no_train,
    )


if __name__ == "__main__":
    main()
