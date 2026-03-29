"""
Model training script.
Usage:
  python -m backend.models.train --city all --synthetic
  python -m backend.models.train --city Bengaluru --data-path ./data/crime_records.csv
"""

from __future__ import annotations

import argparse
import json
import pickle as pickle_module
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

try:
    import xgboost as xgb
    import lightgbm as lgb
    from imblearn.over_sampling import SMOTE
    FULL_DEPS = True
except ImportError:
    FULL_DEPS = False
    print("Warning: xgboost/lightgbm/imbalanced-learn not installed. Install requirements.txt first.")

from backend.models.feature_engineering import build_feature_matrix, FEATURE_COLS
from backend.data.synthetic_generator import generate_records


def load_data(
    data_path: str | None,
    zones_path: str | None,
    use_synthetic: bool,
    synthetic_dir: str = "backend/data/synthetic",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if use_synthetic:
        synth_records = Path(synthetic_dir) / "crime_records.csv"
        synth_zones = Path(synthetic_dir) / "zones.csv"
        if not synth_records.exists():
            print("Synthetic data not found — generating now...")
            generate_records(output_dir=synthetic_dir)
        records_df = pd.read_csv(synth_records)
        zones_df = pd.read_csv(synth_zones)
    else:
        if not data_path or not zones_path:
            raise ValueError("Provide --data-path and --zones-path when not using synthetic data.")
        records_df = pd.read_csv(data_path)
        zones_df = pd.read_csv(zones_path)

    return records_df, zones_df


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    city_filter: str | None = None,
    output_dir: str = "backend/models/saved",
) -> dict:
    if not FULL_DEPS:
        print("Skipping training — dependencies not installed.")
        return {}

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    X_np = X.values.astype(np.float32)
    y_np = y.values

    # Re-encode labels to be contiguous 0..N-1 (handles missing classes)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_np)
    print(f"Classes present: {le.classes_} -> encoded as 0..{len(le.classes_)-1}")

    # Save label encoder alongside model
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/label_encoder.pkl", "wb") as f:
        pickle_module.dump(le, f)

    # Temporal split: last 20% as test
    split = int(len(X_np) * 0.8)
    X_train, X_test = X_np[:split], X_np[split:]
    y_train, y_test = y_encoded[:split], y_encoded[split:]

    # SMOTE for class balance
    smote = SMOTE(random_state=42, k_neighbors=3)
    try:
        X_train, y_train = smote.fit_resample(X_train, y_train)
    except ValueError:
        pass  # Too few samples for SMOTE — skip

    sample_weights = compute_sample_weight("balanced", y_train)

    # --- XGBoost ---
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # --- LightGBM ---
    print("Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        num_leaves=63,
        learning_rate=0.08,
        min_child_samples=10,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
    )

    # --- Evaluate ensemble ---
    xgb_pred = xgb_model.predict(X_test)
    lgb_pred = lgb_model.predict(X_test)
    ensemble_pred = np.where(xgb_pred == lgb_pred, xgb_pred, xgb_pred)  # XGB wins on disagree

    xgb_f1 = f1_score(y_test, xgb_pred, average="weighted", zero_division=0)
    lgb_f1 = f1_score(y_test, lgb_pred, average="weighted", zero_division=0)
    ens_f1 = f1_score(y_test, ensemble_pred, average="weighted", zero_division=0)

    print(f"\nResults:")
    print(f"  XGBoost   weighted F1: {xgb_f1:.4f}")
    print(f"  LightGBM  weighted F1: {lgb_f1:.4f}")
    print(f"  Ensemble  weighted F1: {ens_f1:.4f}")

    # Save
    with open(f"{output_dir}/xgb_model.pkl", "wb") as f:
        pickle_module.dump(xgb_model, f)
    with open(f"{output_dir}/lgb_model.pkl", "wb") as f:
        pickle_module.dump(lgb_model, f)

    feature_names = list(X.columns)
    meta = {
        "feature_names": feature_names,
        "city_filter": city_filter,
        "xgb_f1": xgb_f1,
        "lgb_f1": lgb_f1,
        "ensemble_f1": ens_f1,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "classes": [int(c) for c in le.classes_],
    }
    with open(f"{output_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModels saved to {output_dir}/")
    return meta


def main():
    parser = argparse.ArgumentParser(description="Train CrimeWatch AI models")
    parser.add_argument("--city", default="all", help="City name or 'all'")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--data-path", help="Path to crime_records.csv")
    parser.add_argument("--zones-path", help="Path to zones.csv")
    parser.add_argument("--output-dir", default="backend/models/saved")
    args = parser.parse_args()

    records_df, zones_df = load_data(
        data_path=args.data_path,
        zones_path=args.zones_path,
        use_synthetic=args.synthetic,
    )

    if args.city != "all":
        records_df = records_df[records_df["city"] == args.city]

    print(f"Building feature matrix ({len(records_df):,} raw records)...")
    X, y = build_feature_matrix(records_df, zones_df)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Class distribution:\n{y.value_counts().sort_index()}")

    train_and_evaluate(X, y, city_filter=args.city, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
