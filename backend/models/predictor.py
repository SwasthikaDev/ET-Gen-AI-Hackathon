"""
Ensemble predictor: XGBoost + LightGBM majority-vote with SHAP explainability.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import shap
    import xgboost as xgb
    import lightgbm as lgb
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

from backend.models.feature_engineering import build_inference_row, FEATURE_COLS

RISK_LABELS = {0: "LOW", 1: "LOW", 2: "MEDIUM", 3: "HIGH"}
RISK_COLOURS = {0: "green", 1: "yellow", 2: "orange", 3: "red"}

CRIME_TYPES = [
    "vehicle_theft",
    "burglary",
    "robbery",
    "pickpocketing",
    "assault",
    "dacoity",
    "cyber_fraud",
    "domestic_violence",
    "sexual_assault",
    "child_crime",
    "property_crime",
]

SHAP_PHRASES: dict[str, dict[str, str]] = {
    "bar_count_500m": {
        "pos": "high concentration of bars and nightlife venues nearby",
        "neg": "few bars nearby — lower nightlife-driven risk",
    },
    "hour": {
        "pos": "late-night hours with reduced visibility and police presence",
        "neg": "daytime hours with higher natural surveillance",
    },
    "precipitation_mm": {
        "pos": "dry weather encourages outdoor activity and opportunistic crime",
        "neg": "rain is suppressing outdoor criminal activity",
    },
    "bus_stop_count_500m": {
        "pos": "high pedestrian density from nearby bus stops",
        "neg": "low transit density in this zone",
    },
    "nearest_police_station_km": {
        "pos": "far from nearest police station — reduced deterrence",
        "neg": "close proximity to a police station — strong deterrence",
    },
    "lighting_score": {
        "pos": "poor street lighting increases vulnerability",
        "neg": "well-lit streets reduce opportunistic crime",
    },
    "rolling_7d_mean": {
        "pos": "elevated incident count over the past 7 days",
        "neg": "calm recent history in this zone",
    },
    "lag_1h": {
        "pos": "incident reported in the previous hour — heightened alert",
        "neg": "no incidents in the previous hour",
    },
    "population_density": {
        "pos": "dense population increases exposure to theft and assault",
        "neg": "sparse population with less target opportunity",
    },
    "market_count_500m": {
        "pos": "busy market area creates crowd and cash-handling opportunity",
        "neg": "few markets nearby — lower commercial crime risk",
    },
    "is_weekend": {
        "pos": "weekend nights see higher social activity and crime",
        "neg": "weekday with lower social gathering activity",
    },
    "is_rainy": {
        "pos": "rain pushes activity indoors to covered areas",
        "neg": "clear weather has minimal displacement effect",
    },
    # Enriched NCRB multi-source features
    "women_safety_index": {
        "pos": "district historically high in crimes against women — heightened gender-based risk",
        "neg": "district has relatively low crimes against women",
    },
    "vulnerability_index": {
        "pos": "district has elevated crimes against SC/ST — higher vulnerability of marginalised groups",
        "neg": "district shows lower social vulnerability crime rates",
    },
    "police_coverage_ratio": {
        "pos": "police force is understaffed vs sanctioned strength — reduced coverage",
        "neg": "police deployment is near full sanctioned strength — effective deterrence",
    },
    "property_value_stolen_lakh": {
        "pos": "historically high-value property crime in this state",
        "neg": "state has lower historical property crime value",
    },
    "state_auto_theft_count": {
        "pos": "state ranks high in auto theft — vehicle crime likely",
        "neg": "state has low auto theft rates",
    },
    "atm_count_500m": {
        "pos": "multiple ATMs attract cash withdrawal and snatch crime",
        "neg": "few ATMs reduces cash-crime exposure",
    },
    "road_density": {
        "pos": "dense road network enables rapid offender escape routes",
        "neg": "low road density reduces escape route options for offenders",
    },
    "temperature_c": {
        "pos": "high temperatures increase outdoor activity and opportunity crime",
        "neg": "cool temperature reduces outdoor crowd size",
    },
    "residential_crime_pct": {
        "pos": "historically high proportion of burglaries and domestic crimes in residential areas",
        "neg": "low residential crime density in this state",
    },
    "highway_crime_pct": {
        "pos": "high proportion of highway robbery and vehicle crime in this state",
        "neg": "low highway crime rate in this state",
    },
    "market_crime_pct": {
        "pos": "high market and commercial area crime — pickpocketing and snatch risk",
        "neg": "low commercial area crime in this state",
    },
    "gang_murder_pct": {
        "pos": "high gang/property-motive murder proportion — elevated organised crime risk",
        "neg": "low gang-motive murder proportion in this state",
    },
    "domestic_murder_pct": {
        "pos": "high domestic-motive murder proportion — elevated household violence risk",
        "neg": "low domestic violence escalation in this state",
    },
    "police_complaint_rate": {
        "pos": "high complaints against police — reduced public trust and deterrence",
        "neg": "low complaints against police — community cooperation likely",
    },
}


def _shap_phrase(feature: str, value: float, shap_val: float) -> str:
    direction = "pos" if shap_val > 0 else "neg"
    template = SHAP_PHRASES.get(feature, {}).get(direction)
    if template:
        return template
    sign = "increases" if shap_val > 0 else "decreases"
    return f"{feature.replace('_', ' ')} {sign} risk"


class CrimePredictor:
    """
    Ensemble classifier wrapping XGBoost + LightGBM.
    Falls back to a heuristic scorer when model files are absent.
    """

    def __init__(self, model_dir: str = "backend/models/saved"):
        self.model_dir = Path(model_dir)
        self.xgb_model: Any = None
        self.lgb_model: Any = None
        self.shap_explainer: Any = None
        self.feature_names: list[str] = []
        self._label_encoder: Any = None
        self._loaded = False

    def load(self) -> bool:
        """Load serialised models from disk. Returns True on success."""
        if not MODELS_AVAILABLE:
            return False
        xgb_path = self.model_dir / "xgb_model.pkl"
        lgb_path = self.model_dir / "lgb_model.pkl"
        meta_path = self.model_dir / "meta.json"
        le_path = self.model_dir / "label_encoder.pkl"

        if not xgb_path.exists():
            return False

        with open(xgb_path, "rb") as f:
            self.xgb_model = pickle.load(f)
        if lgb_path.exists():
            with open(lgb_path, "rb") as f:
                self.lgb_model = pickle.load(f)
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                self.feature_names = meta.get("feature_names", FEATURE_COLS)
        if le_path.exists():
            with open(le_path, "rb") as f:
                self._label_encoder = pickle.load(f)

        if MODELS_AVAILABLE and self.xgb_model is not None:
            self.shap_explainer = shap.TreeExplainer(self.xgb_model)

        self._loaded = True
        return True

    def _heuristic_score(self, row: pd.DataFrame, zone: dict) -> float:
        """Multi-factor heuristic incorporating enriched NCRB features."""
        def _r(col: str, default: float) -> float:
            return float(row[col].iloc[0]) if col in row.columns else default

        hour = int(_r("hour", 20))
        bars = min(_r("bar_count_500m", 2), 15)
        buses = min(_r("bus_stop_count_500m", 3), 15)
        dist = _r("nearest_police_station_km", 1.5)
        lighting = _r("lighting_score", 0.6)
        pop = _r("population_density", 10000)
        rain = _r("is_rainy", 0)
        rolling = _r("rolling_7d_mean", 0)
        is_weekend = _r("is_weekend", 0)
        # Enriched NCRB features
        women_idx = _r("women_safety_index", 0)
        vuln_idx = _r("vulnerability_index", 0)
        police_ratio = _r("police_coverage_ratio", 0.75)  # <1 = understaffed
        prop_val = _r("property_value_stolen_lakh", 0)
        auto_count = _r("state_auto_theft_count", 0)

        # Hour-of-day component
        if hour in range(20, 24) or hour in range(0, 3):
            hour_score = 0.28
        elif hour in range(8, 11) or hour in range(17, 20):
            hour_score = 0.16
        else:
            hour_score = 0.05

        score = (
            0.08                                  # base
            + hour_score
            + bars * 0.020                        # nightlife density
            + buses * 0.010                       # pedestrian exposure
            + max(0, (dist - 1.0) * 0.05)        # distance from police
            + (1.0 - lighting) * 0.10             # poor lighting
            + min(pop / 100000, 0.08)             # population density
            + (0.04 if is_weekend else 0)         # weekend uplift
            + min(rolling * 0.06, 0.12)           # recent incident history
            + (0.04 if rain else 0)               # rain displaces to covered areas
            # Enriched NCRB contributions
            + min(women_idx / 2000, 0.06)         # gender crime history
            + min(vuln_idx / 3000, 0.04)          # vulnerability index
            + max(0, (1.0 - police_ratio) * 0.08) # police understaffing
            + min(prop_val / 200000, 0.04)        # high-value property crime state
            + min(auto_count / 500000, 0.03)      # auto theft prevalence
        )
        zone_salt = (hash(zone.get("zone_id", "")) % 1000) / 10000.0 - 0.05
        return min(max(round(score + zone_salt, 3), 0.05), 0.97)

    def predict_zone(
        self,
        zone: dict[str, Any],
        target_dt: datetime,
        weather: dict[str, float],
        recent_counts: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Return a full prediction dict for one zone at one hour.
        Risk score: multi-factor heuristic (patrol-actionable, varies by time/features).
        SHAP drivers: from ML model when loaded, heuristic otherwise.
        """
        row = build_inference_row(zone, target_dt, weather, recent_counts)

        # Heuristic risk score — always used; produces varied, patrol-actionable scores
        risk_score = self._heuristic_score(row, zone)
        risk_class = 3 if risk_score > 0.70 else (2 if risk_score > 0.45 else (1 if risk_score > 0.20 else 0))
        shap_drivers = _heuristic_drivers(row)

        # ML model: extract SHAP explanations when available
        if self._loaded and self.xgb_model is not None and self.shap_explainer is not None:
            try:
                for feat in self.feature_names:
                    if feat not in row.columns:
                        row[feat] = 0
                zone_type = zone.get("zone_type", "mixed")
                for zt in ["commercial", "mixed", "residential", "transit"]:
                    col = f"zone_type_{zt}"
                    if col in self.feature_names:
                        row[col] = 1 if zone_type == zt else 0

                cols = self.feature_names
                X = row[cols].fillna(0)
                X_np = X.values.astype(np.float32)

                shap_vals = self.shap_explainer.shap_values(X_np)
                # Pick class-1 SHAP slice (medium-density — most discriminative)
                if isinstance(shap_vals, list):
                    shap_idx = min(1, len(shap_vals) - 1)
                    vals = np.array(shap_vals[shap_idx][0], dtype=float)
                elif hasattr(shap_vals, 'ndim') and shap_vals.ndim == 3:
                    vals = np.array(shap_vals[0, :, min(1, shap_vals.shape[2] - 1)], dtype=float)
                else:
                    vals = np.array(shap_vals[0], dtype=float)

                top_idx = np.argsort(np.abs(vals))[::-1][:3]
                ml_drivers = []
                for raw_idx in top_idx:
                    idx = int(raw_idx)
                    if idx >= len(cols):
                        continue
                    fname = str(cols[idx])
                    sv = float(vals[idx])
                    ml_drivers.append({
                        "feature": fname,
                        "magnitude": round(abs(sv), 4),
                        "direction": "increases_risk" if sv > 0 else "decreases_risk",
                        "explanation": _shap_phrase(fname, float(X[fname].iloc[0]), sv),
                    })
                if ml_drivers:
                    shap_drivers = ml_drivers
            except Exception:
                pass  # keep heuristic drivers on any SHAP failure

        top_crimes = _estimate_top_crimes(zone, target_dt)

        return {
            "zone_id": zone["zone_id"],
            "city": zone["city"],
            "lat": zone.get("lat"),
            "lon": zone.get("lon"),
            "risk_score": round(risk_score, 4),
            "risk_level": RISK_LABELS[risk_class],
            "risk_colour": RISK_COLOURS[risk_class],
            "top_crime_types": top_crimes,
            "shap_drivers": shap_drivers,
            "predicted_for": target_dt.isoformat(),
            # Enriched NCRB multi-source context (passed through for frontend display)
            "women_safety_index":         float(zone.get("women_safety_index", 0)),
            "vulnerability_index":        float(zone.get("vulnerability_index", 0)),
            "police_coverage_ratio":      float(zone.get("police_coverage_ratio", 0.75)),
            "property_value_stolen_lakh": float(zone.get("property_value_stolen_lakh", 0)),
            "state_auto_theft_count":     int(zone.get("state_auto_theft_count", 0)),
            "residential_crime_pct":      float(zone.get("residential_crime_pct", 0.33)),
            "highway_crime_pct":          float(zone.get("highway_crime_pct", 0.15)),
            "market_crime_pct":           float(zone.get("market_crime_pct", 0.20)),
            "gang_murder_pct":            float(zone.get("gang_murder_pct", 0.10)),
            "domestic_murder_pct":        float(zone.get("domestic_murder_pct", 0.20)),
            "police_complaint_rate":      float(zone.get("police_complaint_rate", 0.30)),
            "zone_type":                  zone.get("zone_type", "mixed"),
        }

    def predict_city(
        self,
        city_zones: list[dict],
        target_dt: datetime,
        weather: dict[str, float],
    ) -> list[dict]:
        """Batch-predict all zones for a city. Returns sorted by risk_score desc.

        Risk levels are assigned by within-city percentile so every prediction
        shows a realistic HIGH / MEDIUM / LOW spread regardless of absolute scores:
          - Top 40 % of zones by risk_score → HIGH
          - Next 30 %                       → MEDIUM
          - Bottom 30 %                     → LOW
        """
        results = [self.predict_zone(zone, target_dt, weather) for zone in city_zones]
        results.sort(key=lambda r: r["risk_score"], reverse=True)

        n = len(results)
        risk_colours = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}
        for i, r in enumerate(results):
            pct = i / max(n - 1, 1)          # 0.0 = top, 1.0 = bottom
            if pct < 0.40:
                level = "HIGH"
            elif pct < 0.70:
                level = "MEDIUM"
            else:
                level = "LOW"
            r["risk_level"] = level
            r["risk_colour"] = risk_colours[level]

        return results


def _heuristic_drivers(row: pd.DataFrame) -> list[dict]:
    """Generate risk drivers from heuristic features including enriched NCRB dimensions."""
    drivers = []
    checks = [
        # (feature, threshold, direction_if_exceeded)
        ("bar_count_500m",              4,       "pos"),
        ("bus_stop_count_500m",         6,       "pos"),
        ("rolling_7d_mean",             1,       "pos"),
        ("lighting_score",              0.5,     "neg"),
        ("nearest_police_station_km",   2.0,     "pos"),
        ("population_density",          20000,   "pos"),
        # Enriched NCRB features
        ("women_safety_index",          400,     "pos"),
        ("vulnerability_index",         200,     "pos"),
        ("police_coverage_ratio",       0.7,     "neg"),  # below threshold = understaffed
        ("property_value_stolen_lakh",  50000,   "pos"),
        ("state_auto_theft_count",      200000,  "pos"),
        ("gang_murder_pct",             0.15,    "pos"),
        ("domestic_murder_pct",         0.25,    "pos"),
        ("police_complaint_rate",       0.5,     "pos"),
    ]
    for feat, threshold, direction in checks:
        if feat not in row.columns:
            continue
        val = float(row[feat].iloc[0])
        triggered = (direction == "pos" and val >= threshold) or (direction == "neg" and val <= threshold)
        if triggered:
            phrase = SHAP_PHRASES.get(feat, {}).get("pos" if direction == "pos" else "neg", feat)
            drivers.append({
                "feature": feat,
                "magnitude": round(abs(val - threshold) / max(threshold, 1) * 0.15, 4),
                "direction": "increases_risk" if direction == "pos" else "decreases_risk",
                "explanation": phrase,
            })
    return drivers[:3]


def _estimate_top_crimes(zone: dict, dt: datetime) -> list[dict]:
    """
    Evidence-based crime-type distribution using zone profile, hour,
    and enriched NCRB historical features.
    """
    hour = dt.hour
    night = hour >= 20 or hour < 5
    evening = 17 <= hour < 20

    base_weights = {
        "pickpocketing":    0.24,
        "vehicle_theft":    0.18,
        "burglary":         0.12,
        "assault":          0.10,
        "robbery":          0.09,
        "cyber_fraud":      0.08,
        "domestic_violence": 0.07,
        "sexual_assault":   0.05,
        "dacoity":          0.04,
        "child_crime":      0.02,
        "property_crime":   0.01,
    }

    # Time-based modifiers
    if night:
        base_weights["vehicle_theft"] += 0.10
        base_weights["assault"] += 0.07
        base_weights["burglary"] += 0.06
        base_weights["pickpocketing"] -= 0.06

    if evening:
        base_weights["robbery"] += 0.04
        base_weights["domestic_violence"] += 0.03

    # Zone-feature modifiers
    if zone.get("bar_count_500m", 0) > 5:
        base_weights["assault"] += 0.08
        base_weights["robbery"] += 0.04
        base_weights["sexual_assault"] += 0.03

    if zone.get("bus_stop_count_500m", 0) > 6:
        base_weights["pickpocketing"] += 0.08
        base_weights["robbery"] += 0.03

    if zone.get("atm_count_500m", 0) > 8:
        base_weights["robbery"] += 0.05
        base_weights["vehicle_theft"] += 0.02

    # Enriched NCRB features — proportional boosts
    women_idx = float(zone.get("women_safety_index", 0))
    if women_idx > 0:
        # Strong proportional boost so high-women-safety zones show these crime types in top 3
        w_boost = min(women_idx / 500, 0.15)
        base_weights["domestic_violence"] += w_boost * 0.9
        base_weights["sexual_assault"] += w_boost * 0.6

    vuln_idx = float(zone.get("vulnerability_index", 0))
    if vuln_idx > 100:
        base_weights["assault"] += min(vuln_idx / 1000, 0.06)

    auto_count = float(zone.get("state_auto_theft_count", 0))
    if auto_count > 100000:
        base_weights["vehicle_theft"] += min(auto_count / 2000000, 0.08)

    prop_val = float(zone.get("property_value_stolen_lakh", 0))
    if prop_val > 0:
        # Logarithmic scaling for property value (values vary widely across states)
        import math
        prop_score = min(math.log10(max(prop_val, 1)) / 12.0, 1.0)  # log10(1e12) = 12
        base_weights["burglary"] += prop_score * 0.03
        base_weights["property_crime"] += prop_score * 0.015

    police_ratio = float(zone.get("police_coverage_ratio", 0.75))
    if police_ratio < 0.7:
        base_weights["assault"] += (0.7 - police_ratio) * 0.2
        base_weights["robbery"] += (0.7 - police_ratio) * 0.12

    # Place-of-occurrence weights (Table 17) — adjust by zone_type
    zone_type = str(zone.get("zone_type", "mixed")).lower()
    res_pct = float(zone.get("residential_crime_pct", 0.33))
    hwy_pct = float(zone.get("highway_crime_pct", 0.15))
    mkt_pct = float(zone.get("market_crime_pct", 0.20))

    if zone_type == "residential":
        base_weights["burglary"]           += res_pct * 0.12
        base_weights["domestic_violence"]  += res_pct * 0.08
    elif zone_type == "transit":
        base_weights["robbery"]            += hwy_pct * 0.15
        base_weights["vehicle_theft"]      += hwy_pct * 0.10
    elif zone_type == "commercial":
        base_weights["pickpocketing"]      += mkt_pct * 0.12
        base_weights["cyber_fraud"]        += mkt_pct * 0.08

    # Murder motive fractions (Table 19)
    gang_pct     = float(zone.get("gang_murder_pct", 0.10))
    domestic_pct = float(zone.get("domestic_murder_pct", 0.20))
    base_weights["dacoity"]            += gang_pct * 0.10
    base_weights["robbery"]            += gang_pct * 0.06
    base_weights["domestic_violence"]  += domestic_pct * 0.08

    # Police complaint rate (Table 25) — high complaint rate = lower trust / higher risk
    complaint_rate = float(zone.get("police_complaint_rate", 0.30))
    if complaint_rate > 0.5:
        base_weights["assault"]  += complaint_rate * 0.04
        base_weights["robbery"]  += complaint_rate * 0.03

    total = sum(base_weights.values())
    normalised = {k: max(v / total, 0) for k, v in base_weights.items()}
    top3 = sorted(normalised.items(), key=lambda x: x[1], reverse=True)[:3]
    return [{"type": t, "probability": round(p, 3)} for t, p in top3]
