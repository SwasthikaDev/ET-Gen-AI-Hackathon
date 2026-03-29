"""
CrimeWatch AI — FastAPI backend.
Provides REST + WebSocket API for predictions, briefings, and feedback.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.models.predictor import CrimePredictor
from backend.services.briefing_service import BriefingService
from backend.services.scheduler import create_scheduler
from backend.data.weather_fetcher import get_current_weather as fetch_weather
from backend.services.news_intelligence import get_city_news_signals

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crimewatch")

# ---------------------------------------------------------------------------
# Shared application state
# ---------------------------------------------------------------------------
app_state: dict[str, Any] = {}

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, ws: WebSocket, city: str):
        await ws.accept()
        self._connections.setdefault(city, []).append(ws)

    def disconnect(self, ws: WebSocket, city: str):
        conns = self._connections.get(city, [])
        if ws in conns:
            conns.remove(ws)

    async def broadcast(self, city: str, message: dict):
        dead = []
        for ws in self._connections.get(city, []):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws, city)


manager = ConnectionManager()


async def ws_broadcast(city: str, message: dict):
    await manager.broadcast(city, message)


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting CrimeWatch AI...")

    predictor = CrimePredictor()
    loaded = predictor.load()
    if not loaded:
        logger.warning("No trained model found — using heuristic scorer. Run `python -m backend.models.train --synthetic` to train.")

    briefing_svc = BriefingService()
    zones_by_city = _load_zones()

    app_state.update(
        predictor=predictor,
        briefing_service=briefing_svc,
        zones_by_city=zones_by_city,
        last_predictions={},
        last_briefings={},
    )

    scheduler = create_scheduler()
    if scheduler:
        scheduler.start()
        app_state["scheduler"] = scheduler

    # Kick off initial predictions for all cities (async, non-blocking)
    asyncio.create_task(_bootstrap_predictions())

    yield

    if scheduler := app_state.get("scheduler"):
        scheduler.shutdown(wait=False)
    logger.info("CrimeWatch AI shut down.")


def _load_zones() -> dict[str, list[dict]]:
    # Prefer OSM-enriched zones, then processed, then synthetic
    candidates = [
        Path("backend/data/processed/zones_osm.csv"),
        Path("backend/data/processed/zones.csv"),
        Path("backend/data/synthetic/zones.csv"),
    ]
    zones_path = next((p for p in candidates if p.exists()), None)
    if not zones_path:
        logger.warning("No zones file found. Run: python -m backend.data.data_pipeline")
        return {}
    df = pd.read_csv(zones_path)
    # Deduplicate: keep first occurrence per (zone_id, city) pair
    before = len(df)
    df = df.drop_duplicates(subset=["zone_id", "city"], keep="first").reset_index(drop=True)
    if len(df) < before:
        logger.info(f"Deduplicated zones: {before} → {len(df)} rows")
    result: dict[str, list[dict]] = {}
    for city, grp in df.groupby("city"):
        result[str(city)] = grp.to_dict("records")
    logger.info(f"Loaded zones from {zones_path} for cities: {list(result.keys())}")
    return result


async def _bootstrap_predictions():
    """Run initial predictions for all cities at startup."""
    await asyncio.sleep(2)  # let server finish starting
    predictor: CrimePredictor = app_state.get("predictor")
    zones_by_city: dict = app_state.get("zones_by_city", {})

    for city, zones in zones_by_city.items():
        try:
            weather = await fetch_weather(city)
            preds = predictor.predict_city(zones, datetime.now(), weather)
            app_state["last_predictions"][city] = preds
        except Exception as e:
            logger.error(f"Bootstrap prediction failed for {city}: {e}")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CrimeWatch AI",
    description="Predictive Crime Hotspot Intelligence System for Indian Cities",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    city: str
    target_hour: str | None = None
    use_live_weather: bool = True


class BriefingRequest(BaseModel):
    city: str
    language: str = "en"
    shift: str | None = None


class FeedbackRequest(BaseModel):
    zone_id: str
    shift_date: str
    shift: str
    incident_confirmed: bool
    crime_type: str | None = None
    officer_id: str | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "cities": list(app_state.get("zones_by_city", {}).keys())}


@app.post("/api/v1/predict")
async def predict(req: PredictRequest):
    predictor: CrimePredictor = app_state.get("predictor")
    zones_by_city = app_state.get("zones_by_city", {})

    zones = zones_by_city.get(req.city)
    if not zones:
        raise HTTPException(404, f"City '{req.city}' not found. Available: {list(zones_by_city.keys())}")

    target_dt = datetime.fromisoformat(req.target_hour) if req.target_hour else datetime.now()

    if req.use_live_weather:
        weather = await fetch_weather(req.city)
    else:
        weather = {"temperature_c": 28.0, "precipitation_mm": 0.0, "wind_speed_kmh": 10.0}

    predictions = predictor.predict_city(zones, target_dt, weather)
    app_state["last_predictions"][req.city] = predictions

    # Broadcast WS update
    await manager.broadcast(req.city, {"event": "prediction_refresh", "city": req.city})

    return {
        "city": req.city,
        "predicted_at": datetime.now().isoformat(),
        "target_hour": target_dt.isoformat(),
        "weather": weather,
        "zones": predictions,
        "summary": _zone_summary(predictions),
    }


@app.get("/api/v1/zones/{city}")
async def get_zones(city: str):
    preds = app_state.get("last_predictions", {}).get(city)
    if preds is None:
        raise HTTPException(404, f"No predictions for '{city}'. Call POST /api/v1/predict first.")
    return {"city": city, "zones": preds, "count": len(preds)}


@app.post("/api/v1/briefing")
async def generate_briefing(req: BriefingRequest, background_tasks: BackgroundTasks):
    briefing_svc: BriefingService = app_state.get("briefing_service")
    zones = app_state.get("last_predictions", {}).get(req.city)

    if not zones:
        raise HTTPException(404, f"No predictions cached for '{req.city}'. Run prediction first.")

    weather = await fetch_weather(req.city)
    briefing = await briefing_svc.generate(
        city=req.city,
        zones=zones,
        target_dt=datetime.now(),
        weather=weather,
        language=req.language,
    )
    app_state.setdefault("last_briefings", {})[req.city] = briefing
    return briefing


@app.get("/api/v1/briefing/{city}")
async def get_briefing(city: str, language: str = "en"):
    cached = app_state.get("last_briefings", {}).get(city)
    if not cached:
        raise HTTPException(404, f"No briefing cached for '{city}'. POST /api/v1/briefing to generate one.")
    return cached


@app.post("/api/v1/feedback")
async def submit_feedback(req: FeedbackRequest):
    """
    Officer confirms or denies a predicted incident.
    In production: write to DB and trigger online model score correction.
    """
    logger.info(
        f"Feedback received — zone={req.zone_id} confirmed={req.incident_confirmed} "
        f"crime={req.crime_type} officer={req.officer_id}"
    )
    # TODO: write to database, trigger drift metric update
    return {"status": "recorded", "zone_id": req.zone_id}


@app.get("/api/v1/weather/{city}")
async def get_weather(city: str):
    """Live weather for a city from Open-Meteo (free, no API key)."""
    weather = await fetch_weather(city)
    return {"city": city, "weather": weather, "fetched_at": datetime.now().isoformat()}


@app.get("/api/v1/news/{city}")
async def get_city_news(city: str):
    """
    NewsWatch Intelligence — live crime news signals for a city.
    Fetches ET, NDTV, TOI, HT RSS feeds → GPT-4o / regex extraction.
    Cached 20 min to avoid hammering news APIs.
    """
    signals = await get_city_news_signals(city)
    high = sum(1 for s in signals if s["severity"] == "HIGH")
    return {
        "city": city,
        "signal_count": len(signals),
        "high_severity": high,
        "signals": signals,
        "fetched_at": datetime.now().isoformat(),
    }


@app.get("/api/v1/forecast/{city}")
async def get_risk_forecast(city: str):
    """
    24-hour risk forecast for a city.
    Runs heuristic predictor at 6 forecast windows (current, +3h, +6h,
    +9h, +12h, +18h, +24h) and returns aggregate HIGH zone count + top zone.
    """
    predictor: CrimePredictor = app_state.get("predictor")
    zones_by_city = app_state.get("zones_by_city", {})
    zones = zones_by_city.get(city)
    if not zones:
        raise HTTPException(404, f"City '{city}' not found.")

    weather = await fetch_weather(city)
    now = datetime.now()
    offsets_h = [0, 3, 6, 9, 12, 18, 24]
    windows = []
    for offset in offsets_h:
        from datetime import timedelta
        target = now.replace(minute=0, second=0, microsecond=0)
        target = target + timedelta(hours=offset)
        preds = predictor.predict_city(zones, target, weather)
        high = sum(1 for p in preds if p["risk_level"] == "HIGH")
        med = sum(1 for p in preds if p["risk_level"] == "MEDIUM")
        low = sum(1 for p in preds if p["risk_level"] == "LOW")
        avg_score = round(sum(p["risk_score"] for p in preds) / max(len(preds), 1), 3)
        top_zone = preds[0]["zone_id"] if preds else ""
        windows.append({
            "hour_offset": offset,
            "timestamp": target.isoformat(),
            "label": target.strftime("%I %p"),
            "high": high,
            "medium": med,
            "low": low,
            "avg_risk_score": avg_score,
            "top_zone": top_zone,
        })

    peak = max(windows, key=lambda w: w["high"])
    return {
        "city": city,
        "total_zones": len(zones),
        "windows": windows,
        "peak_hour": peak["label"],
        "peak_high_zones": peak["high"],
        "generated_at": now.isoformat(),
    }


@app.get("/api/v1/cities")
async def list_cities():
    zones_by_city = app_state.get("zones_by_city", {})
    return {
        "cities": [
            {"name": city, "zone_count": len(zones)}
            for city, zones in zones_by_city.items()
        ]
    }


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------
@app.websocket("/ws/live/{city}")
async def ws_live(websocket: WebSocket, city: str):
    await manager.connect(websocket, city)
    try:
        # Send immediate snapshot
        preds = app_state.get("last_predictions", {}).get(city, [])
        await websocket.send_json({"event": "snapshot", "city": city, "zones": preds})

        while True:
            await asyncio.sleep(30)
            # Refresh predictions and push update
            zones = app_state.get("zones_by_city", {}).get(city, [])
            if zones:
                predictor: CrimePredictor = app_state["predictor"]
                weather = await fetch_weather(city)
                preds = predictor.predict_city(zones, datetime.now(), weather)
                app_state["last_predictions"][city] = preds
                await websocket.send_json(
                    {"event": "zone_update", "city": city, "zones": preds, "timestamp": datetime.now().isoformat()}
                )
    except WebSocketDisconnect:
        manager.disconnect(websocket, city)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _zone_summary(predictions: list[dict]) -> dict:
    levels = [p["risk_level"] for p in predictions]
    return {
        "total_zones": len(predictions),
        "high": levels.count("HIGH"),
        "medium": levels.count("MEDIUM"),
        "low": levels.count("LOW"),
        "top_zone": predictions[0]["zone_id"] if predictions else None,
    }
