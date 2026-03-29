"""
APScheduler-based shift briefing scheduler.
Fires at 06:00, 14:00, and 22:00 for each city.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False

logger = logging.getLogger("scheduler")

SHIFT_HOURS = [6, 14, 22]
CITIES = ["Bengaluru", "Hyderabad", "Mumbai", "Delhi", "Chennai"]


async def run_city_briefing(city: str, hour: int):
    """Called by scheduler for each (city, shift) combination."""
    from backend.main import app_state  # late import to avoid circular

    logger.info(f"Scheduler triggered: {city} shift at {hour:02d}:00")

    try:
        predictor = app_state.get("predictor")
        briefing_svc = app_state.get("briefing_service")
        zones = app_state.get("zones_by_city", {}).get(city, [])

        if not predictor or not zones:
            logger.warning(f"No predictor or zones for {city}. Skipping.")
            return

        weather = await fetch_weather(city)
        target_dt = datetime.now().replace(hour=hour, minute=0, second=0, microsecond=0)
        predictions = predictor.predict_city(zones, target_dt, weather)

        briefing = await briefing_svc.generate(city, predictions, target_dt, weather, language="en")

        # Push to delivery layer
        await deliver_briefing(city, briefing, predictions)

        # Cache in app_state for dashboard
        app_state.setdefault("last_briefings", {})[city] = briefing
        app_state.setdefault("last_predictions", {})[city] = predictions

        logger.info(f"Briefing delivered for {city}")

    except Exception as e:
        logger.error(f"Briefing failed for {city}: {e}", exc_info=True)


async def fetch_weather(city: str) -> dict:
    """Fetch current weather using Open-Meteo (free, no API key needed)."""
    from backend.data.weather_fetcher import get_current_weather
    return await get_current_weather(city)


async def deliver_briefing(city: str, briefing: dict, predictions: list[dict]):
    """Route briefing to WhatsApp and push WS update."""
    from backend.services.whatsapp_service import WhatsAppService
    from backend.main import ws_broadcast

    wa = WhatsAppService()
    if wa.is_configured():
        await wa.send_briefing(city, briefing["text"])

    # Broadcast to connected dashboard WebSocket clients
    try:
        await ws_broadcast(city, {"type": "briefing_ready", "city": city, "shift": briefing.get("shift")})
    except Exception:
        pass


def create_scheduler() -> "AsyncIOScheduler | None":
    if not SCHEDULER_AVAILABLE:
        logger.warning("APScheduler not installed. Scheduling disabled.")
        return None

    scheduler = AsyncIOScheduler(timezone="Asia/Kolkata")

    for hour in SHIFT_HOURS:
        for city in CITIES:
            scheduler.add_job(
                run_city_briefing,
                CronTrigger(hour=hour, minute=0, timezone="Asia/Kolkata"),
                args=[city, hour],
                id=f"briefing_{city}_{hour}",
                replace_existing=True,
            )

    return scheduler
