"""
GenAI briefing service.
Converts zone prediction output + SHAP explanations into
officer-ready shift briefs via GPT-4o.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

SHIFT_NAMES = {0: "Morning", 1: "Afternoon", 2: "Night"}

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
    "ta": "Tamil",
}

SYSTEM_PROMPT = """You are CrimeWatch AI, an intelligent police shift briefing assistant deployed across Indian cities.
You convert predictive crime model outputs — backed by 2.8 million real NCRB crime records (2001–2014), live weather from Open-Meteo, and multi-dimensional crime intelligence (district IPC crimes, crimes against women, crimes against SC/ST, crimes against children, police deployment strength, and property value stolen) — into clear, concise, actionable shift briefings for beat officers.

Rules:
- Always respond in English only.
- Be professional, direct, and specific. Officers need facts and concrete actions, not commentary.
- Keep the full briefing under 320 words.
- Structure:
    1. City-wide risk summary (1-2 sentences)
    2. Top 3 HIGH risk zones — specific patrol action for each (position, patrol route, timing)
    3. Notable patterns (women's safety alert if women_safety_index is elevated; police understaffing note if police_coverage_ratio < 0.7)
    4. Single-line resource reallocation recommendation
- When the women_safety_index or sexual_assault/domestic_violence crimes appear — note "Women Safety Alert" explicitly.
- When police_coverage_ratio < 0.7 — note "Understaffing Advisory".
- Reference the specific risk drivers to explain *why* a zone is high risk.
- Do NOT mention "SHAP", "XGBoost", "machine learning", "model", "algorithm", or any technical terms.
"""

USER_PROMPT = """Generate a {shift_name} shift briefing for {city}.

Current time: {timestamp}
Shift window: {shift_window}

Weather conditions:
- Temperature: {temperature_c}°C  |  Precipitation: {precipitation_mm}mm ({weather_summary})  |  Wind: {wind_kmh} km/h

HIGH RISK ZONES (sorted by risk score):
{high_risk_zones}

MEDIUM RISK ZONES:
{medium_risk_zones}

Context intelligence:
- Women Safety Index (district avg): {women_safety_note}
- Police Coverage Ratio (actual/sanctioned): {police_coverage_note}
- Dominant crime pattern this shift: {dominant_crime}
- 7-day baseline: {baseline_note}

Generate the shift briefing now. Be specific, be brief, be actionable.
"""


def _format_zone_for_prompt(zone: dict) -> str:
    drivers = zone.get("shap_drivers", [])
    reasons = "; ".join(d["explanation"] for d in drivers[:3]) if drivers else "elevated historical activity"
    crimes = ", ".join(
        f"{c['type'].replace('_', ' ')} ({c['probability']:.0%})"
        for c in zone.get("top_crime_types", [])[:3]
    )
    women_flag = ""
    if float(zone.get("women_safety_index", 0)) > 500:
        women_flag = " [WOMEN SAFETY ALERT]"
    police_flag = ""
    if float(zone.get("police_coverage_ratio", 1.0)) < 0.7:
        police_flag = " [UNDERSTAFFING]"
    return (
        f"- Zone {zone['zone_id']}{women_flag}{police_flag}: risk score {zone['risk_score']:.0%} | "
        f"likely crimes: {crimes} | "
        f"risk factors: {reasons}"
    )


def _weather_summary(precipitation_mm: float, temperature_c: float) -> str:
    if precipitation_mm > 5:
        return "heavy rain — activity may push indoors"
    if precipitation_mm > 1:
        return "light rain"
    if temperature_c > 35:
        return "hot and dry — high outdoor activity expected"
    return "clear — expect normal outdoor exposure"


def _baseline_note(high_zones: list[dict]) -> str:
    high_count = len(high_zones)
    if high_count > 5:
        return f"{high_count} zones at HIGH risk — above average for this shift. Recommend increased patrol density."
    if high_count > 2:
        return f"{high_count} zones at HIGH risk — typical for this shift window."
    return "Risk distribution is below average for this shift. Standard patrol recommended."


class BriefingService:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client: Any = None

    def _get_client(self):
        if not OPENAI_AVAILABLE:
            return None
        if not self._client and self.api_key:
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def generate(
        self,
        city: str,
        zones: list[dict],
        target_dt: datetime,
        weather: dict[str, float],
        language: str = "en",
    ) -> dict[str, str]:
        """
        Generate a shift briefing.
        Returns {"text": "...", "language": "en", "city": "...", "shift": "..."}
        """
        hour = target_dt.hour
        shift_slot = 0 if 6 <= hour < 14 else (1 if 14 <= hour < 22 else 2)
        shift_name = SHIFT_NAMES[shift_slot]
        shift_windows = {0: "06:00–14:00", 1: "14:00–22:00", 2: "22:00–06:00"}
        shift_window = shift_windows[shift_slot]

        high_zones = [z for z in zones if z.get("risk_level") == "HIGH"]
        medium_zones = [z for z in zones if z.get("risk_level") == "MEDIUM"]

        high_text = "\n".join(_format_zone_for_prompt(z) for z in high_zones[:5]) or "None"
        medium_text = "\n".join(_format_zone_for_prompt(z) for z in medium_zones[:3]) or "None"

        precip = weather.get("precipitation_mm", 0)
        temp = weather.get("temperature_c", 28)
        wind = weather.get("wind_speed_kmh", 10)

        # Aggregate enriched features from high-risk zones
        all_zones_list = high_zones + medium_zones
        avg_women_idx = sum(float(z.get("women_safety_index", 0)) for z in all_zones_list) / max(len(all_zones_list), 1)
        avg_police_ratio = sum(float(z.get("police_coverage_ratio", 0.75)) for z in all_zones_list) / max(len(all_zones_list), 1)
        from collections import Counter
        crime_counter: Counter = Counter()
        for z in all_zones_list:
            for c in z.get("top_crime_types", []):
                crime_counter[c["type"]] += c["probability"]
        dominant_crime = crime_counter.most_common(1)[0][0].replace("_", " ") if crime_counter else "mixed"

        women_note = f"{avg_women_idx:.0f} — {'ELEVATED (Women Safety Alert active)' if avg_women_idx > 400 else 'within normal range'}"
        police_note = f"{avg_police_ratio:.2f} — {'BELOW 0.7 (Understaffing Advisory)' if avg_police_ratio < 0.7 else 'adequate'}"

        user_msg = USER_PROMPT.format(
            shift_name=shift_name,
            city=city,
            timestamp=target_dt.strftime("%d %b %Y %H:%M"),
            shift_window=shift_window,
            temperature_c=temp,
            precipitation_mm=precip,
            wind_kmh=wind,
            weather_summary=_weather_summary(precip, temp),
            high_risk_zones=high_text,
            medium_risk_zones=medium_text,
            baseline_note=_baseline_note(high_zones),
            women_safety_note=women_note,
            police_coverage_note=police_note,
            dominant_crime=dominant_crime,
        )

        client = self._get_client()
        if client is None:
            return {
                "text": _fallback_brief(city, high_zones, shift_name, target_dt),
                "language": language,
                "city": city,
                "shift": shift_name,
                "generated_by": "fallback",
            }

        system_msg = SYSTEM_PROMPT.format(language=LANGUAGE_NAMES.get(language, "English"))

        try:
            resp = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=600,
                temperature=0.4,
            )
            text = resp.choices[0].message.content.strip()
        except Exception as e:
            text = _fallback_brief(city, high_zones, shift_name, target_dt)
            text += f"\n\n[AI generation failed: {e}. Showing heuristic brief.]"

        return {
            "text": text,
            "language": language,
            "city": city,
            "shift": shift_name,
            "generated_by": "gpt-4o" if client else "fallback",
            "generated_at": target_dt.isoformat(),
        }


def _fallback_brief(city: str, high_zones: list[dict], shift: str, dt: datetime) -> str:
    """Rule-based brief (runs when OpenAI API key is not configured)."""
    lines = [f"**{city.upper()} {shift.upper()} SHIFT BRIEFING — {dt.strftime('%d %b %Y, %H:%M IST')}**\n"]

    if not high_zones:
        lines.append("City risk level is LOW across all monitored zones. Standard patrol is recommended.")
        return "\n".join(lines)

    lines.append(f"**City status: HIGH alert active — {len(high_zones)} zone(s) require priority patrol.**\n")

    for i, zone in enumerate(high_zones[:3], 1):
        crimes = ", ".join(c["type"].replace("_", " ") for c in zone.get("top_crime_types", [])[:2])
        drivers = "; ".join(d["explanation"] for d in zone.get("shap_drivers", [])[:2])
        # Enriched flags
        flags = []
        if float(zone.get("women_safety_index", 0)) > 400:
            flags.append("Women Safety Alert")
        if float(zone.get("police_coverage_ratio", 1.0)) < 0.7:
            flags.append("Understaffing Advisory")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        lines.append(
            f"**Priority {i} — Zone {zone['zone_id']}{flag_str}:** Risk {zone['risk_score']:.0%}. "
            f"Likely crime: {crimes}. Key factor: {drivers}. Deploy 1 constable now."
        )

    # Women safety advisory
    women_zones = [z for z in high_zones if float(z.get("women_safety_index", 0)) > 400]
    if women_zones:
        lines.append(
            f"\n**Women Safety Alert:** {len(women_zones)} zone(s) show elevated gender-based crime history from NCRB data. "
            "Assign female constables and increase visibility at transit points."
        )

    # Understaffing advisory
    understaffed = [z for z in high_zones if float(z.get("police_coverage_ratio", 1.0)) < 0.7]
    if understaffed:
        lines.append(
            f"\n**Understaffing Advisory:** Police actual strength is below 70% of sanctioned strength in this area. "
            "Request reinforcements from the nearest reserve unit."
        )

    lines.append("\n**Resource Recommendation:** Redeploy 2 units from GREEN zones to Priority 1 and Priority 2 zones for this shift window.")
    return "\n".join(lines)
