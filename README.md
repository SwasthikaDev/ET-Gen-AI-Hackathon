# CrimeWatch AI
## Predictive Crime Hotspot Intelligence for Indian Cities
**Team Calyirex · ET Gen AI Hackathon 2026**

---

> India recorded **58.24 lakh IPC crimes in 2022** (NCRB). Every crime is recorded. None of that intelligence reaches the beat officer who could have prevented it — or the journalist who needs to report the pattern.
>
> CrimeWatch AI closes that gap — fusing **11 NCRB datasets** with live weather, real-time news from **Economic Times, NDTV, TOI, HT & The Hindu**, and a 51-feature ML ensemble to generate shift-ready, zone-level crime intelligence for India's top 5 cities.

---

## Live Demo

| Service | URL |
|---------|-----|
| Dashboard | http://localhost:3000 |
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

---

## What Makes This Win-Worthy

| Judging Criterion | What We Built |
|---|---|
| **Innovation & Creativity (20%)** | First system to fuse NCRB government data + live news RSS (ET, NDTV, TOI) + weather + GPT-4o into a real-time crime intelligence map |
| **Technical Implementation (20%)** | XGBoost + LightGBM ensemble (F1=0.851) · 51-feature pipeline · SHAP explainability · GPT-4o NL briefings · WebSocket live push · ESRI satellite map |
| **Practical Impact (20%)** | Women Safety Alerts · Police Understaffing Advisories · 10 crime types · Shift briefings for real officers |
| **User Experience (20%)** | Satellite map with risk circles · Tabbed panel (Zones / Briefing / News / Forecast) · 5-min auto-refresh countdown · Breaking news ticker |
| **Pitch Quality (20%)** | Live prototype · Real NCRB data · Open-Meteo live weather · ET NewsWatch intelligence |

---

## Complete Intelligence Pipeline

```
┌─────────────────────────── DATA SOURCES ───────────────────────────────┐
│                                                                         │
│  11 NCRB Datasets (2.83M records, 2001–2014)                           │
│   ├─ IPC District Crimes (base)                                         │
│   ├─ Crimes Against Women → Women Safety Index                          │
│   ├─ Crimes Against SC → Vulnerability Index                            │
│   ├─ Crimes Against ST → Extends Vulnerability Index                   │
│   ├─ Crimes Against Children → Child Crime Risk                         │
│   ├─ Police Strength (Actual vs Sanctioned) → Understaffing Ratio      │
│   ├─ Property Stolen & Value → Property Crime Magnitude                 │
│   ├─ Auto Theft Detail → Vehicle Crime Probability                      │
│   ├─ Crime by Place (Residential/Highway/Market %)                      │
│   ├─ Murder Motives (Gang/Domestic %)                                   │
│   └─ Police Complaints → Accountability Rate                            │
│                                                                         │
│  Live Open-Meteo Weather API (temperature, rain, wind)                 │
│  OpenStreetMap POI Density (bars, ATMs, bus stops, markets)            │
│                                                                         │
│  📰 ET NewsWatch Intelligence (NEW)                                     │
│   ├─ Economic Times RSS                                                 │
│   ├─ NDTV Top Stories RSS                                               │
│   ├─ Times of India India RSS                                           │
│   ├─ Hindustan Times RSS                                                │
│   └─ The Hindu National RSS                                             │
│       → GPT-4o / regex extraction → city-mapped crime signals           │
│       → Breaking news banner · severity badges · source links           │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────── FEATURE ENGINEERING ────────────────────────────┐
│  51 features across 5 dimensions:                                       │
│   Temporal (hour, day, month, weekend, lag_1h, lag_24h, rolling_7d)    │
│   Weather (temp, precipitation, wind, is_rainy)                         │
│   Spatial/POI (bars, ATMs, bus stops, markets, police dist, lighting)  │
│   NCRB-Enriched (11 multi-source dimensions)                            │
│   Crime Context (zone_type, is_hotspot, population_density)             │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────── ML ENSEMBLE ───────────────────────────────────┐
│  XGBoost + LightGBM majority-vote classifier                            │
│   LightGBM F1 = 0.851 · 10 crime types                                 │
│   vehicle_theft, robbery, assault, dacoity, burglary, cyber_fraud,     │
│   domestic_violence, sexual_assault, child_crime, property_crime        │
│                                                                         │
│  SHAP Explainability → "Why is this zone risky?"                       │
│   Per-zone top-3 risk drivers in plain English                          │
│   Adjusts for women_safety_index, police_coverage, location context    │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────── GENAI LAYER ────────────────────────────────────┐
│  GPT-4o Shift Briefing Generator                                        │
│   Input: top 10 zones + SHAP drivers + women_safety_index +            │
│          police_coverage_ratio + dominant crime + weather               │
│   Output: actionable plain-English shift brief with:                   │
│    [Women Safety Alert] → "Assign female constables"                   │
│    [Understaffing Advisory] → "Request additional patrol unit"          │
│    Fallback rule-based briefing when API key unavailable                │
│                                                                         │
│  GPT-4o NewsWatch Parser (when API key available)                      │
│   Extracts: crime_type, severity, location_hint from news articles      │
│   Regex fallback always available                                        │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────── DASHBOARD ──────────────────────────────────────┐
│  🛰 ESRI Satellite Map + Place Labels overlay                           │
│   Risk circles (HIGH/MED/LOW) · Richer styled popups                   │
│   Layer switcher · Risk legend · Live zone stats overlay                │
│                                                                         │
│  📊 Stats Bar: zones count · risk levels · live weather · alerts        │
│  📰 Breaking News Banner (red ticker when HIGH severity signals found)  │
│  ⏱ 5-min Auto-refresh countdown + WebSocket live push                  │
│                                                                         │
│  Tabbed Right Panel:                                                    │
│   [Zones]    → All zones ranked by risk, expandable NCRB metrics        │
│   [Briefing] → GPT-4o shift briefing + WhatsApp share                  │
│   [News]     → NewsWatch Intelligence panel (ET/NDTV/TOI/HT/TH)        │
│   [Forecast] → 24-hour stacked risk bar chart + peak hour alert        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites
- Python 3.10+, Node.js 18+
- Optional: `OPENAI_API_KEY` (for GPT-4o briefings + news parsing; fallback always active)

### 1 — Backend
```bash
cd crimewatch-ai
python -m venv backend/.venv
backend/.venv/Scripts/activate          # Windows
# or: source backend/.venv/bin/activate  # Linux/Mac

pip install -r backend/requirements.txt

# Generate training data + train model
python -m backend.data.data_pipeline

# Start API server
uvicorn main:app --port 8000 --app-dir backend
```

### 2 — Frontend
```bash
cd frontend
npm install
npm run dev        # development
# or: npm run build && npm start   # production
```

### 3 — Open http://localhost:3000

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health + loaded cities |
| `GET` | `/api/v1/cities` | List cities with zone counts |
| `POST` | `/api/v1/predict` | Run prediction for a city |
| `GET` | `/api/v1/zones/{city}` | Cached zone predictions |
| `GET` | `/api/v1/weather/{city}` | Live weather (Open-Meteo) |
| `GET` | `/api/v1/news/{city}` | **NewsWatch** — live crime signals |
| `GET` | `/api/v1/forecast/{city}` | **24-hour risk forecast** |
| `POST` | `/api/v1/briefing` | Generate GPT-4o shift briefing |
| `POST` | `/api/v1/feedback` | Officer feedback (confirm/deny) |
| `WS` | `/ws/live/{city}` | WebSocket live zone updates |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | XGBoost · LightGBM · SHAP · scikit-learn |
| GenAI | GPT-4o (briefing + news parsing) |
| Backend | FastAPI · Uvicorn · APScheduler · httpx |
| Data | Pandas · NumPy · 11 NCRB CSV datasets |
| Map | Leaflet.js · ESRI World Imagery Satellite |
| Weather | Open-Meteo API (free, no key) |
| News | Economic Times · NDTV · TOI · HT · The Hindu RSS |
| Frontend | Next.js 14 · TypeScript · Tailwind CSS · Lucide |
| Infra | Docker Compose · WebSocket · PWA-ready |

---

## NCRB Data Sources

| NCRB Table | Description | Derived Feature |
|------------|-------------|-----------------|
| IPC District Crimes | State/District year crime count | Base training labels |
| Table 42 | Crimes against Women | `women_safety_index` |
| Table 02 | Crimes against SC | `vulnerability_index` |
| Table 02_01 | Crimes against ST | Extends `vulnerability_index` |
| Table 03 | Crimes against Children | `child_crime_rate` |
| Table 12 | Police Strength | `police_coverage_ratio` |
| Table 10 | Property Stolen Value | `property_value_stolen_lakh` |
| Table 30 | Auto Theft Detail | `state_auto_theft_count` |
| Table 17 | Crime by Place | `residential/highway/market_crime_pct` |
| Table 19 | Murder Motives | `gang/domestic_murder_pct` |
| Table 25 | Police Complaints | `police_complaint_rate` |

---

## Project Structure

```
crimewatch-ai/
├── backend/
│   ├── data/
│   │   ├── ncrb_enriched_loader.py   # 11-source NCRB fusion
│   │   ├── data_pipeline.py          # Training pipeline
│   │   ├── weather_fetcher.py        # Open-Meteo
│   │   └── raw/ncrb/                 # NCRB CSVs (gitignored)
│   ├── models/
│   │   ├── feature_engineering.py    # 51-feature builder
│   │   ├── predictor.py              # XGBoost+LightGBM+SHAP
│   │   └── train.py                  # Training script
│   ├── services/
│   │   ├── briefing_service.py       # GPT-4o briefings
│   │   ├── news_intelligence.py      # NewsWatch RSS → GPT-4o
│   │   └── scheduler.py              # APScheduler jobs
│   └── main.py                       # FastAPI app
├── frontend/
│   ├── app/
│   │   ├── components/
│   │   │   ├── CrimeHeatmap.tsx      # Satellite map
│   │   │   ├── ZoneCard.tsx          # Zone detail card
│   │   │   ├── StatsBar.tsx          # City stats
│   │   │   ├── BriefingPanel.tsx     # GPT-4o briefing
│   │   │   ├── WeatherWidget.tsx     # Live weather
│   │   │   ├── NewsPanel.tsx         # NewsWatch panel
│   │   │   └── ForecastChart.tsx     # 24h forecast
│   │   └── page.tsx                  # Main dashboard
│   └── lib/api.ts                    # API types & fetch
├── CRIMEWATCH_AI.md                  # Technical documentation
└── README.md                         # This file
```

---

## Hackathon Submission

- **GitHub**: https://github.com/SwasthikaDev/ET-Gen-AI-Hackathon
- **Team**: Calyirex
- **Problem Statement**: Smart Cities / Public Safety GenAI
- **Demo Video**: _[record and link here before submission]_

---

*Built with real NCRB data · Real-time weather · Live news intelligence · GPT-4o GenAI*
