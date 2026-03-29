# CrimeWatch AI
## Predictive Crime Hotspot Intelligence for Indian Cities
**Team Calyirex · ET Gen AI Hackathon 2026**

---

> India recorded **58.24 lakh IPC crimes in 2022** (NCRB). Every crime is recorded. None of that intelligence reaches the beat officer who could have prevented it.
>
> CrimeWatch AI closes that gap — fusing **7 NCRB datasets** (2.83M records, 2001–2014) with live weather and geospatial signals, running them through an XGBoost + LightGBM ensemble, and using GPT-4o to generate a plain-English shift briefing — including Women Safety Alerts and Police Understaffing Advisories — delivered to beat officers before every shift starts.

---

## What This Is

A **closed-loop GenAI pipeline** for predictive policing — powered by ALL available government data:

```
7 NCRB Datasets (2.83M records, 2001–2014):
  ├─ IPC District Crimes (base)
  ├─ Crimes Against Women → Women Safety Index
  ├─ Crimes Against SC/ST → Vulnerability Index
  ├─ Crimes Against Children → Child Crime Risk
  ├─ Police Strength (Actual vs Sanctioned) → Understaffing Ratio
  ├─ Property Stolen & Value → Property Crime Magnitude
  └─ Auto Theft Detail → Vehicle Crime Probability
+ Open-Meteo Weather (live)
+ OpenStreetMap POI density (real)
         ↓
  45-feature Engineering (5 new enriched features)
         ↓
  XGBoost + LightGBM Ensemble (LightGBM F1 = 0.846)
  10 crime types including domestic_violence, sexual_assault, child_crime
         ↓
  SHAP Explainer → "Why is this zone risky?"
  Drivers include: police understaffing, women safety index, vulnerability
         ↓
  GPT-4o with enriched prompt:
  Women Safety Alert + Understaffing Advisory + SHAP drivers
         →  Specific, actionable plain-English shift brief
         ↓
  WhatsApp delivery < 60 seconds
         ↓
  Officer feedback → model re-scoring (closed loop)
```

**The GenAI component is not cosmetic.** Without GPT-4o + SHAP, you get a heatmap an analyst might read. With it, you get an actionable brief — including which zones need female constables and where police reinforcements are required — that any beat officer with a WhatsApp can act on.

---

## Quick Demo

```bash
# With real NCRB data
# 1. Download: https://www.kaggle.com/datasets/rajanand/crime-in-india
#    Place CSVs in backend/data/raw/ncrb/

cd crimewatch-ai/backend
pip install -r requirements.txt
python -m backend.data.data_pipeline       # loads NCRB + weather + trains model
uvicorn main:app --reload --port 8000

# OR: instant synthetic demo (no downloads)
python -m backend.data.data_pipeline --synthetic
uvicorn main:app --reload --port 8000

# Frontend
cd ../frontend && npm install && npm run dev
# http://localhost:3000
```

Set `OPENAI_API_KEY` in `.env` (copy from `.env.example`) for live GPT-4o briefings.

---

## What Makes This Win-Worthy

| Standard approach | CrimeWatch AI |
|---|---|
| Uses 1 dataset | Fuses **7 NCRB datasets** — IPC, women's crimes, SC/ST, children, police strength, property value, auto theft |
| Shows 3–4 crime types | Predicts **10 crime types** including domestic violence, sexual assault, child crime |
| Stops at a dashboard | Delivers briefings to WhatsApp with **Women Safety Alerts** and **Understaffing Advisories** |
| Shows risk scores | Explains *why* using SHAP + government data evidence |
| No officer advisory | Tells officers exactly: "Assign female constables" / "Request reinforcements" |
| Static model | Feedback loop → auto-retraining |
| Synthetic/toy data | **2.83M real NCRB records (2001–2014)** — LightGBM F1 = 0.846 |

---

## Pilot Scope

5 cities · 65M+ residents · 125 zones

| City | Districts | Zones |
|---|---|---|
| Bengaluru | Bangalore Urban, Rural | 25 |
| Hyderabad | Hyderabad, Ranga Reddy | 22 |
| Mumbai | Mumbai City, Suburban, Thane | 30 |
| Delhi | 11 districts | 28 |
| Chennai | Chennai, Kancheepuram | 20 |

---

## Docs

| File | Contents |
|---|---|
| `CRIMEWATCH_AI.md` | Full technical documentation |
| `HACKATHON_PITCH.md` | Demo script, judge Q&A, talking points |
| `backend/` | FastAPI + ML pipeline (Python) |
| `frontend/` | Next.js 14 dashboard with Leaflet heatmap |

---

## Judging Alignment

| ET Hackathon Criterion | Our Score |
|---|---|
| Innovation & Creativity | SHAP → LLM causal briefings: novel pipeline |
| Technical Implementation | Real NCRB data, ensemble model, live WebSocket |
| Feasibility & Scalability | SaaS path: ward → city → state; modular stack |
| Relevance to Problem | GenAI for social impact: public safety + govt data |
| Documentation & Presentation | 3 comprehensive docs + live demo |

---

*Prize target: INR 5,00,000 · Built in 48 hours · Team Calyirex*
