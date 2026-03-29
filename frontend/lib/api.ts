const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type RiskLevel = "HIGH" | "MEDIUM" | "LOW";

export interface CrimeType {
  type: string;
  probability: number;
}

export interface ShapDriver {
  feature: string;
  magnitude: number;
  direction: "increases_risk" | "decreases_risk";
  explanation: string;
}

export interface ZonePrediction {
  zone_id: string;
  city: string;
  lat: number;
  lon: number;
  risk_score: number;
  risk_level: RiskLevel;
  risk_colour: string;
  top_crime_types: CrimeType[];
  shap_drivers: ShapDriver[];
  predicted_for: string;
  // Enriched NCRB multi-source features (from zones metadata)
  women_safety_index?: number;
  vulnerability_index?: number;
  police_coverage_ratio?: number;
  property_value_stolen_lakh?: number;
  state_auto_theft_count?: number;
}

export interface PredictionResult {
  city: string;
  predicted_at: string;
  target_hour: string;
  weather: { temperature_c: number; precipitation_mm: number; wind_speed_kmh: number };
  zones: ZonePrediction[];
  summary: { total_zones: number; high: number; medium: number; low: number; top_zone: string };
}

export interface Briefing {
  text: string;
  language: string;
  city: string;
  shift: string;
  generated_by: string;
  generated_at: string;
}

export interface City {
  name: string;
  zone_count: number;
}

export async function fetchCities(): Promise<City[]> {
  const r = await fetch(`${BASE}/api/v1/cities`);
  const data = await r.json();
  return data.cities ?? [];
}

export async function runPrediction(city: string): Promise<PredictionResult> {
  const r = await fetch(`${BASE}/api/v1/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ city, use_live_weather: true }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function fetchBriefing(city: string, language = "en"): Promise<Briefing> {
  // First try GET (cached)
  const get = await fetch(`${BASE}/api/v1/briefing/${city}?language=${language}`);
  if (get.ok) return get.json();

  // Fall back to POST to generate
  const post = await fetch(`${BASE}/api/v1/briefing`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ city, language }),
  });
  if (!post.ok) throw new Error(await post.text());
  return post.json();
}

export async function submitFeedback(data: {
  zone_id: string;
  shift_date: string;
  shift: string;
  incident_confirmed: boolean;
  crime_type?: string;
  officer_id?: string;
}): Promise<void> {
  await fetch(`${BASE}/api/v1/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

export function createWsConnection(city: string, onMessage: (data: unknown) => void): WebSocket {
  const wsBase = BASE.replace(/^http/, "ws");
  const ws = new WebSocket(`${wsBase}/ws/live/${city}`);
  ws.onmessage = (e) => onMessage(JSON.parse(e.data));
  return ws;
}
