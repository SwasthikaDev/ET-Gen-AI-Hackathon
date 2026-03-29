"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import dynamic from "next/dynamic";
import { RefreshCw, Shield, Wifi, WifiOff, MapPin, Clock, Database } from "lucide-react";
import type { City, PredictionResult, ZonePrediction } from "@/lib/api";
import { fetchCities, runPrediction, createWsConnection } from "@/lib/api";
import StatsBar from "./components/StatsBar";
import ZoneCard from "./components/ZoneCard";
import BriefingPanel from "./components/BriefingPanel";
import WeatherWidget from "./components/WeatherWidget";

// SSR-safe map import
const CrimeHeatmap = dynamic(() => import("./components/CrimeHeatmap"), { ssr: false });

const AUTO_REFRESH_SECONDS = 300; // 5 minutes

export default function Dashboard() {
  const [cities, setCities] = useState<City[]>([]);
  const [selectedCity, setSelectedCity] = useState<string>("");
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [liveZones, setLiveZones] = useState<ZonePrediction[]>([]);
  const [countdown, setCountdown] = useState(AUTO_REFRESH_SECONDS);
  const [lastUpdated, setLastUpdated] = useState<string>("");
  const countdownRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Load city list on mount
  useEffect(() => {
    fetchCities()
      .then((c) => {
        setCities(c);
        if (c.length > 0) setSelectedCity(c[0].name);
      })
      .catch(() => {
        const demo: City[] = [
          { name: "Bengaluru", zone_count: 25 },
          { name: "Hyderabad", zone_count: 22 },
          { name: "Mumbai", zone_count: 30 },
          { name: "Delhi", zone_count: 28 },
          { name: "Chennai", zone_count: 20 },
        ];
        setCities(demo);
        setSelectedCity("Bengaluru");
      });
  }, []);

  const predict = useCallback(async () => {
    if (!selectedCity) return;
    setLoading(true);
    try {
      const r = await runPrediction(selectedCity);
      setResult(r);
      setLiveZones(r.zones);
      setLastUpdated(
        new Date().toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", second: "2-digit" })
      );
      setCountdown(AUTO_REFRESH_SECONDS);
    } catch (e) {
      console.error("Prediction failed:", e);
    } finally {
      setLoading(false);
    }
  }, [selectedCity]);

  // Auto-predict when city changes
  useEffect(() => {
    if (selectedCity) predict();
  }, [selectedCity, predict]);

  // Auto-refresh countdown
  useEffect(() => {
    if (countdownRef.current) clearInterval(countdownRef.current);
    countdownRef.current = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          predict();
          return AUTO_REFRESH_SECONDS;
        }
        return prev - 1;
      });
    }, 1000);
    return () => {
      if (countdownRef.current) clearInterval(countdownRef.current);
    };
  }, [predict]);

  // WebSocket live updates
  useEffect(() => {
    if (!selectedCity) return;
    let ws: WebSocket;
    try {
      ws = createWsConnection(selectedCity, (data: unknown) => {
        const msg = data as { event: string; zones?: ZonePrediction[] };
        if (msg.event === "zone_update" && msg.zones) {
          setLiveZones(msg.zones);
          setLastUpdated(
            new Date().toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", second: "2-digit" })
          );
        }
        setWsConnected(true);
      });
      ws.onopen = () => setWsConnected(true);
      ws.onclose = () => setWsConnected(false);
      ws.onerror = () => setWsConnected(false);
    } catch {
      setWsConnected(false);
    }
    return () => ws?.close();
  }, [selectedCity]);

  const zones = liveZones.length ? liveZones : result?.zones ?? [];
  const highZones = zones.filter((z) => z.risk_level === "HIGH");
  const medZones = zones.filter((z) => z.risk_level === "MEDIUM");
  const lowZones = zones.filter((z) => z.risk_level === "LOW");
  const allSorted = [...highZones, ...medZones, ...lowZones];

  const countdownMins = Math.floor(countdown / 60);
  const countdownSecs = countdown % 60;

  return (
    <div className="min-h-screen bg-slate-100 flex flex-col">
      {/* ── Top nav ─────────────────────────────────────────────────────── */}
      <header className="border-b border-slate-200 bg-white/95 backdrop-blur sticky top-0 z-50">
        <div className="max-w-screen-2xl mx-auto px-4 py-2.5 flex items-center justify-between gap-3 flex-wrap">

          {/* Brand */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-blue-700 flex items-center justify-center shadow-sm">
              <Shield className="w-4.5 h-4.5 text-white" />
            </div>
            <div>
              <h1 className="text-sm font-bold text-slate-900 leading-none tracking-tight">CrimeWatch AI</h1>
              <p className="text-[10px] text-slate-500 mt-0.5">Predictive Crime Intelligence · Calyirex · ET GenAI Hackathon</p>
            </div>
          </div>

          {/* Centre: live weather + data badge */}
          <div className="flex items-center gap-2 flex-wrap">
            {selectedCity && <WeatherWidget city={selectedCity} />}
            <span className="hidden md:flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-emerald-50 border border-emerald-200 text-xs text-emerald-700 font-medium">
              <Database className="w-3 h-3" />
              Live NCRB · Open-Meteo
            </span>
          </div>

          {/* Right controls */}
          <div className="flex items-center gap-2.5 flex-wrap">

            {/* City selector */}
            <div className="flex items-center gap-1.5 bg-slate-50 border border-slate-200 rounded-lg px-3 py-1.5">
              <MapPin className="w-3.5 h-3.5 text-slate-500" />
              <select
                value={selectedCity}
                onChange={(e) => setSelectedCity(e.target.value)}
                className="bg-transparent text-sm text-slate-800 outline-none cursor-pointer"
              >
                {cities.map((c) => (
                  <option key={c.name} value={c.name} className="bg-white">
                    {c.name} ({c.zone_count} zones)
                  </option>
                ))}
              </select>
            </div>

            {/* Last updated + countdown */}
            {lastUpdated && (
              <div className="hidden sm:flex items-center gap-1.5 text-xs text-slate-500">
                <Clock className="w-3 h-3" />
                <span>{lastUpdated}</span>
                <span className="text-slate-300">·</span>
                <span className={countdown < 30 ? "text-amber-600 font-semibold" : "text-slate-400"}>
                  next in {countdownMins}:{String(countdownSecs).padStart(2, "0")}
                </span>
              </div>
            )}

            {/* WS indicator */}
            <div className={`flex items-center gap-1 text-xs ${wsConnected ? "text-emerald-600 font-medium" : "text-slate-400"}`}>
              {wsConnected ? <Wifi className="w-3.5 h-3.5" /> : <WifiOff className="w-3.5 h-3.5" />}
              <span>{wsConnected ? "Live" : "Static"}</span>
            </div>

            {/* Predict button */}
            <button
              onClick={predict}
              disabled={loading}
              className="flex items-center gap-2 px-4 py-1.5 rounded-lg bg-blue-700 hover:bg-blue-800 text-white text-sm font-medium transition disabled:opacity-50 shadow-sm"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
              Predict Now
            </button>
          </div>
        </div>
      </header>

      {/* ── Stats bar ───────────────────────────────────────────────────── */}
      <div className="max-w-screen-2xl mx-auto w-full px-4 pt-3 pb-1">
        <StatsBar result={result} loading={loading} />
      </div>

      {/* ── Main content ────────────────────────────────────────────────── */}
      <main className="flex-1 max-w-screen-2xl mx-auto w-full px-4 pb-6 pt-3 grid grid-cols-1 lg:grid-cols-3 gap-4">

        {/* Map — 2/3 width */}
        <div
          className="lg:col-span-2 rounded-xl overflow-hidden border border-slate-200 bg-white shadow-sm"
          style={{ minHeight: 500 }}
        >
          {selectedCity && <CrimeHeatmap zones={zones} city={selectedCity} />}
        </div>

        {/* Right panel: briefing + zone list */}
        <div className="flex flex-col gap-4">
          {/* Briefing */}
          <div className="h-64">
            <BriefingPanel city={selectedCity} />
          </div>

          {/* Zone list */}
          <div className="flex-1 overflow-y-auto scrollbar-hide space-y-2 max-h-[calc(100vh-420px)]">
            <div className="flex items-center justify-between">
              <h2 className="text-xs uppercase tracking-wider text-slate-500 font-semibold">
                All Zones ({allSorted.length})
              </h2>
              {allSorted.length > 0 && (
                <span className="text-[10px] text-slate-400">
                  {highZones.length} high · {medZones.length} med · {lowZones.length} low
                </span>
              )}
            </div>

            {loading && (
              <div className="space-y-2 animate-pulse">
                {[...Array(6)].map((_, i) => (
                  <div key={i} className="h-20 rounded-xl bg-slate-200" />
                ))}
              </div>
            )}
            {!loading &&
              allSorted.map((zone, i) => (
                <ZoneCard key={zone.zone_id} zone={zone} rank={i + 1} />
              ))}
          </div>
        </div>
      </main>
    </div>
  );
}
