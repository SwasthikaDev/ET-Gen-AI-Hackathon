"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import dynamic from "next/dynamic";
import {
  RefreshCw, Shield, Wifi, WifiOff, MapPin, Clock,
  Database, Newspaper, TrendingUp, List, AlertTriangle,
} from "lucide-react";
import type { City, PredictionResult, ZonePrediction, NewsSignal } from "@/lib/api";
import { fetchCities, runPrediction, createWsConnection, fetchNewsSignals } from "@/lib/api";
import StatsBar from "./components/StatsBar";
import ZoneCard from "./components/ZoneCard";
import BriefingPanel from "./components/BriefingPanel";
import WeatherWidget from "./components/WeatherWidget";
import NewsPanel from "./components/NewsPanel";
import ForecastChart from "./components/ForecastChart";

const CrimeHeatmap = dynamic(() => import("./components/CrimeHeatmap"), { ssr: false });

const AUTO_REFRESH_SECONDS = 300;
type RightTab = "zones" | "briefing" | "news" | "forecast";

export default function Dashboard() {
  const [cities, setCities] = useState<City[]>([]);
  const [selectedCity, setSelectedCity] = useState("");
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [liveZones, setLiveZones] = useState<ZonePrediction[]>([]);
  const [countdown, setCountdown] = useState(AUTO_REFRESH_SECONDS);
  const [lastUpdated, setLastUpdated] = useState("");
  const [rightTab, setRightTab] = useState<RightTab>("zones");
  const [breakingSignals, setBreakingSignals] = useState<NewsSignal[]>([]);
  const countdownRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── City list ──────────────────────────────────────────────────────────────
  useEffect(() => {
    fetchCities()
      .then((c) => { setCities(c); if (c.length) setSelectedCity(c[0].name); })
      .catch(() => {
        const demo: City[] = [
          { name: "Bengaluru", zone_count: 25 }, { name: "Hyderabad", zone_count: 22 },
          { name: "Mumbai", zone_count: 30 },    { name: "Delhi", zone_count: 28 },
          { name: "Chennai", zone_count: 20 },
        ];
        setCities(demo);
        setSelectedCity("Bengaluru");
      });
  }, []);

  // ── Predict ────────────────────────────────────────────────────────────────
  const predict = useCallback(async () => {
    if (!selectedCity) return;
    setLoading(true);
    try {
      const r = await runPrediction(selectedCity);
      setResult(r);
      setLiveZones(r.zones);
      setLastUpdated(new Date().toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", second: "2-digit" }));
      setCountdown(AUTO_REFRESH_SECONDS);
    } catch (e) { console.error(e); }
    finally { setLoading(false); }
  }, [selectedCity]);

  useEffect(() => { if (selectedCity) predict(); }, [selectedCity, predict]);

  // ── Auto-refresh countdown ─────────────────────────────────────────────────
  useEffect(() => {
    if (countdownRef.current) clearInterval(countdownRef.current);
    countdownRef.current = setInterval(() => {
      setCountdown((p) => { if (p <= 1) { predict(); return AUTO_REFRESH_SECONDS; } return p - 1; });
    }, 1000);
    return () => { if (countdownRef.current) clearInterval(countdownRef.current); };
  }, [predict]);

  // ── WebSocket live push ────────────────────────────────────────────────────
  useEffect(() => {
    if (!selectedCity) return;
    let ws: WebSocket;
    try {
      ws = createWsConnection(selectedCity, (data: unknown) => {
        const msg = data as { event: string; zones?: ZonePrediction[] };
        if (msg.event === "zone_update" && msg.zones) { setLiveZones(msg.zones); }
        setWsConnected(true);
      });
      ws.onopen = () => setWsConnected(true);
      ws.onclose = () => setWsConnected(false);
      ws.onerror = () => setWsConnected(false);
    } catch { setWsConnected(false); }
    return () => ws?.close();
  }, [selectedCity]);

  // ── Poll breaking news (HIGH severity) every 20 min ───────────────────────
  useEffect(() => {
    if (!selectedCity) return;
    const pollNews = async () => {
      try {
        const d = await fetchNewsSignals(selectedCity);
        setBreakingSignals(d.signals.filter((s) => s.severity === "HIGH").slice(0, 3));
      } catch { /* silent */ }
    };
    pollNews();
    const t = setInterval(pollNews, 20 * 60 * 1000);
    return () => clearInterval(t);
  }, [selectedCity]);

  const zones = liveZones.length ? liveZones : result?.zones ?? [];
  const highZones = zones.filter((z) => z.risk_level === "HIGH");
  const medZones  = zones.filter((z) => z.risk_level === "MEDIUM");
  const lowZones  = zones.filter((z) => z.risk_level === "LOW");
  const allSorted = [...highZones, ...medZones, ...lowZones];

  const countdownMins = Math.floor(countdown / 60);
  const countdownSecs = countdown % 60;

  const TAB_DEFS: { id: RightTab; label: string; icon: React.ReactNode }[] = [
    { id: "zones",    label: "Zones",    icon: <List className="w-3.5 h-3.5" /> },
    { id: "briefing", label: "Briefing", icon: <Shield className="w-3.5 h-3.5" /> },
    { id: "news",     label: "News",     icon: <Newspaper className="w-3.5 h-3.5" /> },
    { id: "forecast", label: "Forecast", icon: <TrendingUp className="w-3.5 h-3.5" /> },
  ];

  return (
    <div className="min-h-screen bg-slate-100 flex flex-col">

      {/* ── Breaking news banner ─────────────────────────────────────────── */}
      {breakingSignals.length > 0 && (
        <div className="bg-red-700 text-white text-xs py-1.5 overflow-hidden">
          <div className="max-w-screen-2xl mx-auto px-4 flex items-center gap-3">
            <span className="font-bold shrink-0 flex items-center gap-1 bg-white text-red-700 px-2 py-0.5 rounded text-[10px]">
              🔴 BREAKING
            </span>
            <div className="overflow-hidden">
              <span className="whitespace-nowrap">
                {breakingSignals.map((s, i) => (
                  <span key={i}>
                    {i > 0 && <span className="mx-3 opacity-50">|</span>}
                    <strong>{s.source}:</strong> {s.title}
                  </span>
                ))}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* ── Top nav ──────────────────────────────────────────────────────── */}
      <header className="border-b border-slate-200 bg-white/95 backdrop-blur sticky top-0 z-50 shadow-sm">
        <div className="max-w-screen-2xl mx-auto px-4 py-2 flex items-center justify-between gap-3 flex-wrap">

          {/* Brand */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-blue-700 flex items-center justify-center shadow">
              <Shield className="w-4 h-4 text-white" />
            </div>
            <div>
              <h1 className="text-sm font-bold text-slate-900 leading-none">CrimeWatch AI</h1>
              <p className="text-[10px] text-slate-500 mt-0.5">
                Predictive Crime Intelligence · ET GenAI Hackathon 2026
              </p>
            </div>
          </div>

          {/* Centre widgets */}
          <div className="flex items-center gap-2 flex-wrap">
            {selectedCity && <WeatherWidget city={selectedCity} />}
            <span className="hidden md:flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-emerald-50 border border-emerald-200 text-xs text-emerald-700 font-medium">
              <Database className="w-3 h-3" /> Live NCRB · Open-Meteo · ET NewsWatch
            </span>
            {breakingSignals.length > 0 && (
              <button
                onClick={() => setRightTab("news")}
                className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-red-50 border border-red-200 text-xs text-red-700 font-semibold animate-pulse"
              >
                <AlertTriangle className="w-3 h-3" /> {breakingSignals.length} Breaking
              </button>
            )}
          </div>

          {/* Right controls */}
          <div className="flex items-center gap-2 flex-wrap">
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

            {lastUpdated && (
              <div className="hidden sm:flex items-center gap-1.5 text-xs text-slate-500">
                <Clock className="w-3 h-3" />
                <span>{lastUpdated}</span>
                <span className="text-slate-300">·</span>
                <span className={countdown < 30 ? "text-amber-600 font-semibold" : "text-slate-400"}>
                  {countdownMins}:{String(countdownSecs).padStart(2, "0")}
                </span>
              </div>
            )}

            <div className={`flex items-center gap-1 text-xs ${wsConnected ? "text-emerald-600 font-medium" : "text-slate-400"}`}>
              {wsConnected ? <Wifi className="w-3.5 h-3.5" /> : <WifiOff className="w-3.5 h-3.5" />}
              <span className="hidden sm:inline">{wsConnected ? "Live" : "Static"}</span>
            </div>

            <button
              onClick={predict}
              disabled={loading}
              className="flex items-center gap-2 px-4 py-1.5 rounded-lg bg-blue-700 hover:bg-blue-800 text-white text-sm font-medium transition disabled:opacity-50 shadow-sm"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
              Predict
            </button>
          </div>
        </div>
      </header>

      {/* ── Stats bar ────────────────────────────────────────────────────── */}
      <div className="max-w-screen-2xl mx-auto w-full px-4 pt-3 pb-1">
        <StatsBar result={result} loading={loading} />
      </div>

      {/* ── Main content ──────────────────────────────────────────────────── */}
      <main className="flex-1 max-w-screen-2xl mx-auto w-full px-4 pb-6 pt-3 grid grid-cols-1 lg:grid-cols-3 gap-4">

        {/* Map — 2/3 */}
        <div className="lg:col-span-2 rounded-xl overflow-hidden border border-slate-200 bg-white shadow-sm" style={{ minHeight: 520 }}>
          {selectedCity && <CrimeHeatmap zones={zones} city={selectedCity} />}
        </div>

        {/* Right panel — tabbed ─────────────────────────────────────────── */}
        <div className="flex flex-col gap-0 rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden">

          {/* Tab bar */}
          <div className="flex border-b border-slate-200">
            {TAB_DEFS.map((t) => (
              <button
                key={t.id}
                onClick={() => setRightTab(t.id)}
                className={`flex-1 flex items-center justify-center gap-1.5 py-2.5 text-xs font-semibold transition border-b-2 ${
                  rightTab === t.id
                    ? "border-blue-700 text-blue-700 bg-blue-50"
                    : "border-transparent text-slate-500 hover:text-slate-700 hover:bg-slate-50"
                }`}
              >
                {t.icon}
                <span className="hidden sm:inline">{t.label}</span>
                {/* News badge */}
                {t.id === "news" && breakingSignals.length > 0 && (
                  <span className="w-4 h-4 rounded-full bg-red-500 text-white text-[9px] flex items-center justify-center font-bold">
                    {breakingSignals.length}
                  </span>
                )}
              </button>
            ))}
          </div>

          {/* Tab content */}
          <div className="flex-1 overflow-hidden p-4" style={{ minHeight: 0 }}>

            {/* Zones */}
            {rightTab === "zones" && (
              <div className="flex flex-col h-full overflow-hidden">
                <div className="flex items-center justify-between mb-2">
                  <h2 className="text-xs uppercase tracking-wider text-slate-500 font-semibold">
                    All Zones ({allSorted.length})
                  </h2>
                  {allSorted.length > 0 && (
                    <span className="text-[10px] text-slate-400">
                      {highZones.length}H · {medZones.length}M · {lowZones.length}L
                    </span>
                  )}
                </div>
                <div className="overflow-y-auto scrollbar-hide space-y-2 flex-1">
                  {loading && (
                    <div className="space-y-2 animate-pulse">
                      {[...Array(5)].map((_, i) => <div key={i} className="h-20 rounded-xl bg-slate-100" />)}
                    </div>
                  )}
                  {!loading && allSorted.map((zone, i) => (
                    <ZoneCard key={zone.zone_id} zone={zone} rank={i + 1} />
                  ))}
                </div>
              </div>
            )}

            {/* Briefing */}
            {rightTab === "briefing" && (
              <div className="h-full">
                <BriefingPanel city={selectedCity} />
              </div>
            )}

            {/* News */}
            {rightTab === "news" && (
              <div className="h-full flex flex-col overflow-hidden" style={{ maxHeight: "calc(100vh - 280px)" }}>
                <NewsPanel city={selectedCity} />
              </div>
            )}

            {/* Forecast */}
            {rightTab === "forecast" && (
              <div className="h-full" style={{ maxHeight: "calc(100vh - 280px)" }}>
                <ForecastChart city={selectedCity} totalZones={allSorted.length} />
              </div>
            )}
          </div>
        </div>
      </main>

      {/* ── Footer data-source bar ────────────────────────────────────────── */}
      <footer className="border-t border-slate-200 bg-white/80 backdrop-blur">
        <div className="max-w-screen-2xl mx-auto px-4 py-2 flex flex-wrap items-center justify-between gap-2 text-[10px] text-slate-400">
          <div className="flex gap-3">
            <span>📊 2.8M+ NCRB records · 11 crime dimensions</span>
            <span>🛰 ESRI Satellite · Open-Meteo weather</span>
            <span>📰 ET · NDTV · TOI · HT · The Hindu</span>
          </div>
          <span>CrimeWatch AI · ET GenAI Hackathon 2026 · Built by Calyirex</span>
        </div>
      </footer>
    </div>
  );
}
