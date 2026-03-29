"use client";

import { useEffect, useState, useCallback } from "react";
import dynamic from "next/dynamic";
import { RefreshCw, Shield, Wifi, WifiOff, MapPin } from "lucide-react";
import type { City, PredictionResult, ZonePrediction } from "@/lib/api";
import { fetchCities, runPrediction, createWsConnection } from "@/lib/api";
import StatsBar from "./components/StatsBar";
import ZoneCard from "./components/ZoneCard";
import BriefingPanel from "./components/BriefingPanel";

// SSR-safe map import
const CrimeHeatmap = dynamic(() => import("./components/CrimeHeatmap"), { ssr: false });

export default function Dashboard() {
  const [cities, setCities] = useState<City[]>([]);
  const [selectedCity, setSelectedCity] = useState<string>("");
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [liveZones, setLiveZones] = useState<ZonePrediction[]>([]);

  // Load city list on mount
  useEffect(() => {
    fetchCities()
      .then((c) => {
        setCities(c);
        if (c.length > 0) setSelectedCity(c[0].name);
      })
      .catch(() => {
        // API not running — show demo cities
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

  // WebSocket live updates
  useEffect(() => {
    if (!selectedCity) return;
    let ws: WebSocket;
    try {
      ws = createWsConnection(selectedCity, (data: unknown) => {
        const msg = data as { event: string; zones?: ZonePrediction[] };
        if (msg.event === "zone_update" && msg.zones) {
          setLiveZones(msg.zones);
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

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col">
      {/* Top nav */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
        <div className="max-w-screen-2xl mx-auto px-4 py-3 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <Shield className="w-6 h-6 text-indigo-400" />
            <div>
              <h1 className="text-base font-bold text-slate-100 leading-none">CrimeWatch AI</h1>
              <p className="text-xs text-slate-500">Predictive Crime Intelligence · Calyirex</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* City selector */}
            <div className="flex items-center gap-1 bg-slate-800 rounded-lg px-3 py-1.5">
              <MapPin className="w-3.5 h-3.5 text-slate-400" />
              <select
                value={selectedCity}
                onChange={(e) => setSelectedCity(e.target.value)}
                className="bg-transparent text-sm text-slate-200 outline-none cursor-pointer"
              >
                {cities.map((c) => (
                  <option key={c.name} value={c.name} className="bg-slate-800">
                    {c.name} ({c.zone_count} zones)
                  </option>
                ))}
              </select>
            </div>

            {/* WS indicator */}
            <div className={`flex items-center gap-1 text-xs ${wsConnected ? "text-green-400" : "text-slate-500"}`}>
              {wsConnected ? <Wifi className="w-3.5 h-3.5" /> : <WifiOff className="w-3.5 h-3.5" />}
              <span>{wsConnected ? "Live" : "Static"}</span>
            </div>

            <button
              onClick={predict}
              disabled={loading}
              className="flex items-center gap-2 px-4 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium transition disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
              Predict Now
            </button>
          </div>
        </div>
      </header>

      {/* Stats bar */}
      <div className="max-w-screen-2xl mx-auto w-full px-4 py-3">
        <StatsBar result={result} loading={loading} />
      </div>

      {/* Main content */}
      <main className="flex-1 max-w-screen-2xl mx-auto w-full px-4 pb-6 grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Map — 2 cols */}
        <div className="lg:col-span-2 rounded-xl overflow-hidden border border-slate-800" style={{ minHeight: 480 }}>
          {selectedCity && (
            <CrimeHeatmap zones={zones} city={selectedCity} />
          )}
        </div>

        {/* Right panel: zone list + briefing */}
        <div className="flex flex-col gap-4">
          {/* Briefing */}
          <div className="h-64">
            <BriefingPanel city={selectedCity} />
          </div>

          {/* Zone list */}
          <div className="flex-1 overflow-y-auto scrollbar-hide space-y-2 max-h-[calc(100vh-400px)]">
            <h2 className="text-xs uppercase tracking-wider text-slate-400 font-semibold">
              All Zones ({allSorted.length})
            </h2>
            {loading && (
              <div className="space-y-2 animate-pulse">
                {[...Array(6)].map((_, i) => (
                  <div key={i} className="h-20 rounded-xl bg-slate-800" />
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
