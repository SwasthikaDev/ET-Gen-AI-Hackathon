"use client";

import { useEffect, useState, useCallback } from "react";
import { Thermometer, Droplets, Wind, RefreshCw } from "lucide-react";
import { fetchLiveWeather } from "@/lib/api";

interface Props {
  city: string;
}

export default function WeatherWidget({ city }: Props) {
  const [weather, setWeather] = useState<{
    temperature_c: number;
    precipitation_mm: number;
    wind_speed_kmh: number;
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [fetchedAt, setFetchedAt] = useState<string>("");

  const load = useCallback(async () => {
    if (!city) return;
    setLoading(true);
    try {
      const w = await fetchLiveWeather(city);
      setWeather(w);
      setFetchedAt(
        new Date().toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" })
      );
    } catch {
      // silently fall back — weather is non-critical
    } finally {
      setLoading(false);
    }
  }, [city]);

  useEffect(() => {
    load();
    const t = setInterval(load, 5 * 60 * 1000); // refresh every 5 min
    return () => clearInterval(t);
  }, [load]);

  if (!weather) {
    return (
      <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-100 border border-slate-200 animate-pulse">
        <div className="w-3 h-3 rounded-full bg-slate-300" />
        <div className="w-16 h-3 rounded bg-slate-300" />
      </div>
    );
  }

  const isRaining = weather.precipitation_mm > 0;
  const isHot = weather.temperature_c > 35;

  return (
    <div className="flex items-center gap-3 px-3 py-1.5 rounded-lg bg-sky-50 border border-sky-200 text-sm text-sky-800">
      {/* Temperature */}
      <span className={`flex items-center gap-1 font-semibold ${isHot ? "text-orange-600" : "text-sky-700"}`}>
        <Thermometer className="w-3.5 h-3.5" />
        {weather.temperature_c.toFixed(1)}°C
      </span>

      {/* Rain */}
      <span className={`flex items-center gap-1 ${isRaining ? "text-blue-700 font-semibold" : "text-sky-500"}`}>
        <Droplets className="w-3.5 h-3.5" />
        {isRaining ? `${weather.precipitation_mm}mm` : "Dry"}
      </span>

      {/* Wind */}
      <span className="flex items-center gap-1 text-sky-500">
        <Wind className="w-3.5 h-3.5" />
        {weather.wind_speed_kmh.toFixed(0)} km/h
      </span>

      {/* Time + refresh */}
      <span className="text-sky-400 text-xs hidden sm:block">{fetchedAt}</span>
      <button
        onClick={load}
        disabled={loading}
        className="text-sky-400 hover:text-sky-600 transition disabled:opacity-40"
        title="Refresh weather"
      >
        <RefreshCw className={`w-3 h-3 ${loading ? "animate-spin" : ""}`} />
      </button>
    </div>
  );
}
