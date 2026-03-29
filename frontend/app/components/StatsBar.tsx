"use client";

import type { PredictionResult } from "@/lib/api";
import { ShieldAlert, Users } from "lucide-react";

interface Props {
  result: PredictionResult | null;
  loading: boolean;
}

const Stat = ({ label, value, colour }: { label: string; value: string | number; colour: string }) => (
  <div className="flex flex-col items-center gap-1 px-4 py-2 rounded-lg bg-slate-800/60">
    <span className={`text-2xl font-bold ${colour}`}>{value}</span>
    <span className="text-xs text-slate-400 uppercase tracking-wider">{label}</span>
  </div>
);

export default function StatsBar({ result, loading }: Props) {
  if (loading) {
    return (
      <div className="flex gap-3 animate-pulse">
        {[...Array(6)].map((_, i) => (
          <div key={i} className="h-16 w-24 rounded-lg bg-slate-800" />
        ))}
      </div>
    );
  }

  if (!result) return null;

  const { summary, weather, predicted_at, zones } = result;
  const time = new Date(predicted_at).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" });

  // Compute enriched city-wide metrics from zone data
  const womenAlertZones = zones.filter(z => (z.women_safety_index ?? 0) > 400).length;
  const understaffedZones = zones.filter(z => (z.police_coverage_ratio ?? 1) < 0.7).length;

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-3 items-center">
        <Stat label="Total Zones" value={summary.total_zones} colour="text-slate-100" />
        <Stat label="High Risk" value={summary.high} colour="text-red-400" />
        <Stat label="Medium Risk" value={summary.medium} colour="text-orange-400" />
        <Stat label="Low Risk" value={summary.low} colour="text-green-400" />
        <Stat label="Temp °C" value={`${weather.temperature_c}°`} colour="text-sky-300" />
        <Stat
          label="Rain mm"
          value={weather.precipitation_mm > 0 ? `${weather.precipitation_mm}mm` : "Dry"}
          colour={weather.precipitation_mm > 0 ? "text-blue-400" : "text-slate-300"}
        />
        <div className="ml-auto text-xs text-slate-500">Updated {time}</div>
      </div>

      {/* Enriched NCRB intelligence alerts row */}
      {(womenAlertZones > 0 || understaffedZones > 0) && (
        <div className="flex flex-wrap gap-2">
          {womenAlertZones > 0 && (
            <span className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full bg-pink-900/40 text-pink-300 border border-pink-700/40">
              <ShieldAlert className="w-3.5 h-3.5" />
              Women Safety Alert active in <b>{womenAlertZones}</b> zone{womenAlertZones > 1 ? "s" : ""}
            </span>
          )}
          {understaffedZones > 0 && (
            <span className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full bg-amber-900/40 text-amber-300 border border-amber-700/40">
              <Users className="w-3.5 h-3.5" />
              Police understaffed in <b>{understaffedZones}</b> zone{understaffedZones > 1 ? "s" : ""}
            </span>
          )}
          <span className="text-xs px-3 py-1.5 rounded-full bg-slate-800/60 text-slate-400 border border-slate-700/30">
            Powered by 2.8M NCRB records · {result.zones.length} crime dimensions
          </span>
        </div>
      )}
    </div>
  );
}
