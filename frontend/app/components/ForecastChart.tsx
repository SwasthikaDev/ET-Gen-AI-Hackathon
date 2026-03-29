"use client";

import { useEffect, useState, useCallback } from "react";
import { TrendingUp, RefreshCw, Clock } from "lucide-react";
import type { ForecastWindow, ForecastResponse } from "@/lib/api";
import { fetchForecast } from "@/lib/api";

interface Props {
  city: string;
  totalZones: number;
}

export default function ForecastChart({ city, totalZones }: Props) {
  const [data, setData] = useState<ForecastResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    if (!city) return;
    setLoading(true);
    try {
      const d = await fetchForecast(city);
      setData(d);
    } catch {
      // silent
    } finally {
      setLoading(false);
    }
  }, [city]);

  useEffect(() => {
    load();
  }, [load]);

  const maxHigh = data ? Math.max(...data.windows.map((w) => w.high), 1) : 1;

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-blue-700" />
          <span className="font-semibold text-slate-900 text-sm">24-Hour Risk Forecast</span>
        </div>
        <button
          onClick={load}
          disabled={loading}
          className="p-1 rounded hover:bg-slate-100 text-slate-400 disabled:opacity-40"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
        </button>
      </div>

      {loading && !data && (
        <div className="flex-1 flex items-center justify-center">
          <div className="space-y-2 animate-pulse w-full">
            <div className="flex gap-2 items-end h-32">
              {[...Array(7)].map((_, i) => (
                <div key={i} className="flex-1 rounded-t bg-slate-200" style={{ height: `${30 + i * 10}%` }} />
              ))}
            </div>
          </div>
        </div>
      )}

      {data && (
        <>
          {/* Peak alert */}
          <div className="mb-3 flex items-center gap-2 px-3 py-2 rounded-lg bg-amber-50 border border-amber-200">
            <Clock className="w-4 h-4 text-amber-600 shrink-0" />
            <p className="text-xs text-amber-800">
              <strong>Peak risk at {data.peak_hour}</strong> — {data.peak_high_zones} of {data.total_zones} zones at HIGH alert
            </p>
          </div>

          {/* Bar chart */}
          <div className="flex-1 flex flex-col justify-end">
            <div className="flex gap-1.5 items-end h-32">
              {data.windows.map((w, i) => (
                <Bar key={i} window={w} maxHigh={maxHigh} totalZones={data.total_zones} isNow={i === 0} />
              ))}
            </div>

            {/* X-axis labels */}
            <div className="flex gap-1.5 mt-1.5">
              {data.windows.map((w, i) => (
                <div key={i} className="flex-1 text-center">
                  <span className={`text-[9px] font-medium ${i === 0 ? "text-blue-700" : "text-slate-400"}`}>
                    {i === 0 ? "Now" : w.label}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Legend */}
          <div className="mt-3 flex gap-3 text-[10px] text-slate-500">
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-sm bg-red-400 inline-block" /> HIGH zones</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-sm bg-amber-300 inline-block" /> MEDIUM</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-sm bg-emerald-400 inline-block" /> LOW</span>
          </div>

          {/* Insight row */}
          <div className="mt-2 pt-2 border-t border-slate-100">
            <div className="flex flex-wrap gap-2">
              {data.windows.map((w, i) => {
                if (w.high === maxHigh && i !== 0) {
                  return (
                    <span key={i} className="text-[10px] px-2 py-0.5 rounded-full bg-red-50 text-red-700 border border-red-200">
                      ⚡ Peak: {w.label}
                    </span>
                  );
                }
                return null;
              })}
              <span className="text-[10px] text-slate-400">
                Avg risk score: {(data.windows.reduce((s, w) => s + w.avg_risk_score, 0) / data.windows.length * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function Bar({
  window: w,
  maxHigh,
  totalZones,
  isNow,
}: {
  window: ForecastWindow;
  maxHigh: number;
  totalZones: number;
  isNow: boolean;
}) {
  const pctHigh = totalZones > 0 ? w.high / totalZones : 0;
  const pctMed = totalZones > 0 ? w.medium / totalZones : 0;
  const pctLow = totalZones > 0 ? w.low / totalZones : 0;
  const totalHeight = 120; // px

  return (
    <div className="flex-1 flex flex-col items-center group cursor-default">
      {/* Tooltip */}
      <div className="hidden group-hover:flex absolute -mt-14 z-10 bg-slate-900 text-white text-[10px] px-2 py-1 rounded shadow-lg whitespace-nowrap flex-col gap-0.5">
        <span>🔴 HIGH: {w.high}</span>
        <span>🟠 MED: {w.medium}</span>
        <span>🟢 LOW: {w.low}</span>
      </div>

      {/* Stacked bar */}
      <div
        className={`w-full rounded-t flex flex-col-reverse overflow-hidden ${isNow ? "ring-1 ring-blue-400" : ""}`}
        style={{ height: totalHeight }}
      >
        <div
          className="w-full bg-emerald-400 transition-all"
          style={{ height: `${pctLow * 100}%` }}
        />
        <div
          className="w-full bg-amber-400 transition-all"
          style={{ height: `${pctMed * 100}%` }}
        />
        <div
          className={`w-full transition-all ${w.high === maxHigh ? "bg-red-500" : "bg-red-400"}`}
          style={{ height: `${pctHigh * 100}%` }}
        />
      </div>

      {/* Zone count label on top */}
      <span className={`text-[9px] mt-0.5 font-bold ${w.high === maxHigh ? "text-red-600" : "text-slate-400"}`}>
        {w.high}
      </span>
    </div>
  );
}
