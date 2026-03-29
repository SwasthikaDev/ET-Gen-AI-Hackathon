"use client";

import { useEffect, useState, useCallback } from "react";
import { Newspaper, RefreshCw, ExternalLink, AlertTriangle, TrendingUp } from "lucide-react";
import type { NewsSignal, NewsResponse } from "@/lib/api";
import { fetchNewsSignals } from "@/lib/api";

const SEVERITY_STYLE: Record<string, { badge: string; dot: string; border: string }> = {
  HIGH:   { badge: "bg-red-50 text-red-700 border-red-200",    dot: "bg-red-500",    border: "border-l-red-400" },
  MEDIUM: { badge: "bg-amber-50 text-amber-700 border-amber-200", dot: "bg-amber-500", border: "border-l-amber-400" },
  LOW:    { badge: "bg-slate-50 text-slate-600 border-slate-200", dot: "bg-slate-400", border: "border-l-slate-300" },
};

const SOURCE_LOGO: Record<string, string> = {
  "Economic Times": "ET",
  "NDTV": "NDTV",
  "Times of India": "TOI",
  "Hindustan Times": "HT",
  "The Hindu": "TH",
};

const CRIME_ICON: Record<string, string> = {
  vehicle_theft: "🚗",
  robbery: "💰",
  assault: "⚠",
  burglary: "🏠",
  cyber_fraud: "💻",
  sexual_assault: "🛡",
  domestic_violence: "🏘",
  child_crime: "👶",
  property_crime: "📦",
  dacoity: "🔫",
};

interface Props {
  city: string;
}

export default function NewsPanel({ city }: Props) {
  const [data, setData] = useState<NewsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [lastFetch, setLastFetch] = useState("");

  const load = useCallback(async () => {
    if (!city) return;
    setLoading(true);
    try {
      const d = await fetchNewsSignals(city);
      setData(d);
      setLastFetch(new Date().toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" }));
    } catch {
      // silent fail — news is non-critical
    } finally {
      setLoading(false);
    }
  }, [city]);

  useEffect(() => {
    load();
    const t = setInterval(load, 20 * 60 * 1000); // refresh every 20 min
    return () => clearInterval(t);
  }, [load]);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Newspaper className="w-4 h-4 text-blue-700" />
          <span className="font-semibold text-slate-900 text-sm">NewsWatch Intelligence</span>
          {data && data.high_severity > 0 && (
            <span className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-red-50 text-red-700 border border-red-200 text-xs font-semibold">
              <AlertTriangle className="w-3 h-3" />
              {data.high_severity} HIGH
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {lastFetch && <span className="text-[10px] text-slate-400">{lastFetch}</span>}
          <button
            onClick={load}
            disabled={loading}
            className="p-1 rounded hover:bg-slate-100 text-slate-400 hover:text-slate-600 transition disabled:opacity-40"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
          </button>
        </div>
      </div>

      {/* Source pills */}
      <div className="flex gap-1.5 mb-3 flex-wrap">
        <span className="text-[10px] text-slate-500 self-center">Sources:</span>
        {["ET", "NDTV", "TOI", "HT", "TH"].map((s) => (
          <span key={s} className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-blue-50 text-blue-700 border border-blue-100">
            {s}
          </span>
        ))}
        <span className="text-[10px] text-slate-400 self-center ml-1">· Live RSS · GPT-4o parsed</span>
      </div>

      {/* Signals list */}
      <div className="flex-1 overflow-y-auto scrollbar-hide space-y-2">
        {loading && !data && (
          <div className="space-y-2 animate-pulse">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-16 rounded-lg bg-slate-100" />
            ))}
          </div>
        )}

        {!loading && data && data.signals.length === 0 && (
          <div className="flex flex-col items-center justify-center h-24 text-slate-400 gap-2">
            <TrendingUp className="w-6 h-6" />
            <p className="text-xs">No crime news in this window.</p>
          </div>
        )}

        {data?.signals.map((signal, i) => (
          <NewsCard key={i} signal={signal} />
        ))}
      </div>

      {/* Footer stat */}
      {data && (
        <div className="mt-2 pt-2 border-t border-slate-100 flex items-center justify-between text-[10px] text-slate-400">
          <span>{data.signal_count} signals parsed · {data.high_severity} high severity</span>
          <span>Powered by GenAI · ET NewsWatch</span>
        </div>
      )}
    </div>
  );
}

function NewsCard({ signal }: { signal: NewsSignal }) {
  const style = SEVERITY_STYLE[signal.severity] ?? SEVERITY_STYLE.LOW;
  const icon = CRIME_ICON[signal.crime_type] ?? "⚠";
  const sourceLabel = SOURCE_LOGO[signal.source] ?? signal.source.slice(0, 3).toUpperCase();

  return (
    <div className={`rounded-lg border bg-white border-l-4 ${style.border} border-slate-200 p-2.5 hover:bg-slate-50 transition`}>
      <div className="flex items-start gap-2">
        <span className="text-base mt-0.5 shrink-0">{icon}</span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5 mb-1 flex-wrap">
            <span className={`px-1.5 py-0.5 rounded-full text-[10px] font-bold border ${style.badge}`}>
              {signal.severity}
            </span>
            <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-slate-100 text-slate-600">
              {sourceLabel}
            </span>
            <span className="text-[10px] text-slate-400 capitalize">
              {signal.crime_type.replace(/_/g, " ")}
            </span>
          </div>

          <p className="text-xs font-semibold text-slate-800 leading-tight line-clamp-2">{signal.title}</p>

          {signal.summary && signal.summary !== signal.title && (
            <p className="text-[11px] text-slate-500 mt-1 line-clamp-2 leading-relaxed">{signal.summary}</p>
          )}

          <div className="flex items-center gap-2 mt-1.5">
            {signal.location_hint && (
              <span className="text-[10px] text-blue-600 font-medium">📍 {signal.location_hint}</span>
            )}
            {signal.url && (
              <a
                href={signal.url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-0.5 text-[10px] text-blue-500 hover:text-blue-700 ml-auto"
                onClick={(e) => e.stopPropagation()}
              >
                Read <ExternalLink className="w-2.5 h-2.5" />
              </a>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
