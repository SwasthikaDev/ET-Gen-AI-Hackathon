"use client";

import { useState } from "react";
import { FileText, RefreshCw, Share2 } from "lucide-react";
import type { Briefing } from "@/lib/api";
import { fetchBriefing } from "@/lib/api";

const LANGUAGES = [
  { code: "en", label: "English" },
];

interface Props {
  city: string;
  initialBriefing?: Briefing | null;
}

export default function BriefingPanel({ city, initialBriefing }: Props) {
  const [briefing, setBriefing] = useState<Briefing | null>(initialBriefing ?? null);
  const [loading, setLoading] = useState(false);
  const [language, setLanguage] = useState("en");

  const load = async (lang = language) => {
    setLoading(true);
    try {
      const b = await fetchBriefing(city, lang);
      setBriefing(b);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const shareWhatsApp = () => {
    if (!briefing) return;
    const text = encodeURIComponent(`*CrimeWatch AI — ${briefing.city} ${briefing.shift} Briefing*\n\n${briefing.text}`);
    window.open(`https://wa.me/?text=${text}`, "_blank");
  };

  return (
    <div className="rounded-xl border border-slate-700/60 bg-slate-900/80 p-4 h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <FileText className="w-5 h-5 text-indigo-400" />
          <h2 className="font-semibold text-slate-100">Shift Briefing</h2>
          {briefing && (
            <span className="text-xs text-slate-500">{briefing.shift} · {briefing.generated_by}</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {/* Language tabs */}
          <div className="flex rounded-lg overflow-hidden border border-slate-700">
            {LANGUAGES.map((l) => (
              <button
                key={l.code}
                onClick={() => { setLanguage(l.code); load(l.code); }}
                className={`px-2 py-1 text-xs transition ${language === l.code ? "bg-indigo-600 text-white" : "text-slate-400 hover:bg-slate-700"}`}
              >
                {l.label}
              </button>
            ))}
          </div>

          <button
            onClick={() => load()}
            disabled={loading}
            className="p-1.5 rounded-lg hover:bg-slate-700 text-slate-400 hover:text-slate-200 transition disabled:opacity-50"
            title="Refresh briefing"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
          </button>

          <button
            onClick={shareWhatsApp}
            disabled={!briefing}
            className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs bg-green-700 hover:bg-green-600 text-white transition disabled:opacity-40"
          >
            <Share2 className="w-3 h-3" />
            WhatsApp
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto scrollbar-hide">
        {loading ? (
          <div className="space-y-3 animate-pulse">
            {[...Array(6)].map((_, i) => (
              <div key={i} className={`h-3 rounded bg-slate-700 ${i % 3 === 2 ? "w-2/3" : "w-full"}`} />
            ))}
          </div>
        ) : briefing ? (
          <div className="prose prose-sm prose-invert max-w-none">
            {briefing.text.split("\n").map((line, i) => {
              if (line.startsWith("**") && line.endsWith("**")) {
                return <p key={i} className="font-bold text-slate-100 mt-3">{line.replace(/\*\*/g, "")}</p>;
              }
              if (line.trim() === "") return <br key={i} />;
              return <p key={i} className="text-slate-300 text-sm leading-relaxed">{line}</p>;
            })}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-center gap-3 text-slate-500">
            <FileText className="w-8 h-8" />
            <p className="text-sm">No briefing yet.</p>
            <button
              onClick={() => load()}
              className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-sm transition"
            >
              Generate briefing
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
