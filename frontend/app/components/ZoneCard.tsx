"use client";

import { useState } from "react";
import { AlertTriangle, CheckCircle, Info, Copy, ShieldAlert, Users, Car } from "lucide-react";
import type { ZonePrediction } from "@/lib/api";
import { submitFeedback } from "@/lib/api";

const RISK_STYLES = {
  HIGH: { border: "border-red-500/60", badge: "bg-red-500/20 text-red-300", icon: <AlertTriangle className="w-4 h-4 text-red-400" /> },
  MEDIUM: { border: "border-orange-500/60", badge: "bg-orange-500/20 text-orange-300", icon: <AlertTriangle className="w-4 h-4 text-orange-400" /> },
  LOW: { border: "border-green-500/40", badge: "bg-green-500/20 text-green-300", icon: <CheckCircle className="w-4 h-4 text-green-400" /> },
};

interface Props {
  zone: ZonePrediction;
  rank: number;
}

export default function ZoneCard({ zone, rank }: Props) {
  const [expanded, setExpanded] = useState(rank <= 3);
  const [feedbackSent, setFeedbackSent] = useState(false);

  const style = RISK_STYLES[zone.risk_level] ?? RISK_STYLES.LOW;

  const patrolNote = `Zone ${zone.zone_id}: ${zone.risk_level} risk (${(zone.risk_score * 100).toFixed(0)}%). Watch for ${zone.top_crime_types
    .slice(0, 2)
    .map((c) => c.type.replace(/_/g, " "))
    .join(", ")}. ${zone.shap_drivers[0]?.explanation ?? ""}`;

  const copyNote = () => navigator.clipboard?.writeText(patrolNote);

  const sendFeedback = async (confirmed: boolean) => {
    await submitFeedback({
      zone_id: zone.zone_id,
      shift_date: new Date().toISOString().split("T")[0],
      shift: new Date().getHours() < 14 ? "morning" : new Date().getHours() < 22 ? "afternoon" : "night",
      incident_confirmed: confirmed,
    });
    setFeedbackSent(true);
  };

  return (
    <div
      className={`rounded-xl border bg-slate-900/80 ${style.border} p-4 transition-all duration-200 cursor-pointer hover:bg-slate-800/80`}
      onClick={() => setExpanded((p) => !p)}
    >
      {/* Header */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="text-slate-400 text-sm font-mono">#{rank}</span>
          {style.icon}
          <span className="font-semibold text-slate-100">{zone.zone_id}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className={`px-2 py-0.5 rounded-full text-xs font-bold ${style.badge}`}>
            {zone.risk_level}
          </span>
          <span className="text-slate-200 font-mono text-sm">
            {(zone.risk_score * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      {/* Crime types mini-bar */}
      <div className="mt-2 flex gap-2 flex-wrap">
        {zone.top_crime_types.slice(0, 3).map((c) => (
          <span key={c.type} className={`text-xs px-2 py-0.5 rounded text-slate-300 ${
            c.type === "domestic_violence" || c.type === "sexual_assault"
              ? "bg-pink-900/60 text-pink-300"
              : c.type === "child_crime"
              ? "bg-purple-900/60 text-purple-300"
              : "bg-slate-700"
          }`}>
            {c.type.replace(/_/g, " ")} · {(c.probability * 100).toFixed(0)}%
          </span>
        ))}
      </div>

      {/* Enriched NCRB alert badges */}
      {expanded && (
        <div className="mt-2 flex gap-2 flex-wrap">
          {(zone as any).women_safety_index > 400 && (
            <span className="flex items-center gap-1 text-xs px-2 py-0.5 rounded-full bg-pink-900/50 text-pink-300 border border-pink-700/40">
              <ShieldAlert className="w-3 h-3" /> Women Safety Alert
            </span>
          )}
          {(zone as any).police_coverage_ratio < 0.7 && (
            <span className="flex items-center gap-1 text-xs px-2 py-0.5 rounded-full bg-amber-900/50 text-amber-300 border border-amber-700/40">
              <Users className="w-3 h-3" /> Understaffed
            </span>
          )}
          {(zone as any).state_auto_theft_count > 200000 && (
            <span className="flex items-center gap-1 text-xs px-2 py-0.5 rounded-full bg-blue-900/50 text-blue-300 border border-blue-700/40">
              <Car className="w-3 h-3" /> High Auto Theft State
            </span>
          )}
        </div>
      )}

      {/* Expanded: SHAP drivers + feedback */}
      {expanded && (
        <div className="mt-3 space-y-2" onClick={(e) => e.stopPropagation()}>
          {zone.shap_drivers.length > 0 && (
            <div>
              <p className="text-xs text-slate-400 uppercase tracking-wider mb-1">Risk drivers</p>
              {zone.shap_drivers.map((d, i) => (
                <div key={i} className="flex items-start gap-2 text-xs text-slate-300 mt-1">
                  <Info className="w-3 h-3 mt-0.5 shrink-0 text-slate-400" />
                  <span
                    className={d.direction === "increases_risk" ? "text-orange-300" : "text-green-300"}
                  >
                    {d.explanation}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Enriched NCRB metrics */}
          {((zone as any).women_safety_index > 0 || (zone as any).police_coverage_ratio) && (
            <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-slate-400 border-t border-slate-700/50 pt-2">
              {(zone as any).women_safety_index > 0 && (
                <span>Women safety index: <b className="text-slate-300">{((zone as any).women_safety_index as number).toFixed(0)}</b></span>
              )}
              {(zone as any).police_coverage_ratio && (
                <span>Police coverage: <b className={(zone as any).police_coverage_ratio < 0.7 ? "text-amber-300" : "text-slate-300"}>{((zone as any).police_coverage_ratio as number * 100).toFixed(0)}%</b></span>
              )}
              {(zone as any).vulnerability_index > 0 && (
                <span>Vulnerability index: <b className="text-slate-300">{((zone as any).vulnerability_index as number).toFixed(0)}</b></span>
              )}
              {(zone as any).property_value_stolen_lakh > 0 && (
                <span>Property value stolen: <b className="text-slate-300">₹{(((zone as any).property_value_stolen_lakh as number) / 100).toFixed(0)}Cr</b></span>
              )}
            </div>
          )}

          <div className="flex gap-2 mt-3 flex-wrap">
            <button
              onClick={copyNote}
              className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs bg-slate-700 hover:bg-slate-600 text-slate-200 transition"
            >
              <Copy className="w-3 h-3" /> Copy patrol note
            </button>

            {!feedbackSent ? (
              <>
                <button
                  onClick={() => sendFeedback(true)}
                  className="px-3 py-1.5 rounded-lg text-xs bg-green-800/50 hover:bg-green-700/60 text-green-300 transition"
                >
                  Confirm incident
                </button>
                <button
                  onClick={() => sendFeedback(false)}
                  className="px-3 py-1.5 rounded-lg text-xs bg-slate-700 hover:bg-slate-600 text-slate-300 transition"
                >
                  False alarm
                </button>
              </>
            ) : (
              <span className="text-xs text-green-400 flex items-center gap-1">
                <CheckCircle className="w-3 h-3" /> Feedback recorded
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
