"use client";

import { useState } from "react";
import { AlertTriangle, CheckCircle, Info, Copy, ShieldAlert, Users, Car } from "lucide-react";
import type { ZonePrediction } from "@/lib/api";
import { submitFeedback } from "@/lib/api";

const RISK_STYLES = {
  HIGH: { border: "border-red-300", badge: "bg-red-50 text-red-700", icon: <AlertTriangle className="w-4 h-4 text-red-600" /> },
  MEDIUM: { border: "border-amber-300", badge: "bg-amber-50 text-amber-700", icon: <AlertTriangle className="w-4 h-4 text-amber-600" /> },
  LOW: { border: "border-emerald-300", badge: "bg-emerald-50 text-emerald-700", icon: <CheckCircle className="w-4 h-4 text-emerald-600" /> },
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
      className={`rounded-xl border bg-white ${style.border} p-4 transition-all duration-200 cursor-pointer hover:bg-slate-50 shadow-sm`}
      onClick={() => setExpanded((p) => !p)}
    >
      {/* Header */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="text-slate-500 text-sm font-mono">#{rank}</span>
          {style.icon}
          <span className="font-semibold text-slate-900">{zone.zone_id}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className={`px-2 py-0.5 rounded-full text-xs font-bold ${style.badge}`}>
            {zone.risk_level}
          </span>
          <span className="text-slate-700 font-mono text-sm">
            {(zone.risk_score * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      {/* Crime types mini-bar */}
      <div className="mt-2 flex gap-2 flex-wrap">
        {zone.top_crime_types.slice(0, 3).map((c) => (
          <span key={c.type} className={`text-xs px-2 py-0.5 rounded text-slate-700 ${
            c.type === "domestic_violence" || c.type === "sexual_assault"
              ? "bg-pink-100 text-pink-700"
              : c.type === "child_crime"
              ? "bg-purple-100 text-purple-700"
              : "bg-slate-100"
          }`}>
            {c.type.replace(/_/g, " ")} · {(c.probability * 100).toFixed(0)}%
          </span>
        ))}
      </div>

      {/* Enriched NCRB alert badges */}
      {expanded && (
        <div className="mt-2 flex gap-2 flex-wrap">
          {(zone as any).women_safety_index > 400 && (
            <span className="flex items-center gap-1 text-xs px-2 py-0.5 rounded-full bg-pink-50 text-pink-700 border border-pink-200">
              <ShieldAlert className="w-3 h-3" /> Women Safety Alert
            </span>
          )}
          {(zone as any).police_coverage_ratio < 0.7 && (
            <span className="flex items-center gap-1 text-xs px-2 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200">
              <Users className="w-3 h-3" /> Understaffed
            </span>
          )}
          {(zone as any).state_auto_theft_count > 200000 && (
            <span className="flex items-center gap-1 text-xs px-2 py-0.5 rounded-full bg-blue-50 text-blue-700 border border-blue-200">
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
              <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">Risk drivers</p>
              {zone.shap_drivers.map((d, i) => (
                <div key={i} className="flex items-start gap-2 text-xs text-slate-700 mt-1">
                  <Info className="w-3 h-3 mt-0.5 shrink-0 text-slate-500" />
                  <span
                    className={d.direction === "increases_risk" ? "text-amber-700" : "text-emerald-700"}
                  >
                    {d.explanation}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Enriched NCRB metrics */}
          {((zone as any).women_safety_index > 0 || (zone as any).police_coverage_ratio) && (
            <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-slate-600 border-t border-slate-200 pt-2">
              {(zone as any).women_safety_index > 0 && (
                <span>Women safety index: <b className="text-slate-800">{((zone as any).women_safety_index as number).toFixed(0)}</b></span>
              )}
              {(zone as any).police_coverage_ratio && (
                <span>Police coverage: <b className={(zone as any).police_coverage_ratio < 0.7 ? "text-amber-700" : "text-slate-800"}>{((zone as any).police_coverage_ratio as number * 100).toFixed(0)}%</b></span>
              )}
              {(zone as any).vulnerability_index > 0 && (
                <span>Vulnerability index: <b className="text-slate-800">{((zone as any).vulnerability_index as number).toFixed(0)}</b></span>
              )}
              {(zone as any).property_value_stolen_lakh > 0 && (
                <span>Property value stolen: <b className="text-slate-800">₹{(((zone as any).property_value_stolen_lakh as number) / 100).toFixed(0)}Cr</b></span>
              )}
            </div>
          )}

          <div className="flex gap-2 mt-3 flex-wrap">
            <button
              onClick={copyNote}
              className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200 transition"
            >
              <Copy className="w-3 h-3" /> Copy patrol note
            </button>

            {!feedbackSent ? (
              <>
                <button
                  onClick={() => sendFeedback(true)}
                  className="px-3 py-1.5 rounded-lg text-xs bg-emerald-100 hover:bg-emerald-200 text-emerald-800 border border-emerald-200 transition"
                >
                  Confirm incident
                </button>
                <button
                  onClick={() => sendFeedback(false)}
                  className="px-3 py-1.5 rounded-lg text-xs bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200 transition"
                >
                  False alarm
                </button>
              </>
            ) : (
              <span className="text-xs text-emerald-700 flex items-center gap-1">
                <CheckCircle className="w-3 h-3" /> Feedback recorded
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
