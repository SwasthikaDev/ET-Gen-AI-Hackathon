"use client";

import { useEffect, useRef } from "react";
import type { ZonePrediction } from "@/lib/api";

const RISK_COLOUR: Record<string, string> = {
  HIGH: "#ef4444",
  MEDIUM: "#f97316",
  LOW: "#22c55e",
};
const RISK_FILL_OPACITY: Record<string, number> = { HIGH: 0.45, MEDIUM: 0.3, LOW: 0.18 };
const RISK_RADIUS: Record<string, number> = { HIGH: 700, MEDIUM: 460, LOW: 280 };
const RISK_WEIGHT: Record<string, number> = { HIGH: 2.5, MEDIUM: 1.8, LOW: 1.2 };

const CITY_CENTERS: Record<string, [number, number]> = {
  Bengaluru: [12.9716, 77.5946],
  Hyderabad: [17.385, 78.4867],
  Mumbai: [19.076, 72.8777],
  Delhi: [28.6139, 77.209],
  Chennai: [13.0827, 80.2707],
};

interface Props {
  zones: ZonePrediction[];
  city: string;
}

export default function CrimeHeatmap({ zones, city }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<unknown>(null);
  const markersRef = useRef<unknown[]>([]);

  // ── Initialise map once ──────────────────────────────────────────────────
  useEffect(() => {
    if (typeof window === "undefined" || !containerRef.current) return;

    import("leaflet").then((L) => {
      if (mapRef.current) return;

      const center = CITY_CENTERS[city] ?? [20.5937, 78.9629];
      const map = L.map(containerRef.current!, {
        zoomControl: true,
        attributionControl: true,
      }).setView(center, 12);

      // ESRI satellite imagery (free, no API key)
      const satellite = L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        {
          maxZoom: 19,
          attribution:
            "Tiles &copy; Esri &mdash; Source: Esri, USGS, NGA, NASA, CGIAR, NLS, OS, NMA, Geodatastyrelsen, Rijkswaterstaat, GSA, Geoland, FEMA, Intermap and the GIS User Community",
        }
      );

      // ESRI city labels overlay (sits on top of satellite)
      const labels = L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        { maxZoom: 19, pane: "overlayPane", opacity: 0.9 }
      );

      // OSM street map (alternate base)
      const streets = L.tileLayer(
        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        { maxZoom: 19, attribution: "&copy; OpenStreetMap contributors" }
      );

      satellite.addTo(map);
      labels.addTo(map);

      // Layer control top-right
      L.control
        .layers(
          { "🛰 Satellite": satellite, "🗺 Street map": streets },
          { "🏷 Place labels": labels },
          { position: "topright", collapsed: false }
        )
        .addTo(map);

      // Risk legend bottom-right
      const legend = (L.control as any)({ position: "bottomright" });
      legend.onAdd = () => {
        const div = L.DomUtil.create("div");
        div.innerHTML = `
          <div style="
            background:rgba(255,255,255,0.97);padding:10px 14px;
            border-radius:10px;font-family:-apple-system,sans-serif;
            font-size:12px;border:1px solid #e2e8f0;
            box-shadow:0 4px 16px rgba(0,0,0,0.18);min-width:130px">
            <div style="font-weight:700;margin-bottom:7px;color:#0f172a;letter-spacing:.4px">RISK LEVEL</div>
            ${[
              ["HIGH", "#ef4444"],
              ["MEDIUM", "#f97316"],
              ["LOW", "#22c55e"],
            ]
              .map(
                ([label, c]) =>
                  `<div style="display:flex;align-items:center;gap:7px;margin-bottom:5px">
                    <span style="width:13px;height:13px;border-radius:50%;background:${c};display:inline-block;border:2px solid ${c}99"></span>
                    <span style="color:#0f172a;font-weight:600">${label}</span>
                  </div>`
              )
              .join("")}
          </div>`;
        return div;
      };
      legend.addTo(map);

      // Live stats overlay bottom-left
      const statsControl = (L.control as any)({ position: "bottomleft" });
      statsControl.onAdd = () => {
        const div = L.DomUtil.create("div");
        div.id = "map-stats-overlay";
        div.innerHTML = `<div style="
          background:rgba(15,23,42,0.88);color:#f8fafc;padding:8px 12px;
          border-radius:10px;font-family:-apple-system,sans-serif;font-size:11px;
          border:1px solid rgba(255,255,255,0.12);backdrop-filter:blur(4px);
          box-shadow:0 4px 16px rgba(0,0,0,0.3)">
          <span style="opacity:0.7">Loading predictions…</span>
        </div>`;
        return div;
      };
      statsControl.addTo(map);

      mapRef.current = map;
    });

    return () => {
      if (mapRef.current) {
        (mapRef.current as { remove: () => void }).remove();
        mapRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Update zone circles + popups when zones/city changes ─────────────────
  useEffect(() => {
    if (!mapRef.current || typeof window === "undefined") return;

    import("leaflet").then((L) => {
      const map = mapRef.current as ReturnType<typeof L.map>;

      markersRef.current.forEach((m) => (m as { remove: () => void }).remove());
      markersRef.current = [];

      zones.forEach((zone) => {
        if (!zone.lat || !zone.lon) return;

        const colour = RISK_COLOUR[zone.risk_level] ?? "#22c55e";
        const radius = RISK_RADIUS[zone.risk_level] ?? 300;

        const circle = L.circle([zone.lat, zone.lon], {
          color: colour,
          fillColor: colour,
          fillOpacity: RISK_FILL_OPACITY[zone.risk_level] ?? 0.2,
          weight: RISK_WEIGHT[zone.risk_level] ?? 1.5,
          radius,
        }).addTo(map);

        // Styled pill chips for crime types
        const crimePills = zone.top_crime_types
          .slice(0, 3)
          .map(
            (c) =>
              `<span style="background:${colour}22;color:${colour};border:1px solid ${colour}55;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:600;display:inline-block;margin:2px">${c.type.replace(/_/g, " ")} · ${(c.probability * 100).toFixed(0)}%</span>`
          )
          .join("");

        const driver = zone.shap_drivers[0]?.explanation ?? "";

        const extraAlerts = [
          (zone as any).women_safety_index > 400
            ? `<div style="margin-top:5px;padding:4px 8px;background:#fce7f3;border:1px solid #f9a8d4;border-radius:6px;color:#be185d;font-size:11px;font-weight:600">⚠ Women Safety Alert</div>`
            : "",
          (zone as any).police_coverage_ratio < 0.7
            ? `<div style="margin-top:4px;padding:4px 8px;background:#fffbeb;border:1px solid #fcd34d;border-radius:6px;color:#92400e;font-size:11px;font-weight:600">⚠ Police Understaffed</div>`
            : "",
          (zone as any).state_auto_theft_count > 200000
            ? `<div style="margin-top:4px;padding:4px 8px;background:#eff6ff;border:1px solid #93c5fd;border-radius:6px;color:#1e40af;font-size:11px;font-weight:600">🚗 High Auto Theft Region</div>`
            : "",
        ]
          .filter(Boolean)
          .join("");

        circle.bindPopup(
          `<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;min-width:230px;max-width:290px">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:5px">
              <span style="font-weight:700;font-size:14px;color:#0f172a">${zone.zone_id}</span>
              <span style="background:${colour}22;color:${colour};border:1px solid ${colour}55;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700">${zone.risk_level}</span>
            </div>
            <div style="font-size:13px;color:#475569;margin-bottom:7px">
              Risk score: <b style="color:${colour};font-size:16px">${(zone.risk_score * 100).toFixed(0)}%</b>
            </div>
            <div style="margin-bottom:7px;line-height:1.8">${crimePills}</div>
            ${driver ? `<div style="font-size:11px;color:#64748b;border-top:1px solid #f1f5f9;padding-top:6px;line-height:1.5">${driver}</div>` : ""}
            ${extraAlerts}
          </div>`,
          { maxWidth: 320, className: "crimewatch-popup" }
        );

        markersRef.current.push(circle);
      });

      // Update stats overlay
      const overlay = document.getElementById("map-stats-overlay");
      if (overlay && zones.length) {
        const high = zones.filter((z) => z.risk_level === "HIGH").length;
        const med = zones.filter((z) => z.risk_level === "MEDIUM").length;
        const low = zones.filter((z) => z.risk_level === "LOW").length;
        overlay.innerHTML = `
          <div style="
            background:rgba(15,23,42,0.88);color:#f8fafc;padding:8px 14px;
            border-radius:10px;font-family:-apple-system,sans-serif;font-size:12px;
            border:1px solid rgba(255,255,255,0.12);backdrop-filter:blur(4px);
            box-shadow:0 4px 16px rgba(0,0,0,0.3);display:flex;gap:12px;align-items:center">
            <span style="font-weight:700;opacity:.7">${city}</span>
            <span style="color:#ef4444;font-weight:700">● HIGH ${high}</span>
            <span style="color:#f97316;font-weight:700">● MED ${med}</span>
            <span style="color:#22c55e;font-weight:700">● LOW ${low}</span>
          </div>`;
      }

      const center = CITY_CENTERS[city];
      if (center) map.setView(center, 12, { animate: true });
    });
  }, [zones, city]);

  return (
    <div ref={containerRef} className="w-full h-full" style={{ minHeight: 460 }} />
  );
}
