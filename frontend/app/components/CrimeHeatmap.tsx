"use client";

import { useEffect, useRef } from "react";
import type { ZonePrediction } from "@/lib/api";

const RISK_COLOURS: Record<string, string> = {
  HIGH: "#ef4444",
  MEDIUM: "#f97316",
  LOW: "#22c55e",
};

interface Props {
  zones: ZonePrediction[];
  city: string;
}

const CITY_CENTERS: Record<string, [number, number]> = {
  Bengaluru: [12.9716, 77.5946],
  Hyderabad: [17.385, 78.4867],
  Mumbai: [19.076, 72.8777],
  Delhi: [28.6139, 77.209],
  Chennai: [13.0827, 80.2707],
};

export default function CrimeHeatmap({ zones, city }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<unknown>(null);
  const markersRef = useRef<unknown[]>([]);

  useEffect(() => {
    if (typeof window === "undefined" || !containerRef.current) return;

    // Dynamically import Leaflet (SSR safe)
    import("leaflet").then((L) => {
      if (mapRef.current) return; // already initialised

      const center = CITY_CENTERS[city] ?? [20.5937, 78.9629];
      const map = L.map(containerRef.current!, { zoomControl: true, attributionControl: false }).setView(center, 12);

      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 18,
        className: "map-tiles",
      }).addTo(map);

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

  // Update markers when zones change
  useEffect(() => {
    if (!mapRef.current || typeof window === "undefined") return;

    import("leaflet").then((L) => {
      const map = mapRef.current as ReturnType<typeof L.map>;

      // Remove old markers
      markersRef.current.forEach((m) => (m as { remove: () => void }).remove());
      markersRef.current = [];

      zones.forEach((zone) => {
        if (!zone.lat || !zone.lon) return;

        const colour = RISK_COLOURS[zone.risk_level] ?? "#22c55e";
        const radius = zone.risk_level === "HIGH" ? 600 : zone.risk_level === "MEDIUM" ? 400 : 250;

        const circle = L.circle([zone.lat, zone.lon], {
          color: colour,
          fillColor: colour,
          fillOpacity: zone.risk_level === "HIGH" ? 0.35 : 0.2,
          weight: 2,
          radius,
        }).addTo(map);

        const crimes = zone.top_crime_types
          .slice(0, 2)
          .map((c) => `${c.type.replace(/_/g, " ")} (${(c.probability * 100).toFixed(0)}%)`)
          .join(", ");

        circle.bindPopup(
          `<div style="font-family:sans-serif;font-size:13px;min-width:200px;">
            <strong>${zone.zone_id}</strong><br/>
            Risk: <strong style="color:${colour}">${zone.risk_level} (${(zone.risk_score * 100).toFixed(0)}%)</strong><br/>
            <em>${crimes}</em><br/>
            ${zone.shap_drivers[0]?.explanation ?? ""}
          </div>`
        );

        markersRef.current.push(circle);
      });

      // Re-center on city
      const center = CITY_CENTERS[city];
      if (center) map.setView(center, 12, { animate: true });
    });
  }, [zones, city]);

  return (
    <div
      ref={containerRef}
      className="w-full h-full rounded-xl overflow-hidden"
      style={{ minHeight: 400 }}
    />
  );
}
