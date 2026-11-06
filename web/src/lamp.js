import { viewDir } from "./refs.js";

// Lamp on a sphere around the dragon; same az/el convention as the camera and pipeline/scene.py.
export function lampPosition({ az_deg, el_deg }, cfg) {
  const d = viewDir({ az_deg, el_deg }), c = cfg.lamp.center, r = cfg.lamp.radius;
  return [c[0] + r * d[0], c[1] + r * d[1], c[2] + r * d[2]];
}

// Keep the lamp inside the domain the network was trained on.
export function clampLamp({ az_deg, el_deg }, cfg) {
  const [lo, hi] = cfg.lamp.elevation_deg;
  return { az_deg: ((az_deg % 360) + 360) % 360, el_deg: Math.min(hi, Math.max(lo, el_deg)) };
}

// "Contre-jour": lamp behind the dragon as seen from the camera, a bit above.
export const backlight = (camAzDeg) => ({ az_deg: (((camAzDeg + 180) % 360) + 360) % 360, el_deg: 20 });
