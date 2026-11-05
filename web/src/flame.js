// Deterministic flicker. Each channel is a weighted sum of sines with weights summing to 1,
// so |value| <= amplitude: the flame never leaves the range the network was trained on.
const wave = (t, f1, f2, f3, p) =>
  0.5 * Math.sin(f1 * t + p) + 0.3 * Math.sin(f2 * t + 2.1 * p) + 0.2 * Math.sin(f3 * t + 0.7);

export function flameOffset(t, cfg) {
  const [iMin, iMax] = cfg.intensity_range;
  return {
    x: cfg.flicker_xz * wave(t, 2.3, 5.7, 13.1, 0.0),
    y: cfg.flicker_y * wave(t, 3.1, 7.9, 17.3, 1.3),
    z: cfg.flicker_xz * wave(t, 1.9, 6.3, 11.7, 2.6),
    intensity: (iMin + iMax) / 2 + (iMax - iMin) / 2 * wave(t, 4.7, 9.3, 23.9, 0.4),
  };
}
