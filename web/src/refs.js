const RAD = Math.PI / 180;

// Unit direction from the orbit target to the camera. Same convention as pipeline/scene.py:
// az around +Y starting at +Z (az=90 -> +X), el above the horizontal.
export function viewDir({ az_deg, el_deg }) {
  const a = az_deg * RAD, e = el_deg * RAD;
  return [Math.cos(e) * Math.sin(a), Math.sin(e), Math.cos(e) * Math.cos(a)];
}

export function nearestRefView(dir, refs) {
  let best = 0, bestDot = -Infinity;
  refs.forEach((v, i) => {
    const d = viewDir(v);
    const dot = d[0] * dir[0] + d[1] * dir[1] + d[2] * dir[2];
    if (dot > bestDot) { bestDot = dot; best = i; }
  });
  return best;
}

// OrbitControls limits = the domain the network was trained on.
export function orbitLimits(cfg) {
  const [elMin, elMax] = cfg.camera.elevation_deg;
  const [rMin, rMax] = cfg.camera.radius;
  return {
    minPolarAngle: Math.PI / 2 - elMax * RAD,
    maxPolarAngle: Math.PI / 2 - elMin * RAD,
    minDistance: rMin,
    maxDistance: rMax,
  };
}

// The user moved the camera off the reference pose: "Path tracé" sides fall back to neural.
export const leaveRef = (sides) =>
  Object.fromEntries(Object.entries(sides).map(([k, v]) => [k, v === "ref" ? "neural" : v]));

// Run fn with the camera at a fixed pose, then restore it (cost scales with candle screen
// coverage, so benchmarks must not depend on where the user left the camera).
export async function withPose(camera, position, target, fn) {
  const saved = camera.position.clone();
  camera.position.copy(position);
  camera.lookAt(target);
  try {
    return await fn();
  } finally {
    camera.position.copy(saved);
    camera.lookAt(target);
  }
}
