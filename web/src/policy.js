// Small decisions the page makes about what it can show right now.

// Bloom needs a half-float render target; without it, render straight to the screen.
export const pickPipeline = ({ halfFloatRT }) => (halfFloatRT ? "hdr" : "ldr");

// A "Path tracé" side shows neural until its (lazy, several-MB) reference image has loaded.
export const sideMode = (requested, refReady) => (requested === "ref" && !refReady ? "neural" : requested);

// Keep the current width on screen until the requested width's latents are downloaded.
export const effectiveWidth = (requested, loadedWidths, current) => (loadedWidths.has(requested) ? requested : current);
