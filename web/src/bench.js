import * as THREE from "three";

// GPU time of a block of GL calls via EXT_disjoint_timer_query_webgl2 (null when unsupported:
// most phones and every iOS browser). Results arrive a few frames late; poll() returns the latest.
export function gpuTimer(gl) {
  const ext = gl.getExtension("EXT_disjoint_timer_query_webgl2");
  if (!ext) return null;
  const pending = [];
  let q = null, last = null;
  return {
    begin() { q = gl.createQuery(); gl.beginQuery(ext.TIME_ELAPSED_EXT, q); },
    end() { gl.endQuery(ext.TIME_ELAPSED_EXT); pending.push(q); },
    poll() {
      while (pending.length && gl.getQueryParameter(pending[0], gl.QUERY_RESULT_AVAILABLE)) {
        const p = pending.shift();
        if (!gl.getParameter(ext.GPU_DISJOINT_EXT)) last = gl.getQueryParameter(p, gl.QUERY_RESULT) / 1e6;
        gl.deleteQuery(p);
      }
      return last;
    },
  };
}

// Wall-clock ms per render of `renderFn` into an offscreen size x size target. Rendering `reps`
// times then reading back 1 pixel forces the GPU to finish, so vsync never caps the result.
export async function bench(renderer, renderFn, { seconds = 3, size = 1024, reps = 20 } = {}) {
  const rt = new THREE.WebGLRenderTarget(size, size);
  const px = new Uint8Array(4);
  const prevTarget = renderer.getRenderTarget(), prevScissor = renderer.getScissorTest();
  renderer.setScissorTest(false);
  renderer.setRenderTarget(rt);
  const sync = () => renderer.readRenderTargetPixels(rt, 0, 0, 1, 1, px);
  renderFn(); sync(); // warm-up: shader compile, uploads
  let n = 0;
  const t0 = performance.now();
  while (performance.now() - t0 < seconds * 1000) {
    for (let i = 0; i < reps; i++) renderFn();
    sync();
    n += reps;
    await new Promise((r) => setTimeout(r, 0)); // keep the page responsive
  }
  const ms = (performance.now() - t0) / n;
  renderer.setRenderTarget(prevTarget);
  renderer.setScissorTest(prevScissor);
  rt.dispose();
  return ms;
}
