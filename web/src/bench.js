import * as THREE from "three";

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

// Wall-clock ms per full frame (post-processing included) drawn by `frameFn` to the screen;
// a 1-pixel readback of the default framebuffer forces completion.
export async function benchFrame(renderer, frameFn, { seconds = 3, reps = 5 } = {}) {
  const gl = renderer.getContext(), px = new Uint8Array(4);
  const sync = () => gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
  frameFn(); sync();
  let n = 0;
  const t0 = performance.now();
  while (performance.now() - t0 < seconds * 1000) {
    for (let i = 0; i < reps; i++) frameFn();
    sync();
    n += reps;
    await new Promise((r) => setTimeout(r, 0));
  }
  return (performance.now() - t0) / n;
}
