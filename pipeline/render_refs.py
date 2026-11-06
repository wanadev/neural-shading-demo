"""Path-traced reference poses (camera + lamp) for the 'Path tracé' side of the curtain.
Writes web/data/refs/view{i}.exr (linear RGBA half float, alpha = geometry; the page draws the blurred
room behind it and applies the same bloom + tone mapping as the live sides) and a tonemapped preview PNG.
python -m pipeline.render_refs --res 1024 --spp 1024"""
import argparse
import time

import numpy as np
import mitsuba as mi

from pipeline.config import EXPOSURE, OUT, REF_POSES, WEB_DATA
from pipeline.gen_data import save_atomic
from pipeline.scene import render
from pipeline.tonemap import aces_filmic, to_srgb8


def save_png(path, rgb_linear):
    mi.Bitmap(to_srgb8(aces_filmic(rgb_linear, EXPOSURE))).write(str(path))


def save_exr(path, rgba):
    tmp = path.with_name(path.stem + ".tmp.exr")
    mi.Bitmap(np.ascontiguousarray(rgba, np.float32), pixel_format=mi.Bitmap.PixelFormat.RGBA) \
        .convert(component_format=mi.Struct.Type.Float16).write(str(tmp))
    tmp.replace(path)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--res", type=int, default=1024)
    p.add_argument("--spp", type=int, default=1024)
    p.add_argument("--views", type=int, nargs="*", default=list(range(len(REF_POSES))))
    a = p.parse_args(argv)
    (WEB_DATA / "refs").mkdir(parents=True, exist_ok=True)
    OUT.mkdir(exist_ok=True)
    for i in a.views:
        pose, t = REF_POSES[i], time.time()
        r = render(pose, (pose["lamp_az_deg"], pose["lamp_el_deg"]), a.res, a.spp, passes=("both", "spec"))
        save_atomic(OUT / f"ref{i}.npz", **r)
        rgba = r["both"].copy()
        rgba[..., :3] += r["spec"]
        save_exr(WEB_DATA / "refs" / f"view{i}.exr", rgba)
        save_png(OUT / f"ref{i}_preview.png", rgba[..., :3])
        print(f"ref {i}: {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
