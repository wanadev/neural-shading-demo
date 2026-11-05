"""Path-traced reference views for the 'Path tracé' side of the curtain.
python -m pipeline.render_refs --res 1024 --spp 4096"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import mitsuba as mi

from pipeline.assets import WEB_DATA
from pipeline.config import EXPOSURE, REF_VIEWS
from pipeline.scene import render
from pipeline.tonemap import aces_filmic, to_srgb8

OUT = Path(__file__).resolve().parent / "out"


def save_png(path, rgb_linear):
    mi.Bitmap(to_srgb8(aces_filmic(rgb_linear, EXPOSURE))).write(str(path))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--res", type=int, default=1024)
    p.add_argument("--spp", type=int, default=4096)
    p.add_argument("--views", type=int, nargs="*", default=list(range(len(REF_VIEWS))))
    a = p.parse_args(argv)
    (WEB_DATA / "refs").mkdir(parents=True, exist_ok=True)
    OUT.mkdir(exist_ok=True)
    for i in a.views:
        t = time.time()
        r = render(REF_VIEWS[i], np.zeros(3), a.res, a.spp)
        np.savez(OUT / f"ref{i}.npz", **r)
        save_png(WEB_DATA / "refs" / f"view{i}.png", r["full"])
        print(f"ref {i}: {time.time() - t:.0f}s", flush=True)
    (WEB_DATA / "refs.json").write_text(json.dumps(REF_VIEWS))


if __name__ == "__main__":
    main()
