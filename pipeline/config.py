"""Single source of truth shared by Mitsuba and three.js (exported to web/data/config.json).
Units: metres. Y up. Dragon standing on the ground (y=0)."""
from pathlib import Path

WEB_DATA = Path(__file__).resolve().parent.parent / "web" / "data"
OUT = Path(__file__).resolve().parent / "out"

DRAGON = dict(url="https://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz",
              member="dragon_recon/dragon_vrip.ply", target_faces=80_000, height=0.15)
JADE = dict(ior=1.61, roughness=0.08, sigma_t=180.0, albedo=(0.35, 0.92, 0.55))
LAMP = dict(center=(0.0, 0.07, 0.0), radius=0.25, sphere_radius=0.008, radiance=(6000.0, 5400.0, 4400.0),
            elevation_deg=(-10.0, 70.0))
ENV = dict(name="anniversary_lounge", scale=0.25)
GROUND = dict(size=4.0, albedo=(0.03, 0.03, 0.03), roughness=0.3)
CAMERA = dict(fov_y=40.0, target=(0.0, 0.07, 0.0), radius=(0.25, 0.6), elevation_deg=(-3.0, 60.0))
EXPOSURE = 0.6
REF_POSES = [dict(az_deg=30, el_deg=15, radius=0.5, lamp_az_deg=60, lamp_el_deg=35),
             dict(az_deg=120, el_deg=10, radius=0.5, lamp_az_deg=300, lamp_el_deg=20),   # backlit
             dict(az_deg=220, el_deg=30, radius=0.5, lamp_az_deg=170, lamp_el_deg=50),
             dict(az_deg=300, el_deg=5, radius=0.5, lamp_az_deg=120, lamp_el_deg=15)]    # backlit
WIDTHS = (16, 32, 64)
LATENT_DIM = 8


def write_json():
    """python -m pipeline.config -> web/data/config.json (keeps thickness_max from pipeline.thickness)."""
    import json
    path = WEB_DATA / "config.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    keep = {k: v for k, v in old.items() if k == "thickness_max"}
    path.write_text(json.dumps({**to_json(), **keep}, indent=1))


def to_json() -> dict:
    return dict(dragon=dict(height=DRAGON["height"]), jade=JADE, lamp=LAMP, env=ENV, ground=GROUND,
                camera=CAMERA, exposure=EXPOSURE, ref_poses=REF_POSES, widths=list(WIDTHS),
                latent_dim=LATENT_DIM)


if __name__ == "__main__":
    write_json()
