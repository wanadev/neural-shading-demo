"""Single source of truth shared by Mitsuba and three.js (exported to web/data/config.json).
Units: metres. Y up. Candle base at y=0."""

CANDLE = dict(radius=0.035, height=0.10, cup_depth=0.006, rim_radius=0.003,
              wobble=0.0008, seed=7, segments=128, rings=96)
FLAME = dict(base=(0.0, 0.110, 0.0), sphere_radius=0.003, radiance=(40.0, 22.0, 8.0),
             flicker_xz=0.003, flicker_y=0.002, intensity_range=(0.8, 1.2))
WAX = dict(ior=1.44, roughness=0.1, sigma_t=800.0, albedo=(0.995, 0.99, 0.975))
TABLE = dict(size=1.0, albedo=(0.08, 0.06, 0.05))
CAMERA = dict(fov_y=35.0, target=(0.0, 0.06, 0.0), radius=(0.2, 0.5), elevation_deg=(-5.0, 60.0))
EXPOSURE = 1.0
REF_VIEWS = [dict(az_deg=30, el_deg=15, radius=0.3), dict(az_deg=120, el_deg=35, radius=0.3),
             dict(az_deg=220, el_deg=5, radius=0.3), dict(az_deg=300, el_deg=55, radius=0.3)]
WIDTHS = (8, 16, 32)


def to_json() -> dict:
    return dict(candle=CANDLE, flame=FLAME, wax=WAX, table=TABLE, camera=CAMERA,
                exposure=EXPOSURE, ref_views=REF_VIEWS, widths=list(WIDTHS))
