"""KSC landing-site coordinates for the alien-KSC client feature.

Ported from ``future_expansions/AP_KSC_Sites/generate.py`` (the offline tool
that builds the Kerbal Konstructs static ``.cfg`` files). Each entry is the
landing coordinate where the cloned KSC cluster is placed on that body:
latitude/longitude plus the terrain altitude at that spot, and whether the
map-decal terrain flattening is skipped.

This is the source of truth for the ``ksc_site`` slot_data row. The C# client
no longer carries a per-body table — ``fill_slot_data`` emits the chosen
starting body's row and the client materialises the cluster from it.

Kerbin is intentionally absent: a Kerbin start uses the stock KSC and needs
no row.
"""
from __future__ import annotations

from .bodies import BodyName


# body -> (lat_deg, lon_deg, terrain_alt_m, skip_map_decal)
KSC_SITES: dict[BodyName, tuple[float, float, float, bool]] = {
    BodyName.TYLO:   (  0.0,  -30.0,  588.2, False),
    BodyName.MOHO:   (  0.0,  163.0, 1055.0, False),
    BodyName.POL:    (  0.0, -112.0,  969.5, False),
    BodyName.LAYTHE: (  0.0, -163.0,  731.8, False),
    BodyName.DRES:   (  0.0, -164.0,  224.4, False),
    BodyName.BOP:    (  0.0, -137.0, 6416.2, False),
    BodyName.EELOO:  (  0.0,  115.0, 1535.9, False),
    BodyName.MUN:    (  0.0, -111.0, 2908.8, False),
    # Greater Flats is already perfectly flat — skip the map decal.
    BodyName.MINMUS: (  0.0,  -17.0,    0.0, True),
    # Off-equator mesa: +536 m vs the equator pick, thinner atmosphere helps
    # Eve ascent (the body where ascent dV matters most).
    BodyName.EVE:    (-25.0, -159.0, 6140.0, False),
    BodyName.IKE:    (  0.0,   73.0, 3179.0, False),
    BodyName.VALL:   (  0.0,  -64.0, 1222.2, False),
    BodyName.GILLY:  (  0.0,   30.0, 3925.5, False),
    BodyName.DUNA:   (  0.0,  -19.0,  417.9, False),
}


def ksc_site_slot_data(home: BodyName) -> dict | None:
    """The ``ksc_site`` slot_data row for ``home``, or ``None`` for a Kerbin
    start (stock KSC, no row) or any body without a defined site."""
    site = KSC_SITES.get(home)
    if site is None:
        return None
    lat, lon, terrain_alt, skip_decal = site
    return {
        "lat": lat,
        "lon": lon,
        "terrain_alt": terrain_alt,
        "skip_decal": skip_decal,
    }
