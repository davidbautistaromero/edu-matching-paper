"""
Convierte parques_bogota.geojson (formato ESRI JSON, coordenadas MAGNA-SIRGAS Bogota)
a GeoJSON estandar con geometria Point en WGS84.

El raw viene del servicio de datos abiertos Bogota. Las geometrias son Polygon
en proyeccion local. Este script extrae el centroide de cada poligono y
lo reproyecta a lon/lat WGS84.

Input:  data/raw/parques_bogota.geojson
Output: data/processed/parques_clean.geojson
"""

import json
import math
from pathlib import Path

RAW_IN   = Path(__file__).parent.parent / "data" / "raw"       / "parques_bogota.geojson"
PROC_OUT = Path(__file__).parent.parent / "data" / "processed" / "parques_clean.geojson"

# Parametros de proyeccion MAGNA-SIRGAS Bogota (del WKT del servicio)
FALSE_EASTING  =  92_334.879
FALSE_NORTHING = 109_320.965
CENTRAL_MERID  = -74.14659167
LAT_ORIGIN     =   4.680486111
SCALE_FACTOR   =   1.0
A              = 6_380_687.0
F_INV          = 298.257222101
B              = A * (1 - 1/F_INV)
E2             = 1 - (B/A)**2


def transverse_mercator_to_latlon(x: float, y: float) -> tuple[float, float]:
    """Convierte TM MAGNA-SIRGAS Bogota a lon/lat WGS84 (precision ~1m)."""
    e2 = E2
    e4 = e2**2
    e6 = e2**3
    x0 = x - FALSE_EASTING
    y0 = y - FALSE_NORTHING
    phi0 = math.radians(LAT_ORIGIN)
    lam0 = math.radians(CENTRAL_MERID)
    M0 = A * (
        (1 - e2/4 - 3*e4/64 - 5*e6/256) * phi0
        - (3*e2/8 + 3*e4/32 + 45*e6/1024) * math.sin(2*phi0)
        + (15*e4/256 + 45*e6/1024) * math.sin(4*phi0)
        - (35*e6/3072) * math.sin(6*phi0)
    )
    M = M0 + y0 / SCALE_FACTOR
    mu = M / (A * (1 - e2/4 - 3*e4/64 - 5*e6/256))
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    phi1 = (mu
            + (3*e1/2 - 27*e1**3/32) * math.sin(2*mu)
            + (21*e1**2/16 - 55*e1**4/32) * math.sin(4*mu)
            + (151*e1**3/96) * math.sin(6*mu)
            + (1097*e1**4/512) * math.sin(8*mu))
    sin_phi1 = math.sin(phi1)
    cos_phi1 = math.cos(phi1)
    tan_phi1 = math.tan(phi1)
    N1 = A / math.sqrt(1 - e2 * sin_phi1**2)
    T1 = tan_phi1**2
    C1 = e2 / (1 - e2) * cos_phi1**2
    R1 = A * (1 - e2) / (1 - e2 * sin_phi1**2)**1.5
    D  = x0 / (N1 * SCALE_FACTOR)
    lat = phi1 - (N1 * tan_phi1 / R1) * (
        D**2/2
        - (5 + 3*T1 + 10*C1 - 4*C1**2 - 9*e2/(1-e2)) * D**4/24
        + (61 + 90*T1 + 298*C1 + 45*T1**2 - 252*e2/(1-e2) - 3*C1**2) * D**6/720
    )
    lon = lam0 + (
        D
        - (1 + 2*T1 + C1) * D**3/6
        + (5 - 2*C1 + 28*T1 - 3*C1**2 + 8*e2/(1-e2) + 24*T1**2) * D**5/120
    ) / cos_phi1
    return math.degrees(lon), math.degrees(lat)


def centroid_of_ring(ring: list) -> tuple[float, float]:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def main():
    PROC_OUT.parent.mkdir(parents=True, exist_ok=True)

    print(f"Leyendo {RAW_IN.name}...")
    with open(RAW_IN, encoding="utf-8") as f:
        data = json.load(f)

    esri_features = data.get("features", [])
    print(f"  Total parques en raw: {len(esri_features):,}")

    print("Convirtiendo coordenadas MAGNA-SIRGAS -> WGS84...")
    geojson_features = []
    skipped = 0

    for feat in esri_features:
        attrs = feat.get("attributes", {})
        geom  = feat.get("geometry", {})

        rings = geom.get("rings", []) if geom else []
        if not rings:
            skipped += 1
            continue

        cx, cy = centroid_of_ring(rings[0])
        try:
            lon, lat = transverse_mercator_to_latlon(cx, cy)
        except Exception:
            skipped += 1
            continue

        # Validar bbox Bogota
        if not (4.3 < lat < 5.0 and -74.5 < lon < -73.8):
            skipped += 1
            continue

        geojson_features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [round(lon, 7), round(lat, 7)],
            },
            "properties": {k: v for k, v in attrs.items() if k != "OBJECTID"},
        })

    out_gj = {"type": "FeatureCollection", "features": geojson_features}
    with open(PROC_OUT, "w", encoding="utf-8") as f:
        json.dump(out_gj, f, ensure_ascii=False)

    print(f"Guardado: {PROC_OUT}")
    print(f"  Parques validos: {len(geojson_features):,} | Omitidos: {skipped}")
    if geojson_features:
        s = geojson_features[0]
        print(f"  Ejemplo: {s['properties'].get('NOMBRE')} @ {s['geometry']['coordinates']}")


if __name__ == "__main__":
    main()
