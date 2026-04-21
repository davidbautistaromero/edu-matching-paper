"""
Limpia el GeoJSON de paraderos SITP.

El archivo raw tiene estructura ESRI no estandar:
  feature = {"attributes": {...}, "geometry": {"x": lon, "y": lat}}

Este script lo convierte a GeoJSON estandar con geometria Point valida.

Input:  data/raw/paraderos_sitp.geojson
Output: data/processed/sitp_clean.geojson
"""

import json
from pathlib import Path

RAW  = Path(__file__).parent.parent / "data" / "raw"  / "paraderos_sitp.geojson"
OUT  = Path(__file__).parent.parent / "data" / "processed" / "sitp_clean.geojson"


def main():
    with open(RAW, encoding="utf-8") as f:
        gj = json.load(f)

    total = len(gj["features"])
    print(f"Features raw: {total:,}")

    features = []
    skipped = 0

    for feat in gj["features"]:
        g = feat.get("geometry") or {}
        props = feat.get("attributes") or feat.get("properties") or {}

        x = g.get("x")
        y = g.get("y")

        if x is None or y is None:
            skipped += 1
            continue

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(x), float(y)],
            },
            "properties": props,
        })

    out_gj = {"type": "FeatureCollection", "features": features}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out_gj, f, ensure_ascii=False)

    print(f"Guardado: {OUT}")
    print(f"  Features validas: {len(features):,} | Omitidas sin coords: {skipped}")


if __name__ == "__main__":
    main()
