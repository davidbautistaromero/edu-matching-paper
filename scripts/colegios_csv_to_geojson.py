"""
Convierte colegios_dataset.csv a GeoJSON (EPSG:4326).
Coordenadas fuente: coord_x (longitud) y coord_y (latitud) con coma decimal.
"""

import json
import re
import unicodedata
import pandas as pd
from pathlib import Path

INPUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "colegios_dataset.csv"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "processed" / "colegios_dataset.geojson"

# Renombrado para columnas que quedan largas tras la normalización automática
RENAME = {
    "objectid":                                      "objectid",
    "nombre_del_establecimiento_educativo":          "nombre_establecimiento",
    "id_del_establecimiento_educativo":              "id_establecimiento",
    "nombre_de_la_sede_educativa":                   "nombre_sede",
    "id_de_la_sede_educativa":                       "id_sede",
    "codigo_de_planta_fisica":                       "codigo_planta_fisica",
    "fecha_de_la_informacion":                       "fecha_informacion",
    "naturaleza_juridica":                           "naturaleza_juridica",
    "orden_de_la_sede":                              "orden_sede",
    "regimen_y_categoria_de_costos":                 "regimen_categoria_costos",
    "nombre_de_la_localidad":                        "nombre_localidad",
    "nombre_de_la_upz":                              "nombre_upz",
    "especialidad_para_la_media":                    "especialidad_media",
    "caracter_para_la_media":                        "caracter_media",
    "enfasis_para_el_caracter_academico_de_la_media": "enfasis_caracter_academico",
    "tipo_de_discapacidad":                          "tipo_discapacidad",
    "talentos_o_capacidades_excepcionales":          "talentos_capacidades_excepcionales",
    "grupos_etnicos":                                "grupos_etnicos",
}


def normalize_col(name: str) -> str:
    name = name.lstrip("\ufeff").strip()
    name = unicodedata.normalize("NFD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = [normalize_col(c) for c in df.columns]
    renamed = [RENAME.get(n, n) for n in normalized]
    df.columns = renamed
    return df


def parse_coord(value) -> float | None:
    try:
        return float(str(value).replace(",", "."))
    except (ValueError, TypeError):
        return None


def row_to_feature(row: dict) -> dict | None:
    lon = parse_coord(row.get("coord_x"))
    lat = parse_coord(row.get("coord_y"))
    if lon is None or lat is None:
        return None
    properties = {k: (None if pd.isna(v) else v) for k, v in row.items()
                  if k not in ("coord_x", "coord_y")}
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": properties,
    }


def main():
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig", dtype=str, encoding_errors="replace")
    df = clean_columns(df)
    df = df[df["sector"] == "Oficial"]
    print(f"Registros leídos: {len(df):,}")
    print("Columnas:", list(df.columns))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    features = []
    skipped = 0
    for row in df.to_dict(orient="records"):
        feature = row_to_feature(row)
        if feature:
            features.append(feature)
        else:
            skipped += 1

    geojson = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }

    OUTPUT_PATH.write_text(json.dumps(geojson, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Features exportadas: {len(features):,}  |  Omitidas (sin coords): {skipped}")
    print(f"Guardado en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
